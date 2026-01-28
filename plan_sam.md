Plan: SAM2/SAM2.1 Integration for McByte                                                                     
                                                                                                              
 Problem Statement                                                                                            
                                                                                                              
 Current SAM2 implementation fails due to API incompatibility. The code uses SAM1-specific methods that don't 
  exist in SAM2/SAM2.1.                                                                                       
                                                                                                              
 Root Cause Analysis                                                                                          
                                                                                                              
 SAM1 API (what the code currently uses):                                                                     
 # Batch box transformation (SAM1-specific)                                                                   
 transformed_boxes = self.sam_predictor.transform.apply_boxes_torch(boxes, img.shape[:2])                     
 # Batch prediction (SAM1-specific) - returns torch.Tensor                                                    
 masks, _, _ = self.sam_predictor.predict_torch(boxes=transformed_boxes, multimask_output=False)              
                                                                                                              
 SAM2 API (what we need):                                                                                     
 # No transformation needed - coordinates auto-normalized                                                     
 # Single box prediction - returns np.ndarray                                                                 
 masks, scores, _ = self.sam_predictor.predict(box=np.array([x1,y1,x2,y2]), multimask_output=False)           
                                                                                                              
 Key Differences:                                                                                             
 ┌────────────────────┬────────────────────────────────────────┬──────────────────────────────┐               
 │      Feature       │                  SAM1                  │         SAM2/SAM2.1          │               
 ├────────────────────┼────────────────────────────────────────┼──────────────────────────────┤               
 │ Box transformation │ transform.apply_boxes_torch() required │ Not needed (auto-normalized) │               
 ├────────────────────┼────────────────────────────────────────┼──────────────────────────────┤               
 │ Prediction method  │ predict_torch()                        │ predict()                    │               
 ├────────────────────┼────────────────────────────────────────┼──────────────────────────────┤               
 │ Batch support      │ Yes (N boxes at once)                  │ No (loop over boxes)         │               
 ├────────────────────┼────────────────────────────────────────┼──────────────────────────────┤               
 │ Input type         │ torch.Tensor                           │ np.ndarray                   │               
 ├────────────────────┼────────────────────────────────────────┼──────────────────────────────┤               
 │ Output type        │ torch.Tensor                           │ np.ndarray                   │               
 └────────────────────┴────────────────────────────────────────┴──────────────────────────────┘               
 Files to Modify                                                                                              
                                                                                                              
 - mask_propagation/mask_manager.py - Core SAM integration (main changes)                                     
 - tools/demo_track.py - Add CLI arguments for SAM selection                                                  
 - INSTALLATION.md - Update installation instructions                                                         
                                                                                                              
 Solution Architecture                                                                                        
                                                                                                              
 1. Create SAM Abstraction Layer                                                                              
                                                                                                              
 Add a unified wrapper method in MaskManager that handles API differences:                                    
                                                                                                              
 def _sam_predict_boxes(self, image: np.ndarray, boxes_xyxy: list) -> torch.Tensor:                           
     """                                                                                                      
     Unified SAM prediction for both SAM1 and SAM2+.                                                          
                                                                                                              
     Args:                                                                                                    
         image: RGB image as numpy array (H, W, 3)                                                            
         boxes_xyxy: List of [x1, y1, x2, y2] bounding boxes                                                  
                                                                                                              
     Returns:                                                                                                 
         masks: Binary masks as tensor (N, 1, H, W)                                                           
     """                                                                                                      
     self.sam_predictor.set_image(image)                                                                      
                                                                                                              
     if self._is_sam2:                                                                                        
         # SAM2/SAM2.1 API - loop over boxes (no batch support)                                               
         masks_list = []                                                                                      
         for box in boxes_xyxy:                                                                               
             box_np = np.array(box)                                                                           
             masks, _, _ = self.sam_predictor.predict(                                                        
                 box=box_np,                                                                                  
                 multimask_output=False                                                                       
             )                                                                                                
             # masks shape: (1, H, W) -> convert to torch                                                     
             masks_list.append(torch.from_numpy(masks).unsqueeze(0))                                          
                                                                                                              
         if masks_list:                                                                                       
             return torch.cat(masks_list, dim=0)  # (N, 1, H, W)                                              
         return torch.empty(0)                                                                                
     else:                                                                                                    
         # SAM1 API - batch processing                                                                        
         image_boxes = torch.tensor(boxes_xyxy, device=self.device)                                           
         transformed_boxes = self.sam_predictor.transform.apply_boxes_torch(                                  
             image_boxes, image.shape[:2]                                                                     
         )                                                                                                    
         masks, _, _ = self.sam_predictor.predict_torch(                                                      
             point_coords=None,                                                                               
             point_labels=None,                                                                               
             boxes=transformed_boxes,                                                                         
             multimask_output=False                                                                           
         )                                                                                                    
         return masks  # (N, 1, H, W)                                                                         
                                                                                                              
 2. Supported Models                                                                                          
 ┌──────────────┬─────────────────────────────┬─────────────────────────────────────────┐                     
 │    Model     │       HuggingFace ID        │                  Notes                  │                     
 ├──────────────┼─────────────────────────────┼─────────────────────────────────────────┤                     
 │ SAM1 Base    │ vit_b                       │ Original SAM, requires local checkpoint │                     
 ├──────────────┼─────────────────────────────┼─────────────────────────────────────────┤                     
 │ SAM1 Large   │ vit_l                       │ Original SAM, requires local checkpoint │                     
 ├──────────────┼─────────────────────────────┼─────────────────────────────────────────┤                     
 │ SAM1 Huge    │ vit_h                       │ Original SAM, requires local checkpoint │                     
 ├──────────────┼─────────────────────────────┼─────────────────────────────────────────┤                     
 │ SAM2 Large   │ facebook/sam2-hiera-large   │ Good balance                            │                     
 ├──────────────┼─────────────────────────────┼─────────────────────────────────────────┤                     
 │ SAM2.1 Large │ facebook/sam2.1-hiera-large │ Recommended                             │                     
 ├──────────────┼─────────────────────────────┼─────────────────────────────────────────┤                     
 │ SAM2 Small   │ facebook/sam2-hiera-small   │ Faster                                  │                     
 ├──────────────┼─────────────────────────────┼─────────────────────────────────────────┤                     
 │ SAM2 Tiny    │ facebook/sam2-hiera-tiny    │ Fastest                                 │                     
 └──────────────┴─────────────────────────────┴─────────────────────────────────────────┘                     
 Implementation Steps                                                                                         
                                                                                                              
 Step 1: Add _sam_predict_boxes() method (NEW)                                                                
                                                                                                              
 Add after _init_sam() method (~line 144):                                                                    
                                                                                                              
 def _sam_predict_boxes(self, image: np.ndarray, boxes_xyxy: list) -> torch.Tensor:                           
     """                                                                                                      
     Unified SAM prediction for both SAM1 and SAM2+.                                                          
                                                                                                              
     Args:                                                                                                    
         image: RGB image as numpy array (H, W, 3)                                                            
         boxes_xyxy: List of [x1, y1, x2, y2] bounding boxes                                                  
                                                                                                              
     Returns:                                                                                                 
         masks: Binary masks as tensor (N, 1, H, W)                                                           
     """                                                                                                      
     self.sam_predictor.set_image(image)                                                                      
                                                                                                              
     if self._is_sam2:                                                                                        
         # SAM2/SAM2.1 API - loop over boxes (no batch support)                                               
         masks_list = []                                                                                      
         for box in boxes_xyxy:                                                                               
             box_np = np.array(box, dtype=np.float32)                                                         
             masks, _, _ = self.sam_predictor.predict(                                                        
                 box=box_np,                                                                                  
                 multimask_output=False                                                                       
             )                                                                                                
             # masks shape: (1, H, W) -> add batch dim                                                        
             masks_list.append(torch.from_numpy(masks).to(self.device))                                       
                                                                                                              
         if masks_list:                                                                                       
             return torch.stack(masks_list, dim=0)  # (N, 1, H, W)                                            
         return torch.empty(0, 1, image.shape[0], image.shape[1], device=self.device)                         
     else:                                                                                                    
         # SAM1 API - batch processing                                                                        
         image_boxes = torch.tensor(boxes_xyxy, device=self.device)                                           
         transformed_boxes = self.sam_predictor.transform.apply_boxes_torch(                                  
             image_boxes, image.shape[:2]                                                                     
         )                                                                                                    
         masks, _, _ = self.sam_predictor.predict_torch(                                                      
             point_coords=None,                                                                               
             point_labels=None,                                                                               
             boxes=transformed_boxes,                                                                         
             multimask_output=False                                                                           
         )                                                                                                    
         return masks  # (N, 1, H, W)                                                                         
                                                                                                              
 Step 2: Update initialize_first_masks() (lines 238-251)                                                      
                                                                                                              
 Replace lines 238-251:                                                                                       
 # OLD CODE:                                                                                                  
 if image_boxes_list == 0:                                                                                    
     # ...                                                                                                    
 else:                                                                                                        
     image_boxes = torch.tensor(image_boxes_list, device=self.sam.device)                                     
     transformed_boxes = self.sam_predictor.transform.apply_boxes_torch(image_boxes,                          
 img_info_prev['raw_img'].shape[:2])                                                                          
                                                                                                              
     masks, _, _ = self.sam_predictor.predict_torch(                                                          
         point_coords=None,                                                                                   
         point_labels=None,                                                                                   
         boxes=transformed_boxes,                                                                             
         multimask_output=False                                                                               
     )                                                                                                        
                                                                                                              
 With:                                                                                                        
 # NEW CODE:                                                                                                  
 if len(image_boxes_list) == 0:  # Fix: was checking == 0, should check len()                                 
     self.init_delay_counter += 1                                                                             
     return None                                                                                              
 else:                                                                                                        
     # Use unified SAM prediction method                                                                      
     masks = self._sam_predict_boxes(img_info_prev['raw_img'], image_boxes_list)                              
                                                                                                              
 Step 3: Update add_new_masks() (lines 330-339)                                                               
                                                                                                              
 Replace lines 330-339:                                                                                       
 # OLD CODE:                                                                                                  
 if len(image_boxes_list) > 0:                                                                                
     image_boxes = torch.tensor(image_boxes_list, device=self.sam.device)                                     
     transformed_boxes = self.sam_predictor.transform.apply_boxes_torch(image_boxes,                          
 img_info_prev['raw_img'].shape[:2])                                                                          
                                                                                                              
     masks, _, _ = self.sam_predictor.predict_torch(                                                          
         point_coords=None,                                                                                   
         point_labels=None,                                                                                   
         boxes=transformed_boxes,                                                                             
         multimask_output=False                                                                               
     )                                                                                                        
                                                                                                              
 With:                                                                                                        
 # NEW CODE:                                                                                                  
 if len(image_boxes_list) > 0:                                                                                
     # Use unified SAM prediction method                                                                      
     masks = self._sam_predict_boxes(img_info_prev['raw_img'], image_boxes_list)                              
                                                                                                              
 Step 4: Remove redundant set_image() calls                                                                   
                                                                                                              
 The _sam_predict_boxes() method calls set_image() internally, so remove:                                     
 - Line 218: self.sam_predictor.set_image(img_info_prev['raw_img'])                                           
 - Line 286: self.sam_predictor.set_image(img_info_prev['raw_img'])                                           
                                                                                                              
 Step 5: Add CLI arguments to demo_track.py                                                                   
                                                                                                              
 Add to make_parser() after line 162:                                                                         
 # SAM model arguments                                                                                        
 parser.add_argument("--sam_type", type=str, default="vit_b",                                                 
     help="SAM model: vit_b, vit_l, vit_h (SAM1) or HuggingFace ID like facebook/sam2.1-hiera-large (SAM2+)") 
 parser.add_argument("--sam_checkpoint", type=str, default=None,                                              
     help="Path to SAM1 checkpoint (only for vit_b/vit_l/vit_h, ignored for SAM2+)")                          
                                                                                                              
 Update MaskManager instantiation at lines 341 and 477:                                                       
 # OLD:                                                                                                       
 mask_menager = MaskManager()                                                                                 
                                                                                                              
 # NEW:                                                                                                       
 mask_menager = MaskManager(                                                                                  
     sam_checkpoint=args.sam_checkpoint,                                                                      
     sam_type=args.sam_type,                                                                                  
 )                                                                                                            
                                                                                                              
 Step 6: Update INSTALLATION.md                                                                               
                                                                                                              
 Add SAM2/SAM2.1 section:                                                                                     
                                                                                                              
 ## SAM2/SAM2.1 (Optional - Better Segmentation)                                                              
                                                                                                              
 For improved mask quality, you can use SAM2.1 instead of SAM1:                                               
                                                                                                              
 ### Installation                                                                                             
                                                                                                              
 ```bash                                                                                                      
 # Install SAM2 package                                                                                       
 pip install sam2                                                                                             
                                                                                                              
 # Login to HuggingFace (required for model download)                                                         
 huggingface-cli login                                                                                        
                                                                                                              
 Accept Model License                                                                                         
                                                                                                              
 Visit and accept the license:                                                                                
 - SAM2.1: https://huggingface.co/facebook/sam2.1-hiera-large                                                 
                                                                                                              
 Usage                                                                                                        
                                                                                                              
 # Run with SAM2.1 (auto-downloads weights + config)                                                          
 python tools/demo_track.py --path video.mp4 --sam_type facebook/sam2.1-hiera-large                           
                                                                                                              
 Available SAM2 Models                                                                                        
 ┌─────────────────────────────┬─────────┬─────────┐                                                          
 │            Model            │  Speed  │ Quality │                                                          
 ├─────────────────────────────┼─────────┼─────────┤                                                          
 │ facebook/sam2-hiera-tiny    │ Fastest │ Lower   │                                                          
 ├─────────────────────────────┼─────────┼─────────┤                                                          
 │ facebook/sam2-hiera-small   │ Fast    │ Good    │                                                          
 ├─────────────────────────────┼─────────┼─────────┤                                                          
 │ facebook/sam2-hiera-large   │ Medium  │ High    │                                                          
 ├─────────────────────────────┼─────────┼─────────┤                                                          
 │ facebook/sam2.1-hiera-large │ Medium  │ Best    │                                                          
 └─────────────────────────────┴─────────┴─────────┘                                                          
 ## Files to Modify Summary                                                                                   
                                                                                                              
 | File | Line(s) | Change |                                                                                  
 |------|---------|--------|                                                                                  
 | `mask_propagation/mask_manager.py` | ~144 | Add `_sam_predict_boxes()` method |                            
 | `mask_propagation/mask_manager.py` | 218 | Remove redundant `set_image()` |                                
 | `mask_propagation/mask_manager.py` | 238-251 | Replace with unified method call |                          
 | `mask_propagation/mask_manager.py` | 286 | Remove redundant `set_image()` |                                
 | `mask_propagation/mask_manager.py` | 330-339 | Replace with unified method call |                          
 | `tools/demo_track.py` | ~163 | Add `--sam_type`, `--sam_checkpoint` args |                                 
 | `tools/demo_track.py` | 341 | Pass SAM args to MaskManager |                                               
 | `tools/demo_track.py` | 477 | Pass SAM args to MaskManager |                                               
 | `INSTALLATION.md` | EOF | Add SAM2 installation section |                                                  
                                                                                                              
 ## Verification                                                                                              
                                                                                                              
 ### Test Commands                                                                                            
                                                                                                              
 ```bash                                                                                                      
 # Test SAM1 (default)                                                                                        
 python tools/demo_track.py --path test_frames/ --sam_type vit_b                                              
                                                                                                              
 # Test SAM2.1 (recommended)                                                                                  
 python tools/demo_track.py --path test_frames/ --sam_type facebook/sam2.1-hiera-large                        
                                                                                                              
 # Test with video input                                                                                      
 python tools/demo_track.py --path test.mp4 --demo video --sam_type facebook/sam2.1-hiera-large               
                                                                                                              
 Verification Checklist                                                                                       
                                                                                                              
 - First frame masks generated correctly                                                                      
 - New object masks added during tracking                                                                     
 - Lost track masks removed properly                                                                          
 - Cutie propagation continues working                                                                        
 - No CUDA/device errors                                                                                      
 - Output visualization shows masks                                                                           
 - Works with both SAM1 and SAM2.1                                                                            
                                                                                                              
 Notes                                                                                                        
                                                                                                              
 - SAM2+ models auto-download from HuggingFace (weights + config)                                             
 - HuggingFace authentication required for SAM2+ (license agreement)                                          
 - SAM1 requires manual checkpoint download                                                                   
 - SAM2 has no batch prediction - loops over boxes (slightly slower but works)  