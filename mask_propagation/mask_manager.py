import torch
import sys
from pathlib import Path
from typing import Optional

import numpy as np

# Enable imports from the Cutie directory
root = Path(__file__).parent
_CUTIE_ROOT = root / 'Cutie'
sys.path.append(str(_CUTIE_ROOT))

# Hydra config path (relative)
cutie_config_rel = "Cutie/cutie/config"

### Cutie ###
from omegaconf import open_dict
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra

from cutie.model.cutie import CUTIE
from cutie.inference.inference_core import InferenceCore
from cutie.inference.utils.args_utils import get_dataset_cfg
from gui.interactive_utils import image_to_torch, torch_prob_to_numpy_mask, index_numpy_to_one_hot_torch

OVERLAP_MEASURE_VARIANT = 1
OVERLAP_VARIANT_2_GRID_STEP = 10
MASK_CREATION_BBOX_OVERLAP_THRESHOLD = 0.6

# Supported SAM model types
SAM_MODEL_TYPES = ["vit_b", "vit_l", "vit_h", "sam2"]


class MaskManager(object):
    def __init__(
        self,
        sam_checkpoint: Optional[str] = None,
        sam_type: str = "vit_b",
        cutie_weights: Optional[str] = None,
        device: Optional[str] = None,
    ):
        """
        Initialize MaskManager with configurable paths and device.

        Args:
            sam_checkpoint: Path to SAM weights. Defaults to ./sam_models/sam_vit_b_01ec64.pth
            sam_type: SAM model type: "vit_b", "vit_l", "vit_h", or "sam2"
            cutie_weights: Path to Cutie weights. Defaults to mask_propagation/Cutie/weights/cutie-base-mega.pth
            device: Device to use ('cuda', 'cuda:0', 'cpu', etc.). Defaults to 'cuda' if available.
        """
        self.masks = None  # masks instances from image segmentation component (SAM)
        self.mask = None  # mask instances related to mask temporal propagator (Cutie)
        self.prediction = None
        self.tracklet_mask_dict = None
        self.mask_color_counter = 0
        self.current_object_list_cutie = []
        self.last_object_number_cutie = 0
        self.awaiting_mask_tracklet_ids = []
        self.init_delay_counter = 0
        self.SAM_START_FRAME = 1

        # Set device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Set default paths
        if sam_checkpoint is None:
            sam_checkpoint = "./sam_models/sam_vit_b_01ec64.pth"

        if cutie_weights is None:
            cutie_weights = str(Path(__file__).parent / "Cutie" / "weights" / "cutie-base-mega.pth")

        # Store for reference
        self.sam_checkpoint = sam_checkpoint
        self.sam_type = sam_type
        self.cutie_weights = cutie_weights

        # Initialize SAM
        np.random.seed(0)
        self._init_sam(sam_checkpoint, sam_type)

        # Initialize Cutie
        self._init_cutie(cutie_weights)

    def _init_sam(self, sam_checkpoint: str, sam_type: str):
        """Initialize SAM or SAM2 model."""
        if sam_type == "sam2":
            # SAM2 initialization
            try:
                from sam2.build_sam import build_sam2
                from sam2.sam2_image_predictor import SAM2ImagePredictor

                self.sam = build_sam2(sam_checkpoint)
                self.sam.to(device=self.device)
                self.sam_predictor = SAM2ImagePredictor(self.sam)
                self._is_sam2 = True
            except ImportError:
                raise ImportError(
                    "SAM2 package not found. Install with: pip install sam2"
                )
        else:
            # Original SAM initialization
            from segment_anything import sam_model_registry, SamPredictor

            if sam_type not in ["vit_b", "vit_l", "vit_h"]:
                raise ValueError(
                    f"Unknown SAM type: {sam_type}. "
                    f"Available: vit_b, vit_l, vit_h, sam2"
                )

            self.sam = sam_model_registry[sam_type](checkpoint=sam_checkpoint)
            self.sam.to(device=self.device)
            self.sam_predictor = SamPredictor(self.sam)
            self._is_sam2 = False

    def _init_cutie(self, cutie_weights: str):
        """Initialize Cutie model."""
        with torch.inference_mode():
            with torch.cuda.amp.autocast(enabled=True):
                # Clear any previous Hydra initialization
                GlobalHydra.instance().clear()

                initialize(
                    version_base='1.3.2',
                    config_path=str(cutie_config_rel),
                    job_name="eval_config"
                )
                cfg = compose(config_name="eval_config")

                weight_path = Path(cutie_weights)
                with open_dict(cfg):
                    cfg['weights'] = str(weight_path)

                # This function from Cutie must be called as it modifies (adds extra values) to cfg
                _ = get_dataset_cfg(cfg)

                # Load the network weights
                self.cutie = CUTIE(cfg).to(self.device).eval()
                model_weights = torch.load(cfg.weights, map_location=self.device)
                self.cutie.load_weights(model_weights)

                # For Cutie
                torch.cuda.empty_cache()
                self.processor = InferenceCore(self.cutie, cfg=cfg)


    def get_updated_masks(self, img_info, img_info_prev, frame_id, online_tlwhs, online_ids, new_tracks, removed_tracks_ids):
        """
        Propagates the existing masks, creates new masks and/or removes redundant masks, depending on the tracker state.

        Args:
            img_info (dict): Frame information dict (generated with detections).
            frame_id (int): Current frame number within the tracking process (starting from 1)
            online_tlwhs (list): Updated tracklet postions, coming from the tracker output
            online_ids (list): Updated tracklet IDs, coming from the tracker output
            new_tracks (list): Newly created tracklets, coming from the tracker output
            removed_tracks_ids (list): Removed tracklet IDs, coming from the tracker output
        """
        
        # This one (for Cutie) is going to be used in both cases, initialization of first masks and adding new masks
        # convert numpy array to pytorch tensor format
        frame_torch = image_to_torch(img_info['raw_img'], device=self.device)
        frame_torch_prev = image_to_torch(img_info_prev['raw_img'], device=self.device)
        
        # Perform SAM segmentation on the first frame bounding boxes
        if frame_id == self.SAM_START_FRAME + 1 + self.init_delay_counter and online_tlwhs is not None:
            # img_info_prev is provided after the first frame, that's why SAM_START_FRAME + 1
            prediction = self.initialize_first_masks(frame_torch, frame_torch_prev, img_info_prev, online_tlwhs, online_ids)

        elif frame_id > self.SAM_START_FRAME + 1 + self.init_delay_counter:
            self.add_new_masks(frame_torch_prev, img_info_prev, online_tlwhs, online_ids, new_tracks)
            self.remove_masks(removed_tracks_ids)

            # Continue the propagation, always for frame_id > SAM_START_FRAME + 1 
            # (The case of frame_id == SAM_START_FRAME + 1 already handled above when initializing first frame masks. Other cases not applicable.)
            prediction = self.processor.step(frame_torch)

        mask_avg_prob_dict = None
        prediction_colors_preserved = None

        # prediction initially set as None, changed after the mask are set at SAM_START_FRAME + 1
        if prediction is not None:
            prediction, mask_avg_prob_dict, prediction_colors_preserved = self.post_process_mask(prediction)

        return prediction, self.tracklet_mask_dict.copy(), mask_avg_prob_dict, prediction_colors_preserved
    

    def initialize_first_masks(self, frame_torch, frame_torch_prev, img_info_prev, online_tlwhs, online_ids):
        self.sam_predictor.set_image(img_info_prev['raw_img']) 
        image_boxes_list = []
        new_tracks_id = []

        # Based on ByteTrack mechanisms, the tracklets created at the first frame will already be considered fully activated, thus as the tracked tracklets (stracks). 
        # Therefore using the online_tlwhs coming from the tracker output
        for i, ot in enumerate(online_tlwhs):

            ### Avoiding creation of the "weird" masks (from overlapped/occluded subjects) ###
            track_BBs_with_lower_bottom = get_tracklets_with_lower_bottom(ot, online_tlwhs)
            overlap = get_overlap_with_lower_bottom_tracklets(ot, track_BBs_with_lower_bottom)

            if overlap >= MASK_CREATION_BBOX_OVERLAP_THRESHOLD:
                self.awaiting_mask_tracklet_ids.append(online_ids[i])
                continue
            ### </> ###

            image_boxes_list.append([ot[0], ot[1], ot[0] + ot[2], ot[1] + ot[3]])
            new_tracks_id.append(online_ids[i])

        if image_boxes_list == 0:
            # Delay the whole process
            self.init_delay_counter += 1
            return None
        else:
            image_boxes = torch.tensor(image_boxes_list, device=self.sam.device)
            transformed_boxes = self.sam_predictor.transform.apply_boxes_torch(image_boxes, img_info_prev['raw_img'].shape[:2])
            
            masks, _, _ = self.sam_predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes,
                multimask_output=False
            )
            # Convert masks
            mask = np.zeros(masks[0].shape)

            for mi in range(len(masks)):
                current_mask = masks[mi].cpu().numpy().astype(int)
                current_mask[current_mask > 0] = mi + 1 # fill first mask part with 1s, second mask part with 2s, etc.                       

                non_occupied = (mask == 0).astype(int)
                mask += (current_mask * non_occupied)

            mask = mask.squeeze(0)
            self.num_objects = len(masks)

            # Needed when adding new masks in the next frames. Thus needs to be updated before adding and after deleting a mask(s)
            self.current_object_list_cutie = list(range(1, self.num_objects+1))
            self.last_object_number_cutie = max(self.current_object_list_cutie, default=0)

            # Not all the BBs get their masks instantly created. Hence using new_tracks_id.
            self.tracklet_mask_dict = dict(zip(new_tracks_id, range(1, self.num_objects+1)))
            
            # A dictionary with id color allocated to each mask. Mask ids will change/shift after removal(s), yet the color ids will stay the same
            self.mask_color_dict = dict(zip(range(1, self.num_objects+1), range(1, self.num_objects+1)))
            self.mask_color_counter = max(list(self.mask_color_dict.values()), default=0)

            # Perform Cutie propagation (frame-by-frame)
            mask_torch = index_numpy_to_one_hot_torch(mask, self.num_objects+1).to(self.device)
            _ = self.processor.step(frame_torch_prev, mask_torch[1:], idx_mask=False)
            prediction = self.processor.step(frame_torch)

            return prediction


    def add_new_masks(self, frame_torch_prev, img_info_prev, online_tlwhs, online_ids, new_tracks):
        if len(new_tracks) > 0 or len(self.awaiting_mask_tracklet_ids)> 0:
            self.sam_predictor.set_image(img_info_prev['raw_img'])
            image_boxes_list = []
            new_tracks_id = []

            ### Try to create masks for the tracklets awaiting from the previous frames. Avoiding creation of the "weird" masks (from overlapped/occluded subjects) ###
            for i, amti in enumerate(self.awaiting_mask_tracklet_ids):
                # Take into account only these tracklets with ids from awaiting_mask_tracklet_ids, that are actually active, not marked as lost. Thus 
                # only these tracklets, that have been returned by the ByteTracker's update function (their IDs in online_ids and coords in online_tlwhs)
                if not amti in online_ids:
                    continue

                amt_index = online_ids.index(amti)
                amt_tlwh = online_tlwhs[amt_index]

                track_BBs_with_lower_bottom = get_tracklets_with_lower_bottom(amt_tlwh, online_tlwhs)
                overlap = get_overlap_with_lower_bottom_tracklets(amt_tlwh, track_BBs_with_lower_bottom)

                # If the overlap is not too big, initiate mask creation for this tracklet and set it to be removed from the awaiting list (awaiting_mask_tracklet_ids)
                if overlap < MASK_CREATION_BBOX_OVERLAP_THRESHOLD:
                    image_boxes_list.append([amt_tlwh[0], amt_tlwh[1], amt_tlwh[0] + amt_tlwh[2], amt_tlwh[1] + amt_tlwh[3]])
                    new_tracks_id.append(amti)

            # Remove from the awaiting list (awaiting_mask_tracklet_ids) the tracklets which are going to have their masks created
            for nti in new_tracks_id:
                self.awaiting_mask_tracklet_ids.remove(nti)
            ### </> ###

            ### Avoiding creation of the "weird" masks (from overlapped subjects) - for the new tracklets from the current frame ###
            for i, nt in enumerate(new_tracks):
                track_BBs_with_lower_bottom = get_tracklets_with_lower_bottom(nt.last_det_tlwh, online_tlwhs)
                
                # For the considered tracklet nt, measure the overlap with the tracklets retrieved as above (2 variants possible)
                overlap = get_overlap_with_lower_bottom_tracklets(nt.last_det_tlwh, track_BBs_with_lower_bottom)

                # If the overlap exceeds the defined threshold, then do not create the mask for this tracklet
                # Keep the tracklet ID in a separate list, awaiting_mask_tracklet_ids
                if overlap >= MASK_CREATION_BBOX_OVERLAP_THRESHOLD:
                    self.awaiting_mask_tracklet_ids.append(nt.track_id)
                    continue
                                        
                image_boxes_list.append([nt.last_det_tlwh[0], nt.last_det_tlwh[1], nt.last_det_tlwh[0] + nt.last_det_tlwh[2], nt.last_det_tlwh[1] + nt.last_det_tlwh[3]])
                new_tracks_id.append(nt.track_id)
            ### </> ###

            if len(image_boxes_list) > 0:
                image_boxes = torch.tensor(image_boxes_list, device=self.sam.device)
                transformed_boxes = self.sam_predictor.transform.apply_boxes_torch(image_boxes, img_info_prev['raw_img'].shape[:2])

                masks, _, _ = self.sam_predictor.predict_torch(
                    point_coords=None,
                    point_labels=None,
                    boxes=transformed_boxes,
                    multimask_output=False
                )

                # Convert masks!
                mask_extra = np.zeros(masks[0].shape)
                max_mask_number = max(self.tracklet_mask_dict.values(), default=0)

                new_masks_numbers = []
                new_object_numbers = []

                for mi in range(len(masks)):
                    current_mask = masks[mi].cpu().numpy().astype(int)
                    next_mask_number = max_mask_number + mi + 1 # Next consecutive number of the new mask to come
                    current_mask[current_mask > 0] = next_mask_number 
                    new_masks_numbers.append(next_mask_number)

                    self.last_object_number_cutie += 1
                    new_object_numbers.append(self.last_object_number_cutie)

                    non_occupied = (mask_extra == 0).astype(int)
                    mask_extra += (current_mask * non_occupied)

                mask_extra = mask_extra.squeeze(0)

                # Add masks to the Cutie
                frame_torch_prev = image_to_torch(img_info_prev['raw_img'], device=self.device)
                
                # Update the existing mask with the new one. Give the priority to the new coming mask
                self.mask_prediction_prev_frame[mask_extra > 0] = mask_extra[mask_extra > 0]

                self.num_objects += len(new_tracks_id)

                mask_prev_extended_torch = index_numpy_to_one_hot_torch(self.mask_prediction_prev_frame, self.num_objects+1).to(self.device) # num_objects + 1 to incorporate the background
                self.current_object_list_cutie.extend(new_object_numbers)

                # Incorporate the new masks. The actual temporal step (from the previous frame to the current) is performed outside this function - common for both adding and removing after removing
                # (In case both adding and removing happends within the same frame, still only one actual temporal step is required.) 
                _ = self.processor.step(frame_torch_prev, mask_prev_extended_torch, objects=self.current_object_list_cutie, idx_mask=False)

                self.mask_color_counter = update_tracklet_mask_dict_after_mask_addition(self.tracklet_mask_dict, self.mask_color_dict, new_tracks_id, new_masks_numbers, self.mask_color_counter)

    
    def remove_masks(self, removed_tracks_ids):
        if len(removed_tracks_ids) > 0:
            mask_ids_to_be_removed = [self.tracklet_mask_dict[i] for i in self.tracklet_mask_dict.keys() if i in removed_tracks_ids]

            purge_activated, tmp_keep_idx, obj_keep_idx = self.processor.object_manager.purge_selected_objects(mask_ids_to_be_removed)

            original_removed_tracks_ids = removed_tracks_ids.copy()

            for rti in original_removed_tracks_ids:
                if rti not in self.tracklet_mask_dict:
                    removed_tracks_ids.remove(rti)

            if purge_activated:
                self.processor.memory.purge_except(obj_keep_idx)

            self.current_object_list_cutie = obj_keep_idx
            self.num_objects = len(self.current_object_list_cutie)

            self.mask_color_counter = update_tracklet_mask_dict_after_mask_removal(self.tracklet_mask_dict, self.mask_color_dict, mask_ids_to_be_removed) 


    def post_process_mask(self, prediction):
        mask_avg_prob_dict = self.get_mask_avg_prob(prediction)
        
        # argmax, convert to numpy
        prediction = torch_prob_to_numpy_mask(prediction)

        # TODO: Ensure that ".copy()" is actually needed
        self.mask_prediction_prev_frame = prediction.copy()

        # Keep the colors of the masks - even after the mask numbers shifted due to deletion
        prediction_colors_preserved = self.adjust_mask_colors(prediction)

        return prediction, mask_avg_prob_dict, prediction_colors_preserved


    def get_mask_avg_prob(self, prediction):
        mask_avg_prob_dict = {}
        mask_maxes = torch.max(prediction, dim=0).indices

        for v in self.tracklet_mask_dict.values():
            # Get the average score of the probabilities - from the spots occupied by the mask 
            # (selected as such by torchmax and indices (like argmax))
            average_mask_v_score = (prediction[v][mask_maxes == v]).mean().item()  
            if average_mask_v_score is not None and not np.isnan(average_mask_v_score):     
                mask_avg_prob_dict[v] = average_mask_v_score

        return mask_avg_prob_dict
    

    def adjust_mask_colors(self, prediction):
        # Keep the colors of the masks - even after the mask numbers shifted due to deletion
        prediction_colors_preserved = prediction.copy()
        keys_descending = sorted(list(self.mask_color_dict.keys()), reverse=True)
        for k in keys_descending:
            prediction_colors_preserved[prediction_colors_preserved == k] = self.mask_color_dict[k]

        return prediction_colors_preserved
    

def get_tracklets_with_lower_bottom(new_tracklet_tlwh, online_tlwhs):
    nt_y = new_tracklet_tlwh[1]
    nt_h = new_tracklet_tlwh[3]

    # Find all the tracklets, both the new ones (new_tracks) and the exisiting ones (online_tlwhs), 
    # with their bottom coordinate higher (thus being at the lower position) than the bottom coordinate of the considered tracklet nt
    track_BBs_with_lower_bottom = []
    nt_bottom = nt_y + nt_h

    for ot in online_tlwhs:
        if ot[1] + ot[3] > nt_bottom: # >, not >= so as to avoid the conflict with itself (also included in online_tlwhs)
            track_BBs_with_lower_bottom.append(ot)

    return track_BBs_with_lower_bottom


def get_overlap_with_lower_bottom_tracklets(new_tracklet_tlwh, track_BBs_with_lower_bottom):
    overlap = 0
    
    if OVERLAP_MEASURE_VARIANT == 1:
        overlap = get_overlap_variant_1(new_tracklet_tlwh, track_BBs_with_lower_bottom)
    elif OVERLAP_MEASURE_VARIANT == 2:
        overlap = get_overlap_variant_2(new_tracklet_tlwh, track_BBs_with_lower_bottom)

    return overlap


# Variant 1: Find the maximal overlap (intersection) of two tracklets, between the new one and exisiting ones from the list. 
# Then divide it by the size of the considered tracklet bounding box
def get_overlap_variant_1(new_tracklet_tlwh, track_BBs_with_lower_bottom):
    nt_x = new_tracklet_tlwh[0]
    nt_y = new_tracklet_tlwh[1]
    nt_w = new_tracklet_tlwh[2]
    nt_h = new_tracklet_tlwh[3]
    
    max_overlap_part = 0 # between 0 and 1

    for lb in track_BBs_with_lower_bottom:
        x_dist = min(nt_x+nt_w, lb[0]+lb[2]) - max(nt_x, lb[0])
        y_dist = min(nt_y+nt_h, lb[1]+lb[3]) - max(nt_y, lb[1])

        if x_dist < 0 or y_dist < 0:
            overlap_area = 0
        else:
            overlap_area = x_dist * y_dist

        overlap_part = overlap_area / (nt_w * nt_h)
        if max_overlap_part < overlap_part:
            max_overlap_part = overlap_part

            if max_overlap_part == 1:
                break

    return max_overlap_part


# Variant 2: Check each pixel (or every 10th pixel vertically and horizontally - imagine a grid) from the considered
# tracklet bounding box if it is also within the bounding box of another tracklet (from all the ones retrieved) - based on their coordinates.
# The overlap measure is then the ratio of the number of occupied pixels to the size of the boudnind box (in terms of each 10 pixel vert. and hor.)
def get_overlap_variant_2(new_tracklet_tlwh, track_BBs_with_lower_bottom):
    nt_x = int(new_tracklet_tlwh[0])
    nt_y = int(new_tracklet_tlwh[1])
    nt_w = int(new_tracklet_tlwh[2])
    nt_h = int(new_tracklet_tlwh[3])

    point_overlap_counter = 0

    for grid_row in range(nt_y, nt_y+nt_h, OVERLAP_VARIANT_2_GRID_STEP):
        for grid_col in range(nt_x, nt_x+nt_w, OVERLAP_VARIANT_2_GRID_STEP):
            for lb in track_BBs_with_lower_bottom:
                if lb[0] <= grid_col and grid_col <= lb[0] + lb[2] and lb[1] <= grid_row and grid_row <= lb[1] + lb[3]:
                    point_overlap_counter += 1
                    break

    overlap_part = point_overlap_counter / (len(range(nt_y, nt_y+nt_h, OVERLAP_VARIANT_2_GRID_STEP)) * len(range(nt_x, nt_x+nt_w, OVERLAP_VARIANT_2_GRID_STEP)))
    
    return overlap_part


def update_tracklet_mask_dict_after_mask_addition(tracklet_mask_dict, mask_color_dict, added_tracklet_ids, added_mask_ids, mask_color_counter): 
    # Part 1/2: Update the mask ids in tracklet_mask_dict
    for k,v in zip(added_tracklet_ids, added_mask_ids):
        tracklet_mask_dict[k] = v

    # Part 2/2: Update the mask ids in mask_color_dict
    for mi in added_mask_ids:
        mask_color_counter += 1
        mask_color_dict[mi] = mask_color_counter

    return mask_color_counter


def update_tracklet_mask_dict_after_mask_removal(tracklet_mask_dict, mask_color_dict, removed_mask_ids):
    # Part 1/2: Update the mask ids in tracklet_mask_dict
    entries_to_be_removed = []
    decrement_mask_id_dict = {}

    for k in tracklet_mask_dict.keys():
        if tracklet_mask_dict[k] in removed_mask_ids:
            entries_to_be_removed.append(k)
        else:
            for rmi in removed_mask_ids:
                if tracklet_mask_dict[k] > rmi:
                    if not k in decrement_mask_id_dict.keys():
                        decrement_mask_id_dict[k] = 1
                    else:
                        decrement_mask_id_dict[k] += 1

    for entry in entries_to_be_removed:
        del tracklet_mask_dict[entry]

    for k in decrement_mask_id_dict.keys():
        tracklet_mask_dict[k] -= decrement_mask_id_dict[k]


    # Part 2/2: Update the mask ids in mask_color_dict

    # Saved and returned in case the last element (with the highest number) was deleted
    mask_color_counter = max(list(mask_color_dict.values()), default=0)

    entries_to_be_removed = []
    decrement_mask_id_dict = {}

    for k in mask_color_dict.keys():
        if k in removed_mask_ids:
            entries_to_be_removed.append(k)
        else:
            for rmi in removed_mask_ids: 
                if k > rmi:
                    if not k in decrement_mask_id_dict.keys():
                        decrement_mask_id_dict[k] = 1
                    else:
                        decrement_mask_id_dict[k] += 1

    for entry in entries_to_be_removed:
        del mask_color_dict[entry]

    mask_color_keys = list(decrement_mask_id_dict.keys())
    for mc in mask_color_keys:
        new_key = mc - decrement_mask_id_dict[mc]
        mask_color_dict[new_key] = mask_color_dict[mc]
        del mask_color_dict[mc]

    return mask_color_counter