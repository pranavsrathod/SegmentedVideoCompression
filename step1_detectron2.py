'''
README:
pip install opencv-python numpy
pip install torch torchvision
pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/{cuda_version}/torch{torch_version}/index.html

For example:pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu118/torch2.1/index.html
'''

import cv2
import numpy as np
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.model_zoo import model_zoo

def read_frames(filename, width, height):
    with open(filename, 'rb') as f:
        data = f.read()
    frame_size = width * height * 3  
    num_frames = len(data) // frame_size
    frames = []
    for i in range(num_frames):
        frame_data = data[i * frame_size:(i + 1) * frame_size]
        frame = np.frombuffer(frame_data, dtype=np.uint8).reshape((height, width, 3)) 
        frames.append(frame)
    return frames

def read_frame(file, width, height):
    frame_data = file.read(width * height * 3)
    if not frame_data:
        return None
    return np.frombuffer(frame_data, dtype=np.uint8).reshape((height, width, 3))

def load_detectron2_model():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Set Detection Threshold
    # cfg.MODEL.DEVICE = "cpu"  # Specify Device GPU（or "cpu"）
    cfg.MODEL.DEVICE = "cuda"
    return DefaultPredictor(cfg)

def object_detection_with_detectron2(predictor, frame):
    outputs = predictor(frame) #frame:in BGR format
    instances = outputs["instances"] #Extract all detected objects
    masks = instances.pred_masks.cpu().numpy()  # (number of objects, image height, image width)
    classes = instances.pred_classes.cpu().numpy()  # object classes
    return masks, classes

#Calculates the optical flow between two video frames
'''
Optical flow is the motion vector of each pixel
between two consecutive frames, describing pixel displacement.
'''
def compute_optical_flow(prev_frame, current_frame):
    #Optical flow calculation requires grayscale images 
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    #dense optical flow: calculates motion vectors for each pixel(Hierarchical Motion Estimation)
    flow = cv2.calcOpticalFlowFarneback(prev_gray, current_gray, None, 
                                        0.5, 3, 15, 3, 5, 1.2, 0)
    return flow #Motion Vector for Each Pixel, (height, width, 2)

def is_camera_moving(flow, motion_threshold=1.5):

    global_motion_vector = np.mean(flow, axis=(0, 1))
    global_magnitude = np.linalg.norm(global_motion_vector)

    if global_magnitude > motion_threshold:
        return True, global_motion_vector
    else:
        return False, global_motion_vector

#Filtering ovjects via optical flow
def filter_foreground_with_optical_flow(flow, masks, is_moving, global_motion_vector, motion_threshold=1.5, angle_threshold=np.pi/8):
    if len(masks) == 0:
        return np.zeros(flow.shape[:2], dtype=bool) 

    #Store the magnitude/angle of motion vector of each pixel
    motion_magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
    motion_angle = np.arctan2(flow[..., 1], flow[..., 0])

    global_magnitude = np.linalg.norm(global_motion_vector)
    global_angle = np.arctan2(global_motion_vector[1], global_motion_vector[0])

    filtered_foreground_mask = np.zeros_like(masks[0], dtype=bool)

    #For each detected object
    for mask in masks:
        mask_motion_magnitude = motion_magnitude[mask]
        mask_motion_angle = motion_angle[mask]

        if is_moving:
            #  Region-Wide Averages
            mask_magnitude_diff = np.abs(np.mean(mask_motion_magnitude) - global_magnitude)
            mask_angle_diff = np.abs(np.mean(mask_motion_angle) - global_angle)

            if mask_magnitude_diff > motion_threshold or mask_angle_diff > angle_threshold:
                filtered_foreground_mask |= mask
        else:
            # camera is stationary, find moving ovjects
            if np.mean(mask_motion_magnitude) > motion_threshold:
                filtered_foreground_mask |= mask

    # Fall back to the original Detectron2 masks if empty result.
    if not filtered_foreground_mask.any():
        for mask in masks:
            filtered_foreground_mask |= mask

    return filtered_foreground_mask


def highlight_microblocks(frame, foreground_mask, block_size=8):
    height, width = frame.shape[:2]
    vis_frame = frame.copy()
    for i in range(0, height, block_size):
        for j in range(0, width, block_size):
            block = foreground_mask[i:i + block_size, j:j + block_size]
            if block.any():
                vis_frame[i:i + block_size, j:j + block_size] = [0, 0, 255]
    return vis_frame


def process_rgb_video(filename, width, height):
    predictor = load_detectron2_model()
    
    with open(filename, 'rb') as file: 
        prev_frame = read_frame(file, width, height)
        if prev_frame is None:
            print("Error: Could not read first frame")
            return

        while True:
            curr_frame = read_frame(file, width, height)
            if curr_frame is None:  # EOF
                break

            frame_bgr = cv2.cvtColor(curr_frame, cv2.COLOR_RGB2BGR)
            prev_frame_bgr = cv2.cvtColor(prev_frame, cv2.COLOR_RGB2BGR)

            flow = compute_optical_flow(prev_frame_bgr, frame_bgr)

            mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])

            flow_vis = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            flow_vis = flow_vis.astype(np.uint8)
            flow_vis = cv2.applyColorMap(flow_vis, cv2.COLORMAP_JET)

            masks, classes = object_detection_with_detectron2(predictor, frame_bgr)
            detectron_vis = frame_bgr.copy()

            for mask in masks:
                detectron_vis[mask] = [0, 255, 0]  # green overlay

            is_moving, global_motion_vector = is_camera_moving(flow)
            foreground_mask = filter_foreground_with_optical_flow(flow, masks, is_moving, global_motion_vector)

            visual_frame = highlight_microblocks(frame_bgr, foreground_mask)

            cv2.putText(frame_bgr, "Original Frame", (20,40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

            cv2.putText(detectron_vis, "Detectron Segmentation", (20,40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

            cv2.putText(flow_vis, "Optical Flow", (20,40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

            cv2.putText(visual_frame, "Final Foreground Blocks", (20,40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

            top = np.hstack((frame_bgr, detectron_vis))
            bottom = np.hstack((flow_vis, visual_frame))
            combined = np.vstack((top, bottom))

            # combined = cv2.resize(combined, (1280, 720))

            cv2.imshow("Segmentation Pipeline Demo", combined)

            if cv2.waitKey(30) & 0xFF == ord('q'):
                break

            prev_frame = curr_frame  

        cv2.destroyAllWindows()

def classify_macroblocks(foreground_mask, block_size=16):
    height, width = foreground_mask.shape
    background = []
    foreground = []
    
    # For each macroblock
    for i in range(0, height // block_size):
        for j in range(0, width // block_size):
            block_mask = foreground_mask[i*block_size:(i+1)*block_size, 
                                      j*block_size:(j+1)*block_size]
            
            #If contains any forground object
            if block_mask.any():
                foreground.append((i, j))
            else:
                background.append((i, j))
    
    return background, foreground

filename = "3.rgb"  
width, height = 960, 540 
process_rgb_video(filename, width, height)
