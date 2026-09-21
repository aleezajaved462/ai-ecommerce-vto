import cv2
import numpy as np
import math

def overlay_watch(frame, hand_landmarks, watch_img):
    """
    Refined 3D Wrist Rotation & Angle Mapping Engine
    """
    h, w, _ = frame.shape
    
    # 1. Wrist (Landmark 0) & Middle Finger MCP (Landmark 9) Coordinates
    wrist = hand_landmarks[0]
    mcp = hand_landmarks[9]
    
    cx, cy = int(wrist.x * w), int(wrist.y * h)
    mx, my = int(mcp.x * w), int(mcp.y * h)
    
    # 2. Dynamic Wrist Scale (Distance between Wrist and Middle Finger Base)
    wrist_width = math.hypot(mx - cx, my - cy)
    if wrist_width == 0:
        return frame
        
    scale = wrist_width * 1.25  # Dynamic Scaling Factor
    
    # 3. Angle Calculation for Precise Wrist Orientation
    angle_rad = math.atan2(my - cy, mx - cx)
    angle_deg = math.degrees(angle_rad) - 90
    
    # 4. Resize and Rotate Watch Overlay Image
    aspect_ratio = watch_img.shape[1] / watch_img.shape[0]
    new_w = int(scale)
    new_h = int(scale / aspect_ratio)
    
    if new_w <= 0 or new_h <= 0:
        return frame
        
    resized_watch = cv2.resize(watch_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Rotation Matrix Setup
    M = cv2.getRotationMatrix2D((new_w // 2, new_h // 2), -angle_deg, 1.0)
    rotated_watch = cv2.warpAffine(
        resized_watch, M, (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0)
    )
    
    # 5. Overlay Blending via Alpha Channel
    top_left_x = int(cx - new_w / 2)
    top_left_y = int(cy - new_h / 2)
    
    for i in range(rotated_watch.shape[0]):
        for j in range(rotated_watch.shape[1]):
            if top_left_y + i >= h or top_left_x + j >= w or top_left_y + i < 0 or top_left_x + j < 0:
                continue
                
            alpha = rotated_watch[i, j, 3] / 255.0 if rotated_watch.shape[2] == 4 else 1.0
            if alpha > 0:
                frame[top_left_y + i, top_left_x + j] = (
                    alpha * rotated_watch[i, j, :3] + (1 - alpha) * frame[top_left_y + i, top_left_x + j]
                )
                
    return frame