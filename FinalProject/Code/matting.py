import os
import time

import cv2
import numpy as np


def _prepare_background(background_path, width, height):
    background = cv2.imread(background_path)
    if background is None:
        raise FileNotFoundError(background_path)

    return cv2.resize(background, (width, height), interpolation=cv2.INTER_LINEAR)


def _binary_mask_to_alpha(binary_mask, feather_radius):
    mask = (binary_mask > 127).astype(np.uint8)

    # calc L2 distance to the nearest zero pixel (inside foreground)
    dist_in = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    # calc L2 distance to the nearest non-zero pixel (outside foreground)
    dist_out = cv2.distanceTransform(1 - mask, cv2.DIST_L2, 5)

    alpha = dist_in / (dist_in + dist_out + 1e-6)

    # gaussian blur
    if feather_radius > 0:
        ksize = feather_radius * 2 + 1
        alpha = cv2.GaussianBlur(alpha, (ksize, ksize), 0)

    return np.clip(alpha, 0.0, 1.0)


def _composite_frame(foreground, background, alpha):
    # expand alpha dimensions from (H, W) to (H, W, 1) 
    alpha_3 = alpha[..., None].astype(np.float32)
    fg = foreground.astype(np.float32)
    bg = background.astype(np.float32)

    # linear blending equation: α * FG + (1 - α) * BG
    matted = alpha_3 * fg + (1.0 - alpha_3) * bg
    
    return np.clip(matted, 0, 255).astype(np.uint8)

def apply_matting( extracted_video_path, binary_video_path, background_image_path, matted_out_path, 
                  alpha_out_path, feather_radius=5):
    
    cap_extracted = cv2.VideoCapture(extracted_video_path)
    cap_binary = cv2.VideoCapture(binary_video_path)

    if not cap_extracted.isOpened():
        raise FileNotFoundError(extracted_video_path)
    if not cap_binary.isOpened():
        raise FileNotFoundError(binary_video_path)

    fps = cap_extracted.get(cv2.CAP_PROP_FPS)
    width = int(cap_extracted.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_extracted.get(cv2.CAP_PROP_FRAME_HEIGHT))

    background = _prepare_background(background_image_path, width, height)

    fourcc = cv2.VideoWriter_fourcc(*"XVID")

    out_matted = cv2.VideoWriter(
        matted_out_path,
        fourcc,
        fps,
        (width, height),
        isColor=True,
    )
    out_alpha = cv2.VideoWriter(
        alpha_out_path,
        fourcc,
        fps,
        (width, height),
        isColor=True,
    )

    if not out_matted.isOpened() or not out_alpha.isOpened():
        cap_extracted.release()
        cap_binary.release()
        out_matted.release()
        out_alpha.release()
        raise IOError("Could not open output video writers")

    alpha_total_time = 0.0
    matting_total_time = 0.0

    frame_idx = 0
    while True:
        ret_fg, foreground = cap_extracted.read()
        ret_mask, binary_frame = cap_binary.read()

        if not ret_fg or not ret_mask:
            break

        if binary_frame.ndim == 3:
            binary_mask = cv2.cvtColor(binary_frame, cv2.COLOR_BGR2GRAY)
        else:
            binary_mask = binary_frame

        # alpha creation
        t0 = time.time()
        alpha = _binary_mask_to_alpha(binary_mask, feather_radius=feather_radius)
        alpha_uint8 = (alpha * 255.0).astype(np.uint8)
        alpha_bgr = cv2.cvtColor(alpha_uint8, cv2.COLOR_GRAY2BGR)
        out_alpha.write(alpha_bgr)
        alpha_total_time += (time.time() - t0)

        # matting
        t1 = time.time()
        matted = _composite_frame(foreground, background, alpha)
        out_matted.write(matted)
        matting_total_time += (time.time() - t1)

        frame_idx += 1

    cap_extracted.release()
    cap_binary.release()
    out_matted.release()
    out_alpha.release()

    if frame_idx == 0:
        raise RuntimeError("No frames read for matting")

    return alpha_total_time, matting_total_time
