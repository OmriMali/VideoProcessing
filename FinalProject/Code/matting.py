import os
import time

import cv2
import numpy as np


def _prepare_background(background_path, width, height):
    """
    Load background.jpg and resize it to match the video frame size.
    """
    background = cv2.imread(background_path)
    if background is None:
        raise FileNotFoundError(background_path)

    return cv2.resize(background, (width, height), interpolation=cv2.INTER_LINEAR)


def _binary_mask_to_alpha(binary_mask, feather_radius=7):
    """
    Convert a binary foreground mask into a soft alpha matte in [0, 1].

    A distance-transform ratio gives smoother boundary transitions than a hard
    mask, which helps reduce jagged compositing edges.
    """
    mask = (binary_mask > 127).astype(np.uint8)

    dist_in = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    dist_out = cv2.distanceTransform(1 - mask, cv2.DIST_L2, 5)

    alpha = dist_in / (dist_in + dist_out + 1e-6)

    if feather_radius > 0:
        ksize = feather_radius * 2 + 1
        alpha = cv2.GaussianBlur(alpha, (ksize, ksize), 0)

    return np.clip(alpha, 0.0, 1.0)


def _composite_frame(foreground, background, alpha):
    """
    Alpha-blend foreground and background:
        output = alpha * foreground + (1 - alpha) * background
    """
    alpha_3 = alpha[..., None].astype(np.float32)
    fg = foreground.astype(np.float32)
    bg = background.astype(np.float32)

    matted = alpha_3 * fg + (1.0 - alpha_3) * bg
    return np.clip(matted, 0, 255).astype(np.uint8)


def apply_matting(
    extracted_video_path,
    binary_video_path,
    background_image_path,
    matted_out_path,
    alpha_out_path,
    feather_radius=7,
):
    """
    Place the extracted foreground onto background.jpg and write matted/alpha videos.

    Returns
    -------
    tuple (float, float)
        Accumulated duration spent processing the alpha channel and the composite blending respectively.
    """
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

    print("[Stage 3] Compositing extracted foreground onto background...")

    # Timers for specific tasks
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

        # Benchmark Alpha Creation
        t0 = time.time()
        alpha = _binary_mask_to_alpha(binary_mask, feather_radius=feather_radius)
        alpha_uint8 = (alpha * 255.0).astype(np.uint8)
        alpha_bgr = cv2.cvtColor(alpha_uint8, cv2.COLOR_GRAY2BGR)
        out_alpha.write(alpha_bgr)
        alpha_total_time += (time.time() - t0)

        # Benchmark Compositing / Matting
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

    print("[Stage 3] Matting complete.")
    print(f" -> Internal Alpha Creation: {alpha_total_time:.2f}s")
    print(f" -> Internal Compositing: {matting_total_time:.2f}s")
    
    return alpha_total_time, matting_total_time


if __name__ == "__main__":
    code_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(code_dir)
    outputs_dir = os.path.join(root_dir, "Outputs")
    inputs_dir = os.path.join(root_dir, "Inputs")

    apply_matting(
        extracted_video_path=os.path.join(outputs_dir, "extracted_212047542_327703013.avi"),
        binary_video_path=os.path.join(outputs_dir, "binary_212047542_327703013.avi"),
        background_image_path=os.path.join(inputs_dir, "background.jpg"),
        matted_out_path=os.path.join(outputs_dir, "matted_212047542_327703013.avi"),
        alpha_out_path=os.path.join(outputs_dir, "alpha_212047542_327703013.avi"),
    )