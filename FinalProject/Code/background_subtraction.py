import cv2
import numpy as np


def _fill_holes(mask):
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return mask

    out = mask.copy()
    for i, c in enumerate(contours):
        if hierarchy[0][i][3] != -1:
            cv2.drawContours(out, [c], -1, 255, -1)
    return out


def _postprocess(mask):
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)
    mask = _fill_holes(mask)

    return mask


def _solidify(mask):
    k1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 21))

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    out = np.zeros_like(mask)
    if contours:
        c = max(contours, key=cv2.contourArea)
        cv2.drawContours(out, [c], -1, 255, -1)

    return out


def _choose_component(mask, prev_center, prev_bbox, min_area=700):
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)

    if num <= 1:
        return np.zeros_like(mask), prev_center, prev_bbox

    best_idx = None
    best_score = -1e18

    for i in range(1, num):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]

        if area < min_area:
            continue

        aspect = w / max(h, 1)
        if aspect > 3.0:
            continue

        cx, cy = centroids[i]

        score = area + 70.0 * h

        if prev_center is not None:
            dist = np.hypot(cx - prev_center[0], cy - prev_center[1])
            score -= 450.0 * dist

        if prev_bbox is not None:
            px, py, pw, ph = prev_bbox
            height_change = abs(h - ph) / max(ph, 1)
            width_change = abs(w - pw) / max(pw, 1)

            score -= 25000.0 * height_change
            score -= 12000.0 * width_change

        if score > best_score:
            best_score = score
            best_idx = i

    if best_idx is None:
        return np.zeros_like(mask), prev_center, prev_bbox

    out = np.zeros_like(mask)
    out[labels == best_idx] = 255

    bbox = (
        int(stats[best_idx, cv2.CC_STAT_LEFT]),
        int(stats[best_idx, cv2.CC_STAT_TOP]),
        int(stats[best_idx, cv2.CC_STAT_WIDTH]),
        int(stats[best_idx, cv2.CC_STAT_HEIGHT]),
    )

    return out, tuple(centroids[best_idx]), bbox


def _train_mog2(frames, history, var_threshold, detect_shadows):
    fgbg = cv2.createBackgroundSubtractorMOG2(
        history=history,
        varThreshold=var_threshold,
        detectShadows=detect_shadows,
    )

    for frame in frames:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv = cv2.GaussianBlur(hsv, (5, 5), 0)
        fgbg.apply(hsv, learningRate=0.005)

    return fgbg


def _extract_direction(frames, history, var_threshold, detect_shadows, margin):
    fgbg = _train_mog2(frames, history, var_threshold, detect_shadows)

    masks = []
    prev_center = None
    prev_bbox = None
    last_good = None

    for frame in frames:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv = cv2.GaussianBlur(hsv, (5, 5), 0)

        raw = fgbg.apply(hsv, learningRate=0.0)

        mask = np.zeros_like(raw)
        mask[raw == 255] = 255

        if margin > 0:
            mask[:margin, :] = 0
            mask[-margin:, :] = 0
            mask[:, :margin] = 0
            mask[:, -margin:] = 0

        mask = _postprocess(mask)

        person, center, bbox = _choose_component(
            mask,
            prev_center,
            prev_bbox,
            min_area=700,
        )

        if cv2.countNonZero(person) < 700 and last_good is not None:
            person = last_good.copy()
        else:
            prev_center = center
            prev_bbox = bbox
            last_good = person.copy()

        person = _solidify(person)
        masks.append(person)

    return masks


def _combine_forward_backward(forward_mask, backward_mask):
    if cv2.countNonZero(forward_mask) == 0:
        return backward_mask

    if cv2.countNonZero(backward_mask) == 0:
        return forward_mask

    overlap = cv2.bitwise_and(forward_mask, backward_mask)

    if cv2.countNonZero(overlap) > 300:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
        support = cv2.dilate(overlap, k)

        combined = cv2.bitwise_or(forward_mask, backward_mask)
        combined = cv2.bitwise_and(combined, support)
    else:
        # Fallback: use smaller mask to avoid background leaks.
        if cv2.countNonZero(forward_mask) <= cv2.countNonZero(backward_mask):
            combined = forward_mask
        else:
            combined = backward_mask

    combined = _solidify(combined)
    return combined


def apply_background_subtraction(
    stabilized_video_path,
    binary_out_path,
    extracted_out_path,
    history=500,
    var_threshold=24,
    detect_shadows=True,
    learning_rate=0.0,
    close_kernel_size=(9, 9),
    open_kernel_size=(5, 5),
    margin=25,
):
    cap = cv2.VideoCapture(stabilized_video_path)
    if not cap.isOpened():
        raise FileNotFoundError(stabilized_video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()

    if not frames:
        raise RuntimeError("No frames read from video")

    print("[Stage 2] Extracting forward masks...")
    masks_fwd = _extract_direction(
        frames,
        history,
        var_threshold,
        detect_shadows,
        margin,
    )

    print("[Stage 2] Extracting backward masks...")
    masks_bwd_rev = _extract_direction(
        list(reversed(frames)),
        history,
        var_threshold,
        detect_shadows,
        margin,
    )

    masks_bwd = list(reversed(masks_bwd_rev))

    fourcc = cv2.VideoWriter_fourcc(*"XVID")

    out_binary = cv2.VideoWriter(
        binary_out_path,
        fourcc,
        fps,
        (width, height),
        isColor=False,
    )

    out_extracted = cv2.VideoWriter(
        extracted_out_path,
        fourcc,
        fps,
        (width, height),
        isColor=True,
    )

    print("[Stage 2] Combining forward/backward masks...")

    for frame, mf, mb in zip(frames, masks_fwd, masks_bwd):
        final_mask = _combine_forward_backward(mf, mb)

        out_binary.write(final_mask)

        extracted = cv2.bitwise_and(frame, frame, mask=final_mask)
        out_extracted.write(extracted)

    out_binary.release()
    out_extracted.release()

    print("[Stage 2] Subtraction complete.")
    print(" -> Binary:", binary_out_path)
    print(" -> Extracted:", extracted_out_path)