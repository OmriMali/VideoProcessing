import cv2
import numpy as np

def _fill_holes(mask):
    contours, hierarchy = cv2.findContours(
        mask,
        cv2.RETR_CCOMP,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if hierarchy is None:
        return mask

    out = mask.copy()

    for i, contour in enumerate(contours):
        # hierarchy[0][i][3] is the parent contour index
        # if it is not -1, this contour is inside another contour means it is a hole inside a foreground object
        if hierarchy[0][i][3] != -1:
            cv2.drawContours(out, [contour], -1, 255, -1)

    return out



def _postprocess(mask):
    # elliptical opening kernel removes tiny white speckles
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    # slightly larger closing kernel reconnects fragmented body parts such as arms, torso, and legs
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))

    # erosion followed by dilation for removing small noise
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open)

    # dilation followed by erosion for filling small gaps
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)

    # fill internal black holes that remain inside the selected foreground
    mask = _fill_holes(mask)

    return mask



def _solidify(mask):
    # closing kernel for small gaps.
    k1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))

    # closing kernel helps connect vertical body parts, especially legs and torso
    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 21))

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k2)

    # avoids drawing holes as separate objects
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    out = np.zeros_like(mask)

    if contours:
        # the selected person should be the largest remaining external contour
        largest_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(out, [largest_contour], -1, 255, -1)

    return out



def _choose_component(mask, prev_center, prev_bbox, min_area=700):
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)

    if num <= 1:
        return np.zeros_like(mask), prev_center, prev_bbox

    best_idx = None
    best_score = -1e18

    for i in range(1, num):
        # extract geometric properties of this connected component.
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]

        # reject tiny blobs caused by noise
        if area < min_area:
            continue

        # reject very flat/wide components bacause they are unlikely to be a person
        aspect = w / max(h, 1)
        if aspect > 3.0:
            continue

        cx, cy = centroids[i]

        # base score- prefer large and tall components
        score = area + 70.0 * h

        if prev_center is not None:
            # prefer components close to the previous person location
            dist = np.hypot(cx - prev_center[0], cy - prev_center[1])
            score -= 450.0 * dist

        if prev_bbox is not None:
            # prefer components whose size is consistent with the previous frame
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
    # mog2 model
    fgbg = cv2.createBackgroundSubtractorMOG2(
        history=history,
        varThreshold=var_threshold,
        detectShadows=detect_shadows,
    )

    for frame in frames:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # gaussian blur for reducing pixel level noise and tiny stabilization
        hsv = cv2.GaussianBlur(hsv, (5, 5), 0)
        fgbg.apply(hsv, learningRate=0.005)

    return fgbg



def _extract_direction(frames, history, var_threshold, detect_shadows, margin):
    # use mog2 model
    fgbg = _train_mog2(frames, history, var_threshold, detect_shadows)

    masks = []
    prev_center = None
    prev_bbox = None
    last_good = None

    for frame in frames:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # gaussian blur for reducing pixel level noise and tiny stabilization
        hsv = cv2.GaussianBlur(hsv, (5, 5), 0)
        raw = fgbg.apply(hsv, learningRate=0.0)

        # convert to binary mask
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

        # fill gaps and keep the person as a single solid silhouette
        person = _solidify(person)
        masks.append(person)

    return masks



def _combine_forward_backward(forward_mask, backward_mask):
    # if one direction fails use the other direction
    if cv2.countNonZero(forward_mask) == 0:
        return backward_mask

    if cv2.countNonZero(backward_mask) == 0:
        return forward_mask

    # agreement between the two temporal directions.
    overlap = cv2.bitwise_and(forward_mask, backward_mask)

    if cv2.countNonZero(overlap) > 300:
        # expand the agreement region so that true pixels that appear in only one pass can still be used if they close to the overlap
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
        support = cv2.dilate(overlap, k)

        # keep the union only where near the reliable support region
        combined = cv2.bitwise_or(forward_mask, backward_mask)
        combined = cv2.bitwise_and(combined, support)
    else:
        # use the smaller mask to avoid background leaks
        if cv2.countNonZero(forward_mask) <= cv2.countNonZero(backward_mask):
            combined = forward_mask
        else:
            combined = backward_mask

    combined = _solidify(combined)
    return combined



def apply_background_subtraction(stabilized_video_path, binary_out_path, extracted_out_path, 
                    history=500, var_threshold=24, detect_shadows=True, margin=25):

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

    # forward pass - extract in normal temporal order
    masks_fwd = _extract_direction(
        frames,
        history,
        var_threshold,
        detect_shadows,
        margin,
    )

    # backward pass - extract in reverse temporal order
    masks_bwd_rev = _extract_direction(
        list(reversed(frames)),
        history,
        var_threshold,
        detect_shadows,
        margin,
    )

    # reverse the backward masks back into the original frame order
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

    if not out_binary.isOpened() or not out_extracted.isOpened():
        out_binary.release()
        out_extracted.release()
        raise IOError("Could not open output video writers")

    for frame, mf, mb in zip(frames, masks_fwd, masks_bwd):
        # combine the two temporal estimates into one final mask
        final_mask = _combine_forward_backward(mf, mb)

        # binary output
        out_binary.write(final_mask)

        # Color extraction output- keeps original pixels only where the mask is 255
        extracted = cv2.bitwise_and(frame, frame, mask=final_mask)
        out_extracted.write(extracted)

    out_binary.release()
    out_extracted.release()
