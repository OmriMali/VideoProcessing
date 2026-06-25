import cv2
import numpy as np


# =============================================================================
# Background Subtraction / Foreground Extraction
# =============================================================================
#
# Goal:
#   Extract one moving person from a stabilized video and write two outputs:
#       1. A binary mask video.
#       2. A color video where only the extracted person is visible.
#
# Main idea:
#   The algorithm uses OpenCV's MOG2 background subtractor, but it does not rely
#   on a single temporal direction. Instead, it extracts masks twice:
#       - once from the video in normal forward order,
#       - once from the video in reverse order,
#   and then combines both masks.
#
# Why forward + backward?
#   Background models can fail at the beginning or end of a sequence because the
#   model has different information depending on the temporal direction. A pixel
#   that is poorly modeled in the forward pass may be better modeled in the
#   backward pass. Combining the two gives a more stable result than one pass.
#
# Important limitation:
#   This is still classical background subtraction. If stabilization has residual
#   jitter, or if the person moves over high-contrast background objects, some
#   background leakage can still occur.
# =============================================================================


def _fill_holes(mask):
    """
    Fill internal holes inside foreground regions.

    Parameters
    ----------
    mask : np.ndarray
        Binary foreground mask. Foreground pixels should be 255 and background
        pixels should be 0.

    Returns
    -------
    np.ndarray
        Binary mask with holes inside connected foreground blobs filled.

    Explanation
    -----------
    cv2.RETR_CCOMP returns a two-level contour hierarchy:
        - external contours are object boundaries,
        - child contours are holes inside those objects.

    For every contour whose parent is not -1, we know it is an internal hole.
    Drawing that hole in white fills it.
    """

    contours, hierarchy = cv2.findContours(
        mask,
        cv2.RETR_CCOMP,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    # If there are no contours, there are no holes to fill.
    if hierarchy is None:
        return mask

    out = mask.copy()

    for i, contour in enumerate(contours):
        # hierarchy[0][i][3] is the parent contour index.
        # If it is not -1, this contour is inside another contour, meaning it is
        # a hole inside a foreground object.
        if hierarchy[0][i][3] != -1:
            cv2.drawContours(out, [contour], -1, 255, -1)

    return out



def _postprocess(mask):
    """
    Clean a raw foreground mask after background subtraction.

    This function performs three operations:
        1. Morphological opening: removes small isolated noise.
        2. Morphological closing: reconnects nearby foreground parts.
        3. Hole filling: fills black gaps inside the foreground body.

    Parameters
    ----------
    mask : np.ndarray
        Raw binary foreground mask.

    Returns
    -------
    np.ndarray
        Cleaned binary foreground mask.
    """

    # Small elliptical opening kernel removes tiny white speckles.
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    # Slightly larger closing kernel reconnects fragmented body parts such as
    # arms, torso, and legs.
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))

    # Opening = erosion followed by dilation. It removes small noise.
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open)

    # Closing = dilation followed by erosion. It fills small gaps.
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)

    # Fill internal black holes that remain inside the selected foreground.
    mask = _fill_holes(mask)

    return mask



def _solidify(mask):
    """
    Convert the selected person component into a more solid silhouette.

    Parameters
    ----------
    mask : np.ndarray
        Binary mask that should already contain mostly the selected person.

    Returns
    -------
    np.ndarray
        A solid single-component mask.

    Explanation
    -----------
    Background subtraction often creates holes inside the person because parts of
    the clothing may be similar to the background. This function strengthens the
    mask after the person component has already been selected.

    It intentionally keeps only the largest external contour, because after
    tracking we assume the dominant component is the person.
    """

    # General closing kernel for small gaps.
    k1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))

    # Taller closing kernel helps connect vertical body parts, especially legs
    # and torso, without expanding too aggressively sideways.
    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 21))

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k2)

    # Keep only external contours. This avoids drawing holes as separate objects.
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    out = np.zeros_like(mask)

    if contours:
        # The selected person should be the largest remaining external contour.
        largest_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(out, [largest_contour], -1, 255, -1)

    return out



def _choose_component(mask, prev_center, prev_bbox, min_area=700):
    """
    Select the connected component that is most likely to be the person.

    Parameters
    ----------
    mask : np.ndarray
        Binary foreground mask after MOG2 and postprocessing.
    prev_center : tuple[float, float] or None
        Center of the person in the previous frame.
    prev_bbox : tuple[int, int, int, int] or None
        Previous bounding box in the form (x, y, width, height).
    min_area : int
        Minimum component area allowed. Small components are treated as noise.

    Returns
    -------
    person_mask : np.ndarray
        Binary mask containing only the chosen component.
    center : tuple[float, float]
        Centroid of the chosen component.
    bbox : tuple[int, int, int, int]
        Bounding box of the chosen component.

    Scoring logic
    -------------
    Each foreground component receives a score. The score rewards:
        - large area,
        - tall height, because a standing person is vertically extended.

    The score penalizes:
        - large distance from the previous person's center,
        - sudden height changes,
        - sudden width changes.

    This makes the extraction temporally stable and prevents the tracker from
    jumping to a random wall/poster artifact.
    """

    num, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)

    # num includes label 0, which is the background. If there is only label 0,
    # no foreground component exists.
    if num <= 1:
        return np.zeros_like(mask), prev_center, prev_bbox

    best_idx = None
    best_score = -1e18

    for i in range(1, num):
        # Extract geometric properties of this connected component.
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]

        # Reject tiny blobs caused by noise or residual background jitter.
        if area < min_area:
            continue

        # Reject very flat/wide components. A walking person may become wider
        # during motion, but a width much larger than height is usually wall or
        # floor leakage.
        aspect = w / max(h, 1)
        if aspect > 3.0:
            continue

        cx, cy = centroids[i]

        # Base score: prefer large and tall components.
        score = area + 70.0 * h

        if prev_center is not None:
            # Prefer components close to the previous person location.
            dist = np.hypot(cx - prev_center[0], cy - prev_center[1])
            score -= 450.0 * dist

        if prev_bbox is not None:
            # Prefer components whose size is consistent with the previous frame.
            px, py, pw, ph = prev_bbox
            height_change = abs(h - ph) / max(ph, 1)
            width_change = abs(w - pw) / max(pw, 1)

            score -= 25000.0 * height_change
            score -= 12000.0 * width_change

        if score > best_score:
            best_score = score
            best_idx = i

    # If all components were rejected, return an empty mask and keep the previous
    # tracker state unchanged.
    if best_idx is None:
        return np.zeros_like(mask), prev_center, prev_bbox

    # Create a mask containing only the selected label.
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
    """
    Train a MOG2 background model on a sequence of frames.

    Parameters
    ----------
    frames : list[np.ndarray]
        Frames used to train the background model.
    history : int
        Number of frames used internally by MOG2 for background statistics.
    var_threshold : float
        MOG2 foreground sensitivity threshold. Lower values are more sensitive.
    detect_shadows : bool
        Whether MOG2 should classify shadows separately.

    Returns
    -------
    cv2.BackgroundSubtractorMOG2
        Trained background subtractor.

    Notes
    -----
    Frames are converted to HSV before training. HSV separates color information
    from brightness better than raw BGR, which can help when illumination changes
    slightly.
    """

    fgbg = cv2.createBackgroundSubtractorMOG2(
        history=history,
        varThreshold=var_threshold,
        detectShadows=detect_shadows,
    )

    for frame in frames:
        # HSV is used consistently both for training and extraction.
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Mild blur reduces pixel-level noise and tiny stabilization jitter.
        hsv = cv2.GaussianBlur(hsv, (5, 5), 0)

        # A small learning rate prevents one frame from dominating the model.
        fgbg.apply(hsv, learningRate=0.005)

    return fgbg



def _extract_direction(frames, history, var_threshold, detect_shadows, margin):
    """
    Extract person masks from a frame sequence in one temporal direction.

    Parameters
    ----------
    frames : list[np.ndarray]
        Input frames. This may be the original order or reversed order.
    history : int
        MOG2 history parameter.
    var_threshold : float
        MOG2 variance threshold.
    detect_shadows : bool
        MOG2 shadow-detection flag.
    margin : int
        Number of pixels to suppress near the image borders.

    Returns
    -------
    list[np.ndarray]
        One binary person mask per input frame.

    Processing steps
    ----------------
    1. Train MOG2 on this temporal direction.
    2. For each frame, compute a raw foreground mask with learningRate=0.
    3. Remove shadows and uncertain values by keeping only value 255.
    4. Remove stabilization-border artifacts.
    5. Clean the mask morphologically.
    6. Choose the component most consistent with the tracked person.
    7. Solidify the selected component.
    """

    # Train a separate model for this temporal direction.
    fgbg = _train_mog2(frames, history, var_threshold, detect_shadows)

    masks = []

    # Tracking state used to keep the selected component temporally consistent.
    prev_center = None
    prev_bbox = None
    last_good = None

    for frame in frames:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv = cv2.GaussianBlur(hsv, (5, 5), 0)

        # learningRate=0 freezes the trained model during extraction. This makes
        # the extraction deterministic for this pass.
        raw = fgbg.apply(hsv, learningRate=0.0)

        # MOG2 output values:
        #   0   = background
        #   127 = shadow, if detectShadows=True
        #   255 = foreground
        # We keep only true foreground pixels.
        mask = np.zeros_like(raw)
        mask[raw == 255] = 255

        # Suppress borders. Stabilization often creates unreliable pixels near
        # the image edges due to warping/cropping.
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

        # If the current frame fails to produce a reasonable person component,
        # reuse the last good mask rather than outputting a completely empty or
        # unstable mask.
        if cv2.countNonZero(person) < 700 and last_good is not None:
            person = last_good.copy()
        else:
            prev_center = center
            prev_bbox = bbox
            last_good = person.copy()

        # Fill gaps and keep the person as a single solid silhouette.
        person = _solidify(person)
        masks.append(person)

    return masks



def _combine_forward_backward(forward_mask, backward_mask):
    """
    Combine masks produced by the forward and backward passes.

    Parameters
    ----------
    forward_mask : np.ndarray
        Mask extracted from the normal temporal direction.
    backward_mask : np.ndarray
        Mask extracted from the reversed temporal direction.

    Returns
    -------
    np.ndarray
        Final combined mask.

    Combination strategy
    --------------------
    If both masks overlap, their overlap is treated as reliable support. The
    overlap is dilated slightly, and the union of both masks is clipped to that
    support region. This keeps pixels that are near the agreement region while
    rejecting distant leakage.

    If the two masks do not overlap, the smaller mask is used, because large masks
    are more likely to contain background leakage.
    """

    # If one direction completely failed, trust the other direction.
    if cv2.countNonZero(forward_mask) == 0:
        return backward_mask

    if cv2.countNonZero(backward_mask) == 0:
        return forward_mask

    # Agreement between the two temporal directions.
    overlap = cv2.bitwise_and(forward_mask, backward_mask)

    if cv2.countNonZero(overlap) > 300:
        # Expand the agreement region so that true body pixels that appear in
        # only one pass can still be kept if they are close to the overlap.
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
        support = cv2.dilate(overlap, k)

        # Keep the union, but only near the reliable support region.
        combined = cv2.bitwise_or(forward_mask, backward_mask)
        combined = cv2.bitwise_and(combined, support)
    else:
        # Fallback: use the smaller mask to avoid background leaks.
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
    """
    Main entry point for Stage 2 foreground extraction.

    Parameters
    ----------
    stabilized_video_path : str
        Path to the stabilized input video.
    binary_out_path : str
        Path where the binary mask video will be saved.
    extracted_out_path : str
        Path where the color extracted-foreground video will be saved.
    history : int
        MOG2 history parameter.
    var_threshold : float
        MOG2 variance threshold. Lower values detect more motion but may also
        create more false foreground.
    detect_shadows : bool
        Whether MOG2 should mark shadows separately.
    learning_rate : float
        Kept for interface compatibility with earlier versions. This specific
        implementation uses fixed learning rates internally in _train_mog2 and
        _extract_direction.
    close_kernel_size : tuple[int, int]
        Kept for interface compatibility with earlier versions. The current
        helper functions define their own kernels internally.
    open_kernel_size : tuple[int, int]
        Kept for interface compatibility with earlier versions. The current
        helper functions define their own kernels internally.
    margin : int
        Number of pixels removed from each image border to avoid stabilization
        artifacts.

    Output
    ------
    The function writes two AVI files:
        - binary_out_path: white foreground on black background.
        - extracted_out_path: original color pixels where the mask is white.
    """

    cap = cv2.VideoCapture(stabilized_video_path)
    if not cap.isOpened():
        raise FileNotFoundError(stabilized_video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Read all frames into memory because the algorithm needs both forward and
    # backward temporal passes. For very long videos, this could be replaced with
    # temporary files or chunked processing.
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()

    if not frames:
        raise RuntimeError("No frames read from video")

    # Forward pass: model and extract in normal temporal order.
    print("[Stage 2] Extracting forward masks...")
    masks_fwd = _extract_direction(
        frames,
        history,
        var_threshold,
        detect_shadows,
        margin,
    )

    # Backward pass: model and extract in reverse temporal order. This helps when
    # the forward background model is weak at the end of the video.
    print("[Stage 2] Extracting backward masks...")
    masks_bwd_rev = _extract_direction(
        list(reversed(frames)),
        history,
        var_threshold,
        detect_shadows,
        margin,
    )

    # Reverse the backward masks back into the original frame order so each frame
    # can be combined with its corresponding forward mask.
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

    print("[Stage 2] Combining forward/backward masks...")

    for frame, mf, mb in zip(frames, masks_fwd, masks_bwd):
        # Combine the two temporal estimates into one final mask.
        final_mask = _combine_forward_backward(mf, mb)

        # Binary output: stores only the foreground mask.
        out_binary.write(final_mask)

        # Color extraction output: keeps original pixels only where the mask is 255.
        extracted = cv2.bitwise_and(frame, frame, mask=final_mask)
        out_extracted.write(extracted)

    out_binary.release()
    out_extracted.release()

    print("[Stage 2] Subtraction complete.")
    print(" -> Binary:", binary_out_path)
    print(" -> Extracted:", extracted_out_path)
