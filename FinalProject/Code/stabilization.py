import cv2
import numpy as np


def extract_harris_corners(gray_img, max_corners=150, thresh=0.01):
    # compute corner harris: det(M) - k * (trace(M))^2
    dst = cv2.cornerHarris(gray_img, blockSize=2, ksize=3, k=0.04)

    h, w = gray_img.shape
    # keep the central exclusion zone to eliminate human tracking vectors
    dst[int(h * 0.2):int(h * 0.8), int(w * 0.25):int(w * 0.75)] = 0

    # threshold the map to retain only prominent local corners
    _, dst_thresh = cv2.threshold(dst, thresh * dst.max(), 255, cv2.THRESH_BINARY)
    dst_thresh = np.uint8(dst_thresh)

    # groups pixel clusters (blobs) and extracts their center-of-mass centroids
    _, _, _, centroids = cv2.connectedComponentsWithStats(dst_thresh)

    corners = centroids[1:]

    if len(corners) > max_corners:
        grid_h, grid_w = h // 4, w // 4
        allocated_corners = []
        corners_per_cell = max_corners // 16

        for r in range(4):
            for c in range(4):
                y_min, y_max = r * grid_h, (r + 1) * grid_h
                x_min, x_max = c * grid_w, (c + 1) * grid_w

                cell_mask = (corners[:, 1] >= y_min) & (corners[:, 1] < y_max) & \
                            (corners[:, 0] >= x_min) & (corners[:, 0] < x_max)
                cell_pts = corners[cell_mask]

                if len(cell_pts) > 0:
                    n_sample = min(len(cell_pts), corners_per_cell)
                    idx = np.random.choice(len(cell_pts), n_sample, replace=False)
                    allocated_corners.append(cell_pts[idx])

        if len(allocated_corners) > 0:
            corners = np.vstack(allocated_corners)

    corners = np.float32(corners).reshape(-1, 1, 2)

    # refine integer centroids to fractional sub-pixel coordinates
    if len(corners) > 0:
        subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)
        # using a small 5x5 local window search neighborhood to maximize sub-pixel convergence
        cv2.cornerSubPix(gray_img, corners, winSize=(5, 5), zeroZone=(-1, -1), criteria=subpix_criteria)

    return corners


def estimate_pair_transform(prev_gray, curr_gray, lk_params):
    # isolate corners on the preceding frame baseline
    prev_pts = extract_harris_corners(prev_gray, max_corners=150)

    if prev_pts.shape[0] == 0:
        return 0.0, 0.0, 0.0

    # use Lucas-Kanade optical flow
    curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None, **lk_params)

    valid_prev = prev_pts[status == 1]
    valid_curr = curr_pts[status == 1]

    # fits a rigid 4-DOF affine matrix using RANSAC to filter out human outlier motion vetors
    if len(valid_prev) >= 4:
        M, _ = cv2.estimateAffinePartial2D(valid_prev, valid_curr, method=cv2.RANSAC, ransacReprojThreshold=1.0)
        if M is not None:
            dx = M[0, 2]
            dy = M[1, 2]
            da = np.arctan2(M[1, 0], M[0, 0])
            return dx, dy, da

    return 0.0, 0.0, 0.0

def stabilize_video_pipeline(input_path, output_path, window_size=11):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Target file not found at: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    lk_params = dict(winSize=(21, 21), maxLevel=4,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.01))

    ret, prev_frame = cap.read()
    if not ret:
        return

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=True)
    out.write(prev_frame)

    T_total = np.eye(3, dtype=np.float32)

    for i in range(num_frames - 1):
        ret, frame = cap.read()
        if not ret:
            break
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # estimate the frame-to-frame step shift
        dx, dy, da = estimate_pair_transform(prev_gray, curr_gray, lk_params)

        # convert the step transform into a 3x3 matrix block
        T_step = np.eye(3, dtype=np.float32)
        T_step[0, 0] = np.cos(da)
        T_step[0, 1] = -np.sin(da)
        T_step[1, 0] = np.sin(da)
        T_step[1, 1] = np.cos(da)
        T_step[0, 2] = dx
        T_step[1, 2] = dy

        T_total = T_total @ T_step
        T_inv = np.linalg.inv(T_total)
        R_warp = T_inv[0:2, :]

        # warp the current frame directly back onto the Frame 0 baseline grid layout
        stabilized_frame = cv2.warpAffine(frame, R_warp, (width, height), flags=cv2.INTER_LINEAR)
        out.write(stabilized_frame)

        prev_gray = curr_gray

    cap.release()
    out.release()