import json
import os
import cv2
import numpy as np

N = 400 # number of particles

s_initial = [250,    # x center
             600,    # y center
              60,    # half width
              180,   # half height
               12,   # velocity x
               0]    # velocity y

SCALE_FACTOR = 0.25


def predict_particles(s_prior: np.ndarray) -> np.ndarray:
    s_prior = s_prior.astype(float)
    
    state_drifted = s_prior.copy()
    state_drifted[0, :] += state_drifted[4, :]
    state_drifted[1, :] += state_drifted[5, :]

    # scale the noise
    noise_std = np.array([[5 * SCALE_FACTOR],     # x center noise 
                          [0 * SCALE_FACTOR],     # y center
                          [0 * SCALE_FACTOR],     # half-width
                          [0 * SCALE_FACTOR],     # half-height
                          [3 * SCALE_FACTOR],     # x velocity
                          [0 * SCALE_FACTOR]])    # y velocity

    white_noise = np.random.normal(0.0, 1.0, size=s_prior.shape) * noise_std
    state_drifted += white_noise

    state_drifted = state_drifted.astype(int)
    return state_drifted


def compute_normalized_histogram(image: np.ndarray, state: np.ndarray) -> np.ndarray:
    state = np.floor(state)
    state = state.astype(int)
    hist = np.zeros((16, 16, 16))
    
    xc, yc, half_w, half_h = state[0], state[1], state[2], state[3]
    img_h, img_w, _ = image.shape

    # calculate valid region of interest (ROI) boundaries clipped to the image frame
    x_min = max(0, xc - half_w)
    x_max = min(img_w, xc + half_w)
    y_min = max(0, yc - half_h)
    y_max = min(img_h, yc + half_h)

    if x_min >= x_max or y_min >= y_max:
        return np.ones(16 * 16 * 16) / (16 * 16 * 16)

    crop = image[y_min:y_max, x_min:x_max, :]

    # calc the histogram
    hist = cv2.calcHist([crop], [0, 1, 2], None, [16, 16, 16], [0, 256, 0, 256, 0, 256])

    # flatten the histogram 
    hist = hist.reshape(16 * 16 * 16)
    hist_sum = np.sum(hist)
    
    # normalize the histogram
    if hist_sum > 0:
        hist = hist / hist_sum
    else:
        hist = np.ones(16 * 16 * 16) / (16 * 16 * 16)

    return hist


def sample_particles(previous_state: np.ndarray, cdf: np.ndarray) -> np.ndarray:
    _, N = previous_state.shape
    S_next = np.zeros(previous_state.shape)
    r_values = np.random.uniform(0.0, 1.0, size=N)

    for n in range(N):
        r = r_values[n]
        j = np.searchsorted(cdf, r)
        j = min(j, N - 1)
        S_next[:, n] = previous_state[:, j]

    return S_next


def bhattacharyya_distance(p: np.ndarray, q: np.ndarray) -> float:
    rho = np.sum(np.sqrt(p * q))
    distance = np.exp(20.0 * rho)
    return distance


def show_particles(image: np.ndarray, state: np.ndarray, W: np.ndarray, frame_index: int, ID: str,
                    frame_index_to_max_state: dict) -> tuple:
    annotated = image.copy()

    # locate the best particle containing the maximum importance weight coefficient
    max_idx = np.argmax(W)
    max_state = state[:, max_idx]
    
    # upscale downsized particle tracking space back up to full frame coordinates
    xc_max, yc_max = max_state[0] / SCALE_FACTOR, max_state[1] / SCALE_FACTOR
    half_w_max, half_h_max = max_state[2] / SCALE_FACTOR, max_state[3] / SCALE_FACTOR

    # convert center coordinates into top-left bounding box origins
    x_max = int(xc_max - half_w_max)
    y_max = int(yc_max - half_h_max)
    w_max = int(2 * half_w_max)
    h_max = int(2 * half_h_max)
    
    cv2.rectangle(annotated, (x_max - s_initial[2], y_max - s_initial[3]), ((x_max + w_max) + s_initial[2], (y_max + h_max) + s_initial[3]), (255, 255, 0), 2)
    cv2.putText(annotated, f"{ID} - Frame {frame_index}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    frame_index_to_max_state[frame_index] = [float(x) for x in [x_max - s_initial[2], y_max - s_initial[3], w_max, h_max]]
    return annotated, frame_index_to_max_state


def apply_tracking(input_video_path: str, output_video_path: str, ID: str):
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open input video: {input_video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    writer = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    frame_index_to_max_state = {}

    ret, image = cap.read()
    if not ret:
        raise RuntimeError(f"Input video contains no frames: {input_video_path}")
        
    resized_image = cv2.resize(image, (0, 0), fx=SCALE_FACTOR, fy=SCALE_FACTOR)
    s_initial_scaled = np.array(s_initial, dtype=float) * SCALE_FACTOR

    # initialize particle arrays by replicating default baseline coordinate parameters across N instances
    state_at_first_frame = np.matlib.repmat(s_initial_scaled, N, 1).T

    S = predict_particles(state_at_first_frame)
    q = compute_normalized_histogram(resized_image, s_initial_scaled)
    W = np.zeros(N)
    
    # calculate starting importance weights based on color vector overlap distance properties
    for n in range(N):
        p = compute_normalized_histogram(resized_image, S[:, n])
        W[n] = bhattacharyya_distance(p, q)

    # normalize weights arrays to guarantee valid cumulative processing steps
    if np.sum(W) > 0:
        W = W / np.sum(W)
    else:
        W[:] = 1.0 / N
        
    # construct an accumulated probability map line for index matching procedures
    C = np.cumsum(W)
    images_processed = 1

    annotated, frame_index_to_max_state = show_particles(
        image, S, W, images_processed, ID, frame_index_to_max_state)
    writer.write(annotated)
    
    while True:
        ret, current_image = cap.read()
        if not ret:
            break

        images_processed += 1
        
        resized_image = cv2.resize(current_image, (0, 0), fx=SCALE_FACTOR, fy=SCALE_FACTOR)

        S_prev = S
        S_next_tag = sample_particles(S_prev, C)
        S = predict_particles(S_next_tag)

        # update and re-weight particles according to new frame color matches
        W = np.zeros(N)
        for n in range(N):
            p = compute_normalized_histogram(resized_image, S[:, n])
            W[n] = bhattacharyya_distance(p, q)

        # renormalize array metrics
        if np.sum(W) > 0:
            W = W / np.sum(W)
        else:
            W[:] = 1.0 / N
        C = np.cumsum(W)

        annotated, frame_index_to_max_state = show_particles(
                current_image, S, W, images_processed, ID, frame_index_to_max_state)
        writer.write(annotated)
        
    cap.release()
    writer.release()
    code_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(code_dir)
    outputs_dir = os.path.join(root_dir, "Outputs")
    RESULTS = 'results'
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(outputs_dir, 'tracking.json'), 'w') as f:
        json.dump(frame_index_to_max_state, f, indent=4)

