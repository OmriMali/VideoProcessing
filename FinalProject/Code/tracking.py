import json
import os
import cv2
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm

ID = "OUTPUT"
RESULTS = 'results'
os.makedirs(RESULTS, exist_ok=True)
code_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(code_dir)
outputs_dir = os.path.join(root_dir, "Outputs")
input_path = os.path.join(outputs_dir, "matted_212047542_327703013.avi")
output_path = os.path.join(outputs_dir, "OUTPUT_212047542_327703013.avi")
VIDEO_PATH = os.path.join(outputs_dir, "matted_212047542_327703013.avi")
OUTPUT_VIDEO_PATH = os.path.join(outputs_dir, "OUTPUT_212047542_327703013.avi")

# SET NUMBER OF PARTICLES
N = 400

# Initial Settings (for original full-size frame)
s_initial = [250,    # x center
             600,    # y center
              120,    # half width
              370,    # half height
               8,    # velocity x
               0]    # velocity y

SCALE_FACTOR = 0.25


def predict_particles(s_prior: np.ndarray) -> np.ndarray:
    s_prior = s_prior.astype(float)
    
    state_drifted = s_prior.copy()
    state_drifted[0, :] += state_drifted[4, :]
    state_drifted[1, :] += state_drifted[5, :]

    # Scale the noise standard deviation to match the downscaled coordinate space
    noise_std = np.array([[0 * SCALE_FACTOR],   # x center noise 
                          [0 * SCALE_FACTOR],    # Y center
                          [0 * SCALE_FACTOR],   # Half-width
                          [0 * SCALE_FACTOR],    # Half-height
                          [0 * SCALE_FACTOR],   # X velocity
                          [0 * SCALE_FACTOR]])   # Y velocity

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

    x_min = max(0, xc - half_w)
    x_max = min(img_w, xc + half_w)
    y_min = max(0, yc - half_h)
    y_max = min(img_h, yc + half_h)

    if x_min >= x_max or y_min >= y_max:
        return np.ones(16 * 16 * 16) / (16 * 16 * 16)

    crop = image[y_min:y_max, x_min:x_max, :]

    hist = cv2.calcHist([crop], [0, 1, 2], None, [16, 16, 16], [0, 256, 0, 256, 0, 256])

    # Flatten and normalize
    hist = hist.reshape(16 * 16 * 16)
    hist_sum = np.sum(hist)
    
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
                  frame_index_to_mean_state: dict, frame_index_to_max_state: dict) -> tuple:
    annotated = image.copy()

    # Calculate average state in low-res, then map to high-res by dividing by SCALE_FACTOR
    mean_state = np.sum(state * W, axis=1)
    xc_avg, yc_avg = mean_state[0] / SCALE_FACTOR, mean_state[1] / SCALE_FACTOR
    half_w_avg, half_h_avg = mean_state[2] / SCALE_FACTOR, mean_state[3] / SCALE_FACTOR

    x_avg = int(xc_avg - half_w_avg)
    y_avg = int(yc_avg - half_h_avg)
    w_avg = int(2 * half_w_avg)
    h_avg = int(2 * half_h_avg)
   
    # Map max state back to high resolution
    max_idx = np.argmax(W)
    max_state = state[:, max_idx]
    xc_max, yc_max = max_state[0] / SCALE_FACTOR, max_state[1] / SCALE_FACTOR
    half_w_max, half_h_max = max_state[2] / SCALE_FACTOR, max_state[3] / SCALE_FACTOR

    x_max = int(xc_max - half_w_max)
    y_max = int(yc_max - half_h_max)
    w_max = int(2 * half_w_max)
    h_max = int(2 * half_h_max)
    
    cv2.rectangle(annotated, (x_max, y_max), (x_max + w_max, y_max + h_max), (255, 255, 0), 2)

    cv2.putText(annotated, f"{ID} - Frame {frame_index}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    frame_index_to_mean_state[frame_index] = [float(x) for x in [x_avg, y_avg, w_avg, h_avg]]
    frame_index_to_max_state[frame_index] = [float(x) for x in [x_max, y_max, w_max, h_max]]
    return annotated, frame_index_to_mean_state, frame_index_to_max_state


def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open input video: {VIDEO_PATH}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    writer = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (frame_width, frame_height))

    frame_index_to_avg_state = {}
    frame_index_to_max_state = {}

    ret, image = cap.read()
    if not ret:
        raise RuntimeError(f"Input video contains no frames: {VIDEO_PATH}")
        
    resized_image = cv2.resize(image, (0, 0), fx=SCALE_FACTOR, fy=SCALE_FACTOR)

    # Convert initial state configuration to the downscaled resolution space
    s_initial_scaled = np.array(s_initial, dtype=float) * SCALE_FACTOR

    state_at_first_frame = np.matlib.repmat(s_initial_scaled, N, 1).T
    S = predict_particles(state_at_first_frame)

    q = compute_normalized_histogram(resized_image, s_initial_scaled)
    W = np.zeros(N)
    for n in range(N):
        p = compute_normalized_histogram(resized_image, S[:, n])
        W[n] = bhattacharyya_distance(p, q)

    if np.sum(W) > 0:
        W = W / np.sum(W)
    else:
        W[:] = 1.0 / N
    C = np.cumsum(W)
    images_processed = 1

    annotated, frame_index_to_avg_state, frame_index_to_max_state = show_particles(
        image, S, W, images_processed, ID, frame_index_to_avg_state, frame_index_to_max_state)
    writer.write(annotated)
    
    with tqdm(desc="Processing frames", unit="frame") as pbar:
        while True:
            ret, current_image = cap.read()
            if not ret:
                break

            images_processed += 1
            pbar.update(1)
            
            resized_image = cv2.resize(current_image, (0, 0), fx=SCALE_FACTOR, fy=SCALE_FACTOR)
            S_prev = S
            S_next_tag = sample_particles(S_prev, C)
            S = predict_particles(S_next_tag)

            W = np.zeros(N)
            for n in range(N):
                p = compute_normalized_histogram(resized_image, S[:, n])
                W[n] = bhattacharyya_distance(p, q)

            if np.sum(W) > 0:
                W = W / np.sum(W)
            else:
                W[:] = 1.0 / N
            C = np.cumsum(W)

            annotated, frame_index_to_avg_state, frame_index_to_max_state = show_particles(
                current_image, S, W, images_processed, ID, frame_index_to_avg_state, frame_index_to_max_state)
            writer.write(annotated)

    cap.release()
    writer.release()

    with open(os.path.join(RESULTS, 'frame_index_to_avg_state.json'), 'w') as f:
        json.dump(frame_index_to_avg_state, f, indent=4)
    with open(os.path.join(RESULTS, 'frame_index_to_max_state.json'), 'w') as f:
        json.dump(frame_index_to_max_state, f, indent=4)

    print(f"Tracking complete. Output video: {OUTPUT_VIDEO_PATH}")


if __name__ == "__main__":
    main()