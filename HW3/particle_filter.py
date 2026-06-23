import json
import os
import cv2
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# change IDs to your IDs.
ID1 = "212047542"
ID2 = "327703013"

ID = "HW3_212047542_327703013".format(ID1, ID2)
RESULTS = 'results'
os.makedirs(RESULTS, exist_ok=True)
IMAGE_DIR_PATH = "Images"

# SET NUMBER OF PARTICLES
N = 100

# Initial Settings
s_initial = [297,    # x center
             139,    # y center
              16,    # half width
              43,    # half height
               0,    # velocity x
               0]    # velocity y


def predict_particles(s_prior: np.ndarray) -> np.ndarray:
    """Progress the prior state with time and add noise.

    Note that we explicitly did not tell you how to add the noise.
    We allow additional manipulations to the state if you think these are necessary.

    Args:
        s_prior: np.ndarray. The prior state.
    Return:
        state_drifted: np.ndarray. The prior state after drift (applying the motion model) and adding the noise.
    """
    s_prior = s_prior.astype(float)

    """ DELETE THE LINE ABOVE AND:
    INSERT YOUR CODE HERE."""
    
    state_drifted = s_prior.copy()
    state_drifted[0, :] += state_drifted[4, :]
    state_drifted[1, :] += state_drifted[5, :]

    noise_std = np.array([[1.0],   # x center noise 
                          [1.0],   # Y center
                          [0.0],   # Half-width
                          [0.0],   # Half-height
                          [1.0],   # X velocity
                          [1.0]])  # Y velocity

    white_noise = np.random.normal(0.0, 1.0, size=s_prior.shape) * noise_std
    state_drifted += white_noise

    state_drifted = state_drifted.astype(int)
    return state_drifted


def compute_normalized_histogram(image: np.ndarray, state: np.ndarray) -> np.ndarray:
    """Compute the normalized histogram using the state parameters.

    Args:
        image: np.ndarray. The image we want to crop the rectangle from.
        state: np.ndarray. State candidate.

    Return:
        hist: np.ndarray. histogram of quantized colors.
    """
    state = np.floor(state)
    state = state.astype(int)
    hist = np.zeros((16, 16, 16))
    """ DELETE THE LINE ABOVE AND:
        INSERT YOUR CODE HERE."""
    xc, yc, half_w, half_h = state[0], state[1], state[2], state[3]
    img_h, img_w, _ = image.shape

    x_min = max(0, xc - half_w)
    x_max = min(img_w, xc + half_w)
    y_min = max(0, yc - half_h)
    y_max = min(img_h, yc + half_h)

    if x_min >= x_max or y_min >= y_max:
        hist = np.ones(16 * 16 * 16) / (16 * 16 * 16)
        return hist

    crop = image[y_min:y_max, x_min:x_max, :]
    quantized = (crop // 16).astype(int)

    for h in range(quantized.shape[0]):
        for w in range(quantized.shape[1]):
            ch_b, ch_g, ch_r = quantized[h, w, 0], quantized[h, w, 1], quantized[h, w, 2]
            hist[ch_b, ch_g, ch_r] += 1

    hist = np.reshape(hist, 16 * 16 * 16)
    hist = hist/sum(hist)

    return hist


def sample_particles(previous_state: np.ndarray, cdf: np.ndarray) -> np.ndarray:
    """Sample particles from the previous state according to the cdf.

    If additional processing to the returned state is needed - feel free to do it.

    Args:
        previous_state: np.ndarray. previous state, shape: (6, N)
        cdf: np.ndarray. cummulative distribution function: (N, )

    Return:
        s_next: np.ndarray. Sampled particles. shape: (6, N)
    """

    """ DELETE THE LINE ABOVE AND:
        INSERT YOUR CODE HERE."""
    
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
    """Calculate Bhattacharyya Distance between two histograms p and q.

    Args:
        p: np.ndarray. first histogram.
        q: np.ndarray. second histogram.

    Return:
        distance: float. The Bhattacharyya Distance.
    """

    """ DELETE THE LINE ABOVE AND:
        INSERT YOUR CODE HERE."""
    
    rho = np.sum(np.sqrt(p * q))
    distance = np.exp(20.0 * rho)

    return distance


def show_particles(image: np.ndarray, state: np.ndarray, W: np.ndarray, frame_index: int, ID: str,
                  frame_index_to_mean_state: dict, frame_index_to_max_state: dict,
                  ) -> tuple:
    fig, ax = plt.subplots(1)
    image = image[:,:,::-1]
    plt.imshow(image)
    plt.title(ID + " - Frame mumber = " + str(frame_index))

    # Avg particle box

    """ DELETE THE LINE ABOVE AND:
        INSERT YOUR CODE HERE."""

    mean_state = np.sum(state * W, axis=1)
    xc_avg, yc_avg, half_w_avg, half_h_avg = mean_state[0], mean_state[1], mean_state[2], mean_state[3]

    x_avg = xc_avg - half_w_avg
    y_avg = yc_avg - half_h_avg
    w_avg = 2 * half_w_avg
    h_avg = 2 * half_h_avg

    rect = patches.Rectangle((x_avg, y_avg), w_avg, h_avg, linewidth=1, edgecolor='g', facecolor='none')
    ax.add_patch(rect)

    # calculate Max particle box

    """ DELETE THE LINE ABOVE AND:
        INSERT YOUR CODE HERE."""

    max_idx = np.argmax(W)
    max_state = state[:, max_idx]
    xc_max, yc_max, half_w_max, half_h_max = max_state[0], max_state[1], max_state[2], max_state[3]

    x_max = xc_max - half_w_max
    y_max = yc_max - half_h_max
    w_max = 2 * half_w_max
    h_max = 2 * half_h_max


    rect = patches.Rectangle((x_max, y_max), w_max, h_max, linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    plt.show(block=False)

    fig.savefig(os.path.join(RESULTS, ID + "-" + str(frame_index) + ".png"))
    frame_index_to_mean_state[frame_index] = [float(x) for x in [x_avg, y_avg, w_avg, h_avg]]
    frame_index_to_max_state[frame_index] = [float(x) for x in [x_max, y_max, w_max, h_max]]
    return frame_index_to_mean_state, frame_index_to_max_state


def main():
    state_at_first_frame = np.matlib.repmat(s_initial, N, 1).T
    S = predict_particles(state_at_first_frame)

    # LOAD FIRST IMAGE
    image = cv2.imread(os.path.join(IMAGE_DIR_PATH, "001.png"))

    # COMPUTE NORMALIZED HISTOGRAM
    q = compute_normalized_histogram(image, s_initial)

    # COMPUTE NORMALIZED WEIGHTS (W) AND PREDICTOR CDFS (C)
    # YOU NEED TO FILL THIS PART WITH CODE:
    """INSERT YOUR CODE HERE."""
    W = np.zeros(N)
    for n in range(N):
        p = compute_normalized_histogram(image, S[:, n])
        W[n] = bhattacharyya_distance(p, q)

    W /= np.sum(W)
    C = np.cumsum(W)
    images_processed = 1

    # MAIN TRACKING LOOP
    image_name_list = os.listdir(IMAGE_DIR_PATH)
    image_name_list.sort()
    frame_index_to_avg_state = {}
    frame_index_to_max_state = {}
    for image_name in image_name_list[1:]:

        S_prev = S

        # LOAD NEW IMAGE FRAME
        image_path = os.path.join(IMAGE_DIR_PATH, image_name)
        current_image = cv2.imread(image_path)

        # SAMPLE THE CURRENT PARTICLE FILTERS
        S_next_tag = sample_particles(S_prev, C)

        # PREDICT THE NEXT PARTICLE FILTERS (YOU MAY ADD NOISE
        S = predict_particles(S_next_tag)

        # COMPUTE NORMALIZED WEIGHTS (W) AND PREDICTOR CDFS (C)
        # YOU NEED TO FILL THIS PART WITH CODE:
        """INSERT YOUR CODE HERE."""
        W = np.zeros(N)
        for n in range(N):
            p = compute_normalized_histogram(current_image, S[:, n])
            W[n] = bhattacharyya_distance(p, q)

        W /= np.sum(W)
        C = np.cumsum(W)

        # CREATE DETECTOR PLOTS
        images_processed += 1
        if 0 == images_processed%10:
            frame_index_to_avg_state, frame_index_to_max_state = show_particles(
                current_image, S, W, images_processed, ID, frame_index_to_avg_state, frame_index_to_max_state)

    with open(os.path.join(RESULTS, 'frame_index_to_avg_state.json'), 'w') as f:
        json.dump(frame_index_to_avg_state, f, indent=4)
    with open(os.path.join(RESULTS, 'frame_index_to_max_state.json'), 'w') as f:
        json.dump(frame_index_to_max_state, f, indent=4)


if __name__ == "__main__":
    main()
