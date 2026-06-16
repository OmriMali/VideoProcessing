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
    # 1. Apply deterministic drift (the dynamic model A)
    state_drifted = s_prior.copy()
    state_drifted[0, :] += state_drifted[4, :]  # X_center = X_center + X_velocity
    state_drifted[1, :] += state_drifted[5, :]  # Y_center = Y_center + Y_velocity

    # 2. Add random additive white noise (diffusion)
    # Standard deviations for: [X_center, Y_center, Width, Height, V_x, V_y]
    # Width and Height have 0 noise since they don't have to change between time steps.
    noise_std = np.array([[5.0],  # X center noise standard deviation
                          [5.0],  # Y center noise standard deviation
                          [0.0],  # Half-width noise standard deviation
                          [0.0],  # Half-height noise standard deviation
                          [1.0],  # X velocity noise standard deviation
                          [1.0]])  # Y velocity noise standard deviation

    # Generate white Gaussian noise matching the shape of the particle matrix
    white_noise = np.random.normal(0.0, 1.0, size=s_prior.shape) * noise_std

    # Combine the deterministic tracking drift with random diffusion
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

    # Calculate bounding box coordinates and clip to image borders
    x_min = max(0, xc - half_w)
    x_max = min(img_w, xc + half_w)
    y_min = max(0, yc - half_h)
    y_max = min(img_h, yc + half_h)

    # Handle boundary case where patch is outside or empty
    if x_min >= x_max or y_min >= y_max:
        hist = np.ones(16 * 16 * 16) / (16 * 16 * 16)
        return hist

    # Crop the target sub-portion
    crop = image[y_min:y_max, x_min:x_max, :]

    # Quantize from 8-bit (0-255) to 4-bit (0-15)
    quantized = (crop // 16).astype(int)

    # Populate the 3D histogram grid
    for b in range(quantized.shape[0]):
        for g in range(quantized.shape[1]):
            # OpenCV images are loaded in BGR channel layout
            ch_b, ch_g, ch_r = quantized[b, g, 0], quantized[b, g, 1], quantized[b, g, 2]
            hist[ch_b, ch_g, ch_r] += 1

    hist = np.reshape(hist, 16 * 16 * 16)

    # normalize
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
    # Get the total number of particles (N) from the matrix dimensions
    _, N = previous_state.shape

    # 1. Generate N random values uniformly distributed between 0 and 1
    r_values = np.random.uniform(0.0, 1.0, size=N)

    # Repeat the search mapping for each generated random threshold
    for n in range(N):
        r = r_values[n]

        # 2. Find the smallest index 'j' where the CDF value is greater than or equal to r
        j = np.searchsorted(cdf, r)

        # Clip index to protect against floating-point edge cases at exactly 1.0
        j = min(j, N - 1)

        # 3. Set the new particle state to mirror the sampled historic particle index
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
    # Compute the Bhattacharyya coefficient similarity measure
    rho = np.sum(np.sqrt(p * q))

    # Calculate weight mapping based on the required equation
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

    # 1. Compute the expected (mean) state across all particles weighted by W
    mean_state = np.sum(state * W, axis=1)

    # 2. Extract components from the mean state vector
    xc_avg, yc_avg, half_w_avg, half_h_avg = mean_state[0], mean_state[1], mean_state[2], mean_state[3]

    # 3. Convert from center/half-size parameters to top-left corner/full-size dimensions
    x_avg = xc_avg - half_w_avg
    y_avg = yc_avg - half_h_avg
    w_avg = 2 * half_w_avg
    h_avg = 2 * half_h_avg

    rect = patches.Rectangle((x_avg, y_avg), w_avg, h_avg, linewidth=1, edgecolor='g', facecolor='none')
    ax.add_patch(rect)

    # calculate Max particle box

    """ DELETE THE LINE ABOVE AND:
        INSERT YOUR CODE HERE."""

    # 1. Find the index of the particle holding the highest tracking weight
    max_idx = np.argmax(W)
    max_state = state[:, max_idx]

    # 2. Extract components from this specific highest-weighted particle
    xc_max, yc_max, half_w_max, half_h_max = max_state[0], max_state[1], max_state[2], max_state[3]

    # 3. Convert parameters to top-left corner and full-size bounding box metrics
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
    # Step 4: Loop through all N columns of S to find individual particle weights
    W = np.zeros(N)
    for n in range(N):
        p = compute_normalized_histogram(image, S[:, n])
        W[n] = bhattacharyya_distance(p, q)

    # Step 5: Normalize vector W so that sum(W) == 1
    W /= np.sum(W)

    # Step 6: Compute cumulative distribution vector C using numpy's cumulative sum
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
