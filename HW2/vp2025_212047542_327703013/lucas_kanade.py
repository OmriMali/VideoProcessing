import cv2
import numpy as np
from tqdm import tqdm
from scipy import signal
from scipy.interpolate import griddata


# FILL IN YOUR ID
ID1 = 212047542
ID2 = 327703013


PYRAMID_FILTER = 1.0 / 256 * np.array([[1, 4, 6, 4, 1],
                                       [4, 16, 24, 16, 4],
                                       [6, 24, 36, 24, 6],
                                       [4, 16, 24, 16, 4],
                                       [1, 4, 6, 4, 1]])
X_DERIVATIVE_FILTER = np.array([[1, 0, -1],
                                [2, 0, -2],
                                [1, 0, -1]])
Y_DERIVATIVE_FILTER = X_DERIVATIVE_FILTER.copy().transpose()

WINDOW_SIZE = 5


def get_video_parameters(capture: cv2.VideoCapture) -> dict:
    """Get an OpenCV capture object and extract its parameters.

    Args:
        capture: cv2.VideoCapture object.

    Returns:
        parameters: dict. Video parameters extracted from the video.

    """
    fourcc = int(capture.get(cv2.CAP_PROP_FOURCC))
    fps = int(capture.get(cv2.CAP_PROP_FPS))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    return {"fourcc": fourcc, "fps": fps, "height": height, "width": width,
            "frame_count": frame_count}


def build_pyramid(image: np.ndarray, num_levels: int) -> list[np.ndarray]:
    """Coverts image to a pyramid list of size num_levels.

    First, create a list with the original image in it. Then, iterate over the
    levels. In each level, convolve the PYRAMID_FILTER with the image from the
    previous level. Then, decimate the result using indexing: simply pick
    every second entry of the result.
    Hint: Use signal.convolve2d with boundary='symm' and mode='same'.

    Args:
        image: np.ndarray. Input image.
        num_levels: int. The number of blurring / decimation times.

    Returns:
        pyramid: list. A list of np.ndarray of images.

    Note that the list length should be num_levels + 1 as the in first entry of
    the pyramid is the original image.
    You are not allowed to use cv2 PyrDown here (or any other cv2 method).
    We use a slightly different decimation process from this function.
    """
    pyramid = [image.copy()]
    """INSERT YOUR CODE HERE."""
   
    for i in range(num_levels):

        blurred = signal.convolve2d(pyramid[i], PYRAMID_FILTER, boundary='symm', mode='same')
        decimated = blurred[::2, ::2]
        pyramid.append(decimated)
        
    return pyramid


def lucas_kanade_step(I1: np.ndarray,
                      I2: np.ndarray,
                      window_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Perform one Lucas-Kanade Step.

    This method receives two images as inputs and a window_size. It
    calculates the per-pixel shift in the x-axis and y-axis. That is,
    it outputs two maps of the shape of the input images. The first map
    encodes the per-pixel optical flow parameters in the x-axis and the
    second in the y-axis.

    (1) Calculate Ix and Iy by convolving I2 with the appropriate filters (
    see the constants in the head of this file).
    (2) Calculate It from I1 and I2.
    (3) Calculate du and dv for each pixel:
      (3.1) Start from all-zeros du and dv (each one) of size I1.shape.
      (3.2) Loop over all pixels in the image (you can ignore boundary pixels up
      to ~window_size/2 pixels in each side of the image [top, bottom,
      left and right]).
      (3.3) For every pixel, pretend the pixel’s neighbors have the same (u,
      v). This means that for NxN window, we have N^2 equations per pixel.
      (3.4) Solve for (u, v) using Least-Squares solution. When the solution
      does not converge, keep this pixel's (u, v) as zero.
    For detailed Equations reference look at slides 4 & 5 in:
    http://www.cse.psu.edu/~rtc12/CSE486/lecture30.pdf

    Args:
        I1: np.ndarray. Image at time t.
        I2: np.ndarray. Image at time t+1.
        window_size: int. The window is of shape window_size X window_size.

    Returns:
        (du, dv): tuple of np.ndarray-s. Each one is of the shape of the
        original image. dv encodes the optical flow parameters in rows and du
        in columns.
    """
    """INSERT YOUR CODE HERE.
    Calculate du and dv correctly.
    """

    du = np.zeros(I1.shape)
    dv = np.zeros(I1.shape)
    Ix = signal.convolve2d(I2, X_DERIVATIVE_FILTER, boundary='symm', mode='same')
    Iy = signal.convolve2d(I2, Y_DERIVATIVE_FILTER, boundary='symm', mode='same')
    
    It = I2 - I1
    
    h, w = I1.shape
    half_win = window_size // 2

    for i in range(half_win, h - half_win):
        for j in range(half_win, w - half_win):
            
            ix_win = Ix[i - half_win:i + half_win + 1, j - half_win:j + half_win + 1].flatten()
            iy_win = Iy[i - half_win:i + half_win + 1, j - half_win:j + half_win + 1].flatten()
            it_win = It[i - half_win:i + half_win + 1, j - half_win:j + half_win + 1].flatten()

            A = np.stack((ix_win, iy_win), axis=1)
            b = -it_win

            ATA = A.T @ A
            ATb = A.T @ b

            if np.linalg.matrix_rank(ATA) == 2:
                nu = np.linalg.solve(ATA, ATb)
                du[i, j] = nu[0]
                dv[i, j] = nu[1]

    return du, dv


def warp_image(image: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Warp image using the optical flow parameters in u and v.

    Note that this method needs to support the case where u and v shapes do
    not share the same shape as of the image. We will update u and v to the
    shape of the image. The way to do it, is to:
    (1) cv2.resize to resize the u and v to the shape of the image.
    (2) Then, normalize the shift values according to a factor. This factor
    is the ratio between the image dimension and the shift matrix (u or v)
    dimension (the factor for u should take into account the number of columns
    in u and the factor for v should take into account the number of rows in v).

    As for the warping, use `scipy.interpolate`'s `griddata` method. Define the
    grid-points using a flattened version of the `meshgrid` of 0:w-1 and 0:h-1.
    The values here are simply image.flattened().
    The points you wish to interpolate are, again, a flattened version of the
    `meshgrid` matrices - don't forget to add them v and u.
    Use `np.nan` as `griddata`'s fill_value.
    Finally, fill the nan holes with the source image values.
    Hint: For the final step, use np.isnan(image_warp).

    Args:
        image: np.ndarray. Image to warp.
        u: np.ndarray. Optical flow parameters corresponding to the columns.
        v: np.ndarray. Optical flow parameters corresponding to the rows.

    Returns:
        image_warp: np.ndarray. Warped image.
    """
    image_warp = image.copy()
    """INSERT YOUR CODE HERE.
    Replace image_warp with something else.
    """
    h, w = image.shape

    if u.shape != (h, w) or v.shape != (h, w):
        u_resized = cv2.resize(u, (w, h))
        v_resized = cv2.resize(v, (w, h))
        
        u = u_resized * (w / u.shape[1])
        v = v_resized * (h / v.shape[0])

    x = np.arange(w)
    y = np.arange(h)
    mesh_x, mesh_y = np.meshgrid(x, y)
    
    points = np.stack((mesh_x.flatten(), mesh_y.flatten()), axis=1)
    
    values = image.flatten()
    
    xi = np.stack(((mesh_x + u).flatten(), (mesh_y + v).flatten()), axis=1)
    
    image_warp = griddata(points, values, xi, method='linear', fill_value=np.nan)
    image_warp = image_warp.reshape((h, w))
    nan_mask = np.isnan(image_warp)
    image_warp[nan_mask] = image[nan_mask]
    return image_warp


def lucas_kanade_optical_flow(I1: np.ndarray,
                              I2: np.ndarray,
                              window_size: int,
                              max_iter: int,
                              num_levels: int) -> tuple[np.ndarray, np.ndarray]:
    """Calculate LK Optical Flow for max iterations in num-levels.

    Args:
        I1: np.ndarray. Image at time t.
        I2: np.ndarray. Image at time t+1.
        window_size: int. The window is of shape window_size X window_size.
        max_iter: int. Maximal number of LK-steps for each level of the pyramid.
        num_levels: int. Number of pyramid levels.

    Returns:
        (u, v): tuple of np.ndarray-s. Each one of the shape of the
        original image. v encodes the optical flow parameters in rows and u in
        columns.

    Recipe:
        (1) Since the image is going through a series of decimations,
        we would like to resize the image shape to:
        K * (2^(num_levels - 1)) X M * (2^(num_levels - 1)).
        Where: K is the ceil(h / (2^(num_levels - 1)),
        and M is ceil(h / (2^(num_levels - 1)).
        (2) Build pyramids for the two images.
        (3) Initialize u and v as all-zero matrices in the shape of I1.
        (4) For every level in the image pyramid (start from the smallest
        image):
          (4.1) Warp I2 from that level according to the current u and v.
          (4.2) Repeat for num_iterations:
            (4.2.1) Perform a Lucas Kanade Step with the I1 decimated image
            of the current pyramid level and the current I2_warp to get the
            new I2_warp.
          (4.3) For every level which is not the image's level, perform an
          image resize (using cv2.resize) to the next pyramid level resolution
          and scale u and v accordingly.
    """
    """INSERT YOUR CODE HERE.
        Replace image_warp with something else.
        """
    h_factor = int(np.ceil(I1.shape[0] / (2 ** (num_levels - 1))))
    w_factor = int(np.ceil(I1.shape[1] / (2 ** (num_levels - 1))))
    IMAGE_SIZE = (w_factor * (2 ** (num_levels - 1)),
                  h_factor * (2 ** (num_levels - 1)))
    if I1.shape != IMAGE_SIZE:
        I1 = cv2.resize(I1, IMAGE_SIZE)
    if I2.shape != IMAGE_SIZE:
        I2 = cv2.resize(I2, IMAGE_SIZE)
    # create a pyramid from I1 and I2
    pyramid_I1 = build_pyramid(I1, num_levels)
    pyarmid_I2 = build_pyramid(I2, num_levels)
    u = np.zeros(pyarmid_I2[-1].shape)
    v = np.zeros(pyarmid_I2[-1].shape)
    """INSERT YOUR CODE HERE.
       Replace u and v with their true value."""
    for level in range(num_levels, -1, -1):
        level_I1 = pyramid_I1[level]
        level_I2 = pyarmid_I2[level]
        
        h_curr, w_curr = level_I1.shape
        if u.shape != (h_curr, w_curr):
            u_resized = cv2.resize(u, (w_curr, h_curr))
            v_resized = cv2.resize(v, (w_curr, h_curr))
            
            u = u_resized * (w_curr / u.shape[1])
            v = v_resized * (h_curr / v.shape[0])
            
        for _ in range(max_iter):
            I2_warp = warp_image(level_I2, u, v)
            
            du, dv = lucas_kanade_step(level_I1, I2_warp, window_size)
            
            u += du
            v += dv

    return u, v

def lucas_kanade_video_stabilization(input_video_path: str,
                                     output_video_path: str,
                                     window_size: int,
                                     max_iter: int,
                                     num_levels: int) -> None:
    """Use LK Optical Flow to stabilize the video and save it to file.

    Args:
        input_video_path: str. path to input video.
        output_video_path: str. path to output stabilized video.
        window_size: int. The window is of shape window_size X window_size.
        max_iter: int. Maximal number of LK-steps for each level of the pyramid.
        num_levels: int. Number of pyramid levels.

    Returns:
        None.

    Recipe:
        (1) Open a VideoCapture object of the input video and read its
        parameters.
        (2) Create an output video VideoCapture object with the same
        parameters as in (1) in the path given here as input.
        (3) Convert the first frame to grayscale and write it as-is to the
        output video.
        (4) Resize the first frame as in the Full-Lucas-Kanade function to
        K * (2^(num_levels - 1)) X M * (2^(num_levels - 1)).
        Where: K is the ceil(h / (2^(num_levels - 1)),
        and M is ceil(h / (2^(num_levels - 1)).
        (5) Create a u and a v which are og the size of the image.
        (6) Loop over the frames in the input video (use tqdm to monitor your
        progress) and:
          (6.1) Resize them to the shape in (4).
          (6.2) Feed them to the lucas_kanade_optical_flow with the previous
          frame.
          (6.3) Use the u and v maps obtained from (6.2) and compute their
          mean values over the region that the computation is valid (exclude
          half window borders from every side of the image).
          (6.4) Update u and v to their mean values inside the valid
          computation region.
          (6.5) Add the u and v shift from the previous frame diff such that
          frame in the t is normalized all the way back to the first frame.
          (6.6) Save the updated u and v for the next frame (so you can
          perform step 6.5 for the next frame.
          (6.7) Finally, warp the current frame with the u and v you have at
          hand.
          (6.8) We highly recommend you to save each frame to a directory for
          your own debug purposes. Erase that code when submitting the exercise.
       (7) Do not forget to gracefully close all VideoCapture and to destroy
       all windows.
    """
    """INSERT YOUR CODE HERE."""
    cap = cv2.VideoCapture(input_video_path)
    params = get_video_parameters(cap)
    fourcc = cv2.VideoWriter_fourcc(*'XVID') 
    fps = params['fps']
    size = (params['width'], params['height'])
    frame_count = params['frame_count']
    out = cv2.VideoWriter(output_video_path, fourcc, fps, size, isColor=False)
    h_factor = int(np.ceil(size[1] / (2 ** (num_levels - 1))))
    w_factor = int(np.ceil(size[0] / (2 ** (num_levels - 1))))
    IMAGE_SIZE = (w_factor * (2 ** (num_levels - 1)),
                  h_factor * (2 ** (num_levels - 1)))
    
    _, first_frame = cap.read()
    gray_first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    out.write(gray_first_frame)

    resize_first_frame = cv2.resize(gray_first_frame, IMAGE_SIZE)
    print(f"Resized first frame shape: {resize_first_frame.shape}")
    sum_u, sum_v = np.zeros((IMAGE_SIZE[1], IMAGE_SIZE[0])), np.zeros((IMAGE_SIZE[1], IMAGE_SIZE[0]))
    print(f"Initialized sum_u and sum_v with shape: {sum_u.shape}")
    prev_frame = resize_first_frame.copy()
    half_win = window_size // 2

    for i in tqdm(range(1, frame_count)):
        ret, frame = cap.read()
        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resize_frame = cv2.resize(gray_frame, IMAGE_SIZE)

        u, v = lucas_kanade_optical_flow(prev_frame, resize_frame, window_size, max_iter, num_levels)

        valid_u = u[half_win:-half_win, half_win:-half_win]
        valid_v = v[half_win:-half_win, half_win:-half_win]

        mean_u = np.mean(valid_u)
        mean_v = np.mean(valid_v)

        u[half_win:-half_win, half_win:-half_win] = mean_u
        v[half_win:-half_win, half_win:-half_win] = mean_v

        sum_u += u
        sum_v += v

        prev_frame = resize_frame.copy()
        warped_frame = warp_image(gray_frame, sum_u, sum_v)
        final_frame = cv2.resize(warped_frame, size)

        final_frame_uint8 = np.clip(final_frame, 0, 255).astype(np.uint8)
        out.write(final_frame_uint8)


    cap.release()
    out.release()
    cv2.destroyAllWindows()


def faster_lucas_kanade_step(I1: np.ndarray,
                             I2: np.ndarray,
                             window_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Faster implementation of a single Lucas-Kanade Step.

    (1) If the image is small enough (you need to design what is good
    enough), simply return the result of the good old lucas_kanade_step
    function.
    (2) Otherwise, find corners in I2 and calculate u and v only for these
    pixels.
    (3) Return maps of u and v which are all zeros except for the corner
    pixels you found in (2).

    Args:
        I1: np.ndarray. Image at time t.
        I2: np.ndarray. Image at time t+1.
        window_size: int. The window is of shape window_size X window_size.

    Returns:
        (du, dv): tuple of np.ndarray-s. Each one of the shape of the
        original image. dv encodes the shift in rows and du in columns.
    """

    du = np.zeros(I1.shape)
    dv = np.zeros(I1.shape)
    
    Ix = signal.convolve2d(I2, X_DERIVATIVE_FILTER, boundary='symm', mode='same')
    Iy = signal.convolve2d(I2, Y_DERIVATIVE_FILTER, boundary='symm', mode='same')
    
    It = I2 - I1
    
    h, w = I1.shape
    half_win = window_size // 2

    if h > 256 and w > 256:
        I2_uint8 = cv2.normalize(I2, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        harris_response = cv2.cornerHarris(I2_uint8, blockSize=2, ksize=3, k=0.04)
        
        is_corner = harris_response > 0.02 * harris_response.max()
    else:
        is_corner = np.ones((h, w), dtype=bool)

    for i in range(half_win, h - half_win):
        for j in range(half_win, w - half_win):
            
            if not is_corner[i, j]:
                continue
            
            ix_win = Ix[i - half_win:i + half_win + 1, j - half_win:j + half_win + 1].flatten()
            iy_win = Iy[i - half_win:i + half_win + 1, j - half_win:j + half_win + 1].flatten()
            it_win = It[i - half_win:i + half_win + 1, j - half_win:j + half_win + 1].flatten()

            A = np.stack((ix_win, iy_win), axis=1)
            b = -it_win

            ATA = A.T @ A
            ATb = A.T @ b

            if np.linalg.matrix_rank(ATA) == 2:
                nu = np.linalg.solve(ATA, ATb)
                du[i, j] = nu[0]
                dv[i, j] = nu[1]
                
    return du, dv


def faster_lucas_kanade_optical_flow(
        I1: np.ndarray, I2: np.ndarray, window_size: int, max_iter: int,
        num_levels: int) -> tuple[np.ndarray, np.ndarray]:
    """Calculate LK Optical Flow for max iterations in num-levels .

    Use faster_lucas_kanade_step instead of lucas_kanade_step.

    Args:
        I1: np.ndarray. Image at time t.
        I2: np.ndarray. Image at time t+1.
        window_size: int. The window is of shape window_size X window_size.
        max_iter: int. Maximal number of LK-steps for each level of the pyramid.
        num_levels: int. Number of pyramid levels.

    Returns:
        (u, v): tuple of np.ndarray-s. Each one of the shape of the
        original image. v encodes the shift in rows and u in columns.
    """
    h_factor = int(np.ceil(I1.shape[0] / (2 ** (num_levels - 1))))
    w_factor = int(np.ceil(I1.shape[1] / (2 ** (num_levels - 1))))
    IMAGE_SIZE = (w_factor * (2 ** (num_levels - 1)),
                  h_factor * (2 ** (num_levels - 1)))
    if I1.shape != IMAGE_SIZE:
        I1 = cv2.resize(I1, IMAGE_SIZE)
    if I2.shape != IMAGE_SIZE:
        I2 = cv2.resize(I2, IMAGE_SIZE)
    pyramid_I1 = build_pyramid(I1, num_levels)
    pyarmid_I2 = build_pyramid(I2, num_levels)
    u = np.zeros(pyarmid_I2[-1].shape)
    v = np.zeros(pyarmid_I2[-1].shape)
    """INSERT YOUR CODE HERE.
       Replace u and v with their true value."""
    for level in range(num_levels, -1, -1):
        level_I1 = pyramid_I1[level]
        level_I2 = pyarmid_I2[level]
        
        h_curr, w_curr = level_I1.shape
        if u.shape != (h_curr, w_curr):
            u_resized = cv2.resize(u, (w_curr, h_curr))
            v_resized = cv2.resize(v, (w_curr, h_curr))
            
            u = u_resized * (w_curr / u.shape[1])
            v = v_resized * (h_curr / v.shape[0])
            
        for _ in range(max_iter):
            I2_warp = warp_image(level_I2, u, v)
            
            du, dv = faster_lucas_kanade_step(level_I1, I2_warp, window_size)
            
            u += du
            v += dv

    return u, v


def lucas_kanade_faster_video_stabilization(
        input_video_path: str, output_video_path: str, window_size: int,
        max_iter: int, num_levels: int) -> None:
    """Calculate LK Optical Flow to stabilize the video and save it to file.

    Args:
        input_video_path: str. path to input video.
        output_video_path: str. path to output stabilized video.
        window_size: int. The window is of shape window_size X window_size.
        max_iter: int. Maximal number of LK-steps for each level of the pyramid.
        num_levels: int. Number of pyramid levels.

    Returns:
        None.
    """
    """INSERT YOUR CODE HERE."""
    cap = cv2.VideoCapture(input_video_path)
    params = get_video_parameters(cap)
    fourcc = cv2.VideoWriter_fourcc(*'XVID') 
    fps = params['fps']
    size = (params['width'], params['height'])
    frame_count = params['frame_count']
    out = cv2.VideoWriter(output_video_path, fourcc, fps, size, isColor=False)
    h_factor = int(np.ceil(size[1] / (2 ** (num_levels - 1))))
    w_factor = int(np.ceil(size[0] / (2 ** (num_levels - 1))))
    IMAGE_SIZE = (w_factor * (2 ** (num_levels - 1)),
                  h_factor * (2 ** (num_levels - 1)))
    
    _, first_frame = cap.read()
    gray_first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    out.write(gray_first_frame)
    resize_first_frame = cv2.resize(gray_first_frame, IMAGE_SIZE)
    print(f"Resized first frame shape: {resize_first_frame.shape}")
    sum_u, sum_v = np.zeros((IMAGE_SIZE[1], IMAGE_SIZE[0])), np.zeros((IMAGE_SIZE[1], IMAGE_SIZE[0]))
    prev_frame = resize_first_frame.copy()
    half_win = window_size // 2

    for i in tqdm(range(1, frame_count)):
        ret, frame = cap.read()
        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resize_frame = cv2.resize(gray_frame, IMAGE_SIZE)

        u, v = faster_lucas_kanade_optical_flow(prev_frame, resize_frame, window_size, max_iter, num_levels)
        valid_u = u[half_win:-half_win, half_win:-half_win]
        valid_v = v[half_win:-half_win, half_win:-half_win]

        mean_u = np.mean(valid_u)
        mean_v = np.mean(valid_v)

        u[half_win:-half_win, half_win:-half_win] = mean_u
        v[half_win:-half_win, half_win:-half_win] = mean_v

        sum_u += u
        sum_v += v

        prev_frame = resize_frame.copy()
        warped_frame = warp_image(gray_frame, sum_u, sum_v)
        final_frame = cv2.resize(warped_frame, size)

        final_frame_uint8 = np.clip(final_frame, 0, 255).astype(np.uint8)
        out.write(final_frame_uint8)


    cap.release()
    out.release()
    cv2.destroyAllWindows()


def lucas_kanade_faster_video_stabilization_fix_effects(
        input_video_path: str, output_video_path: str, window_size: int,
        max_iter: int, num_levels: int, start_rows: int = 10,
        start_cols: int = 2, end_rows: int = 30, end_cols: int = 30) -> None:
    """Calculate LK Optical Flow to stabilize the video and save it to file.

    Args:
        input_video_path: str. path to input video.
        output_video_path: str. path to output stabilized video.
        window_size: int. The window is of shape window_size X window_size.
        max_iter: int. Maximal number of LK-steps for each level of the pyramid.
        num_levels: int. Number of pyramid levels.
        start_rows: int. The number of lines to cut from top.
        end_rows: int. The number of lines to cut from bottom.
        start_cols: int. The number of columns to cut from left.
        end_cols: int. The number of columns to cut from right.

    Returns:
        None.
    """
    """INSERT YOUR CODE HERE."""
    cap = cv2.VideoCapture(input_video_path)
    params = get_video_parameters(cap)
    fourcc = cv2.VideoWriter_fourcc(*'XVID') 
    fps = params['fps']
    size = (params['width'], params['height'])
    frame_count = params['frame_count']
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (size[0] - end_cols - start_cols, size[1] - end_rows - start_rows), isColor=False)
    h_factor = int(np.ceil(size[1] / (2 ** (num_levels - 1))))
    w_factor = int(np.ceil(size[0] / (2 ** (num_levels - 1))))
    IMAGE_SIZE = (w_factor * (2 ** (num_levels - 1)),
                  h_factor * (2 ** (num_levels - 1)))
    
    _, first_frame = cap.read()
    gray_first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    out.write(gray_first_frame[start_rows:size[1] - end_rows, start_cols:size[0] - end_cols])

    resize_first_frame = cv2.resize(gray_first_frame, IMAGE_SIZE)
    sum_u, sum_v = np.zeros((IMAGE_SIZE[1], IMAGE_SIZE[0])), np.zeros((IMAGE_SIZE[1], IMAGE_SIZE[0]))
    prev_frame = resize_first_frame.copy()
    half_win = window_size // 2

    for i in tqdm(range(1, frame_count)):
        ret, frame = cap.read()
        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resize_frame = cv2.resize(gray_frame, IMAGE_SIZE)

        u, v = faster_lucas_kanade_optical_flow(prev_frame, resize_frame, window_size, max_iter, num_levels)

        valid_u = u[half_win:-half_win, half_win:-half_win]
        valid_v = v[half_win:-half_win, half_win:-half_win]

        mean_u = np.mean(valid_u)
        mean_v = np.mean(valid_v)

        u[half_win:-half_win, half_win:-half_win] = mean_u
        v[half_win:-half_win, half_win:-half_win] = mean_v

        sum_u += u
        sum_v += v

        prev_frame = resize_frame.copy()
        warped_frame = warp_image(gray_frame, sum_u, sum_v)
        final_frame = cv2.resize(warped_frame, size)

        final_frame_uint8 = np.clip(final_frame, 0, 255).astype(np.uint8)
        out.write(final_frame_uint8[start_rows:size[1] - end_rows, start_cols:size[0] - end_cols])

    cap.release()
    out.release()
    cv2.destroyAllWindows()

