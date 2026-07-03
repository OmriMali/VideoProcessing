import cv2
import numpy as np
import os

# -----------------------------
# Configuration
# -----------------------------
ID = "OUTPUT"
RESULTS = 'results'
os.makedirs(RESULTS, exist_ok=True)
code_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(code_dir)
outputs_dir = os.path.join(root_dir, "Outputs")
INPUT_VIDEO = os.path.join(outputs_dir, "input_try.mp4")
OUTPUT_VIDEO = os.path.join(outputs_dir, "OUTPUT_TRY.avi")

NUM_BACKGROUND_FRAMES = 600   # Frames used to estimate background
THRESHOLD = 30                # Pixel difference threshold
MIN_BLOB_AREA = 1000           # Remove objects smaller than this

# --------------------------------------------------
# Open video
# --------------------------------------------------
cap = cv2.VideoCapture(INPUT_VIDEO)

if not cap.isOpened():
    raise IOError("Cannot open video.")

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# --------------------------------------------------
# Estimate Background (Median)
# --------------------------------------------------
print("Estimating background...")

frames = []

for _ in range(NUM_BACKGROUND_FRAMES):
    ret, frame = cap.read()

    if not ret:
        break

    frames.append(frame)

frames = np.stack(frames, axis=0)

background = np.median(frames, axis=0).astype(np.uint8)

print("Background estimated.")

# --------------------------------------------------
# Restart video
# --------------------------------------------------
cap.release()
cap = cv2.VideoCapture(INPUT_VIDEO)

# --------------------------------------------------
# Output video
# --------------------------------------------------
fourcc = cv2.VideoWriter_fourcc(*'XVID')

out = cv2.VideoWriter(
    OUTPUT_VIDEO,
    fourcc,
    fps,
    (width, height),
    False  # grayscale
)

kernel = np.ones((5, 5), np.uint8)

# --------------------------------------------------
# Process video
# --------------------------------------------------
while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Absolute difference
    diff = cv2.absdiff(frame, background)

    # Convert to grayscale
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    # Threshold
    _, mask = cv2.threshold(gray, THRESHOLD, 255, cv2.THRESH_BINARY)

    # Remove noise
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Fill holes
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Smooth
    mask = cv2.medianBlur(mask, 5)

    # --------------------------------------------------
    # Remove small connected components
    # --------------------------------------------------
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)

    cleaned = np.zeros_like(mask)

    for i in range(1, num_labels):   # Skip background (label 0)

        area = stats[i, cv2.CC_STAT_AREA]

        if area >= MIN_BLOB_AREA:
            cleaned[labels == i] = 255

    out.write(cleaned)


    if cv2.waitKey(1) == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print("Done.")