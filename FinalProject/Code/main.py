import os
import time
import json
import cv2

# Import your custom modules from the CODE folder
from stabilization import stabilize_video_pipeline
from background_subtraction import apply_background_subtraction
from matting import apply_matting
import tracking

def main():
    # 1. Determine absolute root path dynamically using python's os library
    code_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(code_dir)
    
    # 2. Configure path references strictly matching your workspace template structure
    inputs_dir = os.path.join(root_dir, "Inputs")
    outputs_dir = os.path.join(root_dir, "Outputs")
    os.makedirs(outputs_dir, exist_ok=True)
    
    input_video = os.path.join(inputs_dir, "INPUT.avi")
    background_image = os.path.join(inputs_dir, "background.jpg")
    
    # Define assignment-mandated file export naming conventions
    stabilized_out = os.path.join(outputs_dir, "stabilize_212047542_327703013.avi")
    binary_out = os.path.join(outputs_dir, "binary_212047542_327703013.avi")
    extracted_out = os.path.join(outputs_dir, "extracted_212047542_327703013.avi")
    matted_out = os.path.join(outputs_dir, "matted_212047542_327703013.avi")
    alpha_out = os.path.join(outputs_dir, "alpha_212047542_327703013.avi")
    tracking_out = os.path.join(outputs_dir, "OUTPUT_212047542_327703013.avi")
    
    timing_json_path = os.path.join(outputs_dir, "timing.json")

    # Initialize benchmark stopwatch timers
    timing_dict = {}

    # --- STAGE 1: VIDEO STABILIZATION ---
    print("[Pipeline] Executing Stage 1: Sparse Feature Video Stabilization...")
    start_time = time.time()
    stabilize_video_pipeline(input_video, stabilized_out)
    timing_dict["time_to_stabilize"] = int(time.time() - start_time)

    # --- STAGE 2: BACKGROUND SUBTRACTION ---
    print("[Pipeline] Executing Stage 2: Median Background Subtraction...")
    start_time = time.time()
    apply_background_subtraction(stabilized_out, binary_out, extracted_out)
    timing_dict["time_to_binary"] = int(time.time() - start_time)

    # --- STAGE 3: MATTING ---
    print("[Pipeline] Executing Stage 3: Alpha Matting & Compositing...")
    alpha_time, matting_time = apply_matting(
        extracted_video_path=extracted_out,
        binary_video_path=binary_out,
        background_image_path=background_image,
        matted_out_path=matted_out,
        alpha_out_path=alpha_out,
        feather_radius=7
    )
    timing_dict["time_to_alpha"] = int(alpha_time)
    timing_dict["time_to_matted"] = int(matting_time)

    # --- STAGE 4: PARTICLE FILTER TRACKING ---
    print("[Pipeline] Executing Stage 4: Particle Filter Tracking...")
    start_time = time.time()
    tracking.apply_tracking(input_video_path=matted_out, output_video_path=tracking_out, ID="OUTPUT")
    timing_dict["time_to_output"] = int(time.time() - start_time)

    # 3. Export logged timings directly into timing.json inside Outputs directory
    with open(timing_json_path, "w") as json_file:
        json.dump(timing_dict, json_file, indent=4)
        
    print("\n" + "="*50)
    print(f"Pipeline completed successfully! Benchmarks saved to: {timing_json_path}")
    print(json.dumps(timing_dict, indent=4))
    print("="*50)

if __name__ == "__main__":
    main()