import os
import cv2
from stabilization import stabilize_video_pipeline

def run_local_test():
    print("[+] Test script invoked inside Code folder. Initializing...")
    
    # --- ENTER YOUR STUDENT IDS HERE ---
    ID1 = "212047542"
    ID2 = "327703013"
    
    # Since this file executes inside 'Code', 'Inputs' and 'Outputs' are one level up (..)
    input_video = os.path.join("..", "Inputs", "INPUT.avi")
    output_dir = os.path.join("..", "Outputs")
    output_filename = f"stabilize_{ID1}_{ID2}.avi"
    output_video_path = os.path.join(output_dir, output_filename)
    
    # Safe relative directory creation
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"[+] Created target directory: '{output_dir}'")
        
    if not os.path.exists(input_video):
        print(f"[-] Relative Path Error: Cannot find your video at '{input_video}'")
        print("    Please verify you ran the command from inside the Code folder.")
        return
        
    print(f"[+] Input target verified: {input_video}")
    print(f"[+] Output file location: {output_video_path}")
    
    try:
        # Launch the calculation pipeline
        stabilize_video_pipeline(input_video, output_video_path, window_size=30)
        
        # Verify execution and final size allocation
        if os.path.exists(output_video_path) and os.path.getsize(output_video_path) > 0:
            print("\n" + "="*50)
            print("[+] SUCCESS! Stage 1 output generated cleanly.")
            print(f"[+] File path: {output_video_path}")
            print("="*50)
        else:
            print("[-] Error: Output file path created, but file payload is 0 bytes.")
    except Exception as e:
        print(f"[-] Test failed during runtime. Error details:\n{str(e)}")

if __name__ == "__main__":
    run_local_test()