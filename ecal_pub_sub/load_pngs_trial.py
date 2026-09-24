import cv2
import os
import numpy as np

FOLDER_PATH = "./CamA/"

def load_cam_frames_into_bgr():
    """
    Loads all frames from the specified folder into a BGR numpy array.
    """
    if not os.path.exists(FOLDER_PATH):
        raise FileNotFoundError(f"The folder {FOLDER_PATH} does not exist.")
    
    frames = []
    for filename in sorted(os.listdir(FOLDER_PATH)):
        if filename.endswith('.png'):
            img_path = os.path.join(FOLDER_PATH, filename)
            print(f"Loading image: {img_path}")
            img = cv2.imread(img_path)
            if img is not None:
                frames.append(img)
    
    frames = np.array(frames)
    print("Frames shape:", frames.shape)
    return frames

def play_video_opencv(frames: np.ndarray):
    """
    Plays the video using OpenCV.
    
    Args:
        frames (np.ndarray): Array of frames to be played.
    """
    for frame in frames:
        cv2.imshow('Video', frame)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

def main():
    frames = load_cam_frames_into_bgr()
    play_video_opencv(frames)

if __name__ == "__main__":
    main()