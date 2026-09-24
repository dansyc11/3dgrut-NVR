#!/usr/bin/env python3

import os
import sys
import time

import argparse
import numpy as np

#sys.path.insert(0, '/usr/lib/python3/dist-packages')
import ecal.core.core as ecal_core

sys.path.append('/opt/vilota/bin')
sys.path.append('/opt/vilota/python')


from capnp_publisher import CapnpPublisher
import capnp

sys.path.append(os.path.join(os.path.dirname(__file__), '../vk_sdk/capnp'))
sys.path.append('/opt/vilota/messages')
capnp.add_import_hook()
import cv2

import image_capnp as eCALImage


parser = argparse.ArgumentParser(description="Publish image stream from a sequence of images.")
parser.add_argument('--cam_name', type=str, required=True, help='Camera name to load frames from' \
'Enter CamA, CamB, CamC or CamD, based on what .npz file you want to load from.')

parser.add_argument('--frame_source', type=str, default='png', choices=['npz', 'png'], 
                    help='Source of frames to load. Choose between npz or png folder.')
args = parser.parse_args()

def load_sample_image():
    # if there's a file called bgr_image.npy, load it
    if os.path.exists('./bgr_image.npy'):
        sample_img = np.load('./bgr_image.npy')
    else:
        sample_img = np.array([[[255,255,255],[0,0,0],[255,255,255]],
                         [[0,0,0],[255,255,255],[0,0,0]],
                         [[255,255,255],[0,0,0],[255,255,255]]])
    return sample_img.astype(np.uint8)

def load_frames_npz(cam_name: str = "CamX"):
    filepath = f'./{cam_name}.npz'
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No frames found for camera {cam_name}. Please ensure the file {filepath} exists.")
    data = np.load(filepath)
    frames = data['frames']
    print("Frames shape:", frames.shape)
    return frames
def get_num_images_in_folder(cam_name: str = "CamX") -> int:
    """
    Returns the number of images in a folder.
    """
    folder_path = f'./{cam_name}/'
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"The folder {folder_path} does not exist.")
    
    count = 0
    for filename in os.listdir(folder_path):
        if filename.endswith('.png'):
            count += 1
    return count

def load_frames_bgr_from_png_folder(cam_name: str = "CamX"):
    folder_path = f'./{cam_name}/'
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"No frames found for camera {cam_name}. Please ensure the folder {folder_path} exists.")
    frames = []
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith('.png'):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)
            if img is not None:
                frames.append(img)
    
    frames = np.array(frames)
    print("Frames shape:", frames.shape)
    return frames


def build_image_message(img_array : np.ndarray, index: int, cam_index:int, name:str, encoding:str):
    """
    Builds a Capnp image message from a (h, w, 3) numpy array
    representing one single image in the sequence indexed by an int

    Args:
        bgra_array (np.ndarray): Image array of size H x W x 3 with BGR values from 0-255
        index (int): Integer index in the sequence
    """
    msg = eCALImage.Image.new_message()
    msg_header = msg.header
    msg_header.seq = index
    msg_header.stampMonotonic = msg.header.seq * int(300e6)
    msg.encoding = eCALImage.Image.Encoding.yuv420
    msg.width, msg.height = img_array.shape[1], img_array.shape[0]
    if encoding == "bgr8":
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2YUV_I420)
    msg.data = img_array.tobytes()
    msg.exposureUSec = 180
    msg.gain = 180
    msg.sensorIdx = cam_index
    msg.streamName = name
    #msg.mipMapBrightness = 180
    #print(type(msg))
    #print("Success!")
    return msg   

def register_camera_name(cam_name: str):
    """
    Registers the camera name to initialise the publisher.
    """
    if not cam_name.startswith("Cam"):
        raise ValueError(f"Invalid camera name: {cam_name}. It should start with 'Cam'.")
    topic_name = "S0/" + cam_name.lower()
    return topic_name, cam_name


def main():
    # sample_img= load_sample_image()
    # msg = build_image_message(sample_img, 0)
    # print(msg.width, msg.height, msg.encoding)
    # print(msg.data)
    print("eCAL {} ({})\n".format(ecal_core.getversion(), ecal_core.getdate()))
    
    ecal_core.initialize(sys.argv, "publish_img_stream")
    ecal_core.set_process_state(1, 1, "Image Publisher Running")
    args = parser.parse_args()
    topic_name, cam_name = register_camera_name(args.cam_name)
    pub = CapnpPublisher(topic_name, "Image")

    cam_names = ["CamA", "CamB", "CamC", "CamD"]
    cam_idx = cam_names.index(args.cam_name) if cam_name in cam_names else 0
    #pub = CapnpPublisher("S0/camc", "Image")

    seq = 0
    if args.frame_source == 'npz':
        frames = load_frames_npz(cam_name=args.cam_name)
        num_frames = frames.shape[0]
    else:
        frames = load_frames_bgr_from_png_folder(cam_name=args.cam_name)
        num_frames = get_num_images_in_folder(cam_name=args.cam_name)
    print(f"Number of frames to publish: {num_frames}")
    ended = False
    while ecal_core.ok():
        
        for i, frame in enumerate(frames):
            print(f"Publishing frame {i}")
            msg = build_image_message(frames[i], i, cam_idx, args.cam_name, encoding="bgr8")
            pub.send(msg.to_bytes())
            if i == num_frames - 1:
                ended = True
                break
            time.sleep(1/10)
        if ended:
            print("Published all frames, exiting...")
            break
    
       
           
        
    #     time.sleep(0.01)  # 100 Hz

    ecal_core.finalize()

if __name__ == "__main__":
     main()

