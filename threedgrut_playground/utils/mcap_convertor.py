import json
import os
import time
from mcap.writer import Writer
from time import time_ns, sleep
import cv2
import numpy as np
import sys
import tqdm
from tqdm import trange
# import argparse

# # Add argument
# parser = argparse.ArgumentParser(description="MCAP Writer for VisualKit")
# parser.add_argument('--output', type=str, default='output.mcap', help='Output MCAP file name')
# args = parser.parse_args()

CAMERA_NAMES = ["CamA", "CamB", "CamC", "CamD"]
#image_dir = "./"  # should have subfolders like cama/, camb/, etc.

sys.path.append(os.path.join(os.path.dirname(__file__), '../vk_sdk/capnp'))
sys.path.append('/opt/vilota/messages')
import capnp
capnp.add_import_hook()
import image_capnp as eCALImage


class McapConverter:
    def __init__ (self):
        self.output_filename = "output.mcap"
        self.output_folder = "./mcap_outputs/"
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
        self.output_fullpath = self.output_folder + self.output_filename
        print(f"Starting MCAP Converter")

    def build_image_message(self, img_array : np.ndarray, index: int, cam_index:int, name:str, encoding:str):
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
    
    def load_frame_from_png(self, index:int, folder_path:str):
        filename = folder_path + f"{index:03d}.png"
        if not os.path.exists(filename):
            raise FileNotFoundError(f"No image found at {filename}. Please ensure the file exists.")
        frame = np.array(cv2.imread(filename, cv2.IMREAD_COLOR))
        #print(frame.shape)
        return frame

    def set_filepath(self):
        self.output_folder = "./mcap_outputs/"
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
        self.output_fullpath = self.output_folder + self.output_filename
        print(f"Output file will be saved to: {self.output_fullpath}")

    def load_frames_npz(self, cam_name: str = "CamX"):
        filepath = f'./{cam_name}.npz'
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No frames found for camera {cam_name}. Please ensure the file {filepath} exists.")
        data = np.load(filepath)
        frames = data['frames']
        #print("Frames shape:", frames.shape)
        return frames

    def get_num_frames(self, cam_name: str = "CamX"):
        images_folder = "./cam_streams/" + cam_name + "/"
        if not os.path.exists(images_folder):
            raise FileNotFoundError(f"No images found for camera {cam_name}. Please ensure the folder {images_folder} exists.")
        num_frames = len(os.listdir(images_folder))
        return num_frames

    def write_cam_frame_to_mcap(self, writer, channels, cam_name, frame, index, timestamp):
        if cam_name not in channels:
            # Register a new channel for this camera
            # Assuming schema_id is 0 and message_encoding is "image/jpeg"
            # You can adjust these parameters as needed
            channel_id = writer.register_channel(
                schema_id=0,
                topic=f"S1/{cam_name.lower()}",
                message_encoding="image/jpeg",
                metadata={"camera_name": cam_name}
            )
            channels[cam_name] = channel_id
        
        # Just assume 420 frames, named 0000.jpg to 0419.jpg
        msg = self.build_image_message(frame, index, CAMERA_NAMES.index(cam_name), cam_name, "bgr8")

        writer.add_message(
            channel_id=channels[cam_name],
            log_time=timestamp,
            publish_time=timestamp,
            data=msg.to_bytes(),
        )
        #sleep(1/30)

    def display_simulated_task(self):
        for _ in trange(100, desc="Buffering next frames"):
            # Do stuff here
            time.sleep(0.05)  # simulate a task

    def calculate_time_interval(self, fps= 30):
        # if we have 30 frames per second, each frame lasts
        interval_in_s = 1/fps
        interval_in_ns = int(interval_in_s * 1e9)
        return interval_in_ns

    
    def convert_frame_to_mcap(self, writer, channels, cam_name, frame:np.ndarray, index:int):
        with open(self.output_fullpath, "wb") as stream:
            writer = Writer(stream)
            writer.start(profile="VisualKit")  # ← sets that profile tag

            channels = {}
            # 
            self.write_cam_frame_to_mcap(writer, channels, cam_name )
            
        
            #print(f"Published 420 frames/cam to {self.output_fullpath}.")
            #writer.finish()


# def main():
#     converter = McapConverter()
#     converter.output_filename = "test_2.mcap"
#     converter.set_filepath()
#     interval = converter.calculate_time_interval(30)
#     with open(converter.output_fullpath, "wb") as stream:
#         writer = Writer(stream)
#         writer.start(profile="VisualKit")
#         channels = {}
#         for cam_name in CAMERA_NAMES:
#             # Load frames
#             time = 0
#             num_frames = converter.get_num_frames(cam_name)
#             for i in trange(num_frames, desc = f"Writing frames for {cam_name}"):
#                 frame = converter.load_frame_from_png(i, f"./cam_streams/{cam_name}/")
#                 converter.write_cam_frame_to_mcap(writer, channels, cam_name, frame, i, time)
#                 time += interval

#         writer.finish()
#         print(f"Published {num_frames} frames/cam to {converter.output_fullpath}.")





# if __name__ == "__main__":
#     main()