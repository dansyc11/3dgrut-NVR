# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations
import json
import os
import shutil
from typing import List, Tuple, Union
import numpy as np
import torch
import cv2
from tqdm import tqdm
from scipy.interpolate import splprep, splev
from threedgrut_playground.utils.distortion_camera import DistortionCamera as Camera
from threedgrut.utils.logger import logger
from threedgrut.model.model import MixtureOfGaussians
from threedgrut.model.background import BackgroundColor
from threedgrut_playground.utils.mesh_io import load_mesh, load_materials, load_missing_material_info, create_procedural_mesh
from threedgrut_playground.utils.depth_of_field import DepthOfField
from threedgrut_playground.utils.spp import SPP
from threedgrut_playground.tracer import Tracer
from threedgrut_playground.utils.kaolin_future.transform import ObjectTransform
from threedgrut_playground.utils.kaolin_future.conversions import polyscope_from_kaolin_camera, polyscope_to_kaolin_camera
from threedgrut_playground.utils.kaolin_future.fisheye import generate_fisheye_rays
from threedgrut_playground.utils.kaolin_future.interpolated_cameras import camera_path_generator


class VideoRecorder:
    """ A module for saving video footage of camera trajectories within the Playground"""

    MODES = ['path_smooth', 'path_spline', 'cyclic', 'depth_of_field']
    EXPORT_FORMAT = ['png', 'mcap']

    def __init__(self,
        renderer,
        trajectory_output_path="output.mp4",
        cameras_save_path="cameras.npy",
        mode='path_spline',
        export_format='png',
        frames_between_cameras=4,
        video_fps=30,
        min_dof=2.5,
        max_dof=24
    ):
        """
        Creates a video recorder for saving camera animation along trajectories.
        Args:
            renderer: Reference to the Playground rendering engine, interfaced through a 'render()' function.
            trajectory_output_path (str): Output path for saved video
            cameras_save_path (str): Save path for storing and loading camera trajectories.
            mode (str): Determines how keyframes are interpolated:
                'path_smooth' - shifts the camera in a path along the trajectory, using smoothstep interpolation
                'path_spline' - shifts the camera in a path along the trajectory, using catmull-rom splines
                'cyclic' - shifts the camera in a cyclic trajectory, determined by fitting a bspline
                'depth_of_field' - interpolates the depth of field over a static frame (first frame in trajectory)
            frames_between_cameras (int): How many cameras are inserted between keyframe cameras in the trajectory
            video_fps (int): FPS of exported video
            min_dof (float): For 'depth_of_field' mode only, determines the minimum dof for interpolation start / end
            max_dof (float): For 'depth_of_field' mode only, determines the maximum dof for interpolation start / end
        """
        self.renderer = renderer

        # List of kaolin.render.camera.Camera objects forming the trajectory
        self.trajectory = []

        # Selected interpolation modes
        self.mode = mode

        # Selected export format
        self.export_format = export_format

        # Output path for generated video file
        self.trajectory_output_path = trajectory_output_path

        # For saving and loading trajectory paths
        self.cameras_save_path = cameras_save_path

        # Camera FPS -- how many cameras are interpolated between key views
        self.frames_between_cameras = frames_between_cameras

        # Saved Video FPS
        self.video_fps = video_fps

        # For depth-of-field interpolation mode only - determines the boundary
        self.min_dof = min_dof
        self.max_dof = max_dof

    def add_camera(self, camera: Camera):
        self.trajectory.append(camera)
    
    def get_num_frames(self, list_len:int):
        f_b = self.frames_between_cameras
        num_intervals = list_len - 1
        if num_intervals <= 0:
            raise ValueError("Cannot calculate number of frames for an empty trajectory or a trajectory with a single camera.")
        return num_intervals * f_b
        
    def reset_trajectory(self):
        self.trajectory = []

    def save_trajectory(self):
        torch.save(self.trajectory, self.cameras_save_path)

    def load_trajectory(self):
        self.trajectory = torch.load(self.cameras_save_path)

    def reset_for_new_cam_path_export(self):
        """
        Resets the trajectory and interpolated camera path to be populated with new cameras.
        """
        self.trajectory = []
        self.interpolated_camera_path = None

    def render_dof_trajectory(self):
        out_video = None
        old_use_dof = self.renderer.use_depth_of_field
        old_focus_z = self.renderer.depth_of_field.focus_z
        try:
            camera = self.trajectory[0]
            dofs = np.linspace(self.min_dof, self.max_dof, self.frames_between_cameras)
            for dof in tqdm(dofs):
                self.renderer.use_depth_of_field = True
                self.renderer.depth_of_field.focus_z = dof
                rgb = self.renderer.render(camera)['rgb']

                if out_video is None:
                    out_video = cv2.VideoWriter(self.trajectory_output_path, cv2.VideoWriter_fourcc(*'mp4v'),
                                                self.video_fps, (rgb.shape[2], rgb.shape[1]), True)
                data = rgb[0].clip(0, 1).detach().cpu().numpy()
                data = (data * 255).astype(np.uint8)
                data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
                out_video.write(data)
        finally:
            self.renderer.use_depth_of_field = old_use_dof
            self.renderer.depth_of_field.focus_z = old_focus_z

    def render_linear_trajectory_to_png(self, interpolation_mode='catmull_rom', cam_name: str = "CamX"):
        if len(self.trajectory) < 2:
            raise ValueError('Rendering a path trajectory requires at least 2 cameras.')
        elif interpolation_mode == 'catmull_rom' and len(self.trajectory) < 4:
            raise ValueError('Rendering a path with a spline interpolated trajectory requires at least 4 cameras.')
        
        distortions = self.trajectory[0].distortion_coefficients if hasattr(self.trajectory[0], 'distortion_coefficients') else None
        intrinsics = self.trajectory[0].get_camera_intrinsics() if hasattr(self.trajectory[0], 'get_camera_intrinsics') else None
        if not intrinsics:
            raise ValueError("The first camera in the trajectory does not have intrinsic parameters. "
                             "Ensure all cameras in the trajectory have valid intrinsic parameters.")
        interpolated_path = camera_path_generator(
            trajectory=self.trajectory,
            frames_between_cameras=self.frames_between_cameras,
            interpolation="catmull_rom"
        )

        cam_name = cam_name if cam_name else "CamX" 
        images_filename = f"./ecal_pub_sub/{cam_name}/"
        if os.path.exists(images_filename):
            shutil.rmtree(images_filename)
        os.makedirs(images_filename)
        camera_index = 0
        for camera in tqdm(interpolated_path):
            if not hasattr(camera, 'distortion_coefficients') or camera.distortion_coefficients is None:
                camera.distortion_coefficients = distortions   
            
            if not hasattr(camera, 'intrinsic_params') or camera.intrinsic_params is None:
                fx, fy, cx, cy = intrinsics
                camera.set_cam_intr(fx, fy, cx, cy)

            rgb = self.renderer.render(camera)['rgb']
         
            data = rgb[0].clip(0, 1).detach().cpu().numpy()
            data = (data * 255).astype(np.uint8)
            self.save_to_png(rgb, camera_index, images_filename)
            camera_index += 1
        
        import sys
        sys.exit(f"Saved frames to {images_filename} and exiting for now, please check the folder.")

    def obtain_single_camera_frame(self, frame_index:int, interpolation_mode='catmull_rom', cam_name: str = "CamX"):
        if len(self.trajectory) < 2:
            raise ValueError('Rendering a path trajectory requires at least 2 cameras.')
        elif interpolation_mode == 'catmull_rom' and len(self.trajectory) < 4:
            raise ValueError('Rendering a path with a spline interpolated trajectory requires at least 4 cameras.')
        
        distortions = self.trajectory[0].distortion_coefficients if hasattr(self.trajectory[0], 'distortion_coefficients') else None
        intrinsics = self.trajectory[0].get_camera_intrinsics() if hasattr(self.trajectory[0], 'get_camera_intrinsics') else None
        if not intrinsics:
            raise ValueError("The first camera in the trajectory does not have intrinsic parameters. "
                             "Ensure all cameras in the trajectory have valid intrinsic parameters.")
        
        if self.interpolated_camera_path is None:
            interpolated_path = camera_path_generator(
                trajectory=self.trajectory,
                frames_between_cameras=self.frames_between_cameras,
                interpolation=interpolation_mode
            )
            self.interpolated_camera_path = list(interpolated_path)

        cam_name = cam_name if cam_name else "CamX"
        
        camera = self.interpolated_camera_path[frame_index]
        if not hasattr(camera, 'distortion_coefficients') or camera.distortion_coefficients is None:
            camera.distortion_coefficients = distortions   
        
        if not hasattr(camera, 'intrinsic_params') or camera.intrinsic_params is None:
            fx, fy, cx, cy = intrinsics
            camera.set_cam_intr(fx, fy, cx, cy)

        rgb = self.renderer.render(camera)['rgb']
        
        bgr_frame = self.convert_to_bgr(rgb)       
        return bgr_frame

    def render_continuous_trajectory(self):
        if len(self.trajectory) < 4:
            raise ValueError('Rendering a continuous trajectory requires at least 4 cameras.')

        def _fit_bspline(data_points, frames_between_cameras):
            stacked_data_points = torch.stack([dp for dp in data_points], dim=1).cpu().numpy()
            tck, u = splprep(stacked_data_points, u=None, s=0.0, per=1)
            u_new = np.linspace(u.min(), u.max(), frames_between_cameras * len(data_points))
            interpolated_cam_param = np.stack(splev(u_new, tck, der=0)).T
            interpolated_cam_param = torch.tensor(interpolated_cam_param)
            return interpolated_cam_param

        cam_poses = _fit_bspline(
            data_points=[c.cam_pos().squeeze() for c in self.trajectory],
            frames_between_cameras=self.frames_between_cameras
        )
        cam_ats = _fit_bspline(
            data_points=[c.cam_pos().squeeze() - c.cam_forward().squeeze() for c in self.trajectory],
            frames_between_cameras=self.frames_between_cameras
        )
        cam_ups = _fit_bspline(
            data_points=[c.cam_up().squeeze() for c in self.trajectory],
            frames_between_cameras=self.frames_between_cameras
        )

        first_cam = self.trajectory[0]
        fx, fy, cx, cy = first_cam.get_camera_intrinsics()
        interpolated_path = []
        for eye, at, up in tqdm(zip(cam_poses, cam_ats, cam_ups)):
            interpolated_path.append(
                Camera.from_args(
                    eye=eye,
                    at=at,
                    up=up,
                    fov=first_cam.fov(in_degrees=False),
                    focal_x = fx,
                    focal_y = fy,
                    x0=cx, y0=cy,
                    width=first_cam.width, height=first_cam.height,
                    device=first_cam.device
                )
            )

        out_video = None
        frames = np.array([])
        for camera in tqdm(interpolated_path):
            rgb = self.renderer.render(camera)['rgb']
            if not hasattr(camera, 'distortion_coefficients') or camera.distortion_coefficients is None:
                ValueError(
                    f"Camera {camera} does not have distortion co`efficients. "
                    "Ensure all cameras in the trajectory have valid distortion coefficients."
                )
            else: 
                print(f"Camera {camera} has distortion coefficients: {camera.distortion_coefficients}") 
            if out_video is None:
                out_video = cv2.VideoWriter(self.trajectory_output_path, cv2.VideoWriter_fourcc(*'mp4v'),
                                            self.video_fps, (rgb.shape[2], rgb.shape[1]), True)
            data = rgb[0].clip(0, 1).detach().cpu().numpy()
            data = (data * 255).astype(np.uint8)
            data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
            out_video.write(data)

            # also save the bgr buffers to a npz file
            #bgr_array = self.convert_to_bgr(rgb)
            yuv420_arr = self.convert_to_yuv420(rgb)
            frames.append(yuv420_arr)
        
        frames = np.stack(frames, axis=0)
        # Save the frames to a npz file
        np.savez_compressed("./frames.npz", frames=frames)
        print(f"Saved frames to ./frames.npz")



        out_video.release()
        print(f"Saved video to {self.trajectory_output_path}")
    
    #def render_linear_trajectory_to_mcap_all_cams()

    def render_video(self, cam_name: str = "CamX"):
        """
        Renders the trajectory to a video file, according to the set mode (see init()).
        """
        if self.export_format == 'png':
            self.render_linear_trajectory_to_png(interpolation_mode=self.mode, cam_name=cam_name)
        else:
            raise ValueError(f'Unknown export format: {self.ex}')
        

    def convert_to_bgr(self, rgb: torch.Tensor) -> np.ndarray:
        """
        Converts the RGB tensor to BGR numpy array to be published as imageMsg.data
        with encoding bgr8.
        """
        bgr = rgb.clone().detach().cpu().numpy()
        bgr.clip(0, 1, out=bgr)
        bgr = (bgr * 255).astype(np.uint8).squeeze(0) # Convert to numpy for OpenCV compatibility
        # Convert RGB to BGR for OpenCV compatibility
        bgr = cv2.cvtColor(bgr, cv2.COLOR_RGB2BGR)
        #print(f"BGR shape: {bgr.shape}, First pixel: {bgr[0, 0, :]}")
        return bgr
    
    def convert_to_yuv420(self, rgb: torch.Tensor) -> np.ndarray:
        """
        Converts the RGB tensor to YUV420 numpy array to be published as imageMsg.data
        with encoding bgr8.
        """
        rgb_tensor= rgb.clone().detach().cpu().numpy()
        rgb_tensor.clip(0, 1, out=rgb_tensor)
        rgb_tensor = (rgb_tensor * 255).astype(np.uint8).squeeze(0) # Convert to numpy for OpenCV compatibility
        # Convert RGB to BGR for OpenCV compatibility
        yuv420 = cv2.cvtColor(rgb_tensor, cv2.COLOR_RGB2YUV_I420)
        #print(f"BGR shape: {bgr.shape}, First pixel: {bgr[0, 0, :]}")
        return yuv420
    
    def save_to_png(self, rgb: torch.Tensor, index: int, filepath:str):
        """
        Converts the RGB tensor to BGR numpy array to be published as imageMsg.data
        with encoding bgr8.
        """
        bgr = rgb.clone().detach().cpu().numpy()
        bgr.clip(0, 1, out=bgr)
        bgr = (bgr * 255).astype(np.uint8).squeeze(0) # Convert to numpy for OpenCV compatibility
        # Convert RGB to BGR for OpenCV compatibility
        bgr = cv2.cvtColor(bgr, cv2.COLOR_RGB2BGR)
        #print(f"BGR shape: {bgr.shape}, First pixel: {bgr[0, 0, :]}")
        save_path = filepath + str(index).zfill(5) +".png"
        cv2.imwrite(save_path, bgr)
        