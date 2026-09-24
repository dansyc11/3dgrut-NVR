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
import copy
import json
import math
import sys
from typing import List
import numpy as np
from threedgrut_playground.utils.distortion_camera import DistortionCamera
import torch
import os
import kaolin
import polyscope as ps
import polyscope.imgui as psim
import traceback
from threedgrut.utils.logger import logger
from threedgrut.gui.ps_extension import initialize_cugl_interop
from threedgrut_playground.utils.video_out import VideoRecorder
from threedgrut_playground.utils.novel_view_renderer import NovelViewRenderer
from threedgrut_playground.utils.kaolin_future.conversions import polyscope_from_kaolin_camera, polyscope_to_kaolin_camera
from threedgrut_playground.engine import Engine3DGRUT, OptixPrimitiveTypes
from threedgrut_playground.utils.mcap_convertor import McapConverter
import csv
from mcap.writer import Writer
import tqdm
from tqdm import trange

#################################
##    --- Polyscope Gui ---    ##
#################################

class Playground:
    AVAILABLE_CONTROLLERS = ['Turntable', 'First Person', 'Free']

    def __init__(self, gs_object, mesh_assets_folder, default_config, buffer_mode="device2device"):

        self.engine = Engine3DGRUT(gs_object, mesh_assets_folder, default_config)
        self.scene_mog = self.engine.scene_mog
        self.primitives = self.engine.primitives
        self.video_recorder = self.engine.video_recorder
        self.novel_view_renderer = self.engine.novel_view_renderer
        self.mcap_convertor = McapConverter()
        self.video_h = 800
        self.video_w = 1280
        self.selected_pose_idx = 0
       

        """ When this flag is toggled on, the state of the canvas have changed and it needs to be re-rendered """
        self.is_running = True
        self.is_force_canvas_dirty = False
        self.controller_type = 'Turntable'
        self.gui_aux_fields = dict()
        self.density_buffer = copy.deepcopy(self.scene_mog.density)  # For slicing planes
        self.ps_buffer_mode = buffer_mode
        self.viz_do_train = False
        self.viz_bbox = False
        self.live_update = True  # if disabled , will skip rendering updates to accelerate background training loop
        self.viz_render_styles = ['color', 'density']
        self.viz_render_style_ind = 0
        self.viz_curr_render_size = None
        self.viz_curr_render_style_ind = None
        self.viz_render_color_buffer = None
        self.viz_render_scalar_buffer = None
        self.viz_render_name = 'render'
        self.viz_render_enabled = True
        self.selected_camera_idx = None
        self.distortions = None  # Distortion coefficients for the camera, if any

        self.slice_planes = self.slice_plane_enabled = self.slice_plane_pos = self.slice_plane_normal = None
        self.init_polyscope(buffer_mode)

    def run(self):
        """ Run infinite gui loop"""
        while self.is_running:
            ps.frame_tick()

            if ps.window_requests_close():
                self.is_running = False
                os._exit(0)

    @torch.cuda.nvtx.range("init_polyscope")
    def init_polyscope(self, buffer_mode):
        def set_polyscope_buffering_mode():
            if buffer_mode == "host2device":
                logger.info("polyscope set to host2device mode.")
            else:  # device2device
                initialize_cugl_interop()
                logger.info("polyscope set to device2device mode.")
        set_polyscope_buffering_mode()
        ps.set_use_prefs_file(False)
        ps.set_up_dir("neg_y_up")
        ps.set_front_dir("neg_z_front")
        ps.set_navigation_style("free")
        ps.set_enable_vsync(False)
        ps.set_max_fps(-1)
        ps.set_background_color((0., 0., 0.))
        ps.set_ground_plane_mode("none")
        # PLAYGROUND_WIN=WxH overrides the default window size. The window
        # stays user-resizable; nothing re-forces its size per frame any more.
        try:
            win_w, win_h = (int(v) for v in
                            os.environ.get("PLAYGROUND_WIN", "").lower().split("x"))
        except ValueError:
            win_w, win_h = 1600, 1000
        ps.set_window_resizable(True)
        ps.set_window_size(win_w, win_h)
        ps.set_give_focus_on_show(True)
        ps.set_automatically_compute_scene_extents(False)
        ps.set_bounding_box(np.array([-1.5, -1.5, -1.5]), np.array([1.5, 1.5, 1.5]))

        # Toggle off default polyscope menus
        ps.set_build_default_gui_panels(False)

        ps.init()
        ps.set_user_callback(self.ps_ui_callback)

        self.slice_planes = [ps.add_scene_slice_plane() for _ in range(6)]
        self.slice_plane_enabled = [False for _ in range(6)]
        self.slice_plane_pos = [
            np.array([-5.0, 0.0, 0.0]),
            np.array([0.0, -5.0, 0.0]),
            np.array([0.0, 0.0, -5.0]),
            np.array([5.0, 0.0, 0.0]),
            np.array([0.0, 5.0, 0.0]),
            np.array([0.0, 0.0, 5.0]),
        ]
        self.slice_plane_normal = [
            180.0 * np.array([1.0, 0.0, 0.0]),
            180.0 * np.array([0.0, 1.0, 0.0]),
            180.0 * np.array([0.0, 0.0, 1.0]),
            180.0 * np.array([-1.0, 0.0, 0.0]),
            180.0 * np.array([0.0, -1.0, 0.0]),
            180.0 * np.array([0.0, 0.0, -1.0]),
        ]

        # Update once to popualte lazily-created structures
        self.update_render_view_viz(force=True)

    @torch.cuda.nvtx.range("render_from_current_ps_view")
    @torch.no_grad()
    def render_from_current_ps_view(self, window_w=None, window_h=None, distortion=None):
        """ Render a frame using the polyscope gui camera and window size """
        if window_w is None or window_h is None:
            window_w, window_h = ps.get_window_size()
        if self.novel_view_renderer.is_loaded():
            camera = self.novel_view_renderer.get_camera_at_index(self.selected_camera_idx)
            window_w, window_h = camera.width, camera.height

        # No ps.set_window_size here: the render buffers are sized from the
        # camera above, so the output stays camera-exact and the fullscreen
        # image quantity stretches to whatever window the user chose. The
        # per-frame resize used to snap the OS window to the camera
        # resolution and made it impossible to resize. A one-shot snap
        # remains in the "Fly to" paths (_snap_window_to_camera).
        # Update polyscope camera with params from gui
        view_params = ps.CameraParameters(
            ps.CameraIntrinsics(fov_vertical_deg=self.engine.camera_fov, aspect=window_w / window_h),
            ps.get_view_camera_parameters().get_extrinsics()
        )
        
        ps.set_view_camera_parameters(view_params)

        # If window size changed since the last render call, mark the canvas as dirty.
        # We check it here since the event comes from the windowing system and could prompt in between frame renders
        last_window_size = self.engine.last_state.get('canvas_size')
        if last_window_size:
            if (last_window_size[0] != window_h) or (last_window_size[1] != window_w):
                self.is_force_canvas_dirty = True

        camera = polyscope_to_kaolin_camera(view_params, window_w, window_h, distortion_coefficients= distortion, device=self.engine.device)
        if self.novel_view_renderer.is_loaded() and self.selected_camera_idx is not None:
            fx, fy, cx, cy = self.novel_view_renderer.get_camera_intrinsics_at_index(self.selected_camera_idx)
            camera = polyscope_to_kaolin_camera(
                view_params, window_w, window_h, fx = fx, fy = fy, cx = cx, cy = cy, distortion_coefficients= distortion, device = self.engine.device, is_vilota=True)

        is_first_pass = self.is_dirty(camera)
        if not is_first_pass and not self.engine.has_progressive_effects_to_render():
            return self.engine.last_state['rgb'], self.engine.last_state['opacity']

        # Render a frame pass
        outputs = self.engine.render_pass(camera, is_first_pass)

        self.is_force_canvas_dirty = False
        return outputs['rgb'], outputs['opacity']

    def update_data_on_device(self, buffer, tensor_array):
        if self.ps_buffer_mode == 'host2device':
            buffer.update_data(tensor_array.detach().cpu().numpy())
        else:
            buffer.update_data_from_device(tensor_array.detach())

    @torch.cuda.nvtx.range("update_render_view_viz")
    @torch.no_grad()
    def update_render_view_viz(self, force=False):
        """ Renders a single pass using the polyscope camera and updates the polyscope canvas buffers
        with rendered information.
        """
        # Get the current kaolin cam hfov and vfov
        

        if self.novel_view_renderer.is_loaded():
            camera = self.novel_view_renderer.get_camera_at_index(self.selected_camera_idx)
            window_w, window_h = camera.width, camera.height
        else: window_w, window_h = ps.get_window_size()
        if self.distortions:
            distortions = self.distortions[self.selected_camera_idx] if self.selected_camera_idx is not None else None
            # if distortions[5] == 0.0:
            #     intrinsics = self.novel_view_renderer.get_camera_intrinsics_at_index(self.selected_camera_idx) 
            #     distortions = distortions + intrinsics

        style = self.viz_render_styles[self.viz_render_style_ind]
        if force or self.viz_curr_render_style_ind != self.viz_render_style_ind or self.viz_curr_render_size != (
                window_w, window_h):
            self.viz_curr_render_style_ind = self.viz_render_style_ind
            self.viz_curr_render_size = (window_w, window_h)

            if style in ("color",):

                dummy_image = np.ones((window_h, window_w, 4), dtype=np.float32)

                ps.add_color_alpha_image_quantity(
                    self.viz_render_name,
                    dummy_image,
                    enabled=self.viz_render_enabled,
                    image_origin="upper_left",
                    show_fullscreen=True,
                    show_in_imgui_window=False,
                )
                self.viz_render_color_buffer = ps.get_quantity_buffer(self.viz_render_name, "colors")
                self.viz_render_scalar_buffer = None

            elif style == "density":

                dummy_vals = np.zeros((window_h, window_w), dtype=np.float32)
                dummy_vals[0] = 1.0  # hack so the default polyscope scale gets set more nicely

                ps.add_scalar_image_quantity(
                    self.viz_render_name,
                    dummy_vals,
                    enabled=self.viz_render_enabled,
                    image_origin="upper_left",
                    show_fullscreen=True,
                    show_in_imgui_window=False,
                    cmap="blues",
                    vminmax=(0, 1),
                )
                self.viz_render_color_buffer = None
                self.viz_render_scalar_buffer = ps.get_quantity_buffer(self.viz_render_name, "values")

        # do the actual rendering
        try:
            sple_orad, sple_odns = self.render_from_current_ps_view(distortion=distortions if self.distortions else None)
            sple_orad = sple_orad[0]
            sple_odns = sple_odns[0]
        except Exception:
            print("Rendering error occurred.")
            traceback.print_exc()
            return

        # update the data
        if style in ("color",):
            # append 1s for alpha
            sple_orad = torch.cat((sple_orad, torch.ones_like(sple_orad[:, :, 0:1])), dim=-1)
            self.update_data_on_device(self.viz_render_color_buffer, sple_orad)

        elif style == "density":
            self.update_data_on_device(self.viz_render_scalar_buffer, sple_odns)

    def is_dirty(self, camera: kaolin.render.camera.Camera):
        """ Returns true if the state of the scene have changed since last time the canvas was rendered. """
        # Force dirty flag is on or engine specifies canvas is dirty
        return self.is_force_canvas_dirty or self.engine.is_dirty(camera)

    def _draw_preset_settings_widget(self):
        psim.SetNextItemOpen(True, psim.ImGuiCond_FirstUseEver)
        if psim.TreeNode("Quick Settings"):
            if (psim.Button("Fast")):
                self.engine.use_spp = False
                self.engine.antialiasing_mode = '4x MSAA'
                self.engine.spp.mode = 'msaa'
                self.engine.spp.spp = 4
                self.engine.spp.reset_accumulation()
                self.engine.use_optix_denoiser = False
                self.is_force_canvas_dirty = True
            psim.SameLine()
            if (psim.Button("Balanced")):
                self.engine.use_spp = True
                self.engine.antialiasing_mode = '4x MSAA'
                self.engine.spp.mode = 'msaa'
                self.engine.spp.spp = 4
                self.engine.spp.reset_accumulation()
                self.engine.use_optix_denoiser = True
                self.is_force_canvas_dirty = True
            psim.SameLine()
            if (psim.Button("High Quality")):
                self.engine.use_spp = True
                self.engine.antialiasing_mode = 'Quasi-Random (Sobol)'
                self.engine.spp.mode = 'low_discrepancy_seq'
                self.engine.spp.spp = 64
                self.engine.spp.reset_accumulation()
                self.engine.use_optix_denoiser = True
                self.is_force_canvas_dirty = True

            psim.TreePop()

    def _draw_render_widget(self):
        window_w, window_h = ps.get_window_size()
        psim.SetNextItemOpen(True, psim.ImGuiCond_FirstUseEver)
        if psim.TreeNode("Render"):
            render_channel_changed, self.viz_render_style_ind = psim.Combo(
                "Style", self.viz_render_style_ind, self.viz_render_styles
            )
            if render_channel_changed:
                self.engine.rebuild_bvh(self.scene_mog)
                self.is_force_canvas_dirty = True

            psim.PushItemWidth(110)
            cam_idx = Engine3DGRUT.AVAILABLE_CAMERAS.index(self.engine.camera_type)
            is_cam_changed, new_cam_idx = psim.Combo("Camera", cam_idx, Engine3DGRUT.AVAILABLE_CAMERAS)
            if is_cam_changed:
                self.engine.camera_type = Engine3DGRUT.AVAILABLE_CAMERAS[new_cam_idx]
                self.is_force_canvas_dirty = self.is_force_canvas_dirty or is_cam_changed
                self.engine.camera_fov = 120.0 if self.engine.camera_type == 'Fisheye' else 60.0

            if self.engine.camera_type == 'Fisheye':
                psim.SameLine()
                is_fov_changed, self.engine.camera_fov = psim.SliderFloat(
                    "FoV", self.engine.camera_fov, v_min=60.0, v_max=180.0
                )
                self.is_force_canvas_dirty = self.is_force_canvas_dirty or is_fov_changed
            elif self.engine.camera_type == 'Pinhole':
                psim.SameLine()
                is_fov_changed, self.engine.camera_fov = psim.SliderFloat(
                    "FoV", self.engine.camera_fov, v_min=30.0, v_max=90
                )
                new_cam = ps.CameraParameters(
                    ps.CameraIntrinsics(
                        fov_vertical_deg=self.engine.camera_fov,
                        fov_horizontal_deg=None,
                        aspect=window_w / window_h
                    ),
                    ps.get_view_camera_parameters().get_extrinsics()
                )
                ps.set_view_camera_parameters(new_cam)
                self.is_force_canvas_dirty = self.is_force_canvas_dirty or is_fov_changed
            psim.PopItemWidth()

            psim.SameLine()
            if psim.Button("Reset View"):
                ps.reset_camera_to_home_view()
            if psim.IsItemHovered():
                psim.SetNextWindowPos([window_w - psim.GetWindowWidth() - 120, 20])
                psim.Begin("Reset View", psim.ImGuiWindowFlags_NoTitleBar)
                psim.TextUnformatted("View Navigation:")
                psim.TextUnformatted("      Rotate: [left click drag]")
                psim.TextUnformatted("   Translate: [shift] + [left click drag] OR [right click drag]")
                psim.TextUnformatted("        Zoom: [scroll] OR [ctrl] + [shift] + [left click drag]")
                psim.TextUnformatted("   Use [ctrl-c] and [ctrl-v] to save and restore camera poses")
                psim.TextUnformatted("     via the clipboard.")
                psim.TextUnformatted("\nMenu Navigation:")
                psim.TextUnformatted("   Menu headers with a '>' can be clicked to collapse and expand.")
                psim.TextUnformatted("   Use [ctrl] + [left click] to manually enter any numeric value")
                psim.TextUnformatted("     via the keyboard.")
                psim.TextUnformatted("   Press [space] to dismiss popup dialogs.")
                psim.End()

            psim.PushItemWidth(115)

            controller_idx = Playground.AVAILABLE_CONTROLLERS.index(self.controller_type)
            is_controller_changed, new_controller_idx = psim.Combo("Nav.", controller_idx, Playground.AVAILABLE_CONTROLLERS)
            if is_controller_changed:
                self.controller_type = Playground.AVAILABLE_CONTROLLERS[new_controller_idx]
                self.is_force_canvas_dirty = self.is_force_canvas_dirty or is_controller_changed

                if Playground.AVAILABLE_CONTROLLERS[new_controller_idx] == 'Turntable':
                    ps.set_navigation_style("turntable")
                elif Playground.AVAILABLE_CONTROLLERS[new_controller_idx] == 'First Person':
                    ps.set_navigation_style("first_person")
                elif Playground.AVAILABLE_CONTROLLERS[new_controller_idx] == 'Free':
                    ps.set_navigation_style("free")

            psim.SameLine()

            up_dirs = ["x_up", "neg_x_up", "y_up", "neg_y_up", "z_up", "neg_z_up"]
            front_dirs = ["x_front", "neg_x_front", "y_front", "neg_y_front", "z_front", "neg_z_front"]

            up_dir_idx = up_dirs.index(ps.get_up_dir())
            is_cam_up_changed, new_up_idx = psim.Combo("Up", up_dir_idx, up_dirs)
            if is_cam_up_changed:
                ps.set_up_dir(up_dirs[new_up_idx])
                self.is_force_canvas_dirty = self.is_force_canvas_dirty or is_cam_up_changed
            psim.SameLine()
            front_dir_idx = front_dirs.index(ps.get_front_dir())
            is_cam_front_changed, new_front_idx = psim.Combo("Front", front_dir_idx, front_dirs)
            if is_cam_front_changed:
                ps.set_front_dir(front_dirs[new_front_idx])
                self.is_force_canvas_dirty = self.is_force_canvas_dirty or is_cam_front_changed
            psim.PopItemWidth()

            psim.PushItemWidth(100)
            settings_changed, self.engine.gamma_correction = psim.SliderFloat(
                "Gamma Correction", self.engine.gamma_correction, v_min=0.5, v_max=3.0
            )
            self.is_force_canvas_dirty = self.is_force_canvas_dirty or settings_changed
            psim.SameLine()

            settings_changed, self.engine.max_pbr_bounces = psim.SliderInt(
                "Max PBR Bounces", self.engine.max_pbr_bounces, v_min=1, v_max=15
            )
            self.is_force_canvas_dirty = self.is_force_canvas_dirty or settings_changed
            psim.PopItemWidth()

            settings_changed, self.engine.use_optix_denoiser = psim.Checkbox("Use Optix Denoiser", self.engine.use_optix_denoiser)
            self.is_force_canvas_dirty = self.is_force_canvas_dirty or settings_changed

            psim.Text("Camera Coordinates:")
            camera = ps.get_view_camera_parameters()
            camera_pos = camera.get_position()
            camera_rot = camera.get_R()
            psim.PushItemWidth(100)
            # psim.Text(f"Position: [{camera_pos[0]:.2f}, {camera_pos[1]:.2f}, {camera_pos[2]:.2f}]")
            # psim.Text(f"Rotation: {camera_rot}")
            curr_view_mat = camera.get_view_mat()
            curr_six6of = self.novel_view_renderer.convert_view_matrix_to_6dof_pose(curr_view_mat)
            psim.Text(f"6DOF:{str(curr_six6of)}")
         

            psim.PopItemWidth()
            psim.TreePop()

    def _draw_video_recording_controls(self):
        psim.SetNextItemOpen(False, psim.ImGuiCond_FirstUseEver)
        if psim.TreeNode("Record Trajectory Video"):
            _, self.video_recorder.trajectory_output_path = psim.InputText("Video Output Path",
                                                                           self.video_recorder.trajectory_output_path)
            if (psim.Button("Add Camera")):
                if self.selected_camera_idx is None:
                    ps.warning("Please select a camera first.")
                else:self.add_cam_to_vid_recorder(self.selected_camera_idx)
                #view_mat, sixdof_pose, view_mat_string = self.get_view_matrix_and_pose()
                #self.save_pose_to_csv(view_mat, sixdof_pose, view_mat_string, csv_filename="video_trajectory.csv")
                
            psim.SameLine()
            if (psim.Button("Reset")):
                self.video_recorder.reset_trajectory()
            psim.SameLine()
            if (psim.Button("Render Video")):
                try:
                    if self.novel_view_renderer.is_loaded() and self.selected_camera_idx is not None:
                        idx = self.selected_camera_idx
                        cam_name = self.novel_view_renderer.get_cam_name_at_index(idx)
                    self.video_recorder.render_video(cam_name=cam_name if self.novel_view_renderer.is_loaded() else None)
                except ValueError as e: # Catch and display any warnings for incorrect input
                    ps.warning(f'{e}')

            psim.PushItemWidth(150)
            mode_index = VideoRecorder.MODES.index(self.video_recorder.mode)
            _, mode_index = psim.Combo(
                "Interpolation", mode_index, VideoRecorder.MODES
            )
            self.video_recorder.mode = VideoRecorder.MODES[mode_index]
            psim.PopItemWidth()

            if self.video_recorder.mode == 'depth_of_field':
                psim.PushItemWidth(75)
                _, self.video_recorder.min_dof = psim.SliderFloat(
                    "Min FoV", self.video_recorder.min_dof, v_min=0.0, v_max=24.0
                )
                psim.SameLine()
                _, self.video_recorder.max_dof = psim.SliderFloat(
                    "Max FoV", self.video_recorder.max_dof, v_min=0.0, v_max=24.0
                )
                psim.PopItemWidth()

            psim.PushItemWidth(75)
            _, self.video_recorder.frames_between_cameras = psim.SliderInt(
                "Frames Between", self.video_recorder.frames_between_cameras, v_min=1, v_max=120
            )
            psim.SameLine()
            _, self.video_recorder.video_fps = psim.SliderInt(
                "FPS", self.video_recorder.video_fps, v_min=1, v_max=120
            )
            _, self.video_w = psim.SliderInt(
                "Width", self.video_w, v_min=1, v_max=8192
            )
            psim.SameLine()
            _, self.video_h = psim.SliderInt(
                "Height", self.video_h, v_min=1, v_max=8192
            )
            psim.PopItemWidth()

            trajectory = self.video_recorder.trajectory
            psim.Text(f"There are {len(trajectory)} cameras in the trajectory.")

            if len(trajectory) > 0 and psim.TreeNode("Cameras"):
                remained_cameras = []
                for i, cam in enumerate(trajectory):
                    is_not_removed = self._draw_single_trajectory_camera(i, cam)
                    remained_cameras.append(is_not_removed)
                self.video_recorder.trajectory = [trajectory[i] for i in range(len(trajectory)) if remained_cameras[i]]

                psim.TreePop()

            if psim.TreeNode("Load/Save a Trajectory"):
                _, self.video_recorder.cameras_save_path = psim.InputText("Cameras' Path",
                                                                          self.video_recorder.cameras_save_path)
                if (psim.Button("Save Trajectory")):
                    self.video_recorder.save_trajectory()
                psim.SameLine()
                if (psim.Button("Load Trajectory")):
                    self.video_recorder.load_trajectory()

                psim.TreePop()

            psim.TreePop()

    def add_cam_to_vid_recorder(self, index):
        fx, fy, cx, cy = self.novel_view_renderer.get_camera_intrinsics_at_index(index)
        cam = self.novel_view_renderer.get_camera_at_index(index)
        cam_w, cam_h = cam.width, cam.height
        ps_cam_params = polyscope_from_kaolin_camera(cam)
        camera = polyscope_to_kaolin_camera(
                    ps_cam_params, width=cam_w, height=cam_h, distortion_coefficients=self.distortions[index] if index is not None else None,
                    fx = fx, fy = fy, cx = cx, cy = cy
                )
        camera.set_cam_intr(fx, fy, cx, cy)
        self.video_recorder.add_camera(camera)

    
    def _draw_novel_view_from_calib_controls(self):
        """ Draws a widget to render novel views from a camera calibration file. User can fly to different camera views in
        the unit to see how the simulated view from that particular Vilota device looks"""
        psim.SetNextItemOpen(False, psim.ImGuiCond_FirstUseEver)

        # Add a persistent status message attribute if not present
        if not hasattr(self, "_novel_view_calib_status"):
            self._novel_view_calib_status = ""

        # Add an attribute for successfully loaded calibration
        if not hasattr(self, "calibration_loaded"):
            self.calibration_loaded = False

        if not hasattr(self, "_trajectory_status"):
            self._trajectory_status = "Trajectory not loaded"

        if psim.TreeNode("Novel View from Vilota Calibration file"):
            #psim.Text("Place your calibration file into a folder called ./calibrations")
            
            # CALIB DROPDOWN. Scanned once, refreshable. Each entry shows the
            # deviceName inside the file, because the filename and the serial
            # do not always agree: vk180.json holds DP180IP-2404-0004.
            if not hasattr(self, "_calib_files"):
                self._calib_files, self._calib_labels = self._scan_calibration_files()
                cur = self.novel_view_renderer.calibration_filename
                self._calib_idx = (self._calib_files.index(cur)
                                   if cur in self._calib_files else 0)
            if self._calib_files:
                changed, self._calib_idx = psim.Combo(
                    "Calibration file", self._calib_idx, self._calib_labels)
                if changed:
                    self.novel_view_renderer.calibration_filename = \
                        self._calib_files[self._calib_idx]
                psim.SameLine()
                if psim.Button("Rescan"):
                    del self._calib_files
            else:
                psim.Text("No .json files in ./calibration_files/")

            _, self.novel_view_renderer.calibration_filename = psim.InputText(
                "Calib path",
                self.novel_view_renderer.calibration_filename
            )
            self.novel_view_renderer.set_filepath()
            calibration_path = self.novel_view_renderer.calibration_fullpath
            psim.TextWrapped(f"Loads from: {calibration_path}")

            # Disable "Load Calibration" button if calibration is already loaded
            psim.BeginDisabled(self.calibration_loaded == True)
            if psim.Button("Load Calibration"):
                try:
                    self.novel_view_renderer.load_device()
                    self.calibration_loaded = True  # Set the flag to indicate that calibration is loaded
                    self._novel_view_calib_status = f"Loaded {self.novel_view_renderer.get_camera_count()} cameras from {calibration_path}"
                    self.distortions = self.novel_view_renderer.get_cam_distortions()
                except FileNotFoundError as e:
                    self._novel_view_calib_status = f"Calibration file not found. \nPlease make sure that file exists."
                    import traceback; traceback.print_exc()
                except ValueError as e:
                    self._novel_view_calib_status = f"{e}"
                    import traceback; traceback.print_exc()
                except Exception as e:
                    self._novel_view_calib_status = f"{e}"
                    import traceback; traceback.print_exc()
                    
            psim.EndDisabled()

            if self.novel_view_renderer.is_loaded():
                psim.SameLine()
                if psim.Button("Reload Calibration"):
                    try:
                        self.selected_camera_idx = None
                        self.calibration_loaded = False  # Reset the flag to allow reloading
                        self.novel_view_renderer.load_device(reload=True)
                        self._novel_view_calib_status = f"Reloaded {self.novel_view_renderer.get_camera_count()} cameras from {calibration_path}"
                        self.distortions = self.novel_view_renderer.get_cam_distortions()
                        self.calibration_loaded = True  # Set the flag to indicate that the device is loaded
                    except FileNotFoundError as e:
                        self._novel_view_calib_status = f"Calibration file not found. \nPlease make sure that file exists."
                    except ValueError as e:
                        self._novel_view_calib_status = f"{e}"
                    except Exception as e:
                        self._novel_view_calib_status = f"{e}"
            # Always display the status message if present
            if self._novel_view_calib_status:
                psim.TextWrapped(self._novel_view_calib_status)
            
            
            if getattr(self, "selected_camera_idx", None) is None:
                # If no camera is selected, select the first one by default
                self.selected_camera_idx = 0    
            if getattr(self, "trajectory_loaded", None) is None:
                 self.trajectory_loaded = False
            
            if getattr(self, "poses", None) is None:
                # Initialize poses_list if it doesn't exist
                self.poses = []
            
            if getattr(self, "export_cam_index", None) is None:
                # Initialize export_cam_index if it doesn't exist
                self.export_cam_index = 0

            # If calibration is loaded, show all controls
            if self.calibration_loaded:
                if psim.TreeNode("Device loaded"):
                    psim.Text(self.novel_view_renderer.get_device_name_and_serial_no())
                    _, self.flyto_relative = psim.Checkbox(
                        "Fly to relative to current view",
                        getattr(self, "flyto_relative", False))
                    for idx, (camera) in self.novel_view_renderer.get_all_cameras().items():
                        # Draw a button to select this camera
                        if psim.Button(f"Select {self.novel_view_renderer.get_cam_name_at_index(idx)}"):
                            self.selected_camera_idx = idx
                            psim.SameLine()
                            self._draw_single_vk_cam(idx, camera)
                        elif self.selected_camera_idx == idx:
                            psim.SameLine()
                            self._draw_single_vk_cam(idx, camera)
                    psim.Text(f"Origin camera = {self.novel_view_renderer.get_origin_camera_index()}")
                    psim.TreePop()

                if psim.TreeNode("Save/Load Video trajectory"):
                    if self._trajectory_status:
                        psim.TextWrapped(self._trajectory_status)
                    _, self.novel_view_renderer.trajectory_filename = psim.InputText(
                        "Trajectory file",
                        self.novel_view_renderer.trajectory_filename
                    )
                    psim.TextWrapped("in ./video_trajectories/")
                    if not self.trajectory_loaded and psim.Button("Load Trajectory"):
                        try:
                            self.novel_view_renderer.set_trajectory_filepath()
                            self.poses = self.novel_view_renderer.get_trajectory_poses()
                            self.trajectory_loaded = True  # Set the flag to indicate that the trajectory is loaded
                        except FileNotFoundError as e:
                            self._trajectory_status = f"Trajectory file not found. \nPlease make sure that file exists."
                        except ValueError as e:
                            self._trajectory_status = f"{e}"
                        except Exception as e:
                            self._trajectory_status = f"{e}"
                    psim.SameLine()
                    if psim.Button("Build Orbit Trajectory"):
                        from threedgrut_playground.utils.orbit_trajectory import build_orbit_trajectory
                        try:
                            self.poses = build_orbit_trajectory(
                                self, flip=getattr(self, "orbit_flip", False)
                            )
                            self._trajectory_status = f"Orbit built: {len(self.poses)} poses"
                            self.trajectory_loaded = True
                            # Park the rig at the selected camera's first orbit
                            # pose, so "Fly to" shows the render's first view
                            # instead of the world origin.
                            if self.poses:
                                first = getattr(self, "orbit_first_pose", {}).get(
                                    self.selected_camera_idx, 0)
                                self._fly_to_pose(first, self.poses[first])
                                self._trajectory_status += f", rig at pose {first}"
                        except Exception as e:
                            self._trajectory_status = f"Orbit failed: {e}"
                    psim.SameLine()
                    _, self.orbit_flip = psim.Checkbox(
                        "Flip side", getattr(self, "orbit_flip", False)
                    )

                    if psim.Button("Create New Trajectory"):
                        self.poses = self.novel_view_renderer.create_new_trajectory()
                        self._trajectory_status = "Created a new trajectory. Add poses to it."
                        self.trajectory_loaded = True
                    
                    if self.trajectory_loaded:
                        if psim.Button("Add current view to trajectory"):
                            current_cam_params = ps.get_view_camera_parameters()
                            current_cam_view_mat = current_cam_params.get_view_mat()
                            curr_cam_index = self.selected_camera_idx
                            self.poses = self.novel_view_renderer.add_pose_to_trajectory(current_cam_view_mat, curr_cam_index)
                        if psim.TreeNode(f"Trajectory: {len(self.poses)} poses"):
                            _, self.export_cam_index = psim.SliderInt(
                                "Choose Cam", self.export_cam_index, v_min=0, v_max=self.novel_view_renderer.get_camera_count() - 1
                            )
                            
                            psim.NewLine()
                            if psim.Button("Render Current Cam Trajectory PNGS"):
                                self.video_recorder.export_format = "png"
                                self.populate_vid_trajectory(self.poses, self.export_cam_index)
                                cam_name = self.novel_view_renderer.get_cam_name_at_index(self.export_cam_index)
                                self.video_recorder.render_video(cam_name=cam_name if self.novel_view_renderer.is_loaded() else None)
                            psim.SameLine()
                            if psim.Button("Render Device Trajectory MCAP"):
                                self.render_mcap_trajectory(self.poses)
                            psim.NewLine()
                            psim.Text(f"Cam {self.export_cam_index} to render.")
                            poses_list = self.novel_view_renderer.get_trajectory_poses()
                            self.poses = poses_list
                            self._draw_cam_trajectory_view(self.poses)
                            psim.TreePop()

                    psim.TreePop()

            
            # if self.calibration_loaded:
            #     origin = self.novel_view_renderer.get_origin_camera_index()
            #     if self.calibration_loaded and self.selected_camera_idx == origin and psim.Button("Save World to Cam D"):
            #         cam_d_view_matrix = ps.get_camera_view_matrix()
            #         self.novel_view_renderer.world_to_camd = cam_d_view_matrix
            #         print(f"Saved world to cam D view matrix: {cam_d_view_matrix}")
                
            #     if len(self.novel_view_renderer.world_to_camd) != 0 and self.selected_camera_idx != origin and psim.Button("Save world to cam i"):
            #         cam_i_view_matrix = ps.get_camera_view_matrix()
            #         self.novel_view_renderer.world_to_cami = cam_i_view_matrix
            #         self.novel_view_renderer.i = self.selected_camera_idx
            #         print(f"Saved world to cam {self.novel_view_renderer.i} view matrix : {self.novel_view_renderer.world_to_cami}")
                
            #     if len(self.novel_view_renderer.world_to_camd) != 0 and len(self.novel_view_renderer.world_to_cami) != 0 and psim.Button("Check rig layout"):
            #         self.novel_view_renderer.check_rig_layout()

            # if getattr(self, "scene_center", None) is None:
            #     # If no center is defined, select this by default
            #     self.scene_center = [0.999029278755188, 0.00626010866835713, 0.043603427708149, 0.0137977302074432, 
            #                          0.00638079596683383, -0.99997615814209, -0.00262921908870339, -0.265371531248093, 
            #                          0.0435859337449074, 0.00290489150211215, -0.999045491218567, 2.19675970077515, 
            #                          0.0, 0.0, 0.0, 1.0]
            

            # if self.calibration_loaded and psim.Button("Register New Scene Center"):
            #     view_mat, sixdof_pose, view_mat_string = self.get_view_matrix_and_pose()
            #     print(f"View matrix: {view_mat}")
            #     self.save_pose_to_csv(view_mat, sixdof_pose, view_mat_string)
            #     print(f"6DOF pose: {sixdof_pose}")
            
            # psim.SameLine()        
                
            # if self.calibration_loaded and psim.Button("Move to Scene center"):
            #     if hasattr(self, "scene_center") and self.scene_center is not None:
            #         # Move the camera to the saved scene center
            #         index = self.selected_camera_idx
            #         self.novel_view_renderer.move_rig_to_pose(self.scene_center,cam_index= index, is_6dof=False)

            #     cam = self.novel_view_renderer.get_camera_at_index(self.selected_camera_idx)
            #     view_params = polyscope_from_kaolin_camera(cam)
            #     eye = view_params.get_position()
            #     target = view_params.get_position() + view_params.get_look_dir()
            #     up = view_params.get_up_dir()
            #     ps.look_at_dir(eye, target, up, fly_to=True)
            
            # #set indicator for trajectory loaded
            # if getattr(self, "trajectory_loaded", None) is None:
            #     self.trajectory_loaded = False
            
            # if getattr(self, "poses_list", None) is None:
            #     self.poses_list = []
            
            # _, self.novel_view_renderer.trajectory_filename = psim.InputText(
            #     "Trajectory filename (in ./calibration_files/trajectories)",
            #     self.novel_view_renderer.trajectory_filename
            # )





            # self.novel_view_renderer.set_trajectory_filepath()
            # trajectory_path = self.novel_view_renderer.trajectory_fullpath
            # psim.Text(f"Trajectory file will be loaded from:\n{trajectory_path}")


            # if self.calibration_loaded and psim.TreeNode("Load Trajectory"):
            #     if psim.Button("Load Trajectory"):
            #         self.trajectory_loaded = True
            #         poses_list = self.novel_view_renderer.get_trajectory_poses()
            #         self.poses_list = poses_list
            #         psim.Text(f"Loaded {len(poses_list)} poses from the trajectory.")
            #     if self.trajectory_loaded:
            #         self._draw_cam_trajectory_view(self.poses_list)

            
            
            # if self.calibration_loaded and psim.TreeNode("Move device to pose"):
            #     psim.Text("Enter your 6DOF pose (angles in degrees):")
            #     # Persist pose values as an attribute so they don't reset every frame
            #     if not hasattr(self, "_move_pose_values"):
            #         self._move_pose_values = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            #     labels = ["x", "y", "z", "Roll", "Pitch", "Yaw"]
            #     changed_any = False
            #     for i, label in enumerate(labels):
            #         changed, value = psim.InputFloat(label, self._move_pose_values[i])
            #         if changed:
            #             self._move_pose_values[i] = value
            #             changed_any = True
            #     if psim.Button("Set 6dof pose"):
            #         pose = self._move_pose_values.copy()
            #         self.novel_view_renderer.move_rig_to_pose(pose)
            #         cam = self.novel_view_renderer.get_camera_at_index(self.selected_camera_idx)
            #         view_params = polyscope_from_kaolin_camera(cam)
            #         eye = view_params.get_position()
            #         target = view_params.get_position() + view_params.get_look_dir()
            #         up = view_params.get_up_dir()
            #         ps.look_at_dir(eye, target, up, fly_to=True)

            psim.TreePop()
    
    def populate_vid_trajectory(self, poses_list, idx=None):
        """ Populates the video recorder trajectory with poses from the provided list.
            Moves rig to each pose and adds the camera view to the video recorder to 
            be interpolated later.
         Args:
            poses_list (list): List of poses to populate the video recorder with (from
            origin camera, since all trajectories are saved as origin camera/body poses )
            idx (int, optional): Camera index to use for the poses. If None, uses
                the currently selected camera index.
        """
        self.video_recorder.trajectory = []
        if idx is None:
            idx = self.selected_camera_idx
        for pose in poses_list:
            self.novel_view_renderer.move_rig_to_pose(pose, is_6dof=True)
            self.add_cam_to_vid_recorder(idx)
            


    # def save_pose_to_csv(self, view_mat, sixdof_pose, view_mat_string, csv_filename="new_scene_center.csv"):
    #     csv_path = "./trajectories/" + csv_filename
    #     # Ensure sixdof_pose is a flat list or tuple of 6 values
    #     if sixdof_pose is not None and len(sixdof_pose) == 6:
    #         file_exists = os.path.isfile(csv_path)
    #         with open(csv_path, "a", newline='') as f:
    #             writer = csv.writer(f)
    #             if not file_exists:
    #                 writer.writerow(["x", "y", "z", "roll", "pitch", "yaw"])
    #             writer.writerow(sixdof_pose)
    #     psim.Text(f"6DOF pose: {sixdof_pose}")

    # def get_view_matrix_and_pose(self):
    #     params_string = ps.get_view_as_json()
    #     params_json = json.loads(params_string)
    #             # Grab the view matrix
    #     view_mat = params_json.get("viewMat", None)
    #     self.scene_center = view_mat if view_mat is not None else self.scene_center
    #     self.novel_view_renderer.center_view_matrix = self.scene_center
    #     sixdof_pose = self.novel_view_renderer.convert_scene_center_to_6dof_pose()
    #     view_mat_string = str(view_mat) if view_mat is not None else None
    #     return view_mat,sixdof_pose,view_mat_string



    def _draw_slice_plane_controls(self):

        psim.SetNextItemOpen(False, psim.ImGuiCond_FirstUseEver)
        if psim.TreeNode("Slice Planes"):
            any_plane_changed = False
            for sp_idx, slice_plane in enumerate(self.slice_planes):
                self.slice_planes[sp_idx].set_draw_widget(False)

                changed, is_enabled = psim.Checkbox(f"Slice Plane {sp_idx}", self.slice_plane_enabled[sp_idx])
                if changed:
                    # self.slice_planes[sp_idx].set_draw_plane(is_enabled)
                    self.slice_plane_enabled[sp_idx] = is_enabled
                    self.slice_planes[sp_idx].set_draw_plane(False)
                    self.slice_planes[sp_idx].set_pose(self.slice_plane_pos[sp_idx], self.slice_plane_normal[sp_idx])
                any_plane_changed |= changed

                psim.PushItemWidth(psim.GetContentRegionAvail()[0] * 0.65)
                changed, values = psim.SliderFloat3(
                    f"SPPos{sp_idx}",
                    [self.slice_plane_pos[sp_idx][0], self.slice_plane_pos[sp_idx][1], self.slice_plane_pos[sp_idx][2]],
                    v_min=-10.0, v_max=10.0,
                    format="%.2f",
                    
                )
                any_plane_changed |= changed
                if changed:
                    self.slice_plane_pos[sp_idx] = values
                    self.slice_planes[sp_idx].set_pose(self.slice_plane_pos[sp_idx], self.slice_plane_normal[sp_idx])

                changed, values = psim.SliderFloat3(
                    f"SPNorm{sp_idx}",
                    [self.slice_plane_normal[sp_idx][0], self.slice_plane_normal[sp_idx][1],
                     self.slice_plane_normal[sp_idx][2]],
                    v_min=-180.0, v_max=180.0,
                    format="%.2f",
                    
                )
                any_plane_changed |= changed
                if changed:
                    self.slice_plane_normal[sp_idx] = values
                    self.slice_planes[sp_idx].set_pose(self.slice_plane_pos[sp_idx], self.slice_plane_normal[sp_idx])

                psim.PopItemWidth()

            if any_plane_changed:
                self._recompute_slice_planes()

            psim.TreePop()

    @torch.no_grad()
    def _recompute_slice_planes(self):
        p1 = self.scene_mog.positions
        enabled_planes = p1.new_tensor(self.slice_plane_enabled, dtype=torch.bool)
        self.scene_mog.density = copy.deepcopy(self.density_buffer)
        if enabled_planes.any():
            p0 = p1.new_tensor(self.slice_plane_pos)[None, enabled_planes]
            n = p1.new_tensor(self.slice_plane_normal)[None, enabled_planes]
            is_inside = (torch.sum((p1[:, None] - p0) * n, dim=-1) > 0).all(dim=1)
            self.scene_mog.density[~is_inside] = -100000  # Empty density
        self.is_force_canvas_dirty = True

    def _draw_antialiasing_widget(self):
        psim.SetNextItemOpen(True, psim.ImGuiCond_FirstUseEver)
        if psim.TreeNode("Antialiasing"):
            psim.PushItemWidth(150)
            settings_changed, self.engine.use_spp = psim.Checkbox(
                "Enable", self.engine.use_spp
            )
            self.is_force_canvas_dirty = self.is_force_canvas_dirty or settings_changed

            psim.SameLine()
            aa_index = Engine3DGRUT.ANTIALIASING_MODES.index(self.engine.antialiasing_mode)
            psim.PushItemWidth(200)
            is_antialiasing_changed, aa_index = psim.Combo(
                "Mode", aa_index, Engine3DGRUT.ANTIALIASING_MODES
            )
            psim.PopItemWidth()
            if is_antialiasing_changed:
                self.engine.antialiasing_mode = Engine3DGRUT.ANTIALIASING_MODES[aa_index]
                # '4x MSAA', '8x MSAA', '16x MSAA', 'Quasi-Random (Sobol)'
                if self.engine.antialiasing_mode == '4x MSAA':
                    self.engine.spp.mode = 'msaa'
                    self.engine.spp.spp = 4
                elif self.engine.antialiasing_mode == '8x MSAA':
                    self.engine.spp.mode = 'msaa'
                    self.engine.spp.spp = 8
                elif self.engine.antialiasing_mode == '16x MSAA':
                    self.engine.spp.mode = 'msaa'
                    self.engine.spp.spp = 16
                elif self.engine.antialiasing_mode == 'Quasi-Random (Sobol)':
                    self.engine.spp.mode = 'low_discrepancy_seq'
                    self.engine.spp.spp = 64
                self.engine.spp.reset_accumulation()
            self.is_force_canvas_dirty = self.is_force_canvas_dirty or is_antialiasing_changed

            psim.SameLine()
            psim.PushItemWidth(75)
            spp_min, spp_max = 1, 256
            if self.engine.antialiasing_mode == '4x MSAA':
                spp_min, spp_max = 4, 4
            elif self.engine.antialiasing_mode == '8x MSAA':
                spp_min, spp_max = 8, 8
            elif self.engine.antialiasing_mode == '16x MSAA':
                spp_min, spp_max = 16, 16
            settings_changed, self.engine.spp.spp = psim.SliderInt(
                "SPP", self.engine.spp.spp, v_min=spp_min, v_max=spp_max
            )
            psim.PopItemWidth()
            self.is_force_canvas_dirty = self.is_force_canvas_dirty or settings_changed

            settings_changed, self.engine.spp.batch_size = psim.SliderInt(
                "Batch Size (#Frames)", self.engine.spp.batch_size, v_min=1, v_max=min(1024, self.engine.spp.spp)
            )
            self.is_force_canvas_dirty = self.is_force_canvas_dirty or settings_changed

            if self.engine.use_spp:
                panel_width = psim.GetContentRegionAvail()[0]
                progress_width = round(panel_width / 1.2)
                progress_label = f'{str(self.engine.spp.spp_accumulated_for_frame)}/{str(self.engine.spp.spp)}'
                psim.ProgressBar(fraction=self.engine.spp.spp_accumulated_for_frame / self.engine.spp.spp,
                                 size_arg=(progress_width, 0))

            psim.PopItemWidth()
            psim.TreePop()

    def _draw_depth_of_field_widget(self):
        psim.SetNextItemOpen(True, psim.ImGuiCond_FirstUseEver)
        if psim.TreeNode("Depth of Field"):
            settings_changed, self.engine.use_depth_of_field = psim.Checkbox(
                "Enable", self.engine.use_depth_of_field
            )
            self.is_force_canvas_dirty = self.is_force_canvas_dirty or settings_changed

            psim.SameLine()
            settings_changed, self.engine.depth_of_field.spp = psim.SliderInt(
                "SPP", self.engine.depth_of_field.spp, v_min=1, v_max=256
            )
            self.is_force_canvas_dirty = self.is_force_canvas_dirty or settings_changed

            settings_changed, self.engine.depth_of_field.focus_z = psim.SliderFloat(
                "Focus Z", self.engine.depth_of_field.focus_z, v_min=0.25, v_max=24.0
            )
            self.is_force_canvas_dirty = self.is_force_canvas_dirty or settings_changed

            psim.SameLine()
            settings_changed, self.engine.depth_of_field.aperture_size = psim.SliderFloat(
                "Aperture Size", self.engine.depth_of_field.aperture_size, v_min=1e-5, v_max=1e-1)
            self.is_force_canvas_dirty = self.is_force_canvas_dirty or settings_changed

            if self.engine.use_depth_of_field:
                panel_width = psim.GetContentRegionAvail()[0]
                progress_width = round(panel_width / 1.2)
                progress_label = f'{str(self.engine.depth_of_field.spp_accumulated_for_frame)}/{str(self.engine.depth_of_field.spp)}'
                psim.ProgressBar(fraction=self.engine.depth_of_field.spp_accumulated_for_frame / self.engine.depth_of_field.spp,
                                 size_arg=(progress_width, 0))

            psim.TreePop()

    def _draw_primitives_widget(self):
        removed_objs = []
        duped_objs = []
        psim.SetNextItemOpen(True, psim.ImGuiCond_FirstUseEver)
        if psim.TreeNode("Primitives"):
            self._draw_general_primitive_settings_widget()
            for obj_name, obj in self.primitives.objects.items():

                psim.SetNextItemOpen(True, psim.ImGuiCond_FirstUseEver)
                if psim.TreeNode(obj_name):

                    is_retained, is_duplicated = self._draw_single_primitive_main_settings_widget(obj_name, obj)
                    if not is_retained:
                        removed_objs.append(obj_name)
                    if is_duplicated:
                        duped_objs.append(obj_name)

                    psim.SetNextItemOpen(True, psim.ImGuiCond_FirstUseEver)
                    if psim.TreeNode("Properties"):

                        if obj.primitive_type == OptixPrimitiveTypes.DIFFUSE:
                            self._draw_diffuse_pbr_settings_widget(obj)
                        if obj.primitive_type == OptixPrimitiveTypes.GLASS:
                            self._draw_glass_settings_widget(obj)
                        elif obj.primitive_type == OptixPrimitiveTypes.MIRROR:
                            pass
                        psim.TreePop()

                    self._draw_transform_widget(obj)
                    psim.TreePop()
            psim.TreePop()

        for obj_name in removed_objs:
            self.primitives.remove_primitive(obj_name)
            self.is_force_canvas_dirty = True

        for obj_name in duped_objs:
            self.primitives.duplicate_primitive(obj_name)
            self.is_force_canvas_dirty = True

    def _draw_materials_widget(self):
        material_changed = False
        psim.SetNextItemOpen(False, psim.ImGuiCond_FirstUseEver)
        if psim.TreeNode("Materials"):

            for mat_name, material in self.primitives.registered_materials.items():
                psim.SetNextItemOpen(False, psim.ImGuiCond_FirstUseEver)
                if psim.TreeNode(mat_name + f' [ID #{material.material_id}]'):
                    changed, values = psim.SliderFloat3(
                        "Diffuse Factor",
                        [material.diffuse_factor[0], material.diffuse_factor[1], material.diffuse_factor[2]],
                        v_min=0.0, v_max=1.4,
                        format="%.3f",
                        
                    )
                    if changed:
                        material.diffuse_factor[0] = values[0]
                        material.diffuse_factor[1] = values[1]
                        material.diffuse_factor[2] = values[2]
                        material_changed = True

                    changed, values = psim.SliderFloat3(
                        "Emissive Factor",
                        [material.emissive_factor[0], material.emissive_factor[1], material.emissive_factor[2]],
                        v_min=0.0, v_max=1.0,
                        format="%.3f",
                        
                    )
                    if changed:
                        material.emissive_factor[0] = values[0]
                        material.emissive_factor[1] = values[1]
                        material.emissive_factor[2] = values[2]
                        material_changed = True

                    changed, value = psim.SliderFloat("Metallic Factor", material.metallic_factor,
                                                      v_min=0.0, v_max=1.0)
                    if changed:
                        material.metallic_factor = value
                        material_changed = True

                    changed, value = psim.SliderFloat("Roughness Factor", material.roughness_factor,
                                                      v_min=0.0, v_max=1.0)
                    if changed:
                        material.roughness_factor = value
                        material_changed = True

                    changed, value = psim.SliderFloat("Transmission Factor", material.transmission_factor,
                                                      v_min=0.0, v_max=1.0)
                    if changed:
                        material.transmission_factor = value
                        material_changed = True

                    changed, value = psim.SliderFloat("IOR", material.ior,
                                                      v_min=0.2, v_max=2.0)
                    if changed:
                        material.ior = value
                        material_changed = True

                    if material.diffuse_map is not None:
                        psim.Text(f"Diffuse Texture: {material.diffuse_map.shape[0]}x{material.diffuse_map.shape[1]}")
                    else:
                        psim.Text(f"Diffuse Texture: No")

                    if material.emissive_map is not None:
                        psim.Text(
                            f"Emissive Texture: {material.emissive_map.shape[0]}x{material.emissive_map.shape[1]}")
                    else:
                        psim.Text(f"Emissive Texture: No")

                    if material.metallic_roughness_map is not None:
                        psim.Text(
                            f"Metal-Rough Texture: {material.metallic_roughness_map.shape[0]}x{material.metallic_roughness_map.shape[1]}")
                    else:
                        psim.Text(f"Metal-Rough Texture: No")

                    if material.normal_map is not None:
                        psim.Text(f"Normal Texture: {material.normal_map.shape[0]}x{material.normal_map.shape[1]}")
                    else:
                        psim.Text(f"Normal Texture: No")

                    psim.TreePop()
            psim.TreePop()

        if material_changed:
            self.is_force_canvas_dirty = True
            self.engine.invalidate_materials_on_gpu()

    def _draw_general_primitive_settings_widget(self):
        primitives_disabled = not self.primitives.enabled
        settings_changed, self.primitives.enabled = psim.Checkbox(
            "Disable Primitives", primitives_disabled
        )
        self.primitives.enabled = not self.primitives.enabled
        self.is_force_canvas_dirty = self.is_force_canvas_dirty or settings_changed

        psim.SameLine()
        settings_changed, self.primitives.use_smooth_normals = psim.Checkbox(
            "Smooth Normals", self.primitives.use_smooth_normals
        )
        self.is_force_canvas_dirty = self.is_force_canvas_dirty or settings_changed

        psim.SameLine()
        settings_changed, self.primitives.disable_pbr_textures = psim.Checkbox(
            "Disable Textures", self.primitives.disable_pbr_textures
        )
        self.is_force_canvas_dirty = self.is_force_canvas_dirty or settings_changed

        settings_changed, self.engine.disable_gaussian_tracing = psim.Checkbox(
            "Disable Gaussians", self.engine.disable_gaussian_tracing
        )
        self.is_force_canvas_dirty = self.is_force_canvas_dirty or settings_changed

        psim.SameLine()

        settings_changed, self.engine.force_white_bg = psim.Checkbox(
            "Force White BG", self.engine.force_white_bg
        )
        self.is_force_canvas_dirty = self.is_force_canvas_dirty or settings_changed
        if self.engine.envmap is not None:
            psim.SameLine()
            settings_changed, self.primitives.enable_envmap = psim.Checkbox(
                "Enable Envmap", self.primitives.enable_envmap
            )
            self.is_force_canvas_dirty = self.is_force_canvas_dirty or settings_changed
            psim.SameLine()
            settings_changed, self.primitives.use_envmap_as_background = psim.Checkbox(
                "Use Envmap As Background", self.primitives.use_envmap_as_background
            )
            self.is_force_canvas_dirty = self.is_force_canvas_dirty or settings_changed

        psim.PushItemWidth(100)
        is_add_primitive = psim.Button("Add Primitive")
        psim.PopItemWidth()

        psim.SameLine()
        if 'add_geom_select_type' not in self.gui_aux_fields:
            self.gui_aux_fields['add_geom_select_type'] = 0
        available_geometries = sorted(list(self.primitives.assets.keys()))
        _, new_geom_select_type_idx = psim.Combo(
            "Geometry", self.gui_aux_fields['add_geom_select_type'], available_geometries
        )
        self.gui_aux_fields['add_geom_select_type'] = new_geom_select_type_idx

        if is_add_primitive:
            geom_idx = self.gui_aux_fields['add_geom_select_type']
            self.primitives.add_primitive(
                geometry_type=available_geometries[geom_idx],
                primitive_type=OptixPrimitiveTypes.GLASS,
                device=self.scene_mog.device
            )
            self.primitives.rebuild_bvh_if_needed(True, True)
            self.is_force_canvas_dirty = True
            self.engine.invalidate_materials_on_gpu()

        psim.SameLine()

        psim.PushItemWidth(100)
        scenes_folder = 'playground_scenes'
        scene_path = os.path.join(scenes_folder, self.engine.scene_name) + '.pt'
        if psim.Button("Save"):
            os.makedirs(scenes_folder, exist_ok=True)
            data = dict(
                objects=self.primitives.objects,
                materials=self.primitives.registered_materials,
                slice_plane_enabled=self.slice_plane_enabled,
                slice_plane_pos=self.slice_plane_pos,
                slice_plane_normal=self.slice_plane_normal
            )
            torch.save(data, scene_path)

            print(f'Scene saved to {scene_path}')
        psim.SameLine()
        if psim.Button("Load"):
            if not os.path.exists(scene_path):
                ps.warning(f'No data stored for scene under {scene_path}')
            else:
                data = torch.load(scene_path)
                self.primitives.objects = data['objects']
                self.primitives.registered_materials = data.get('materials', dict())
                self.slice_plane_enabled = data['slice_plane_enabled']
                self.slice_plane_pos = data['slice_plane_pos']
                self.slice_plane_normal = data['slice_plane_normal']
                self._recompute_slice_planes()
                self.primitives.rebuild_bvh_if_needed(force=True, rebuild=True)
                self.is_force_canvas_dirty = True
                self.engine.invalidate_materials_on_gpu()
                print(f'Scene loaded from {scene_path}')
        psim.PopItemWidth()

    def _draw_single_primitive_main_settings_widget(self, obj_name, obj):
        available_primitive_modes = OptixPrimitiveTypes.names()
        settings_changed, new_prim_type_idx = psim.Combo(
            "Type", obj.primitive_type.value, available_primitive_modes
        )
        if settings_changed:
            obj.primitive_type = OptixPrimitiveTypes(new_prim_type_idx)
            self.primitives.recompute_stacked_buffers()
            self.primitives.rebuild_bvh_if_needed(True, True)  # Rebuild so None types are truly ignored in BVH
        self.is_force_canvas_dirty = self.is_force_canvas_dirty or settings_changed

        psim.SameLine()

        is_retained = True
        is_duplicated = False
        psim.PushItemWidth(100)  # button_width
        if psim.Button("Remove"):
            is_retained = False
        psim.SameLine()
        if psim.Button("Duplicate"):
            is_duplicated = True
        psim.PopItemWidth()

        return is_retained, is_duplicated


    def _scan_calibration_files(self):
        """List .json files in ./calibration_files/ with their device name.

        Returns (filenames, labels). A file without cameraData is marked, so
        the dropdown never offers something the loader will reject.
        """
        import json
        import os
        folder = "./calibration_files/"
        names, labels = [], []
        try:
            found = sorted(f for f in os.listdir(folder) if f.endswith(".json"))
        except OSError:
            return [], []
        for fn in found:
            try:
                with open(os.path.join(folder, fn)) as fh:
                    d = json.load(fh)
                cams = len(d.get("cameraData", []))
                dev = d.get("deviceName") or "no deviceName"
                label = (f"{fn}  [{dev}, {cams} cams]" if cams
                         else f"{fn}  [no cameraData]")
            except Exception:
                label = f"{fn}  [unreadable]"
            names.append(fn)
            labels.append(label)
        return names, labels

    def _draw_transform_widget(self, obj):

        psim.SetNextItemOpen(True, psim.ImGuiCond_FirstUseEver)
        if psim.TreeNode("Transform"):
            object_transform = obj.transform
            transform_changed = False
            psim.PushItemWidth(100)  # button_width
            # if psim.Button("Reset"):
            #     object_transform.reset()
            #     transform_changed = True
            # psim.PopItemWidth()
            if psim.Button("Reset to 30cm square size"):
                # Change the size, keep the board where it is. reset() also
                # zeroes translation and rotation, which loses the placement.
                _t = (object_transform.tx, object_transform.ty, object_transform.tz)
                _r = (object_transform.rx, object_transform.ry, object_transform.rz)
                object_transform.reset()
                object_transform.tx, object_transform.ty, object_transform.tz = _t
                object_transform.rx, object_transform.ry, object_transform.rz = _r
                object_transform.sx = 1.410
                object_transform.sy = 0.825
                transform_changed = True
            psim.PopItemWidth()

            if not hasattr(self, "square_size"):
                self.square_size = 15.0
            
            _, self.square_size = psim.InputFloat(
                "Set Square Size:",
                self.square_size,
                format="%.2f"
            )
            psim.SameLine()
            sized = self.size_any_object(obj, self.square_size)
            if psim.Button("Square size (cm)"):
                if sized is None:
                    print("[playground] no texture on this object, cannot size it")
                else:
                    object_transform.sx, object_transform.sy = sized[0], sized[1]
                    transform_changed = True
            psim.SameLine()
            psim.TextDisabled(sized[2] if sized else "no texture")

            # SLIDERS RESTORED
            psim.PushItemWidth(psim.GetContentRegionAvail()[0] * 0.65)
            changed, values = psim.SliderFloat3(
                "Translate",
                [object_transform.tx, object_transform.ty, object_transform.tz],
                v_min=-15.0, v_max=15.0,
                format="%.4f",
            )
            if changed:
                object_transform.tx = values[0]
                object_transform.ty = values[1]
                object_transform.tz = values[2]
                transform_changed = True

            changed, values = psim.SliderFloat3(
                "Rotate",
                [object_transform.rx, object_transform.ry, object_transform.rz],
                v_min=-180.0, v_max=180.0,
                format="%.3f",
            )
            if changed:
                object_transform.rx = values[0]
                object_transform.ry = values[1]
                object_transform.rz = values[2]
                transform_changed = True

            changed, values = psim.SliderFloat3(
                "Scale",
                [object_transform.sx, object_transform.sy, object_transform.sz],
                v_min=-5.0, v_max=5.0,
                format="%.4f",
            )
            if changed:
                object_transform.sx = values[0]
                object_transform.sy = values[1]
                object_transform.sz = values[2]
                transform_changed = True
            psim.PopItemWidth()

            # Type exact numbers. The sliders are for finding a position,
            # these are for recording one.
            psim.PushItemWidth(220)
            _, tvals = psim.InputFloat3(
                "Translate (exact)",
                [object_transform.tx, object_transform.ty, object_transform.tz],
                format="%.4f")
            if psim.Button("Apply translate"):
                object_transform.tx = tvals[0]
                object_transform.ty = tvals[1]
                object_transform.tz = tvals[2]
                transform_changed = True
            _, rvals = psim.InputFloat3(
                "Rotate (exact)",
                [object_transform.rx, object_transform.ry, object_transform.rz],
                format="%.3f")
            if psim.Button("Apply rotate"):
                object_transform.rx = rvals[0]
                object_transform.ry = rvals[1]
                object_transform.rz = rvals[2]
                transform_changed = True
            psim.PopItemWidth()

            if transform_changed:
                self.primitives.rebuild_bvh_if_needed(force=True, rebuild=False)
                self.is_force_canvas_dirty = True
            
            psim.TreePop()

    def calculate_sx_and_sy(self, square: float, rows: int = 4, cols: int = 7):
        """Half-extents for a rows x cols board. square is the tag pitch in cm.

        The old version had rows and cols swapped and hardcoded 4 and 7. The
        call site swapped them back, so the two faults cancelled for a 4x7
        board and for nothing else. It also ignored the one-gap border that
        add_padding draws.
        """
        return ((cols + 0.3 * (cols - 1) + 0.6) * square * 0.01 / 2.0,
                (rows + 0.3 * (rows - 1) + 0.6) * square * 0.01 / 2.0)

    def material_for(self, obj):
        """Return (name, material) for the object, or (None, None)."""
        try:
            mat_id = obj.material_id[0].item()
        except Exception:
            return None, None
        for name, mat in self.primitives.registered_materials.items():
            if mat.material_id == mat_id:
                return name, mat
        return None, None

    def board_shape_for(self, obj):
        """ANY SHAPE. Return (rows, cols) if the material is a known board."""
        name, _mat = self.material_for(obj)
        if name is None:
            return None
        try:
            from threedgrut_playground.utils.boards import BOARD_SPECS
        except Exception:
            return None
        for spec_name, rows, cols, _ids in BOARD_SPECS:
            if spec_name == name:
                return rows, cols
        return None

    def texture_aspect_for(self, obj):
        """Width over height of the object's diffuse map, or None."""
        _name, mat = self.material_for(obj)
        dm = getattr(mat, "diffuse_map", None) if mat is not None else None
        if dm is None or dm.ndim < 2 or dm.shape[0] == 0:
            return None
        return float(dm.shape[1]) / float(dm.shape[0])

    def size_any_object(self, obj, value_cm):
        """Half-extents for any object. Returns (sx, sy, rule) or None."""
        shape = self.board_shape_for(obj)
        if shape is not None:
            rows, cols = shape
            sx, sy = self.calculate_sx_and_sy(value_cm, rows, cols)
            return sx, sy, f"tag pitch, {rows}x{cols} board"
        aspect = self.texture_aspect_for(obj)
        if aspect is None:
            return None
        sx = value_cm * 0.01 / 2.0
        return sx, sx / aspect, f"quad width, aspect {aspect:.3f}"



            

    def _draw_diffuse_pbr_settings_widget(self, obj):
        has_single_material = torch.min(obj.material_id) == torch.max(obj.material_id)
        if not has_single_material:
            psim.Text('Multiple Materials')
        else:
            current_mat_id = obj.material_id[0].item()
            mat_id_to_mat_idx = {m.material_id: idx for idx, (m_name, m) in
                                 enumerate(self.primitives.registered_materials.items())}
            mat_name_to_mat_idx = {m_name: idx for idx, (m_name, m) in
                                   enumerate(self.primitives.registered_materials.items())}
            mat_idx_to_mat_id = {v: k for k, v in mat_id_to_mat_idx.items()}
            settings_changed, new_mat_idx = psim.Combo("Material", current_mat_id, list(mat_name_to_mat_idx.keys()))
            if settings_changed:
                obj.material_id[:] = mat_idx_to_mat_id[new_mat_idx]
                self.primitives.recompute_stacked_buffers()
                self.primitives.rebuild_bvh_if_needed(True, True)  # Rebuild so None types are truly ignored in BVH
            self.is_force_canvas_dirty = self.is_force_canvas_dirty or settings_changed

    def _draw_glass_settings_widget(self, obj):
        settings_changed, obj.refractive_index = psim.SliderFloat(
            "Refractive Index", obj.refractive_index, v_min=0.5, v_max=2.0)
        if settings_changed:
            self.primitives.recompute_stacked_buffers()
        self.is_force_canvas_dirty = self.is_force_canvas_dirty or settings_changed

    def _relative_fly_target(self, cam_index):
        """Fly-to target if the rig origin sat at the current viewport.
        Viewport and result are polyscope/OpenGL axes; the rig extrinsic
        is OpenCV axes, hence the F flips. Does not move the rig."""
        F = np.diag([1.0, -1.0, -1.0, 1.0])
        V = np.array(ps.get_view_camera_parameters().get_view_mat(), dtype=float)
        E = np.array(self.novel_view_renderer.v_device.extrinsics[cam_index], dtype=float)
        view = F @ (np.linalg.inv(E) @ (F @ V @ F)) @ F
        rot, t = view[:3, :3], view[:3, 3]
        eye = -rot.T @ t
        # OpenCV +y points down, so the camera y axis is a DOWN vector.
        # Polyscope wants UP. Negate it, or the preview shows the room
        # inverted. This is preview only: the render builds its own
        # matrices in novel_view_renderer and does not call look_at_dir.
        return eye, eye - rot[2, :], -rot[1, :]

    def _snap_window_to_camera(self, cam):
        """One-shot window resize to the camera's native resolution, so the
        fullscreen render preview maps 1:1 right after a Fly to. The window
        stays resizable and nothing re-forces the size afterwards."""
        ps.set_window_size(int(cam.width), int(cam.height))

    def _draw_single_vk_cam(self, i, cam: DistortionCamera):
        view_params = polyscope_from_kaolin_camera(cam)
        eye = view_params.get_position()
        target = view_params.get_position() + view_params.get_look_dir() #TODO: print eye target and up
        # OpenCV +y is DOWN, Polyscope wants UP. Preview only: the render builds its own matrices in novel_view_renderer.
        up = -view_params.get_up_dir()
        dist = cam.distortion_coefficients
        psim.PushItemWidth(200)
        if (psim.Button(f"Fly to")):
            if getattr(self, "flyto_relative", False):
                eye, target, up = self._relative_fly_target(i)
            self._snap_window_to_camera(cam)
            ps.look_at_dir(eye, target, up, fly_to=True)

        xi_alpha = "Not Fisheye"
        if cam.distortion_coefficients[5] != 0.0:
            xi = round(float(cam.distortion_coefficients[9]), 2)
            alpha = round(float(cam.distortion_coefficients[10]), 2)
            xi_alpha = f"xi: {xi}, alpha: {alpha}"
        
        # if cam is not None:
        #     fx, fy, cx, cy = cam.intrinsics.focal_x, cam.intrinsics.focal_y, cam.intrinsics.x0, cam.intrinsics.y0
        #     window_w, window_h = cam.intrinsics.width, cam.intrinsics.height
        #     h_fov = 2 * math.atan2(window_w / 2, fx) * (180 / math.pi)
        #     v_fov = 2 * math.atan2(window_h / 2, fy) * (180 / math.pi)
            
        #     psim.Text(f"Horizontal FOV: {h_fov:.2f} degrees")
        #     psim.Text(f"Vertical FOV: {v_fov:.2f} degrees")
        
        psim.SameLine()
        psim.Text(f"(Selected), {xi_alpha}")
        psim.PopItemWidth()
    
    
    def _fly_to_pose(self, i, pose):
        """Move the rig to trajectory pose i and fly the viewport to the
        selected camera's view of it. Shared by "Move to Pose" and the
        automatic park after "Build Orbit Trajectory"."""
        # All poses in file are from origin camera
        self.selected_pose_idx = i
        self.novel_view_renderer.move_rig_to_pose(pose)
        cam = self.novel_view_renderer.get_camera_at_index(self.selected_camera_idx)
        view_params = polyscope_from_kaolin_camera(cam)
        eye = view_params.get_position()
        target = view_params.get_position() + view_params.get_look_dir()
        # OpenCV +y is DOWN, Polyscope wants UP. Preview only: the render builds its own matrices in novel_view_renderer.
        up = -view_params.get_up_dir()
        self._snap_window_to_camera(cam)
        ps.look_at_dir(eye, target, up, fly_to=True)

    def _draw_cam_trajectory_view(self, poses_list):
        if poses_list is not None:
            for i in range(len(poses_list)):
                pose = poses_list[i]
                if psim.Button(f"Move to Pose {i}"):
                    self._fly_to_pose(i, pose)
                
                
                # if self.selected_pose_idx == i and psim.Button(f"Add to output trajectory"):
                #     camera = polyscope_to_kaolin_camera(
                #         ps.get_view_camera_parameters(), width=self.video_w, height=self.video_h
                #     )
                #     self.video_recorder.trajectory.append(camera)
                

    

    def _draw_single_trajectory_camera(self, i, cam):
        view_params = polyscope_from_kaolin_camera(cam)
        eye = view_params.get_position()
        target = view_params.get_position() + view_params.get_look_dir()
        # OpenCV +y is DOWN, Polyscope wants UP. Preview only: the render builds its own matrices in novel_view_renderer.
        up = -view_params.get_up_dir()
        psim.PushItemWidth(200)
        if (psim.Button(f"View {i + 1}")):
            ps.look_at_dir(eye, target, up, fly_to=True)

        psim.SameLine()
        if (psim.Button(f"Remove {i + 1}")):
            is_not_removed = False
        else:
            is_not_removed = True

        psim.SameLine()
        if psim.Button(f"Replace {i + 1}"):
            camera = polyscope_to_kaolin_camera(
                ps.get_view_camera_parameters(), width=self.video_w, height=self.video_h
            )
            self.video_recorder.trajectory[i] = camera

        psim.PopItemWidth()
        return is_not_removed
    from mcap.writer import Writer
    def render_mcap_trajectory(self, poses):
        """ Renders the trajectory to an mcap file from all cameras 
        using the provided poses.
        Args:
            poses (list): List of poses (loaded trajectory from origin) to render.
        """
        self.mcap_convertor = McapConverter()
        self.video_recorder.export_format = "mcap"
        self.mcap_convertor.output_filename = "long_final_path.mcap"
        self.mcap_convertor.set_filepath()
        interval = self.mcap_convertor.calculate_time_interval()
        cam_names = ["CamA", "CamB", "CamC", "CamD"]
        # Render a subset. Set to None for all four.
        only_cams = None
        with open (self.mcap_convertor.output_fullpath, 'wb') as stream:
            writer = Writer(stream)
            writer.start(profile = "VisualKit")
            channels ={}
            num_frames = self.video_recorder.get_num_frames(len(poses)) # change to get this frm self.vid_recorder
            for cam_name in cam_names:
                if only_cams and cam_name not in only_cams:
                    continue
                time_stamp = 0
                self.video_recorder.reset_for_new_cam_path_export()
                
                # Now populate according to the cam name
                index = cam_names.index(cam_name)
                self.populate_vid_trajectory(poses, idx=index)
                for i in trange(num_frames, desc = f"Writing frames for {cam_name}"):
                    frame = self.video_recorder.obtain_single_camera_frame(frame_index=i, cam_name=cam_name)
                    self.mcap_convertor.write_cam_frame_to_mcap(writer,channels,cam_name,frame,i, time_stamp)
                    time_stamp += interval
            
            writer.finish()
            sys.exit(f"MCAP file written to {self.mcap_convertor.output_fullpath}")
                    
    @torch.cuda.nvtx.range("ps_ui_callback")
    def ps_ui_callback(self):
        """ Polyscope custom UI callback - used to draw gui menu"""
        # Default widget width: half the panel, so labels keep the other
        # half instead of being truncated by ImGui's window-based default.
        psim.PushItemWidth(max(psim.GetContentRegionAvail()[0] * 0.5, 200.0))
        self._draw_preset_settings_widget()
        psim.Separator()
        self._draw_render_widget()
        psim.Separator()
        self._draw_video_recording_controls()
        psim.Separator()
        self._draw_novel_view_from_calib_controls()
        psim.Separator()
        self._draw_slice_plane_controls()
        psim.Separator()
        self._draw_antialiasing_widget()
        psim.Separator()
        self._draw_depth_of_field_widget()
        psim.Separator()
        self._draw_materials_widget()
        psim.Separator()
        self._draw_primitives_widget()
        psim.PopItemWidth()

        # Finally refresh the canvas by rendering the next pass, if needed
        if self.live_update:
            self.update_render_view_viz()

