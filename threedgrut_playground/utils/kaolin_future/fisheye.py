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

import math
import torch
from typing import Optional, Tuple
from threedgrut_playground.utils.distortion_camera import DistortionCamera as Camera
from kaolin.render.camera import generate_centered_pixel_coords, CameraFOV
import numpy as np

"""
This module is to be included in next version of kaolin 0.18.0.
As of March 26, 2025 the latest public release is kaolin 0.17.0, hence it's included here independently.
"""

# Private function copied from kaolin, to be deleted
def _to_ndc_coords(pixel_x, pixel_y, camera):
    pixel_x = 2 * (pixel_x / camera.width) - 1.0
    pixel_y = 2 * (pixel_y / camera.height) - 1.0
    return pixel_x, pixel_y


def generate_fisheye_rays(
    camera: Camera,
    coords_grid: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    eps: float = 1e-9
):
    r"""Default ray generation function for perfect wide-angle fisheye cameras.

    Fisheye cameras map wide angles to screen space by introducing distortion towards the edge of the image,
    e.g. straight lines get mapped to curves in the projected image.

    This raygen function uses equidistant mapping, which preserves angular distances
    e.g. if two locations are THETA angles apart in world coordinates, they will be equally spaced in the
    projected image.
    The exact mapping is characterized by the field of view.

    Note: This function does not concern non-ideal distortion parameters, such as
    radial, tangential distortion parameters.

    Args:
        camera (kaolin.render.camera.Camera): A single camera object (batch size 1).
        coords_grid (Tuple[torch.FloatTensor, torch.FloatTensor], optional):
            x and y pixel grid of ray-intersecting coordinates of shape :math:`(\text{H, W})`.
            Coordinates integer parts represent the pixel :math:`(\text{i, j})` coords, and the fraction part of
            :math:`[\text{0,1}]` represents the location within the pixel itself.
            For example, a coordinate of :math:`(\text{0.5, 0.5})` represents the center of the top-left pixel.
        eps (float):
            Numerical sensitivity parameter, used to prevent division by zero.

    Returns:
        (torch.FloatTensor, torch.FloatTensor, torch.BoolTensor):
            First and second entries are the generated rays for the camera, as ray origins and ray direction tensors of
            :math:`(\text{HxW, 3})`.

            The third entry is a boolean mask that masks in which rays fall within the field of view of the camera
            (rays outside the fov region shouldn't be rendered), of shape
            :math:`(\text{HxW, 1})`.
    """
    assert len(camera) == 1, "generate_fisheye_rays() supports only camera input of batch size 1"
    if coords_grid is None:
        coords_grid = generate_centered_pixel_coords(camera.width, camera.height, device=camera.device)
    else:
        assert camera.device == coords_grid[0].device, \
            f"Expected camera and coords_grid[0] to be on the same device, " \
            f"but found {camera.device} and {coords_grid[0].device}."

        assert camera.device == coords_grid[1].device, \
            f"Expected camera and coords_grid[1] to be on the same device, " \
            f"but found {camera.device} and {coords_grid[1].device}."

    # coords_grid should remain immutable (a new tensor is implicitly created here)
    pixel_y, pixel_x = coords_grid
    pixel_x = pixel_x.to(camera.device, camera.dtype)
    pixel_y = pixel_y.to(camera.device, camera.dtype)

    # Account for principal point (offsets from the center)
    pixel_x = pixel_x - camera.x0
    pixel_y = pixel_y + camera.y0

    # pixel values are now in range [-1, 1], both tensors are of shape res_y x res_x
    u, v = _to_ndc_coords(pixel_x, pixel_y, camera)
    r = torch.sqrt(u * u + v * v)
    out_of_fov_mask = (r > 1.0)[:, :, None]

    phi_cos = torch.where(torch.abs(r) > eps, u / r, 0.0)
    phi_cos = torch.clamp(phi_cos, -1.0, 1.0)
    phi = torch.arccos(phi_cos)
    phi = torch.where(v < 0, -phi, phi)
    theta = r * camera.fov(in_degrees=False) * 0.5

    rays_dir = torch.stack(
        [torch.cos(phi) * torch.sin(theta), torch.sin(phi) * torch.sin(theta), torch.cos(theta)], dim=2
    )
    mock_dir = torch.zeros_like(rays_dir)
    mock_dir[:, :, 0] = -1.0
    mock_dir[:, :, 1] = -.05
    rays_dir = torch.where(out_of_fov_mask, mock_dir, rays_dir).unsqueeze(0)

    # Generate ray origins in world coordinates
    cam_center = camera.cam_pos()
    rays_ori = (torch.tensor(cam_center, device=camera.device, dtype=torch.float32)
                .reshape(1, 1, 1, 3)
                .expand(1, camera.height, camera.width, 3))
    #

    rays_ori = rays_ori.reshape(-1, 3)

    rays_dir = rays_dir.reshape(-1, 3)
    out_of_fov_mask = out_of_fov_mask.reshape(-1, 1)

    return rays_ori, rays_dir, out_of_fov_mask


def generate_fisheye_rays_double_sphere(
    camera: Camera,
    distortion_params: list[float],  # [fx, fy, cx, cy, xi, alpha]
    coords_grid: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    eps: float = 1e-9
):
    """Double-sphere fisheye ray generation (camera-to-world, like pinhole rays).

    Args:
        camera: a batch of 1 camera (with attributes width, height, device, dtype, cam_pos(), etc.).
        distortion_params: [fx, fy, cx, cy, xi, alpha].
        coords_grid: optional (y_grid, x_grid) of shape (H, W), giving pixel coordinates (in image pixel units).
        eps: small epsilon to avoid div/zero.

    Returns:
        rays_ori: (H*W, 3), all origins (camera center).
        rays_dir: (H*W, 3), unit directions in world space.
        out_of_fov_mask: (H*W, 1) bool mask, True = outside FOV.
    """
    assert len(camera) == 1, "generate_fisheye_rays_double_sphere supports only camera input of batch size 1"
    if coords_grid is None:
        coords_grid = generate_centered_pixel_coords(camera.width, camera.height, device=camera.device)
    else:
        assert camera.device == coords_grid[0].device, \
            f"Expected camera and coords_grid[0] to be on the same device, " \
            f"but found {camera.device} and {coords_grid[0].device}."
        assert camera.device == coords_grid[1].device, \
            f"Expected camera and coords_grid[1] to be on the same device, " \
            f"but found {camera.device} and {coords_grid[1].device}."

    pixel_y, pixel_x = coords_grid
    pixel_x = pixel_x.to(camera.device, camera.dtype)
    pixel_y = pixel_y.to(camera.device, camera.dtype)

    fx, fy, cx, cy, xi, alpha = distortion_params[5:11]
    #fx, fy, cx, cy = camera.get_camera_intrinsics()
    #fx, fy = 614.8464965820313, 613.5963134765625

    # Compute normalized image coordinates (m_x, m_y)
    m_x = (pixel_x - cx) / fx
    m_y = (pixel_y - cy) / fy

    r2 = (m_x * m_x) + (m_y * m_y)
    r = torch.sqrt(r2)

    # Double sphere model unprojection (see Usenko et al., 2018)
    numerator = 1.0 - ((alpha * alpha) * r2)
    inside = 1 - ((2 * alpha - 1) * r2)
    denominator = alpha * torch.sqrt(torch.clamp(inside, min=0.0)) + 1 - alpha
    m_z = torch.where(denominator.abs() > eps, numerator / denominator, 0.0)
    #m_z = torch.clamp(m_z, min=-1.0, max=1.0)

    # Unprojection scalar
    scalar_numerator = m_z * xi + torch.sqrt((m_z * m_z) + (1 - xi * xi) * r2)
    scalar_denominator = m_z * m_z + r2
    scalar = torch.where(scalar_denominator.abs() > eps, scalar_numerator / scalar_denominator, 0.0)

    # Camera-space direction
    rays_dir = torch.stack(
        [
            m_x * scalar,
            -m_y * scalar,
            -m_z * scalar
        ],
        dim=-1
    )  # (H, W, 3)

    rays_dir = rays_dir.reshape(-1, 3)  # shape (H*W, 3)
    xi_shift = torch.tensor([0.0, 0.0, xi], device=rays_dir.device, dtype=rays_dir.dtype)
    xi_shift = xi_shift.view(1, 3)  # shape (1, 3)
    #z-convention is opposite of the formulation in the paper so xi shift is added
    rays_dir = rays_dir + xi_shift

    ray_dir = rays_dir.reshape(-1, 3)
    ray_orig = torch.zeros_like(ray_dir)

    # Normalize direction
    ray_dir = ray_dir / torch.linalg.norm(ray_dir, dim=-1, keepdim=True)
    # Transform from camera to world coordinates
    ray_orig, ray_dir = camera.extrinsics.inv_transform_rays(ray_orig, ray_dir)

    # Out-of-FOV mask: mask points where denominator or inside sqrt is invalid, or r is too large
    out_of_fov_mask = (inside < 0) | (denominator.abs() <= eps) | (scalar_denominator.abs() <= eps) | (r > 5.0)
    out_of_fov_mask = out_of_fov_mask.reshape(-1, 1)

    return ray_orig, ray_dir, out_of_fov_mask

def generate_pinhole_rays(
        camera: Camera,
        coords_grid: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
):
    """Kaolin-math-matching pinhole ray generation."""
    assert len(camera) == 1, "generate_pinhole_rays supports only camera input of batch size 1"
    if coords_grid is None:
        coords_grid = generate_centered_pixel_coords(camera.width, camera.height, device=camera.device)
    else:
        assert camera.device == coords_grid[0].device, \
            f"Expected camera and coords_grid[0] to be on the same device, " \
            f"but found {camera.device} and {coords_grid[0].device}."
        assert camera.device == coords_grid[1].device, \
            f"Expected camera and coords_grid[1] to be on the same device, " \
            f"but found {camera.device} and {coords_grid[1].device}."
    
    pixel_y, pixel_x = coords_grid
    pixel_x = pixel_x.to(camera.device, camera.dtype)
    pixel_y = pixel_y.to(camera.device, camera.dtype)


    pixel_x = pixel_x - camera.x0
    pixel_y = pixel_y + camera.y0
    

    # follow kaolin 
    pixel_x = 2 * (pixel_x / camera.width) - 1.0
    pixel_y = 2 * (pixel_y / camera.height) - 1.0

    # Scale by tangent of half FOV
    ray_dir = torch.stack(
        (
            pixel_x * camera.tan_half_fov(CameraFOV.HORIZONTAL),
            -pixel_y * camera.tan_half_fov(CameraFOV.VERTICAL),
            -torch.ones_like(pixel_x)
        ),
        dim=-1
    )

    ray_dir = ray_dir.reshape(-1, 3)
    ray_orig = torch.zeros_like(ray_dir)

    # Normalize direction
    ray_dir = ray_dir / torch.linalg.norm(ray_dir, dim=-1, keepdim=True)
    # Transform from camera to world coordinates
    ray_orig, ray_dir = camera.extrinsics.inv_transform_rays(ray_orig, ray_dir)

    return ray_orig, ray_dir

def estimate_theta_star(
    k1, k2, k3, k4,
    ru: torch.Tensor,
    #theta_range = (0.0, 180.0),
    step_size = 0.001,
    device=None,
    dtype=None
):
    """
    Helper function with implementation of Lut to compute 
    theta_star = d_inverse(ru)
    """
    #start, end = theta_range[0], theta_range[1]

    num_steps = int((math.pi - 0.0) / step_size)
    theta_vals = torch.linspace(0.0, math.pi, steps=num_steps, device=device, dtype=dtype)
    d = lambda theta: theta + k1 * theta**3 + k2 * theta**5 + k3 * theta**7 + k4 * theta**9
    R = d(theta_vals)  # (num_steps,)

    # ru: (...), need to find theta_star for each ru
    # Use torch.searchsorted (PyTorch >= 1.6)
    ru_clamped = torch.clamp(ru, R[0], R[-1])
    idx = torch.searchsorted(R, ru_clamped) - 1
    idx = torch.clamp(idx, 0, len(R) - 2)

    r0 = R[idx]
    r1 = R[idx + 1]
    theta0 = theta_vals[idx]
    theta1 = theta_vals[idx + 1]

    theta_star = theta0 + (ru_clamped - r0) * (theta1 - theta0) / (r1 - r0 + 1e-12) #interpolate to get the theta in between by that ratio

    return theta_star

def generate_rays_kb4(
    camera: Camera,
    distortion_params: list[float], 
    coords_grid: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    eps: float = 1e-9
):
    """KB4 raygen

    Args:
        camera: a batch of 1 camera (with attributes width, height, device, dtype, cam_pos(), etc.).
        distortion_params: full list of distortion params ( slice to get first 4 kb4)
        coords_grid: optional (y_grid, x_grid) of shape (H, W), giving pixel coordinates (in image pixel units).
        eps: small epsilon to avoid div/zero.

    Returns:
        rays_ori: (H*W, 3), all origins (camera center).
        rays_dir: (H*W, 3), unit directions in camera space.
        out_of_fov_mask: (H*W, 1) bool mask, True = outside FOV.
    """
    assert len(camera) == 1, "generate_rays_kb4() supports only camera input of batch size 1"
    if coords_grid is None:
        coords_grid = generate_centered_pixel_coords(camera.width, camera.height, device=camera.device)
    else:
        assert camera.device == coords_grid[0].device, \
            f"Expected camera and coords_grid[0] to be on the same device, " \
            f"but found {camera.device} and {coords_grid[0].device}."

        assert camera.device == coords_grid[1].device, \
            f"Expected camera and coords_grid[1] to be on the same device, " \
            f"but found {camera.device} and {coords_grid[1].device}."
    
    # Unpack = params
    k1, k2, k3, k4 = distortion_params[:4]
    #k1, k2, k3, k4 = [0.0]*4

    fx, fy, cx, cy = camera.get_camera_intrinsics()
    #fx, fy, cx, cy = camera.get_camera_intrinsics()
    # import sys
    # fx, fy, cx, cy, k1, k2, k3, k4 = [614.8464965820313, 613.5963134765625, 939.4267578125, 585.2528076171875, -3.0282579245977104e-05,
    #                 0.0002524006995372474,
    #                 0.0007139311637729406,
    #                 -9.387189493281767e-05,]
    #fx, fy, cx, cy = distortion_params[-4:]
    pixel_y, pixel_x = coords_grid
    pixel_x = pixel_x.to(camera.device, camera.dtype)
    pixel_y = pixel_y.to(camera.device, camera.dtype)

    pixel_x = pixel_x - cx
    pixel_y = pixel_y - cy


    m_x = (pixel_x) / fx 
    m_y = (pixel_y) / fy

   

    # Calculate r^2
    r2 = (m_x * m_x) + (m_y * m_y)
    r2 = torch.clamp(r2, min=0.0)
    ru = torch.sqrt(r2)
    

    theta_star = estimate_theta_star(
        k1, k2, k3, k4,
        ru=ru,
        device=ru.device,
        dtype=ru.dtype
    )

    sin_t = torch.sin(theta_star)
    cos_t = torch.cos(theta_star)
    
    
    rays_dir = torch.stack([
        sin_t * (m_x / ru),
        -sin_t * (m_y / ru),
        -cos_t
    ], dim=-1)

    ray_dir = rays_dir.reshape(-1, 3)
    ray_orig = torch.zeros_like(ray_dir)

    # Normalize direction
    ray_dir = ray_dir / torch.linalg.norm(ray_dir, dim=-1, keepdim=True)
    # Transform from camera to world coordinates
    ray_orig, ray_dir = camera.extrinsics.inv_transform_rays(ray_orig, ray_dir)

    return ray_orig, ray_dir
