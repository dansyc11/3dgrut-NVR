from kaolin.render.camera import Camera
import torch
from typing import List, Tuple
# Extends the kaolin camera class to include distortion coefficients
class DistortionCamera(Camera):
    def __init__(self, *args, distortion_coefficients=None, intrinsic_params =None, **kwargs):
        """
        Initialises the DistortionCamera, which extends the kaolin Camera class to include distortion coefficients.
        Args:
            *args: Positional arguments for the Camera class.
            distortion_coefficients (list or None): Distortion coefficients to be used for the camera.
            **kwargs: Keyword arguments for the Camera class.
        """
        super().__init__(*args, **kwargs)
        self.distortion_coefficients = distortion_coefficients if distortion_coefficients is not None else None
        self.intrinsic_params = intrinsic_params

    def set_distortion_coefficients_ds(self, distortion_coefficients):
        """
        Sets the distortion coefficients for the camera.
        Args:
            distortion_coefficients (list): List of distortion coefficients.
        """
        if len(distortion_coefficients) != 6:
            raise ValueError("Distortion coefficients must be a list of 6 values.")
        self.distortion_coefficients = distortion_coefficients

    def get_distortion_coefficients(self):
        """
        Returns the distortion coefficients of the camera.
        Returns:
            list or None: List of distortion coefficients, or None if no coefficients are set.
        """
        return self.distortion_coefficients
    
    @classmethod
    def from_args(cls, *args, distortion_coefficients=None, intrinsic_params = None ,**kwargs):
        # First get the “raw” Camera so you pick up all the
        # built‑in logic for extrinsics/intrinsics.
        base_cam = super(DistortionCamera, cls).from_args(*args, **kwargs)
        # Now build *your* subclass, seeding it with everything
        return cls(
            extrinsics=base_cam.extrinsics,
            intrinsics=base_cam.intrinsics,
            distortion_coefficients=distortion_coefficients,
            intrinsic_params = intrinsic_params
        )
    
    def cuda(self, device=None):
        # Move base camera to cuda
        base = super().cuda()
        # Re-create as DistortionCamera, copying over extra attributes
        return DistortionCamera(
            extrinsics=base.extrinsics,
            intrinsics=base.intrinsics,
            distortion_coefficients=torch.tensor(self.distortion_coefficients, device = device),
            intrinsic_params=torch.tensor(self.intrinsic_params, device=device)
            # ...copy any other custom attributes...
        )
    
    def set_intrinsic_params(self, intrinsic_params: List[float]):
        """
        Sets the intrinsic parameters of the camera.
        Args:
            intrinsic_params (list): List of intrinsic parameters [fx, fy, cx, cy].
        """
        if len(intrinsic_params) != 4:
            raise ValueError("Intrinsic parameters must be a list of 4 values: [fx, fy, cx, cy].")
        self.intrinsic_params = intrinsic_params

    
    def get_camera_intrinsics(self) -> Tuple[float, float, float, float]:

        fx = float(self.intrinsics.focal_x)
        fy = float(self.intrinsics.focal_y)
        cx = float(self.intrinsics.x0)
        cy = float(self.intrinsics.y0)
        #if cx == 0.0 or cy == 0.0:
            #raise ValueError(f"Camera {index} has invalid intrinsic parameters: cx={cx}, cy={cy}.")
        self.intrinsic_params = [fx, fy, cx, cy]
        return [fx, fy, cx, cy]
    
    def set_cam_intr(self, fx: float, fy: float, cx: float, cy: float):
        """
        Sets the camera intrinsic parameters.
        Args:
            fx (float): Focal length in x direction.
            fy (float): Focal length in y direction.
            cx (float): Principal point x coordinate.
            cy (float): Principal point y coordinate.
        """
        self.intrinsics.focal_x = fx
        self.intrinsics.focal_y = fy
        self.intrinsics.x0 = cx
        self.intrinsics.y0 = cy
    
    def get_intrinsic_params(self) -> List[float]:
        """
        Returns the intrinsic parameters of the camera.
        Returns:
            list: List of intrinsic parameters [fx, fy, cx, cy].
        """
        if self.intrinsic_params is None:
            raise ValueError("Intrinsic parameters are not set.")
        return self.intrinsic_params 
    
    def get_view_matrix(self) -> torch.Tensor:
        """
        Returns the view matrix of the camera.
        Returns:
            torch.Tensor: The view matrix of the camera.
        """
        return self.extrinsics.view_matrix()
    

       
def wtf():
    """
    A placeholder function that does nothing.
    """
    pass