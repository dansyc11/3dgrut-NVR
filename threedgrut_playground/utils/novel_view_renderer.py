from copy import Error
import csv
from threedgrut_playground.utils.distortion_camera import DistortionCamera as Camera
import torch
import os
import json
from typing import List, Tuple, Union
import numpy as np
import functools
from scipy.spatial.transform import Rotation as R

# UTILITIES:
# 1. view_matrix_from_rotation_translation
# - Constructs a 4x4 view matrix from rotation and translation, 
# 2. build_extrinsic_mat_from_rotation_translation
#   Constructs a 4x4 view matrix from rotation and translation, overriding zero-rotation for Cam 0 if needed,
#   and applying a global offset so all cameras shift together
# 3. six_dof_pose_to_4x4_world_t_body_mat
#   Converts a 6-DOF pose (x, y, z, roll, pitch, yaw) into a 4x4 world_T_body transformation matrix.


VILOTA_CAM_MAP = {
    "DP180-": {
        "device_prefix": "DP180-",
        "num_cams": 3,
        "cam_names": ["CamB", "CamC", "CamD"],        
    },
    "DP180IP": {
        "device_prefix": "DP180-",
        "num_cams": 4,
        "cam_names": ["CamA", "CamB", "CamC", "CamD"],
    },
    # Yet to add support for VKL to account for to camera socket extrinsics
    "VKL": {
        "device_prefix": "VKL-",
        "num_cams": 4,
        "cam_names": ["CamA", "CamB", "CamC", "CamD"],
    }
}

def view_matrix_from_rotation_translation(
            rotation: List,
            translation: List,
            offset: 'List[float]' = [0.0, 0.0, 0.0]
        ) -> np.ndarray:
        """
        Constructs a 4×4 view matrix from rotation and translation, overriding zero-rotation for Cam 0 if needed,
        and applying a global offset so all cameras shift together.
        """
        import numpy as np

        # Convert to arrays
    
        R = np.array(rotation, dtype=float).reshape(3,3)   # shape (3,3)
        t = np.array(translation, dtype=float).reshape(3,1)  # shape (3,1)
        
        # Override zero rotation to identity
        if np.allclose(R, 0):
            # This is likely Cam 0 with zero extrinsics; give it identity orientation
            R = np.eye(3, dtype=float)
            # Optionally: ensure t is zero so offset purely determines new center
            t = np.zeros((3,1), dtype=float)

        # Apply global offset: shift camera center by `offset` in world space
        offset_arr = np.array(offset, dtype=float).reshape(3,1)
        # Assuming rotation/translation denote world-to-camera extrinsic:
        #t = t - R @ offset_arr
        
        #t = t - offset_arr # TODO: remove this lol after seeing if it works in polyscope

      
        # Build view matrix
        upper = np.hstack([R, t])               # shape (3,4)
        bottom = np.array([[0.0, 0.0, 0.0, 1.0]], dtype=upper.dtype)
        view_matrix = np.vstack([upper, bottom])  # shape (4,4)
        return view_matrix

def build_extrinsic_mat_from_rotation_translation(
            rotation: List,
            translation: List,
        ) -> np.ndarray:
        """
        Constructs a 4×4 extrinsic matrix from rotation and translation.
        Overrides camD's 0 rotation to identity matrix
        """
        # Convert to arrays
        R = np.array(rotation, dtype=float).reshape(3,3)   # shape (3,3)
        t = np.array(translation, dtype=float).reshape(3,1)  # shape (3,1)
        t = np.divide(t, 100.0) # to convert from cm to m

        # Override zero rotation to identity
        if np.allclose(R, 0):
            # This is likely Cam 0 with zero extrinsics; give it identity orientation
            R = np.eye(3, dtype=float)
            # Optionally: ensure t is zero so offset purely determines new center
            t = np.zeros((3,1), dtype=float)
      
        # Build view matrix
        upper = np.hstack([R, t])               # shape (3,4)
        bottom = np.array([[0.0, 0.0, 0.0, 1.0]], dtype=upper.dtype)
        extrinsic_matrix = np.vstack([upper, bottom])  # shape (4,4)
        return extrinsic_matrix

def six_dof_pose_to_view_mat (six_dof_pose: List[float]) -> np.ndarray:
    """
    Converts a 6-DOF pose (x, y, z, roll, pitch, yaw) into a 4x4 view matrix.
    Euler angles MUST be in degree format

    Args:
        six_dof_pose (List[float]): the 6-DOF pose (x, y, z, roll, pitch, yaw) of the origin camera
    """
    # We can construct the rotation matrix using scipy's from euler
    rotation_matrix = R.from_euler('xyz', six_dof_pose[3:], degrees=True).as_matrix()
    translation_vector = np.array(six_dof_pose[:3], dtype=float).reshape(3, 1)  # shape (3,1)
    # Stack them
    view_matrix = view_matrix_from_rotation_translation(rotation_matrix, translation_vector)
    view_matrix = np.linalg.inv(view_matrix)  # Invert to get world to body transformation
    return view_matrix


class NovelViewRenderer:
    """ Manages the rendering of a simulated view from a Vilota device loaded from any Vilota calibration file"""

    def __init__(self, renderer=None):
        self.renderer = renderer
        # Name of the calibration file to initialise the view from (default: vkl.json)
        self.calibration_filename = "vk180.json"

        # Full path to the calibation file (defaults from ./calibration_files/, no feature to change this yet)
        self.calibration_fullpath = "./calibration_files/" + self.calibration_filename

        # The Vilota Device object that handles the camera rig's
        # - loading: VilotaDevice.load_from_path()
        # - accessing cameras
        # - movement to a defined point: VilotaDevice.move_rig_to()
        self.v_device = None
        self.trajectory_filename = "video_trajectory"  # Default trajectory filename
        self.trajectory_folder = "./video_trajectories/"  # Default trajectory folder
        self.trajectory_parser = None
        self.trajectory_fullpath = "./video_trajectories/video_trajectory.csv"  # Default trajectory full path
        self.world_to_camd =[]
        self.world_to_cami = []
        self.i = 0  # Camera index to check

        self.center_view_matrix = None  # The center view matrix of the rig, used for scene center calculations

        self.trajectory = []
    
    def set_filepath(self):
        """Sets the full path to the calibration file based on the current filename."""
        if not self.calibration_filename:
            raise ValueError("Calibration filename is not set.")
        self.calibration_fullpath = "./calibration_files/" + self.calibration_filename
    
    def set_trajectory_filepath(self):
        """Sets the trajectory filename."""
        if not self.trajectory_filename:
            self.trajectory_fullpath = self.trajectory_folder + "video_trajectory.csv"
        self.trajectory_fullpath = self.trajectory_folder + self.trajectory_filename + ".csv"

    def is_loaded(self) -> bool:
        # You may cache or directly call underlying:
        return getattr(self.v_device, 'isLoaded', lambda: False)()
    
    def set_device_is_loaded(self, is_loaded: bool):
        """Sets the loaded state of the Vilota device."""
        if self.v_device is not None:
            self.v_device.loaded = is_loaded
        else:
            raise Error("Vilota device is not initialized.")
    
    def ensure_loaded(method):
        """Decorator: raise Error if `self.is_loaded()` is False."""
        @functools.wraps(method)
        def wrapper(self, *args, **kwargs):
            if not self.is_loaded() or self.v_device is None:
                raise Error(f"{method.__name__}(): vilota device not loaded.")
            return method(self, *args, **kwargs)
        return wrapper
    
    def load_device(self, reload: bool = False):
        """Loads the Vilota device from the calibration file."""
        if reload or self.v_device is None:
            self.v_device = VilotaDevice.load_from_path(self.calibration_fullpath)
    
    @ensure_loaded
    def get_camera_at_index(self, index: int) -> Camera:
        """Returns the camera at the specified index."""
        return self.v_device.get_camera_at_index(index)
    
    @ensure_loaded
    def get_cam_name_at_index(self, index: int) -> str:
        """Returns the camera name at the specified index."""
        if index < 0 or index >= self.v_device.get_camera_count():
            raise IndexError(f"Camera index {index} out of range.")
        return self.v_device.get_cam_name(index)
    
    @ensure_loaded
    def get_all_cameras(self) -> List[Camera]:
        """Returns all cameras in the device."""
        return self.v_device.get_all_cameras()
    
    @ensure_loaded
    def get_camera_count(self) -> int:
        """Returns the number of cameras in the device."""
        return self.v_device.get_camera_count()
    
    @ensure_loaded
    def move_rig_to_pose(
        self,
        new_pose: List[float],
        cam_index= None,
        is_6dof= True
    ):
        """
        Moves the origin camera to pose specified, and updates all cameras accordingly.
        If cam_index is not specified, pose is assumed to be that of origin camera (Cam D)
        If is_6dof is not specified, pose is assumed to be a 4x4 view matrix

        Args:
            new_pose (List[float]): The new pose in question. It can be:
                                    - 6DOF pose: [x, y, z, roll, pitch, yaw]
                                    - View matrix: List of length 16 (flattened)
            cam_index (Optional[int]): Index of camera where pose is taken from. Defaults to None
                                        (Origin camera index is taken when it is not specified)
            is_6
        """

        # If no cam_index is specified, assume the pose is for the origin camera.
        if not cam_index:
            cam_index = self.get_origin_camera_index()

        #If is_6dof is not specified, pose is assumed to be a 4x4 view matrix
        if is_6dof or len(new_pose) == 6:
            view_matrix = six_dof_pose_to_view_mat(new_pose)
            self.__move_rig_to_pose(new_view_matrix = view_matrix, cam_index = cam_index, is_reshaped = True)

        else: self.__move_rig_to_pose(new_pose, cam_index)

    
    @ensure_loaded
    def __move_rig_to_pose(
        self,
        new_view_matrix : List[float],
        cam_index : int,
        is_reshaped = False
    ):
        """
        Private method to move rig to new view matrix
        """
        # If it's not reshaped (given raw len 16 array), we reshape it
        if not is_reshaped:
            new_view_matrix = np.array(new_view_matrix, dtype=float).reshape(4, 4)
        
        self.v_device.move_rig_to_view(new_view_matrix, cam_index)

    @ensure_loaded
    def check_rig_layout(
        self
    ):
        """
        Checks the rig layout by comparing the extrinsics of camera i to the extrinsics of camera D.
        Args:
            world_to_camd (np.ndarray): The world to cam D transformation matrix.
            world_to_cami (np.ndarray): The world to cam i transformation matrix.
            i (int): The index of the camera to check.
        """
        if len(self.world_to_camd) == 0:
            raise Error("world_to_camd is not set. Please set it before checking the rig layout.")
        
        self.v_device.check_rig_layout(self.world_to_camd, self.world_to_cami, self.i)
    
    @ensure_loaded
    def get_origin_view_matrix(self, cam_index:int, new_pose:np.ndarray) -> np.ndarray:
        return self.v_device.get_view_from_origin_cam(new_pose, cam_index)

    
    def convert_view_matrix_to_6dof_pose(self, new_pose = None) -> List[float]:
        """
        Converts the scene center to a 6-DOF pose.
        Takes this NVR's scene center view matrix and converts it into a 6DOF pose.
        View matrix is 4x4 matrix representing the cam's world to body transformation.
        To get the body to world 6dof pose, we need to invert the view matrix then convert it to a 6DOF pose.
        Returns:
            List[float]: The 6-DOF pose in the format [tx, ty, tz, roll, pitch, yaw].
        """
        if self.center_view_matrix is None and new_pose is None:
            raise Error("center_view_matrix is not set. Please set it before converting to 6DOF pose.")
        if new_pose is not None:
            world_to_body = new_pose
        else: world_to_body = self.center_view_matrix
        world_to_body = np.array(world_to_body, dtype=float).reshape(4, 4)
        body_to_world = np.linalg.inv(world_to_body)  # Invert the view matrix to get body to world
        r1 = body_to_world[0, :3]  # First row (x-axis)
        r2 = body_to_world[1, :3]  # Second row (y-axis)
        r3 = body_to_world[2, :3]  # Third row (z-axis)
        translation = body_to_world[:3, 3]  # Translation vector (tx, ty, tz)
        # Convert rotation matrix to Euler angles (roll, pitch, yaw)
        rotation = R.from_matrix(np.array([r1, r2, r3]))
        roll_pitch_yaw = rotation.as_euler('xyz', degrees=True)  # Convert to Euler angles in degrees
        # Combine translation and rotation into a 6-DOF pose
        six_dof_pose = np.concatenate((translation, roll_pitch_yaw))
        return six_dof_pose.tolist()
    
    @ensure_loaded
    def get_trajectory_poses(self) -> List[List[float]]:
        """Returns a list of 6-DOF poses from the trajectory parser."""
        if not self.trajectory_parser: 
            self.trajectory_parser = TrajectoryPathParser(self.trajectory_fullpath)
        if not self.trajectory_parser.poses:
            self.trajectory_parser.parse()
            if self.trajectory_parser.poses not in self.trajectory:
                self.trajectory = self.trajectory_parser.poses
        self.trajectory = self.trajectory_parser.poses
        return self.trajectory_parser.poses
    
    @ensure_loaded
    def add_pose_to_trajectory(self, pose:np.ndarray, cam_index:int):
        if cam_index != self.get_origin_camera_index():
            pose = self.v_device.get_view_from_origin_cam(pose, cam_index)
        pose = self.convert_view_matrix_to_6dof_pose(pose)
        self.trajectory = self.trajectory_parser.append_pose_to_file(pose)
        return self.trajectory
    
    @ensure_loaded
    def create_new_trajectory(self):
        """
        Creates a new trajectory file with the given name.
        """
        self.set_trajectory_filepath()
        if not self.trajectory_parser:
            self.trajectory_parser = TrajectoryPathParser(self.trajectory_fullpath)
        self.trajectory = self.trajectory_parser.create_new_trajectory_file(self.trajectory_fullpath)
        return self.trajectory
    
    
    @ensure_loaded
    def get_cam_distortions(self) -> List[float]:
        """Returns a list of distortion coefficients for all cameras."""
        return self.v_device.get_cam_distortions()
    
    @ensure_loaded
    def get_origin_camera_pose(self) -> np.ndarray:
        """Returns the origin camera's pose as a 4x4 view matrix."""
        return self.v_device.get_origin_camera_pose()
    
    @ensure_loaded
    def get_origin_camera_index(self) -> int:
        """Returns the index of the origin camera."""
        return self.v_device.get_origin_camera_index()
    
    @ensure_loaded
    def get_camera_intrinsics_at_index(self, index: int) -> List[float]:
        camera = self.get_camera_at_index(index)
        return self.get_camera_intrinsics(camera)
        
    
    def get_camera_intrinsics(self, camera) -> Tuple[float, float, float, float]:
        fx = float(camera.intrinsics.focal_x)
        fy = float(camera.intrinsics.focal_y)
        cx = float(camera.intrinsics.x0)
        cy = float(camera.intrinsics.y0)
        #if cx == 0.0 or cy == 0.0:
            #raise ValueError(f"Camera {index} has invalid intrinsic parameters: cx={cx}, cy={cy}.")
        return [fx, fy, cx, cy]
    
    def get_string_representation(self) -> str:
        """Returns a string representation of the Vilota device."""
        if not self.is_loaded() or self.v_device is None:
            return "Vilota device not loaded."
        return self.v_device.get_string_rep()
    
    def get_device_name_and_serial_no(self) -> str:
        """Returns the name and serial number of the Vilota device."""
        if not self.is_loaded() or self.v_device is None:
            return "Vilota device not loaded."
        return f"Device Name: {self.v_device.name},\nProduct name: {self.v_device.product_name}"
    

class VilotaDevice:
    """ Represents a Vilota device with multiple cameras, loaded from a calibration file. """

    def __init__(self, calibration_file: str):
        self.calibration_file = calibration_file
        self.name = os.path.basename(calibration_file).split('.')[0]
        self.product_name = "Unknown"  # Placeholder for serial number, if available
        self.cameras = {}
        self.extrinsics = {}
        self.loaded = False
        self.loader = None
        self.offset_from_origin =[1.0, 2.0, 3.0]  # Default offset from origin camera

    @classmethod
    def load_from_path(cls, path: str) -> 'VilotaDevice':
        """ Loads the Vilota device from a calibration file. """
        device = cls(path)
        device.loader = Loader(path)
        try:
            extrinsics, cameras = device.loader.load_all_cameras()
            device.name, device.product_name = device.loader.load_device_info()
            device.loaded = True
            device.cameras = cameras
            device.extrinsics = extrinsics
            device.origin_camera_pose = device.get_origin_camera_pose()
            
        except Exception as e:
            import logging
            logging.error(f"Failed to load cameras: {e}")
            print(f"Failed to load cameras: {e}")
            device.loaded = False
        return device

    def isLoaded(self) -> bool:
        return self.loaded

    def get_camera_at_index(self, index: int) -> Camera:
        return self.cameras[index]

    def get_all_cameras(self) -> List[Camera]:
        return self.cameras

    def get_camera_count(self) -> int:
        return len(self.cameras)
    
    def get_cam_name(self, index: int) -> str:
        if self.name.startswith("DP180-"):
            return VILOTA_CAM_MAP["DP180-"]["cam_names"][index]
        elif self.name.startswith("DP180IP"):
            
            return VILOTA_CAM_MAP["DP180IP"]["cam_names"][index]
        elif self.product_name_name.startswith("VKL-"):
            return VILOTA_CAM_MAP["VKL"]["cam_names"][index]
        else:
            raise ValueError(f"Unknown camera index {index} for device {self.name}.")
    
    def get_cam_distortions(self) -> List[float]:
        """ Returns a list of distortion coefficients for all cameras. """
        distortions = []
        for camera in self.cameras.values():
            if hasattr(camera, 'distortion_coefficients'):
                dist = camera.distortion_coefficients
                distortions.append(dist)
                
            else:
                distortions.append(None)

        return distortions
    
    def check_rig_layout(self, world_to_camd, world_to_cami, i):
    # world_to_camd : Get the world to cam of cam D
    # world_to_cama : get the world to cam of cam A
        cam_i_to_camd = np.array(world_to_camd) @ np.linalg.inv(np.array(world_to_cami))
        # create a new file to store the cam_i_to_camd and compare it with cam i's extrinsics
        #first create and open a new file called check_extrinsics.txt
        with open("check_extrinsics.txt", "a") as f:
            f.write(f"Camera {i} to Cam D extrinsics:\n")
            f.write(f"{cam_i_to_camd}\n")
            f.write(f"Camera {i} extrinsics:\n")
            f.write(f"{self.extrinsics[i]}\n")
            f.write("\n")
        #     # cam_a_to_camd : world_to_camd @ inv(world_to_cama) --> cam a extrinsic frm file ( verify this)

    def move_rig_to_view(self, new_view_matrix: np.ndarray, view_cam_index: int):
        """
        Moves the origin camera to the new view matrix/ view from origin camera given the new matrix.
        Updates all other cameras by their respective translations and rotations from the updated
        orientation of the origin camera

        Args:
            new_view_matrix (np.ndarray): The new view matrix to set
            view_cam_index (int): Indicates which camera the view matrix is from
        """
        # Check if new_view_matrix is from origin, if not get the origin view matrix relative to this
        
        og_index = self.get_origin_camera_index() 
        is_view_from_origin = view_cam_index == og_index
        if not is_view_from_origin:
            new_view_matrix = self.get_view_from_origin_cam(new_view_matrix, view_cam_index)
        
        # Update each camera to new calculated view
        for index, camera in self.cameras.items():
            F = np.diag([1, -1, -1, 1])
            world_to_cam0 = F@ new_view_matrix @F
            cam0_to_world = np.linalg.inv(world_to_cam0)
            cam_i_to_cam0 = self.extrinsics[index]
            cami_to_world = cam0_to_world @ np.array(cam_i_to_cam0)
            
            world_to_cami = np.linalg.inv(cami_to_world)  # Invert to get the view matrix
            new_view_mat = F @ world_to_cami @ F
            camera.update(torch.tensor(new_view_mat, dtype=torch.float64)) 
        
        # Update the origin camera pose
        # self.cameras[og_index].update(torch.tensor(new_view_matrix, dtype=torch.float64))  # Update the origin camera's view matrix
        # self.origin_camera_pose = new_view_matrix
    

    def get_view_from_origin_cam(self, world_to_cami: np.ndarray, cam_index : int) -> np.ndarray:
        """
        Helper function to calculate view matrix of origin camera given the
        view matrix of another camera

        Args:
            world_to_cami (np.ndarray): View matrix to selected camera
            cam_index (int): Index of selected camera in the rig

        Returns:
            world_to_camd (np.ndarray): View matrix to origin camera
        """
        cami_to_cam0 = self.extrinsics[cam_index]
        world_to_camd = cami_to_cam0 @ world_to_cami

        return world_to_camd

    def get_origin_camera_pose(self):
        return self.loader.get_origin_camera_info()[1]
    
    def get_origin_camera_index(self):
        return self.loader.get_origin_camera_info()[0]
    
    def get_string_rep(self):
        strng = "Vilota Device:\n"
        if self.cameras != {}:
            for index, camera in self.cameras.items():
                strng += f"Camera {index}: {camera.named_params()}\n"
                strng += f"Distortion Coefficients: {camera.distortion_coefficients}\n"
                if self.get_origin_camera_index() == index:
                    strng += " (Origin Camera)\n"
        else:
            strng += "No cameras loaded.\n"
        
        return strng
    

class Loader:
    """ Loader for Vilota calibration files. """
    def __init__(self, calibration_file: str):
        self.calibration_file = calibration_file
    
    def load_camera_info(self) -> Tuple[int, dict]:
        """
        Loads a tuple of number of cameras
        and the camera data list from the json file."""
        if not os.path.exists(self.calibration_file):
            raise FileNotFoundError(f"Calibration file {self.calibration_file} does not exist.")
        
        with open(self.calibration_file, 'r') as file:
            data = json.load(file)
        
        camera_data = data.get('cameraData', [])
        num_cameras = len(camera_data)
        
        if num_cameras == 0:
            raise ValueError("No camera data found in the calibration file.")
        
        return num_cameras, camera_data
    
    def load_device_info(self) -> Tuple[str, str]:
        """
        Loads the device name and serial number from the calibration file.
        Returns:
            Tuple[str, str]: Device name and serial number.
        """
        if not os.path.exists(self.calibration_file):
            raise FileNotFoundError(f"Calibration file {self.calibration_file} does not exist.")
        
        with open(self.calibration_file, 'r') as file:
            data = json.load(file)
        
        device_name = data.get('deviceName', 'Unknown')
        product_name = data.get('productName', 'Unknown')
        
        return device_name, product_name

    def parse_cali_json_to_camera_params(
            self,
            data: List,
            index: int = 0          
        ) -> Tuple[int, dict]:
        """
        Parses a .json calibration file and returns the camera parameters.

        Args:
            index (int): Index of the camera to parse from the calibration file. Default is 0.
            data (List): The loaded JSON data from the calibration file.

        Returns:
            dict: A dictionary containing the camera parameters.
        """
        cam_data = data
        
        cam_rig_index = cam_data[index][0]
        camera_data = cam_data[index][1]

        translation = camera_data.get('extrinsics', {}).get('translation', [])
        trans_vector = list(translation.values())
        cam_type = camera_data.get('cameraType', 9)
        if cam_type != 0:
            has_distortion = False
        else:
            has_distortion = True
        
        camera_params = {
            'cameraType': camera_data.get('cameraType', 'unknown'),
            'distortionCoeff': camera_data.get('distortionCoeff', []),
            'ext_rotation' : camera_data.get('extrinsics', {}).get('rotationMatrix', []),
            'ext_translation' : trans_vector,
            'intrinsics': camera_data.get('intrinsicMatrix', {}),
            'width': camera_data.get('width', 1920),
            'height': camera_data.get('height', 1200),
            'hasDistortion': has_distortion
        }

        return cam_rig_index, camera_params

    def cam_params_to_distortion_camera(
            self,
            cam_rig_index: int,
            camera_params: dict, 
            device: Union[torch.device, str] = 'cpu'
    ) -> Tuple[np.ndarray, Camera]:
        """
        Converts loaded camera parameters to a Distortion Camera object.

        Args:
            camera_params (dict): Dictionary containing camera parameters.
            device (optional, torch.device or str): the device on which camera parameters will be allocated. Default: cpu
        Returns:
            Camera: A Kaolin Camera object initialized with the provided parameters.
        """
        # Extract intrinsic params
        intrinsic_matrix = camera_params['intrinsics']
        f_x = intrinsic_matrix[0][0]
        f_y = intrinsic_matrix[1][1]
        c_x = intrinsic_matrix[0][2]
        c_y = intrinsic_matrix[1][2]

        # Extract extrinsic params
        rotation_matrix = camera_params['ext_rotation']
        translation_vector = camera_params['ext_translation']
        print(f"Rot {cam_rig_index}:{rotation_matrix}")
        print(f"Trans {cam_rig_index}:{translation_vector}")
        

        if rotation_matrix is None or len(rotation_matrix) == 0:
            rotation_matrix = [[0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0]]
            self.origin_camera_pose = build_extrinsic_mat_from_rotation_translation(rotation_matrix, translation_vector)
            self.origin_camera = cam_rig_index

        elif rotation_matrix[0][0] == 0.0 and rotation_matrix[1][1] == 0.0 and rotation_matrix[2][2] == 0.0:
            self.origin_camera_pose = build_extrinsic_mat_from_rotation_translation(rotation_matrix, translation_vector)
            self.origin_camera = cam_rig_index
         
        extrinsic_matrix = build_extrinsic_mat_from_rotation_translation(rotation_matrix, translation_vector)
        print(f"Extrinsic Matrix {cam_rig_index}:\n{extrinsic_matrix}")

        # Convert to Distortion Camera object
        distortion_camera = Camera.from_args(
            view_matrix = torch.tensor(extrinsic_matrix, dtype=torch.float64, device=device),
            focal_x = f_x,
            focal_y = f_y,
            x0 = c_x,
            y0 = c_y,
            width = camera_params['width'],
            height = camera_params['height'],
            distortion_coefficients = camera_params['distortionCoeff'],
            intrinsic_params=[f_x, f_y, c_x, c_y],
            dtype=torch.float64,
            device = device
        )
        distortion_camera.set_intrinsic_params([f_x, f_y, c_x, c_y])

        return extrinsic_matrix, distortion_camera
    
    def load_all_cameras(self) -> Tuple[dict, dict]:
        """
        Loads all cameras from the calibration file and returns a list of Distortion Camera objects.

        Returns:
        Tuple:
            extrinsics (dict): A dictionary where keys are camera rig indices and values are 4x4 extrinsic matrices.
            cameras (dict): A dictionary where keys are camera rig indices and
            
        """
        num_cameras, camera_data = self.load_camera_info()
        cameras = {}
        extrinsics = {}
        for i in range(num_cameras):
            cam_rig_index, cam_params = self.parse_cali_json_to_camera_params(camera_data, i)
            extrinsic_mat, distortion_camera = self.cam_params_to_distortion_camera(cam_rig_index, cam_params)

            cameras[cam_rig_index] = distortion_camera
            extrinsics[cam_rig_index] = extrinsic_mat # 
            
        # Sort cameras and extrinsics by camera rig index
        extrinsics = dict(sorted(extrinsics.items(), key=lambda x: x[0]))
        cameras = dict(sorted(cameras.items(), key=lambda x: x[0]))

        return extrinsics, cameras

    def get_origin_camera_info(self) -> Tuple[int, np.ndarray]:
        """
        Returns the origin camera's index and its pose as a 4x4 view matrix.
        
        Returns:
            Tuple[int, np.ndarray]: The index of the origin camera and its view matrix.
        """
        return self.origin_camera, self.origin_camera_pose
    

class TrajectoryPathParser:
    """ Parses a trajectory path file and returns a list of 6-DOF poses. """
    
    def __init__(self, path: str):
        self.trajectory_file = path if path else "./video_trajectories/test_1.csv"
        self.poses = []
    
    def parse(self) -> List[List[float]]:
        data =[]
        if not os.path.exists(self.trajectory_file):
            raise FileNotFoundError(f"Trajectory file {self.trajectory_file} does not exist.")
        with open(self.trajectory_file, newline='') as f:
            reader = csv.reader(f)
            data = list(reader)
        #save the poses as floats
        data = data[1:]  # Skip the header row
        data = [[float(x) for x in row] for row in data if len(row) == 6]  # Ensure each row has exactly 6 elements
        if data is None:
            raise Error("Trajectory CSV has no entries.")
        self.poses = data
        return data
    
    def create_new_trajectory_file(self, filepath: str = None):
        """
        Creates a new trajectory file with the specified filepath.
        If no filepath is provided, uses the default trajectory file path.
        """
        if filepath:
            self.trajectory_file = filepath
        if not self.trajectory_file.endswith('.csv'):
            self.trajectory_file += '.csv'
        
        # Create the file and write the header
        with open(self.trajectory_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['x', 'y', 'z', 'roll', 'pitch', 'yaw'])

        poses = self.parse()
        return poses
    
    def append_pose_to_file(self, pose):
        """
        Appends a new pose to the trajectory file.
        """  
        with open(self.trajectory_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(pose)  # Write the pose as a new row
        poses = self.parse()
        return poses

############# TEST FUNCTIONS ######################

def test_novel_view_renderer_load(path:str = "vkl.json"):
    """
    Test function to demonstrate the usage of NovelViewRenderer.
    """
    renderer = None  # Replace with actual renderer instance
    nvr = NovelViewRenderer(renderer)
    #nvr.calibration_folder = "~/3dgrut/calibration/"
    # Set calibration path
    nvr.calibration_filename = path
    nvr.set_filepath()
    nvr.calibration_fullpath = "../../calibration_files/" + nvr.calibration_filename
    print("Loading calibration file:", nvr.calibration_fullpath)
    # Load cameras from calibration file
    
    nvr.load_device()
    print(nvr.get_string_representation())
    

    #print(f"Params: {nvr.camera_distortion_params}")

devices = ["vkl.json", "dp180_1.json", "dp180_2.json", "vk180.json"]

# for device in devices:
#     print(f"Testing device: {device}")
#     test_novel_view_renderer_load(device)

def test_novel_view_renderer_move_rig(path:str = "vkl.json"):
    """
    Test function to demonstrate the usage of NovelViewRenderer.
    """
    renderer = None  # Replace with actual renderer instance
    nvr = NovelViewRenderer(renderer)
    #nvr.calibration_folder = "~/3dgrut/calibration/"
    # Set calibration path
    nvr.calibration_filename = path
    nvr.set_filepath()
    print("Loading calibration file:", nvr.calibration_fullpath)
    # Load cameras from calibration file
    
    nvr.load_device()
    print(nvr.get_string_representation())
    
    nvr.move_rig_to(
        new_pose = [0.0, 0.0, 0.0, 90.0, 180.0, 0.0]  # Example pose: [tx, ty, tz, roll, pitch, yaw]
    )
    print(nvr.get_string_representation())

def main():
    test_novel_view_renderer_load("vk180.json")

if __name__ == "__main__":
    main()
    #test_novel_view_renderer_move_rig()
    #test_novel_view_renderer_load("dp180_1.json")
    #test_novel_view_renderer_load("dp180_2.json")
    #test_novel_view_renderer_load("vk180.json")

