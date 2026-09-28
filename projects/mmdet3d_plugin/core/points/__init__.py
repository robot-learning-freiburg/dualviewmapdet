from .base_points import BasePoints
from .cam_points import CameraPoints
from .lidar_points import LiDARPoints

__all__ = ["BasePoints", "CameraPoints", "LiDARPoints"]


def get_points_type(points_type):
    """Get the class of points according to coordinate type.

    Args:
        points_type (str): The type of points coordinate.
            The valid value are "CAMERA", "LIDAR", or "DEPTH".

    Returns:
        class: Points type.
    """
    if points_type == "CAMERA":
        points_cls = CameraPoints
    elif points_type == "LIDAR":
        points_cls = LiDARPoints
    else:
        raise ValueError(
            'Only "points_type" of "CAMERA", or "LIDAR",'
            f" are supported, got {points_type}"
        )

    return points_cls