import numpy as np

def npcal_pose0to1(pose0, pose1):
    """
    Compute relative transform: T_0to1 = inv(pose1) @ pose0
    Args:
        pose0: (4,4) np.ndarray
        pose1: (4,4) np.ndarray
    Returns:
        (4,4) np.ndarray, float32
    """
    pose1_inv = np.eye(4, dtype=np.float64)
    pose1_inv[:3, :3] = pose1[:3, :3].T
    pose1_inv[:3, 3] = (pose1[:3, :3].T * -pose1[:3, 3]).sum(axis=1)
    pose_0to1 = pose1_inv @ pose0.astype(np.float64)
    return pose_0to1.astype(np.float32)


def _build_category_to_index():
    """
    Prefer AV2 official category enum if available.
    Fallback to a minimal mapping required by nuscenes preprocessing.
    """
    try:
        from av2.datasets.sensor.constants import AnnotationCategories
        return {
            **{"NONE": 0},
            **{k.value: i + 1 for i, k in enumerate(AnnotationCategories)},
        }
    except Exception:
        # Minimal fallback for NusNamMap outputs used in dataprocess/misc_data.py
        names = [
            "ANIMAL",
            "PEDESTRIAN",
            "STROLLER",
            "WHEELCHAIR",
            "CONSTRUCTION_CONE",
            "BICYCLE",
            "ARTICULATED_BUS",
            "BUS",
            "REGULAR_VEHICLE",
            "LARGE_VEHICLE",
            "MOTORCYCLE",
            "VEHICULAR_TRAILER",
            "TRUCK",
        ]
        mapping = {"NONE": 0}
        for i, n in enumerate(names, start=1):
            mapping[n] = i
        return mapping


CATEGORY_TO_INDEX = _build_category_to_index()
