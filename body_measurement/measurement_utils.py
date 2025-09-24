import math
from typing import Dict, Tuple

def euclidean(p: Tuple[int,int], q: Tuple[int,int]) -> float:
    return math.sqrt((p[0]-q[0])**2 + (p[1]-q[1])**2)

def compute_body_length(landmarks: Dict[str, Tuple[int,int]]) -> float:
    # length from withers to rump as proxy
    return euclidean(landmarks["withers"], landmarks["rump"])

def compute_height_at_withers(landmarks: Dict[str, Tuple[int,int]]) -> float:
    # vertical distance from head (top) to withers
    return abs(landmarks["head"][1] - landmarks["withers"][1])

def compute_chest_width(landmarks: Dict[str, Tuple[int,int]]) -> float:
    return euclidean(landmarks["chest_left"], landmarks["chest_right"])

def compute_rump_angle(landmarks: Dict[str, Tuple[int,int]]) -> float:
    # angle between line (withers->rump) relative to horizontal
    p = landmarks["withers"]
    q = landmarks["rump"]
    dx = q[0] - p[0]
    dy = q[1] - p[1]
    angle = math.degrees(math.atan2(dy, dx))
    return angle

def compute_classification_score(measures: dict) -> float:
    """
    Simple scoring: weighted sum or formula.
    Example: Normalize each measure and produce a score [0, 1]
    """
    # dummy weights
    w_len = 0.3
    w_ht = 0.3
    w_ch = 0.2
    w_ang = 0.2
    # naive normalization (you should calibrate with data)
    score = (w_len * measures["body_length"]
           + w_ht * measures["height"]
           + w_ch * measures["chest_width"]
           + w_ang * abs(measures["rump_angle"]))
    return score
