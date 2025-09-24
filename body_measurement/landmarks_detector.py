import cv2
import numpy as np

def detect_landmarks(image: np.ndarray) -> dict:
    """
    Dummy / example: detect keypoints on cattle/buffalo body.
    Return dict of named points in pixel coordinates:
      e.g. {
        "head": (x, y),
        "withers": (x, y),
        "rump": (x, y),
        "chest_left": (x, y),
        "chest_right": (x, y)
      }
    """
    h, w = image.shape[:2]
    # Dummy example: pick approximate points relative to image size
    landmarks = {
        "withers": (int(w * 0.3), int(h * 0.4)),
        "rump":    (int(w * 0.7), int(h * 0.4)),
        "chest_left":  (int(w * 0.4), int(h * 0.6)),
        "chest_right": (int(w * 0.6), int(h * 0.6)),
        "head": (int(w * 0.5), int(h * 0.2))
    }
    return landmarks
