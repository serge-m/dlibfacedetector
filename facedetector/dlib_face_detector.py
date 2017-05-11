from typing import List

import numpy as np
import dlib


class DlibFaceDetectorException(Exception):
    pass


class DlibFaceDetector:
    def __init__(self):
        self._detector = dlib.get_frontal_face_detector()

    def detect(self, img_rgb: np.ndarray) -> List:
        if img_rgb.ndim != 3 or img_rgb.shape[-1] != 3:
            raise DlibFaceDetectorException("wrong image dimensions. Current shape {}".format(img_rgb.shape))
        detected = self._detector(img_rgb.copy(), 1)
        return list(detected)
