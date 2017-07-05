from typing import List

import numpy as np
import dlib


class DlibFaceDetectorException(Exception):
    pass


_right_number_of_dimensions = 3


class DlibFaceDetector:
    def __init__(self):
        self._detector = dlib.get_frontal_face_detector()

    def detect(self, img_rgb: np.ndarray) -> List:
        if img_rgb.ndim != _right_number_of_dimensions or img_rgb.shape[-1] != 3:
            raise DlibFaceDetectorException("wrong image dimensions. Current shape {}".format(img_rgb.shape))

        img_copy_to_make_detector_work = img_rgb.copy()
        try:
            detected = self._detector(img_copy_to_make_detector_work, 1)
        except Exception as e:
            raise DlibFaceDetectorException from e

        return list(detected)
