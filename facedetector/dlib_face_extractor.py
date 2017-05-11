from typing import List

import numpy as np
import dlib

from kolasimagesearch.impl.feature_engine.subimage import SubImage
from kolasimagesearch.impl.feature_engine.subimage_extractor import SubimageExtractor


class DlibFaceExtractorException(Exception):
    pass


class DlibFaceExtractor(SubimageExtractor):
    def __init__(self, path_face_detector_model):
        pass

    def extract(self, img_rgb: np.ndarray) -> List[SubImage]:
        if img_rgb.ndim != 3 or img_rgb.shape[-1] != 3:
            raise DlibFaceExtractorException("wrong image dimensions. Current shape {}".format(img_rgb.shape))
        return []
