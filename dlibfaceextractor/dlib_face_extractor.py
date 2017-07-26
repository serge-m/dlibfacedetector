from typing import List

import logging
import numpy as np

from dlibfaceextractor.aligned_face_extractor import AlignedFaceExtractor
from dlibfaceextractor.dlib_face_detector import DlibFaceDetector
from kolasimagecommon import SubImage
from kolasimagecommon import SubimageExtractor


class DlibFaceExtractorException(Exception):
    pass


class DlibFaceExtractor(SubimageExtractor):
    def __init__(self, path_face_detector_model: str, dst_img_size: int):
        logger = logging.getLogger(__name__)
        logger.info("Init DlibFaceExtractor "
                    "with path_face_detector_model {}, dst_img_size {}".format(path_face_detector_model, dst_img_size))
        self._face_detector = DlibFaceDetector()
        self._aligned_face_extractor = AlignedFaceExtractor(path_face_detector_model, dst_img_size)

    def extract(self, img_rgb: np.ndarray) -> List[SubImage]:
        logger = logging.getLogger(__name__)
        logger.info("Detecting faces for image of shape {}".format(img_rgb.shape))

        if img_rgb.ndim != 3 or img_rgb.shape[-1] != 3:
            raise DlibFaceExtractorException("wrong image dimensions. Current shape {}".format(img_rgb.shape))
        rectangles = self._face_detector.detect(img_rgb)
        result = [SubImage(self._aligned_face_extractor.extract_aligned_face(face_rectange, img_rgb))
                  for face_rectange in rectangles]
        logger.info("Found {} faces".format(len(result)))
        return result
