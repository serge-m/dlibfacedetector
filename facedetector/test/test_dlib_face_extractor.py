import dlib
import numpy as np
import pytest
from PIL import Image

from facedetector.dlib_face_detector import DlibFaceDetector
from facedetector.dlib_face_extractor import DlibFaceExtractor, DlibFaceExtractorException


class TestDlibFaceExtractor:
    def test_dlib_face_detector_with_wrong_dimensions(self):
        extractor = DlibFaceExtractor("./model/shape_predictor_68_face_landmarks.dat", 100)
        with pytest.raises(DlibFaceExtractorException):
            extractor.extract(np.array([]))

        with pytest.raises(DlibFaceExtractorException):
            extractor.extract(np.zeros([10, 10]))

        with pytest.raises(DlibFaceExtractorException):
            extractor.extract(np.zeros([10, 10, 20]))

        with pytest.raises(DlibFaceExtractorException):
            extractor.extract(np.zeros([10, 10, 3, 2]))

    def test_dlib_face_detector_with_no_face(self):
        extractor = DlibFaceExtractor("./model/shape_predictor_68_face_landmarks.dat", 100)
        result = extractor.extract(np.zeros([100, 100, 3]))
        assert len(result) == 0

    def test_detector_works(self):
        image = Image.open("./faces-pair-family-asia-huging-pretty-1822539.jpg")
        image = np.asarray(image)

        extractor = DlibFaceExtractor("./model/shape_predictor_68_face_landmarks.dat", 100)
        result = extractor.extract(image)

        assert result == [dlib.rectangle(384, 73, 446, 135), dlib.rectangle(361, 146, 436, 221)]
