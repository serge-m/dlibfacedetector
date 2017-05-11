import numpy as np
import pytest

from facedetector.dlib_face_detector import DlibFaceDetector, DlibFaceDetectorException


class TestDlibFaceDetector:
    def test_dlib_face_detector_with_wrong_dimensions(self):
        detector = DlibFaceDetector()
        with pytest.raises(DlibFaceDetectorException):
            detector.detect(np.array([]))

        with pytest.raises(DlibFaceDetectorException):
            detector.detect(np.zeros([10, 10]))

        with pytest.raises(DlibFaceDetectorException):
            detector.detect(np.zeros([10, 10, 20]))

        with pytest.raises(DlibFaceDetectorException):
            detector.detect(np.zeros([10, 10, 3, 2]))

    def test_dlib_face_detector_with_no_face(self):
        detector = DlibFaceDetector()
        result = detector.detect(np.zeros([100, 100, 3]))
        assert len(result) == 0

