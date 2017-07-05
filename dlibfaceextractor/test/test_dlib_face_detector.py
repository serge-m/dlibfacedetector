import numpy as np
import pytest

from dlibfaceextractor.dlib_face_detector import DlibFaceDetector, DlibFaceDetectorException
import dlib

from dlibfaceextractor.test.common_test_objects import get_path_for_test_object, load_image_as_numpy_array


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

    def test_detector_works(self):
        image = load_image_as_numpy_array(get_path_for_test_object("./faces-pair-family-asia-huging-pretty-1822539.jpg"))

        detector = DlibFaceDetector()
        result = detector.detect(image)

        assert result == [dlib.rectangle(384, 73, 446, 135), dlib.rectangle(361, 146, 436, 221)]


