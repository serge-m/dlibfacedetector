from unittest import mock
from unittest.mock import call

import numpy as np
import pytest

from dlibfaceextractor.dlib_face_extractor import DlibFaceExtractor, DlibFaceExtractorException
from dlibfaceextractor.test.common_test_objects import get_path_for_test_object, load_image_as_numpy_array
from kolasimagecommon import SubImage


@mock.patch('dlibfaceextractor.dlib_face_extractor.DlibFaceDetector')
@mock.patch('dlibfaceextractor.dlib_face_extractor.AlignedFaceExtractor')
class TestDlibFaceExtractor:
    def test_dlib_face_detector_with_wrong_dimensions(self, class_AlignedFaceExtractor, class_DlibFaceDetector):
        extractor = DlibFaceExtractor("path_model", 100)
        with pytest.raises(DlibFaceExtractorException):
            extractor.extract(np.array([]))

        with pytest.raises(DlibFaceExtractorException):
            extractor.extract(np.zeros([10, 10]))

        with pytest.raises(DlibFaceExtractorException):
            extractor.extract(np.zeros([10, 10, 20]))

        with pytest.raises(DlibFaceExtractorException):
            extractor.extract(np.zeros([10, 10, 3, 2]))

    def test_dlib_face_detector_with_no_face(self, class_AlignedFaceExtractor, class_DlibFaceDetector):
        extractor = DlibFaceExtractor("path_model", 100)
        result = extractor.extract(np.zeros([100, 100, 3]))
        assert len(result) == 0

    def test_detector_works(self, class_AlignedFaceExtractor, class_DlibFaceDetector):
        face1 = np.array([1, 2])
        face2 = np.array([3, 4])
        rect1 = [1,2,3,4]
        rect2 = [3, 4, 5, 6]
        class_AlignedFaceExtractor.return_value.extract_aligned_face.side_effect = [face1, face2]
        class_DlibFaceDetector.return_value.detect.return_value = [rect1, rect2]
        dst_img_size = 100
        image = load_image_as_numpy_array(get_path_for_test_object("faces-pair-family-asia-huging-pretty-1822539.jpg"))

        extractor = DlibFaceExtractor("path_model", dst_img_size)
        result = extractor.extract(image)

        assert result == [SubImage(face1), SubImage(face2)]
        class_AlignedFaceExtractor.assert_called_once_with("path_model", dst_img_size)
        class_AlignedFaceExtractor.return_value.extract_aligned_face.assert_has_calls([call(rect1, image), call(rect2, image)])
        class_DlibFaceDetector.assert_called_once_with()
        class_DlibFaceDetector.return_value.detect.assert_called_once_with(image)
