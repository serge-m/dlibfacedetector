import dlib
import numpy as np

from dlibfaceextractor.aligned_face_extractor import AlignedFaceExtractor
from dlibfaceextractor.test.common_test_objects import get_path_for_test_object, load_image_as_numpy_array


class TestAlignedFaceExtractor:
    def test_face_extractor(self):
        image = load_image_as_numpy_array(get_path_for_test_object("./faces-pair-family-asia-huging-pretty-1822539.jpg"))
        expected_face = load_image_as_numpy_array(get_path_for_test_object("./expected_face1.png"))

        dst_size = 100
        extractor = AlignedFaceExtractor(get_path_for_test_object("./model/shape_predictor_68_face_landmarks.dat"), dst_size)
        face = extractor.extract_aligned_face(dlib.rectangle(384, 73, 446, 135), image)

        assert isinstance(face, np.ndarray)
        assert face.shape == (dst_size, dst_size, 3)
        assert np.allclose(face.astype('int'), expected_face.astype('int'), atol=0)

