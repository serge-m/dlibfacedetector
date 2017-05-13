import dlib
import numpy as np
from PIL import Image

from facedetector.aligned_face_extractor import AlignedFaceExtractor


class TestAlignedFaceExtractor:
    def test_face_extractor(self):
        image = Image.open("./faces-pair-family-asia-huging-pretty-1822539.jpg")
        image = np.asarray(image)

        expected_face = np.asarray(Image.open("./expected_face1.png"))

        dst_size = 100
        extractor = AlignedFaceExtractor("./model/shape_predictor_68_face_landmarks.dat", dst_size)
        face = extractor.extract_aligned_face(dlib.rectangle(384, 73, 446, 135), image)

        assert isinstance(face, np.ndarray)
        assert face.shape == (dst_size, dst_size, 3)
        assert np.allclose(face.astype('int'), expected_face.astype('int'), atol=0)

