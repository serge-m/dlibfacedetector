from PIL import Image
import numpy as np
import os

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))


def get_path_for_test_object(relative_path: str) -> str:
    return os.path.join(CURRENT_DIR, relative_path)


def load_image_as_numpy_array(path: str) -> np.ndarray:
    image = Image.open(path)
    image = np.asarray(image)
    return image
