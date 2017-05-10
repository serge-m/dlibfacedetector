from typing import List

import numpy as np

from kolasimagesearch.impl.feature_engine.subimage import SubImage
from kolasimagesearch.impl.feature_engine.subimage_extractor import SubimageExtractor


class DlibFaceDetector(SubimageExtractor):
    def extract(self, image: np.ndarray) -> List[SubImage]:
        return []