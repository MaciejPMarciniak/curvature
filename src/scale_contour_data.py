from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

import contour_io


@dataclass
class ScaleContourData:
    """Class to scale contours according to the size of the original image of the heart.

    Attributes:
        case_index:
            Index of interest.
        project_path:
            The directory with the image and segmentation data.
        scale_width_factor:
            The factor for horizontal coordinate to scale to original image.
        scale_height_factor:
            The factor for vertical coordinate to scale to original image.
    """

    case_index: int
    project_path: Path
    scale_width_factor: float = 1.0
    scale_height_factor: float = 1.0

    def scale_contour(self, contour: NDArray, adjust_with_info_file: bool = True) -> NDArray:
        if adjust_with_info_file:
            self._calculate_scaling_factors()
        scaled_contour = contour * np.repeat(
            [[self.scale_width_factor, self.scale_height_factor]], len(contour), axis=0
        )
        return scaled_contour

    def _calculate_scaling_factors(self) -> None:
        io = contour_io.ContourIO(self.case_index, self.project_path)
        image = io.read_image()
        dimensions = io.read_image_size()
        self.scale_width_factor = dimensions["image_width"] / image.shape[1]
        self.scale_height_factor = dimensions["image_height"] / image.shape[0]
