import os
from dataclasses import dataclass
from pathlib import Path

import imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from numpy.typing import NDArray


class DirectoryNotFoundError(FileNotFoundError):
    """Thrown if directory does not exist"""


def create_directory(directory: str) -> str:
    if not os.path.isdir(directory):
        os.mkdir(directory)
    return directory


def check_directory_exists(directory: Path) -> bool:
    if not directory.is_dir():
        raise DirectoryNotFoundError("Check the path to the directory")
    return True


def check_file_exists(file_path: Path) -> bool:
    if not file_path.is_file():
        raise FileNotFoundError("Check case index and path to the file")
    return True


@dataclass
class ContourIO:
    """A class for input/output operations"""

    case_index: int
    project_path: Path
    segmentations_folder: str = "Segmentations"
    image_folder: str = "Images"
    image_information_folder: str = "ImageInfo"
    image_information_file: str = "Dimensions.csv"
    output_folder: str = "Output"

    def __post_init__(self) -> None:
        if not isinstance(self.case_index, int):
            raise ValueError(f"Wrong type of the case index: {type(self.case_index)=}")
        if self.case_index < 0:
            raise ValueError("Provide a non-negative integer as an index")
        if check_directory_exists(self.project_path):
            logger.info(f"Data is read from {self.project_path}")

    def read_segmentation(self) -> NDArray:
        """Reads the segmentation image file, if exists.

        Returns:
            Segmentation image.
        """
        segmentations_folder_path = self.project_path / self.segmentations_folder
        if check_directory_exists(segmentations_folder_path):
            logger.info(f"Segmentations read from {segmentations_folder_path}")

        segmentation_file = "Segmentation" + str(self.case_index) + ".png"
        if check_file_exists(segmentations_folder_path / segmentation_file):
            logger.info(f"Reading mask {segmentation_file}")

        gray_mask = imageio.v3.imread(segmentations_folder_path / segmentation_file)
        return gray_mask

    def read_image(self) -> NDArray:
        """Reads the modality image file, if exists.

        Returns:
            Modality image.
        """
        images_folder_path = self.project_path / self.image_folder
        if check_directory_exists(images_folder_path):
            logger.info(f"Images read from {images_folder_path}")

        image_file = str(self.case_index) + ".png"
        if check_file_exists(images_folder_path / image_file):
            logger.info(f"Reading image {image_file}")

        image = imageio.v3.imread(images_folder_path / image_file)
        return image

    def read_image_size(self) -> dict[str, float]:
        """Reads the original size of the modality image.

        Raises:
            ValueError: Thrown if no information on the given index exists.

        Returns:
            Width and height of the image.
        """
        info_folder_path = self.project_path / self.image_information_folder
        if check_directory_exists(info_folder_path):
            logger.info(f"Image information read from {info_folder_path}")

        if check_file_exists(info_folder_path / self.image_information_file):
            logger.info(f"Reading image information from {self.image_information_file}")

        df_image_info = pd.read_csv(info_folder_path / self.image_information_file, index_col="id")
        if self.case_index not in df_image_info.index:
            raise ValueError(f"Case index {self.case_index} not found in the voxel info file.")

        dimensions = {}
        for dimension in ["image_width", "image_height"]:
            dimensions[dimension] = df_image_info.loc[self.case_index][dimension]
        return dimensions

    def save_contours(self, contours: list[NDArray], contour_names: list[str]) -> None:
        """Saves the contours as arrays into a file.

        Args:
            contours: Contours to be saved.
            contour_names: Names of files for contours to be saved to.
        """
        output_path = self.project_path / self.output_folder
        if check_directory_exists(output_path):
            logger.info(f"Contours are saved in {output_path}")

        assert len(contours) == len(contour_names), (
            f"Inconsistency between provided contours {len(contours)=} "
            f"and contour names {len(contour_names)=}"
        )

        for contour, contour_name in zip(contours, contour_names):
            contour_file = output_path / (contour_name + str(self.case_index) + ".csv")
            np.savetxt(contour_file, contour)

    def save_image_with_contours(
        self, image: NDArray, endo_contour: NDArray, epi_contour: NDArray
    ) -> None:
        """Saves the image with provided contours on top.

        Args:
            image: Image
            endo_contour: Endocardial contour
            epi_contour: Epicardial contour
        """
        output_path = self.project_path / self.output_folder
        if check_directory_exists(output_path):
            logger.info(f"Image is saved in {output_path}")

        figure_size = self.read_image_size()
        plt.figure(figsize=(figure_size["image_width"], figure_size["image_height"]))
        plt.imshow(image, cmap="gray")
        plt.plot(endo_contour[:, 0], endo_contour[:, 1], "r-")
        plt.plot(epi_contour[:, 0], epi_contour[:, 1], "b-")

        plt.savefig(output_path / ("ContouredImage" + str(self.case_index) + ".png"))
        plt.clf()

    def save_image_with_wall_thickness(
        self, image: NDArray, endo_contour: NDArray, epi_contour: NDArray, bld: NDArray
    ) -> None:
        """Saves the image with provided contours and wall thickness measurements.

        Args:
            image: Image
            endo_contour: Endocardial contour
            epi_contour: Epicardial contour
            bld: bidirectional local distance matrix
        """
        output_path = self.project_path / self.output_folder
        if check_directory_exists(output_path):
            logger.info(f"Image is saved in {output_path}")

        plt.imshow(image, cmap="gray")
        for key in bld:
            xs = endo_contour[key][0], epi_contour[bld[key]][0]
            ys = endo_contour[key][1], epi_contour[bld[key]][1]
            plt.plot(xs, ys, "y--")

        plt.plot(endo_contour[:, 0], endo_contour[:, 1], "r-")
        plt.plot(epi_contour[:, 0], epi_contour[:, 1], "b-")
        plt.savefig(output_path / ("WallThicknessImage" + str(self.case_index) + ".png"))
        plt.clf()
