import numpy as np
from loguru import logger
from numpy.typing import NDArray
from scipy.spatial.distance import cdist


class WallThickness:
    """Class to calculate myocardial wall thickness from the endocardial and
    epicardial contours, as seen in a 2D slice of the heart.

    The calculation relies on the bidirectional local distance proposed by
    Kim et al. (https://pubmed.ncbi.nlm.nih.gov/23127072/). The thickness is
    measured from the endocardial points towards the epicardial points of
    the contours. For the best results it is recommended that the epicardial
    contour has at least twice as many points as the endocardial contour.

    According to the general practice, the myocardium viewed in the long axis
    can be divided into six segments: basal, mid and apical on each side. The
    SEGMENT_x limits allow for extraction of the regional myocardial wall
    thicknesses from the contours.

    Attributes:
        endo_contour:
            The delineation of the endocardium.
        epi_contour:
            The delineation of the epicardium.
        segments:
            The boundaries of the segments of the contour.
        thickness:
            The thickness calculated for each endocardial point.
        bld:
            Bidirectional local distance matrix, useful for plotting
    """

    def __init__(self, endo_contour: NDArray, epi_contour: NDArray) -> None:
        self.endo_contour = endo_contour
        self.epi_contour = epi_contour
        self.segments = self._assign_segments()
        self.thickness: NDArray = np.zeros_like(self.endo_contour)
        self.bld: dict[int, int]

    def _assign_segments(self) -> dict[int, tuple[int, int]]:
        """Assigns the boundaries of the segments."""
        endo_len = len(self.endo_contour)
        segment_len = np.round(endo_len / 6)
        segments = {}
        # Set boundaries for the middle segments (2-5)
        for i in range(1, 5):
            segments[i + 1] = (segment_len * i, segment_len * (i + 1))

        # In segment 1 ignore the first 5 thickness values - they are artifacts
        segments[1] = (5, segment_len)

        # Same in segment 6 but ignore the last 5 values
        segments[6] = (segment_len * 5, endo_len - 5)
        for k, v in segments.items():
            segments[k] = (int(v[0]), int(v[1]))

        return segments

    def calculate_wall_thickness(self) -> None:
        """Calculates the distances between endocardial and epicardial contours' points.

        Returns:
            The myocardial thickness values.
        """
        logger.info("Calculating wall thickness")

        # Calculate distances between all combinations of points
        cost_matrix = cdist(np.array(self.endo_contour), np.array(self.epi_contour))

        # Assign the correct points for thickness calculation
        self.bld = self._calculate_bidirectional_local_distance_matrix(cost_matrix=cost_matrix)

        # Extract correct thickness values
        for point, _ in enumerate(self.thickness):
            self.thickness[point] = cost_matrix[point, self.bld[point]]

    def extract_thickness_parameters(self, segment_id: int) -> dict[str, NDArray]:
        """Calculates the segmental aggregates.

        Args:
            segment_id: The id of the segment to calculate values for, between 1 and 6.

        Returns:
            Max, mean and median thickness of the segment.
        """
        if segment_id not in range(1, 7):
            raise ValueError(
                "The segment id must be one of (1, 2, 3, 4, 5, 6). " f"{segment_id} was provided."
            )
        aggregate = {}
        boundary = self.segments[segment_id]
        logger.debug(boundary)
        aggregate["max"] = np.max(self.thickness[boundary[0] : boundary[1]])
        aggregate["mean"] = np.mean(self.thickness[boundary[0] : boundary[1]])
        aggregate["median"] = np.median(self.thickness[boundary[0] : boundary[1]])

        return aggregate

    def _calculate_bidirectional_local_distance_matrix(
        self, cost_matrix: NDArray
    ) -> dict[int, int]:
        """Assigns the closest points according to the bld algorithm (see class description).

        Returns:
            Pairs of endocardial and epicardial points found by bld. The number of pairs is equal
            to the number of endocardial points.
        """
        bidirectional_local_distance = {}

        # First iteration = forward distance
        forward_min_distance = np.argmin(cost_matrix, axis=1)
        for p_ref_f, p_t_f in enumerate(forward_min_distance):
            bidirectional_local_distance[p_ref_f] = p_t_f

        # Second iteration = backward distance
        backward_min_distance = np.argmin(cost_matrix, axis=0)
        for p_t_b, p_ref_b in enumerate(backward_min_distance):
            if (
                cost_matrix[p_ref_b, p_t_b]
                > cost_matrix[p_ref_b, bidirectional_local_distance[p_ref_b]]
            ):
                bidirectional_local_distance[p_ref_b] = p_t_b

        return bidirectional_local_distance
