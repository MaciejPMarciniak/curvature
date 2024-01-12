# Analysis script
from pathlib import Path

import contour_io
import myocardial_contour
import scale_contour_data
import wall_thickness

# Add click


def main() -> None:
    data_path = Path("/home/mm/Downloads/Curvature_data")
    case_id = 0

    io = contour_io.ContourIO(case_index=case_id, project_path=data_path)
    segmentation = io.read_segmentation()

    contour = myocardial_contour.MyocardialContour(segmentation)
    contour.generate_myocardial_contours(True)

    scaling = scale_contour_data.ScaleContourData(case_index=case_id, project_path=data_path)
    scaled_endo_contour = scaling.scale_contour(
        contour.endocardial_contour, adjust_with_info_file=True
    )
    scaled_epi_contour = scaling.scale_contour(
        contour.epicardial_contour, adjust_with_info_file=False
    )

    wt = wall_thickness.WallThickness(scaled_endo_contour, scaled_epi_contour)
    wt.calculate_wall_thickness()

    io.save_contours([scaled_endo_contour, scaled_epi_contour], ["Endocardium", "Epicardium"])
    io.save_image_with_contours(
        io.read_image(),
        contour.endocardial_contour,
        contour.epicardial_contour,
    )
    io.save_image_with_wall_thickness(
        io.read_image(), contour.endocardial_contour, contour.epicardial_contour, wt.bld
    )


if __name__ == "__main__":
    main()
