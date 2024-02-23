import matplotlib.pyplot as plt
import numpy as np
import vtkmodules.all as vtk
from loguru import logger

# pylint: disable=import-error
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy


class SurfaceCurvature:
    """Class for computing the surface curvature of a self.mesh."""

    def __init__(self, mesh: vtk.vtkPolyData) -> None:
        self.mesh = mesh
        self.invert_mean_curvature = False
        self.poly_data = mesh

    @property
    def n_points(self) -> int:
        return self.poly_data.GetNumberOfPoints()

    @property
    def n_cells(self) -> int:
        return self.poly_data.GetNumberOfCells()

    def _has_triangle_strip(self) -> bool:
        has_triangle_strip = False
        for cell_id in range(self.mesh.GetNumberOfCells()):
            if self.mesh.GetCellType(cell_id) == vtk.VTK_TRIANGLE_STRIP:
                has_triangle_strip = True
                logger.info("Mesh has triangle strip")
                break
        return has_triangle_strip

    def _triangulate_filter(self) -> vtk.vtkPolyData:
        triangulate_filter = vtk.vtkTriangleFilter()
        triangulate_filter.SetInputData(self.mesh)
        triangulate_filter.Update()
        logger.info("Triangulation complete")
        return triangulate_filter.GetOutput()

    def get_mean_curvature(self) -> None:
        """Calculates mean curvature of a mesh"""
        logger.debug("Start GetMeanCurvature")

        # if self._has_triangle_strip():
        self.poly_data = self._triangulate_filter()

        # Empty array check
        if self.poly_data.GetNumberOfPolys() == 0 or self.poly_data.GetNumberOfPoints() == 0:
            logger.error("No points/cells to operate on")
            return

        self.poly_data.BuildLinks()

        vertices = vtk.vtkIdList()
        vertices_n = vtk.vtkIdList()
        neighbours = vtk.vtkIdList()

        mean_curvature = vtk.vtkDoubleArray()
        mean_curvature.SetName("Mean_Curvature")
        mean_curvature.SetNumberOfComponents(1)
        mean_curvature.SetNumberOfTuples(self.n_points)

        # Create and allocate
        n_f = np.zeros(3)  # normal of facet (could be stored for later?)
        n_n = np.zeros(3)  # normal of edge
        t = np.zeros(3)  # to store the cross product of n_f n_n
        ore = np.zeros(3)  # origin of e
        end = np.zeros(3)  # end of e
        oth = np.zeros(3)  # third vertex necessary for comp of n
        vn0 = np.zeros(3)
        vn1 = np.zeros(3)  # vertices for computation of neighbour's n
        vn2 = np.zeros(3)
        e = np.zeros(3)  # edge (oriented)

        # Init, preallocate the mean curvature
        num_neighb = np.zeros(self.n_points, dtype=int)

        # Main loop
        logger.debug(
            "Main loop: loop over facets such that id > id of neighbour so that every edge"
            " comes only once"
        )

        for f in range(self.n_cells):
            self.poly_data.GetCellPoints(f, vertices)
            nv = vertices.GetNumberOfIds()

            for v in range(nv):
                # Get neighbour
                v_l = vertices.GetId(v)
                v_r = vertices.GetId((v + 1) % nv)
                v_o = vertices.GetId((v + 2) % nv)

                self.poly_data.GetCellEdgeNeighbors(f, v_l, v_r, neighbours)

                # Compute only if there is really ONE neighbour
                # AND meanCurvature has not been computed yet!
                # (ensured by n > f)
                if neighbours.GetNumberOfIds() == 1 and (n := neighbours.GetId(0)) > f:
                    hf = 0.0  # temporary store

                    # Find 3 corners of f: in order!
                    self.poly_data.GetPoint(v_l, ore)
                    self.poly_data.GetPoint(v_r, end)
                    self.poly_data.GetPoint(v_o, oth)

                    # Compute normal of f
                    vtk.vtkTriangle.ComputeNormal(ore, end, oth, n_f)

                    # Compute common edge
                    e[0] = end[0] - ore[0]
                    e[1] = end[1] - ore[1]
                    e[2] = end[2] - ore[2]
                    length = vtk.vtkMath.Normalize(e)

                    af = vtk.vtkTriangle.TriangleArea(ore, end, oth)

                    # Find 3 corners of n: in order!
                    self.poly_data.GetCellPoints(n, vertices_n)
                    self.poly_data.GetPoint(vertices_n.GetId(0), vn0)
                    self.poly_data.GetPoint(vertices_n.GetId(1), vn1)
                    self.poly_data.GetPoint(vertices_n.GetId(2), vn2)
                    af += float(vtk.vtkTriangle.TriangleArea(vn0, vn1, vn2))

                    # Compute normal of n
                    vtk.vtkTriangle.ComputeNormal(vn0, vn1, vn2, n_n)

                    # The cosine is n_f * n_n
                    cs = vtk.vtkMath.Dot(n_f, n_n)

                    # The sin is (n_f x n_n) * e
                    vtk.vtkMath.Cross(n_f, n_n, t)
                    sn = vtk.vtkMath.Dot(t, e)

                    # Signed angle in [-pi, pi]
                    if sn != 0.0 or cs != 0.0:
                        angle = np.arctan2(sn, cs)
                        hf = length * angle
                    else:
                        hf = 0.0

                    # Add weighted hf to scalar at v_l and v_r
                    if af != 0.0:
                        hf = hf / af * 3.0

                    mean_curvature.SetValue(v_l, mean_curvature.GetValue(v_l) + hf)
                    mean_curvature.SetValue(v_r, mean_curvature.GetValue(v_r) + hf)

                    num_neighb[v_l] += 1
                    num_neighb[v_r] += 1

        # Put curvature in vtkArray
        for v in range(self.n_points):
            if num_neighb[v] > 0:
                hf = 0.5 * mean_curvature.GetValue(v) / num_neighb[v]
                if self.invert_mean_curvature:
                    mean_curvature.SetValue(v, -hf)
                else:
                    mean_curvature.SetValue(v, hf)
            else:
                mean_curvature.SetValue(v, 1.0)

        self.poly_data.GetPointData().AddArray(mean_curvature)
        self.poly_data.GetPointData().SetActiveScalars("Mean_Curvature")

        logger.debug("Set Values of Mean Curvature: Done")

    # Example usage
    # self.mesh = vtk.vtkPolyData()  # Replace this with your actual vtkPolyData
    # get_mean_curvature(self.mesh)


if __name__ == "__main__":
    # Create a named color object
    colors = vtk.vtkNamedColors()

    # Create a sphere
    obj_reader = vtk.vtkOBJReader()
    obj_reader.SetFileName("/home/mm/Projects/curvature/mesh/15736_Spiral_Twist_v1_NEW.obj")

    # Read the OBJ file
    obj_reader.Update()

    print(type(obj_reader.GetOutput()))
    mc = SurfaceCurvature(obj_reader.GetOutput())
    mc.get_mean_curvature()
    sphere_source = mc.poly_data

    # Create a mapper
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(sphere_source)
    mean_curvature_array = mc.poly_data.GetPointData().GetArray("Mean_Curvature")

    mean_curvature_values = vtk_to_numpy(mean_curvature_array)
    mean_curvature_values[1::2] = mean_curvature_values[::2]
    mean_curvature_array = numpy_to_vtk(mean_curvature_values)
    logger.debug(mean_curvature_values)
    plt.plot(mean_curvature_values)
    plt.show()
    # Set the scalar values for the mapper
    mapper.GetInput().GetPointData().SetScalars(mean_curvature_array)

    # Create an actor
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    # Create a renderer
    renderer = vtk.vtkRenderer()
    renderer.AddActor(actor)
    renderer.SetBackground(colors.GetColor3d("DarkGreen"))

    # Create a render window
    render_window = vtk.vtkRenderWindow()
    render_window.SetWindowName("Sphere")
    render_window.AddRenderer(renderer)

    # Create a render window interactor
    render_window_interactor = vtk.vtkRenderWindowInteractor()
    render_window_interactor.SetRenderWindow(render_window)

    # Render and start the interaction
    render_window.Render()
    render_window_interactor.Start()

    obj_writer = vtk.vtkPolyDataWriter()
    obj_writer.SetInputData(mc.poly_data)
    obj_writer.SetFileName("/home/mm/Projects/curvature/mesh/curve.vtk")

    # Read the OBJ file
    obj_writer.Update()
