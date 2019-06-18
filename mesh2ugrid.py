#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2016-2019                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
### This function inspired by Miguel A. Rodriguez                            ###
###  https://fenicsproject.org/qa/12933/                                     ###
###            making-vtk-python-object-from-solution-object-the-same-script ###
###                                                                          ###
################################################################################

import dolfin
import glob
import numpy
import vtk

import myPythonLibrary as mypy
import myVTKPythonLibrary as myvtk

import dolfin_dic as ddic

################################################################################

def mesh2ugrid(
        mesh,
        function_space):

    n_dim = mesh.geometry().dim()
    # n_dim = mesh.ufl_domain().geometric_dimension()
    assert (n_dim in (2,3))
    # print "n_dim = "+str(n_dim)

    n_verts = mesh.num_vertices()
    # print "n_verts = "+str(n_verts)
    n_cells = mesh.num_cells()
    # print "n_cells = "+str(n_cells)

    # Store nodes coordinates as numpy array
    n_dofs = function_space.dim()
    np_coordinates = function_space.tabulate_dof_coordinates().reshape([n_dofs, n_dim])
    # print "np_coordinates = "+str(np_coordinates)

    if (n_dim == 2):
        np_coordinates = numpy.hstack((np_coordinates, numpy.zeros([n_dofs, 1])))
        # print "np_coordinates = "+str(np_coordinates)

    # Convert nodes coordinates to VTK
    vtk_coordinates = vtk.util.numpy_support.numpy_to_vtk(np_coordinates)
    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(vtk_coordinates)

    # Check element type
    dofmap = function_space.dofmap()
    # print "dofmap = "+str(dofmap)
    n_nodes_per_element = dofmap.num_element_dofs(0)
    # print "n_nodes_per_element = "+str(n_nodes_per_element)
    if   (n_dim == 2):
        assert (n_nodes_per_element in (3,  6))
        if   (n_nodes_per_element == 3):
            vtk_cell_type = vtk.VTK_TRIANGLE
        elif (n_nodes_per_element == 6):
            vtk_cell_type = vtk.VTK_QUADRATIC_TRIANGLE
    elif (n_dim == 3):
        assert (n_nodes_per_element in (4, 10))
        if   (n_nodes_per_element == 4):
            vtk_cell_type = vtk.VTK_TETRA
        elif (n_nodes_per_element == 10):
            vtk_cell_type = vtk.VTK_QUADRATIC_TETRA

    # Store connectivity as numpy array
    np_connectivity = numpy.empty(
        [n_cells, n_nodes_per_element],
        dtype=numpy.int)
    for i in range(n_cells):
        np_connectivity[i,:] = dofmap.cell_dofs(i)

    # Permute connectivity
    # (Because node numbering scheme in FEniCS is different from VTK for 2nd order.)
    # (Explains why the connectivity is first filled as an array and then flattened.)
    if   (vtk_cell_type == vtk.VTK_QUADRATIC_TRIANGLE):
        assert (0), "ToCheck. Aborting."
    elif (vtk_cell_type == vtk.VTK_QUADRATIC_TETRA):
        PERM_DOLFIN_TO_VTK = [0, 1, 2, 3, 9, 6, 8, 7, 5, 4]
        np_connectivity = np_connectivity[:, PERM_DOLFIN_TO_VTK]

    # Add left column specifying number of nodes per cell and flatten array
    np_connectivity = numpy.hstack((numpy.ones([n_cells, 1], dtype=numpy.int)*n_nodes_per_element, np_connectivity)).flatten()

    # Convert connectivity to VTK
    vtk_connectivity = vtk.util.numpy_support.numpy_to_vtkIdTypeArray(np_connectivity)

    # Create cell array
    vtk_cells = vtk.vtkCellArray()
    vtk_cells.SetCells(n_cells, vtk_connectivity)

    # Create unstructured grid and set points and connectivity
    ugrid = vtk.vtkUnstructuredGrid()
    ugrid.SetPoints(vtk_points)
    ugrid.SetCells(vtk_cell_type, vtk_cells)

    return ugrid

################################################################################

def add_function_to_ugrid(
        function,
        ugrid):

    # Convert function values and add as scalar data
    np_array = function.vector().get_local()
    vtk_array = vtk.util.numpy_support.numpy_to_vtk(np_array)
    vtk_array.SetName(function.name())

    # There are multiple ways of adding this
    ugrid.GetPointData().SetScalars(vtk_array)
