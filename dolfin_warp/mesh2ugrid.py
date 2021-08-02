#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2016-2021                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
### This function inspired by Miguel A. Rodriguez                            ###
###  https://fenicsproject.org/qa/12933/                                     ###
###            making-vtk-python-object-from-solution-object-the-same-script ###
###                                                                          ###
################################################################################

from builtins import range

import dolfin
import glob
import numpy
import vtk

import myPythonLibrary    as mypy
import myVTKPythonLibrary as myvtk

import dolfin_warp as dwarp

################################################################################

def mesh2ugrid(
        mesh,
        verbose=0):

    if (verbose): print("mesh2ugrid")

    n_dim = mesh.geometry().dim()
    assert (n_dim in (2,3))
    if (verbose): print("n_dim = "+str(n_dim))

    n_verts = mesh.num_vertices()
    if (verbose): print("n_verts = "+str(n_verts))
    n_cells = mesh.num_cells()
    if (verbose): print("n_cells = "+str(n_cells))

    # Create function space
    fe = dolfin.FiniteElement(
        family="CG",
        cell=mesh.ufl_cell(),
        degree=1,
        quad_scheme='default')
    fs = dolfin.FunctionSpace(
        mesh,
        fe)

    # Store nodes coordinates as numpy array
    n_nodes = fs.dim()
    assert (n_nodes == n_verts)
    if (verbose): print("n_nodes = "+str(n_nodes))
    global np_coordinates # MG20190621: if it disappears the vtk objects is broken
    np_coordinates = fs.tabulate_dof_coordinates().reshape([n_nodes, n_dim])
    if (verbose): print("np_coordinates = "+str(np_coordinates))

    if (n_dim == 2):
        np_coordinates = numpy.hstack((np_coordinates, numpy.zeros([n_nodes, 1])))
        if (verbose): print("np_coordinates = "+str(np_coordinates))

    # Convert nodes coordinates to VTK
    vtk_coordinates = vtk.util.numpy_support.numpy_to_vtk(
        num_array=np_coordinates,
        deep=1)
    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(vtk_coordinates)
    if (verbose): print("n_points = "+str(vtk_points.GetNumberOfPoints()))

    # Store connectivity as numpy array
    if   (n_dim == 2):
        n_nodes_per_cell = 3
    elif (n_dim == 3):
        n_nodes_per_cell = 4
    if (verbose): print("n_nodes_per_cell = "+str(n_nodes_per_cell))
    global np_connectivity # MG20190621: if it disappears the vtk objects is broken
    np_connectivity = numpy.empty(
        [n_cells, 1+n_nodes_per_cell],
        dtype=numpy.int)
    for i in range(n_cells):
        np_connectivity[i, 0] = n_nodes_per_cell
        np_connectivity[i,1:] = fs.dofmap().cell_dofs(i)
    # if (verbose): print("np_connectivity = "+str(np_connectivity))

    # Add left column specifying number of nodes per cell and flatten array
    np_connectivity = np_connectivity.flatten()
    # if (verbose): print("np_connectivity = "+str(np_connectivity))

    # Convert connectivity to VTK
    vtk_connectivity = vtk.util.numpy_support.numpy_to_vtkIdTypeArray(np_connectivity)

    # Create cell array
    vtk_cells = vtk.vtkCellArray()
    vtk_cells.SetCells(n_cells, vtk_connectivity)
    if (verbose): print("n_cells = "+str(vtk_cells.GetNumberOfCells()))

    # Create unstructured grid and set points and connectivity
    if   (n_dim == 2):
        vtk_cell_type = vtk.VTK_TRIANGLE
    elif (n_dim == 3):
        vtk_cell_type = vtk.VTK_TETRA
    ugrid = vtk.vtkUnstructuredGrid()
    ugrid.SetPoints(vtk_points)
    ugrid.SetCells(vtk_cell_type, vtk_cells)

    return ugrid

################################################################################

def add_function_to_ugrid(
        function,
        ugrid,
        verbose=0):

    if (verbose): print("add_function_to_ugrid")
    if (verbose): print(ugrid.GetPoints())

    # Convert function values and add as scalar data
    n_dofs = function.function_space().dim()
    if (verbose): print("n_dofs = "+str(n_dofs))
    n_dim = function.ufl_element().value_size()
    if (verbose): print("n_dim = "+str(n_dim))
    assert (n_dofs//n_dim == ugrid.GetNumberOfPoints()),\
        "Only CG1 functions can be converted to VTK. Aborting."
    global np_array # MG20190621: if it disappears the vtk objects is broken
    np_array = function.vector().get_local()
    if (verbose): print("np_array = "+str(np_array))
    np_array = np_array.reshape([n_dofs//n_dim, n_dim])
    if (verbose): print("np_array = "+str(np_array))
    vtk_array = vtk.util.numpy_support.numpy_to_vtk(
        num_array=np_array,
        deep=1)
    vtk_array.SetName(function.name())

    if (verbose): print(ugrid.GetPoints())
    ugrid.GetPointData().AddArray(vtk_array)
    if (verbose): print(ugrid.GetPoints())

################################################################################

def add_functions_to_ugrid(
        functions,
        ugrid):

    for function in functions:
        dwarp.add_function_to_ugrid(
            function=function,
            ugrid=ugrid)
