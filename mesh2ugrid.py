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
        mesh):

    n_dim = mesh.geometry().dim()
    assert (n_dim in (2,3))
    # print "n_dim = "+str(n_dim)

    n_verts = mesh.num_vertices()
    # print "n_verts = "+str(n_verts)
    n_cells = mesh.num_cells()
    # print "n_cells = "+str(n_cells)

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
    # print "n_nodes = "+str(n_nodes)
    np_coordinates = fs.tabulate_dof_coordinates().reshape([n_nodes, n_dim])
    # print "np_coordinates = "+str(np_coordinates)

    if (n_dim == 2):
        np_coordinates = numpy.hstack((np_coordinates, numpy.zeros([n_nodes, 1])))
        # print "np_coordinates = "+str(np_coordinates)

    # Convert nodes coordinates to VTK
    vtk_coordinates = vtk.util.numpy_support.numpy_to_vtk(
        num_array=np_coordinates,
        deep=1)
    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(vtk_coordinates)
    # print "n_points = "+str(vtk_points.GetNumberOfPoints())

    # Store connectivity as numpy array
    n_nodes_per_cell = fs.dofmap().num_element_dofs(0)
    # print "n_nodes_per_cell = "+str(n_nodes_per_cell)
    np_connectivity = numpy.empty(
        [n_cells, n_nodes_per_cell+1],
        dtype=numpy.int)
    for i in range(n_cells):
        np_connectivity[i, 0] = n_nodes_per_cell
        np_connectivity[i,1:] = fs.dofmap().cell_dofs(i)
    # print "np_connectivity = "+str(np_connectivity)

    # Add left column specifying number of nodes per cell and flatten array
    np_connectivity = np_connectivity.flatten()
    # print "np_connectivity = "+str(np_connectivity)

    # Convert connectivity to VTK
    vtk_connectivity = vtk.util.numpy_support.numpy_to_vtkIdTypeArray(np_connectivity)

    # Create cell array
    vtk_cells = vtk.vtkCellArray()
    vtk_cells.SetCells(n_cells, vtk_connectivity)

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
        ugrid):

    # print ugrid.GetPoints()

    # Convert function values and add as scalar data
    n_dofs = function.function_space().dim()
    # print "n_dofs = "+str(n_dofs)
    n_dim = function.value_size()
    # print "n_dim = "+str(n_dim)
    assert (n_dofs/n_dim == ugrid.GetNumberOfPoints()),\
        "Only CG1 functions can be converted to VTK. Aborting."
    np_array = function.vector().get_local()
    # print "np_array = "+str(np_array)
    np_array = np_array.reshape([n_dofs/n_dim, n_dim])
    # print "np_array = "+str(np_array)
    vtk_array = vtk.util.numpy_support.numpy_to_vtk(
        num_array=np_array,
        deep=1)
    vtk_array.SetName(function.name())

    # print ugrid.GetPoints()
    ugrid.GetPointData().AddArray(vtk_array)
    # print ugrid.GetPoints()

################################################################################

def add_functions_to_ugrid(
        functions,
        ugrid):

    for function in functions:
        ddic.add_function_to_ugrid(
            function=function,
            ugrid=ugrid)
