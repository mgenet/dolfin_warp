#coding=utf8

########################################################################
###                                                                  ###
### Created by Martin Genet, 2016                                    ###
###                                                                  ###
### École Polytechnique, Palaiseau, France                           ###
###                                                                  ###
########################################################################

import dolfin
import numpy

import myPythonLibrary as mypy
import myVTKPythonLibrary as myvtk

import dolfin_dic as ddic

########################################################################

def compute_quadrature_degree_from_points_count(
        image_filename,
        mesh_filebasename,
        mesh_ext="vtk",
        deg_min=1,
        deg_max=20,
        verbose=1):

    image = myvtk.readImage(
        filename=image_filename,
        verbose=verbose)
    n_points = image.GetNumberOfPoints()

    mesh = myvtk.readUGrid(
        filename=mesh_filebasename+"."+mesh_ext,
        verbose=verbose)
    n_cells = mesh.GetNumberOfCells()

    (cell_locator,
     closest_point,
     generic_cell,
     k_cell,
     subId,
     dist) = myvtk.getCellLocator(
        mesh=mesh,
        verbose=verbose)

    point = numpy.empty(3)
    n_pixels_per_cell = numpy.zeros(n_cells)
    for k_point in xrange(n_points):
        image.GetPoint(k_point, point)

        k_cell = cell_locator.FindCell(point)
        if (k_cell == -1): continue
        else: n_pixels_per_cell[k_cell] += 1
    n_pixels_per_cell_max = int(max(n_pixels_per_cell))
    n_pixels_per_cell_avg = int(sum(n_pixels_per_cell)/n_cells)

    if (verbose):
        #print "n_pixels_per_cell = "+str(n_pixels_per_cell)
        #print "sum(n_pixels_per_cell) = "+str(sum(n_pixels_per_cell))
        print "n_pixels_per_cell_max = "+str(n_pixels_per_cell_max)
        print "n_pixels_per_cell_avg = "+str(n_pixels_per_cell_avg)

    mesh = dolfin.Mesh(mesh_filebasename+"."+"xml")

    for degree in xrange(deg_min,deg_max+1):
        if (verbose): print "degree = "+str(degree)
        n_quad = len(dolfin.FunctionSpace(
            mesh,
            dolfin.FiniteElement(
                family="Quadrature",
                cell=mesh.ufl_cell(),
                degree=degree,
                quad_scheme="default")).dofmap().dofs())/len(mesh.cells())
        if (verbose): print "n_quad = "+str(n_quad)
        #if (n_quad > n_pixels_per_cell_max): break
        if (n_quad > n_pixels_per_cell_avg): break

    return degree

########################################################################

def compute_quadrature_degree_from_integral(
        image_filename,
        mesh,
        deg_min=1,
        deg_max=10,
        tol=1e-2,
        n_under_tol=1,
        verbose=1):

    image = myvtk.readImage(
        filename=image_filename,
        verbose=verbose)
    image_dimension = myvtk.getImageDimensionality(
        image=image,
        verbose=verbose)
    if (verbose): print "image_dimension = " + str(image_dimension)

    dX = dolfin.dx(mesh)

    first_time = True
    k_under_tol = 0
    for degree in xrange(deg_min,deg_max+1):
        if (verbose): print "degree = " + str(degree)
        fe = dolfin.FiniteElement(
            family="Quadrature",
            cell=mesh.ufl_cell(),
            degree=degree,
            quad_scheme="default")
        if (image_dimension == 2):
            I0 = ddic.ExprIm2(
                filename=image_filename,
                element=fe)
        elif (image_dimension == 3):
            I0 = ddic.ExprIm3(
                filename=image_filename,
                element=fe)
        else:
            assert (0), "image_dimension must be 2 or 3. Aborting."
        if not (first_time):
            I0_norm_old = I0_norm
        I0_norm = dolfin.assemble(I0**2 * dX, form_compiler_parameters={'quadrature_degree':degree})**(1./2)
        if (verbose): print "I0_norm = " + str(I0_norm)
        if (first_time):
            first_time = False
            continue
        I0_norm_err = abs(I0_norm-I0_norm_old)/I0_norm_old
        if (verbose): print "I0_norm_err = " + str(I0_norm_err)
        if (I0_norm_err < tol):
            k_under_tol += 1
        else:
            k_under_tol = 0
        if (k_under_tol >= n_under_tol):
            break

    return degree
