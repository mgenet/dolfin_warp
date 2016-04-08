#coding=utf8

########################################################################
###                                                                  ###
### Created by Martin Genet, 2016                                    ###
###                                                                  ###
### École Polytechnique, Palaiseau, France                           ###
###                                                                  ###
########################################################################

import dolfin

import myFEniCSPythonLibrary as myFEniCS
import myVTKPythonLibrary    as myVTK

########################################################################

def compute_quadrature_degree(
        image_filename,
        mesh,
        deg_min=1,
        deg_max=10,
        tol=1e-2,
        n_under_tol=1,
        verbose=1):

    image_dimension = myVTK.computeImageDimensionality(
        image_filename=image_filename,
        verbose=0)
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
            I0 = myFEniCS.ExprIm2(
                filename=image_filename,
                element=fe)
        elif (image_dimension == 3):
            I0 = myFEniCS.ExprIm3(
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
