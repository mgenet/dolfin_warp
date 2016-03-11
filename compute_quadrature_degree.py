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

########################################################################

def compute_quadrature_degree(
        image_filename,
        dX,
        image_dimension=3,
        deg_min=1,
        deg_max=10,
        tol=1e-2,
        n_under_tol=1,
        verbose=1):

    first_time = True
    k_under_tol = 0
    for degree in xrange(deg_min,deg_max+1):
        if (verbose): print "degree = " + str(degree)
        fe = dolfin.FiniteElement(
            family="Quadrature",
            degree=degree)
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
        I0_norm = dolfin.assemble(I0**2 * dX)**(1./2)
        if (verbose): print "I0_norm = " + str(I0_norm)
        if (first_time):
            first_time = False
        else:
            I0_norm_err = abs(I0_norm-I0_norm_old)/I0_norm_old
            if (verbose): print "I0_norm_err = " + str(I0_norm_err)
            if (I0_norm_err < tol):
                k_under_tol += 1
            else:
                k_under_tol = 0
            if (k_under_tol >= n_under_tol):
                break
        I0_norm_old = I0_norm
    return degree
