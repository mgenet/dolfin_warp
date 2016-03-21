#coding=utf8

########################################################################
###                                                                  ###
### Created by Martin Genet, 2016                                    ###
###                                                                  ###
### École Polytechnique, Palaiseau, France                           ###
###                                                                  ###
########################################################################

import dolfin
import glob
import numpy
import os

import myFEniCSPythonLibrary as myFEniCS
import myVTKPythonLibrary    as myVTK

########################################################################

def fedic(
        working_folder,
        working_basename,
        images_folder,
        images_basename,
        mesh_folder,
        mesh_basename,
        mesh_degree=1,
        images_dimension=3,
        k_ref=0,
        penalty=0.9,
        use_I0_tangent=0,
        relax_type="const",
        relax_init=0.9,
        relax_n_iter=10,
        tol_res=None,
        tol_disp=None,
        n_iter_max=100,
        print_iterations=0,
        continue_after_fail=0,
        verbose=1):

    if not os.path.exists(working_folder):
        os.mkdir(working_folder)

    print "Loading mesh…"
    mesh = dolfin.Mesh(mesh_folder+"/"+mesh_basename+".xml")

    fs = dolfin.VectorFunctionSpace(
        mesh=mesh,
        family="Lagrange",
        degree=mesh_degree)
    dX = dolfin.dx(mesh)

    print "Computing quadrature degree…"
    degree = myFEniCS.compute_quadrature_degree(
        image_filename=images_folder+"/"+images_basename+"_"+str(k_ref).zfill(2)+".vti",
        mesh=mesh,
        image_dimension=images_dimension)
    print "degree = " + str(degree)

    fe = dolfin.FiniteElement(
        family="Quadrature",
        cell=mesh.ufl_cell(),
        degree=degree,
        quad_scheme="default")
    ve = dolfin.VectorElement(
        family="Quadrature",
        cell=mesh.ufl_cell(),
        degree=degree,
        quad_scheme="default")

    print "Loading reference image…"
    if (images_dimension == 2):
        I0 = myFEniCS.ExprIm2(
            filename=images_folder+"/"+images_basename+"_"+str(k_ref).zfill(2)+".vti",
            element=fe)
        DI0 = myFEniCS.ExprGradIm2(
            filename=images_folder+"/"+images_basename+"_"+str(k_ref).zfill(2)+".vti",
            element=ve)
    elif (images_dimension == 3):
        I0 = myFEniCS.ExprIm3(
            filename=images_folder+"/"+images_basename+"_"+str(k_ref).zfill(2)+".vti",
            element=fe)
        DI0 = myFEniCS.ExprGradIm3(
            filename=images_folder+"/"+images_basename+"_"+str(k_ref).zfill(2)+".vti",
            element=ve)
    else:
        assert (0), "images_dimension must be 2 or 3. Aborting."
    I0_norm = dolfin.assemble(I0**2 * dX, form_compiler_parameters={'quadrature_degree':degree})**(1./2)
    assert (I0_norm > 0.), "I0_norm = " + str(I0_norm)
    print "I0_norm = " + str(I0_norm)

    print "Defining functions…"
    penalty = dolfin.Constant(penalty)
    U = dolfin.Function(
        fs,
        name="displacement")
    U_old = dolfin.Function(
        fs,
        name="old displacement")
    dU = dolfin.Function(
        fs,
        name="ddisplacement")
    ddU = dolfin.TrialFunction(fs)
    V = dolfin.TestFunction(fs)

    print "Printing initial solution…"
    file_pvd = dolfin.File(working_folder+"/"+working_basename+"_.pvd")
    for vtu in glob.glob(working_folder+"/"+working_basename+"_*.vtu"):
        os.remove(vtu)
    file_pvd << (U, float(k_ref))

    print "Defining variational forms…"
    if (images_dimension == 2):
        I1 = myFEniCS.ExprDefIm2(
            U=U,
            element=fe)
        DI1 = myFEniCS.ExprGradDefIm2(
            U=U,
            element=ve)
        if (use_I0_tangent):
            a =     penalty  * dolfin.inner(dolfin.dot(DI0, ddU),
                                            dolfin.dot(DI0,   V))*dX\
              + (1.-penalty) * dolfin.inner(dolfin.grad(ddU),
                                            dolfin.grad(  V))*dX
        else:
            a =     penalty  * dolfin.inner(dolfin.dot(DI1, ddU),
                                            dolfin.dot(DI1,   V))*dX\
              + (1.-penalty) * dolfin.inner(dolfin.grad(ddU),
                                            dolfin.grad(  V))*dX
        b =     penalty  * dolfin.inner(I0-I1,
                                        dolfin.dot(DI1, V))*dX\
          - (1.-penalty) * dolfin.inner(dolfin.grad(U),
                                        dolfin.grad(V))*dX
        b0 =     penalty  * dolfin.inner(I0,
                                         dolfin.dot(DI0, V))*dX
    elif (images_dimension == 3):
        I1 = myFEniCS.ExprDefIm3(
            U=U,
            element=fe)
        DI1 = myFEniCS.ExprGradDefIm3(
            U=U,
            element=ve)
        if (use_I0_tangent):
            a =     penalty  * dolfin.inner(dolfin.dot(DI0, ddU),
                                            dolfin.dot(DI0,   V))*dX\
              + (1.-penalty) * dolfin.inner(dolfin.grad(ddU),
                                            dolfin.grad(  V))*dX
        else:
            a =     penalty  * dolfin.inner(dolfin.dot(DI1, ddU),
                                            dolfin.dot(DI1,   V))*dX\
              + (1.-penalty) * dolfin.inner(dolfin.grad(ddU),
                                            dolfin.grad(  V))*dX
        b =     penalty  * dolfin.inner(I0-I1,
                                        dolfin.dot(DI1, V))*dX\
          - (1.-penalty) * dolfin.inner(dolfin.grad(U),
                                        dolfin.grad(V))*dX
        b0 =     penalty  * dolfin.inner(I0,
                                         dolfin.dot(DI0, V))*dX
    else:
        assert (0), "images_dimension must be 2 or 3. Aborting."
    B0 = dolfin.assemble(b0, form_compiler_parameters={'quadrature_degree':degree})
    B0_norm = numpy.linalg.norm(B0)
    assert (B0_norm > 0.), "B0_norm = " + str(B0_norm)
    print "B0_norm = " + str(B0_norm)

    # linear system
    if (use_I0_tangent):
        A = dolfin.assemble(a, form_compiler_parameters={'quadrature_degree':degree})
    else:
        A = None
    B = None

    print "Checking number of frames…"
    n_frames = len(glob.glob(images_folder+"/"+images_basename+"_*.vti"))
    assert (n_frames > 1), "n_frames = " + str(n_frames)
    print "n_frames = " + str(n_frames)

    assert (abs(k_ref) < n_frames), "k_ref = " + str(k_ref)
    k_ref = k_ref%n_frames

    n_iter_tot = 0
    global_success = True
    for forward_or_backward in ["forward", "backward"]: # not sure if this still works…
        print "forward_or_backward = " + forward_or_backward

        if (forward_or_backward == "forward"):
            k_frames = range(k_ref+1, n_frames, +1)
        elif (forward_or_backward == "backward"):
            k_frames = range(k_ref-1, -1, -1)
        print "k_frames = " + str(k_frames)

        if (forward_or_backward == "backward"):
            U.vector()[:] = 0.

        for k_frame in k_frames:
            print "k_frame = " + str(k_frame)

            if (print_iterations):
                if (os.path.exists(working_folder+"/"+working_basename+"-frame="+str(k_frame).zfill(2)+".pdf")):
                    os.remove(working_folder+"/"+working_basename+"-frame="+str(k_frame).zfill(2)+".pdf")
                file_dat_iter = open(working_folder+"/"+working_basename+"-frame="+str(k_frame).zfill(2)+".dat", "w")

                file_pvd_iter = dolfin.File(working_folder+"/"+working_basename+"-frame="+str(k_frame).zfill(2)+"_.pvd")
                for vtu in glob.glob(working_folder+"/"+working_basename+"-frame="+str(k_frame).zfill(2)+"_*.vtu"):
                    os.remove(vtu)
                file_pvd_iter << (U, 0.)

            print "Loading image and image gradient…"
            if (images_dimension == 2):
                I1.init_image( filename=images_folder+"/"+images_basename+"_"+str(k_frame).zfill(2)+".vti")
                DI1.init_image(filename=images_folder+"/"+images_basename+"_"+str(k_frame).zfill(2)+".vti")
            elif (images_dimension == 3):
                I1.init_image( filename=images_folder+"/"+images_basename+"_"+str(k_frame).zfill(2)+".vti")
                DI1.init_image(filename=images_folder+"/"+images_basename+"_"+str(k_frame).zfill(2)+".vti")
            else:
                assert (0), "images_dimension must be 2 or 3. Aborting."

            print "Running registration…"
            k_iter = 0
            success = False
            while (True):
                print "k_iter = " + str(k_iter)
                n_iter_tot += 1

                # linear system: matrix
                if not (use_I0_tangent):
                    A = dolfin.assemble(a, tensor=A, form_compiler_parameters={'quadrature_degree':degree})
                    #print "A = " + str(A.array())
                    #A_norm = numpy.linalg.norm(A.array())
                    #print "A_norm = " + str(A_norm)

                # linear system: residual
                if (relax_type == "aitken"):
                    if (k_iter == 1):
                        B_old = B.copy()
                    elif (k_iter > 1):
                        B_old[:] = B[:]
                    #print "B_old = " + str(B_old[0])
                B = dolfin.assemble(b, tensor=B, form_compiler_parameters={'quadrature_degree':degree})
                #print "B = " + str(B.array())
                B_norm = numpy.linalg.norm(B.array())
                #print "B_norm = " + str(B_norm)
                err_res = B_norm/B0_norm
                print "err_res = " + str(err_res)

                # linear system: solve
                dolfin.solve(A, dU.vector(), B)
                #print "dU = " + str(dU.vector().array())

                # relaxation
                U_old.vector()[:] = U.vector()[:]
                if (relax_type == "constant"):
                    if (k_iter == 0):
                        relax = relax_init
                elif (relax_type == "aitken"):
                    if (k_iter == 0):
                        relax = relax_init
                    else:
                        if (k_iter == 1):
                            dB = B - B_old
                        elif (k_iter > 1):
                            dB[:] = B[:] - B_old[:]
                        relax *= (-1.) * B_old.inner(dB) / dB.inner(dB)
                    print "relax = " + str(relax)
                elif (relax_type == "automatic"):
                    B_relax = numpy.empty(relax_n_iter)
                    for k_relax in xrange(relax_n_iter):
                        #print "k_relax = " + str(k_relax)
                        relax = float(k_relax+1)/relax_n_iter
                        U.vector()[:] = U_old.vector()[:] + relax * dU.vector()[:]
                        B = dolfin.assemble(b, tensor=B, form_compiler_parameters={'quadrature_degree':degree})
                        B_relax[k_relax] = numpy.linalg.norm(B)
                        #print "B_relax = " + str(B_relax[k_relax])
                    #print "B_relax = " + str(B_relax)
                    k_relax = numpy.argmin(B_relax)
                    relax = float(k_relax+1)/relax_n_iter
                    print "relax = " + str(relax)

                # solution update
                U.vector()[:] = U_old.vector()[:] + relax * dU.vector()[:]

                if (print_iterations):
                    #print "U = " + str(U.vector().array())
                    file_pvd_iter << (U, float(k_iter+1))

                # displacement error
                dU_norm = numpy.linalg.norm(dU.vector().array())
                print "dU_norm = " + str(dU_norm)
                U_norm = numpy.linalg.norm(U.vector().array())
                assert (U_norm > 0.), "U_norm = " + str(U_norm)
                print "U_norm = " + str(U_norm)
                err_disp = dU_norm/U_norm
                print "err_disp = " + str(err_disp)

                # image error
                if (k_iter > 0):
                    I1I0_norm_old = I1I0_norm
                I1I0_norm = dolfin.assemble((I1-I0)**2 * dX, form_compiler_parameters={'quadrature_degree':degree})**(1./2)
                #print "I1I0_norm = " + str(I1I0_norm)
                err_im = I1I0_norm/I0_norm
                print "err_im = " + str(err_im)
                if (k_iter == 0):
                    err_im_rel = 1.
                else:
                    err_im_rel = abs(I1I0_norm-I1I0_norm_old)/I1I0_norm_old
                    print "err_im_rel = " + str(err_im_rel)

                if (print_iterations):
                    file_dat_iter.write(" ".join([str(val) for val in [k_iter, B_norm, err_res, relax, dU_norm, U_norm, err_disp, I1I0_norm, err_im, err_im_rel]])+"\n")

                # exit test
                if (tol_disp is not None) and (tol_res is not None):
                    if ((err_disp < tol_disp) or ((dU_norm < tol_disp) and (U_norm < tol_disp))) and (err_res < tol_res):
                        success = True
                elif (tol_disp is not None):
                    if (err_disp < tol_disp) or ((dU_norm < tol_disp) and (U_norm < tol_disp)):
                        success = True
                elif (tol_res is not None):
                    if (err_res < tol_res):
                        success = True
                if (success):
                    print "Nonlinear solver converged…"
                    break

                if (k_iter == n_iter_max-1):
                    global_success = False
                    print "Warning! Nonlinear solver failed to converge… (k_frame = "+str(k_frame)+")"
                    break

                # increment counter
                k_iter += 1

            if (print_iterations):
                os.remove(working_folder+"/"+working_basename+"-frame="+str(k_frame).zfill(2)+"_.pvd")
                file_dat_iter.close()
                os.system("gnuplot -e \"set terminal pdf; set output '"+working_folder+"/"+working_basename+"-frame="+str(k_frame).zfill(2)+".pdf'; set logscale y; set yrange [1e-4:1e1]; plot '"+working_folder+"/"+working_basename+"-frame="+str(k_frame).zfill(2)+".dat' u 1:3 pt 1 lw 3 title 'err_res', '' u 1:4 pt 1 lw 3 title 'relax', '' u 1:6 pt 1 lw 3 title 'err_disp', '' using 1:8 pt 1 lw 3 title 'err_im', '' using 1:9 pt 1 lw 3 title 'err_im_rel', 1e-3 lt 0 notitle\"")

            if not (success) and not (continue_after_fail):
                break

            print "Printing solution…"
            file_pvd << (U, float(k_frame))

        if not (success) and not (continue_after_fail):
            break

    print "n_iter_tot = " + str(n_iter_tot)

    os.remove(working_folder+"/"+working_basename+"_.pvd")

    return global_success
