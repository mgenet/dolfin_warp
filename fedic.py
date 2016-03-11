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
        images_dimension=3,
        k_ref=0,
        penalty=0.9,
        use_I0_tangent=0,
        relax_type="const",
        relax_init=0.9,
        tol_disp=1e-3,
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
        degree=1)
    dX = dolfin.dx(mesh)

    print "Computing quadrature degree…"
    degree = myFEniCS.compute_quadrature_degree(
        image_filename=images_folder+"/"+images_basename+"_"+str(k_ref).zfill(2)+".vti",
        dX=dX,
        image_dimension=images_dimension)
    print "degree = " + str(degree)

    fe = dolfin.FiniteElement(
        family="Quadrature",
        degree=degree)

    print "Loading reference image…"
    if (images_dimension == 2):
        I0 = myFEniCS.ExprIm2(
            filename=images_folder+"/"+images_basename+"_"+str(k_ref).zfill(2)+".vti",
            element=fe)
        if (use_I0_tangent):
            DXI0 = myFEniCS.ExprGradXIm2(
                filename=images_folder+"/"+images_basename+"_"+str(k_ref).zfill(2)+".vti",
                element=fe)
            DYI0 = myFEniCS.ExprGradYIm2(
                filename=images_folder+"/"+images_basename+"_"+str(k_ref).zfill(2)+".vti",
                element=fe)
    elif (images_dimension == 3):
        I0 = myFEniCS.ExprIm3(
            filename=images_folder+"/"+images_basename+"_"+str(k_ref).zfill(2)+".vti",
            element=fe)
        if (use_I0_tangent):
            DXI0 = myFEniCS.ExprGradXIm3(
                filename=images_folder+"/"+images_basename+"_"+str(k_ref).zfill(2)+".vti",
                element=fe)
            DYI0 = myFEniCS.ExprGradYIm3(
                filename=images_folder+"/"+images_basename+"_"+str(k_ref).zfill(2)+".vti",
                element=fe)
            DZI0 = myFEniCS.ExprGradZIm3(
                filename=images_folder+"/"+images_basename+"_"+str(k_ref).zfill(2)+".vti",
                element=fe)
    else:
        assert (0), "images_dimension must be 2 or 3. Aborting."
    I0_norm = dolfin.assemble(I0**2 * dX)**(1./2)

    print "Defining functions…"
    penalty = dolfin.Constant(penalty)
    U = dolfin.Function(
        fs,
        name="displacement")
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
        DXI1 = myFEniCS.ExprGradXDefIm2(
            U=U,
            element=fe)
        DYI1 = myFEniCS.ExprGradYDefIm2(
            U=U,
            element=fe)
        if (use_I0_tangent):
            a =     penalty  * dolfin.inner(dolfin.dot(dolfin.as_vector((DXI0, DYI0)), ddU),
                                            dolfin.dot(dolfin.as_vector((DXI0, DYI0)),   V))*dX\
              + (1.-penalty) * dolfin.inner(dolfin.grad(ddU),
                                            dolfin.grad(  V))*dX
        else:
            a =     penalty  * dolfin.inner(dolfin.dot(dolfin.as_vector((DXI1, DYI1)), ddU),
                                            dolfin.dot(dolfin.as_vector((DXI1, DYI1)),   V))*dX\
              + (1.-penalty) * dolfin.inner(dolfin.grad(ddU),
                                            dolfin.grad(  V))*dX
        b =     penalty  * dolfin.inner(I0-I1,
                                        dolfin.dot(dolfin.as_vector((DXI1, DYI1)), V))*dX\
          - (1.-penalty) * dolfin.inner(dolfin.grad(U),
                                        dolfin.grad(V))*dX
    elif (images_dimension == 3):
        I1 = myFEniCS.ExprDefIm3(
            U=U,
            element=fe)
        DXI1 = myFEniCS.ExprGradXDefIm3(
            U=U,
            element=fe)
        DYI1 = myFEniCS.ExprGradYDefIm3(
            U=U,
            element=fe)
        DZI1 = myFEniCS.ExprGradZDefIm3(
            U=U,
            element=fe)
        if (use_I0_tangent):
            a =     penalty  * dolfin.inner(dolfin.dot(dolfin.as_vector((DXI0, DYI0, DZI0)), ddU),
                                            dolfin.dot(dolfin.as_vector((DXI0, DYI0, DZI0)),   V))*dX\
              + (1.-penalty) * dolfin.inner(dolfin.grad(ddU),
                                            dolfin.grad(  V))*dX
        else:
            a =     penalty  * dolfin.inner(dolfin.dot(dolfin.as_vector((DXI1, DYI1, DZI1)), ddU),
                                            dolfin.dot(dolfin.as_vector((DXI1, DYI1, DZI1)),   V))*dX\
              + (1.-penalty) * dolfin.inner(dolfin.grad(ddU),
                                            dolfin.grad(  V))*dX
        b =     penalty  * dolfin.inner(I0-I1,
                                        dolfin.dot(dolfin.as_vector((DXI1, DYI1, DZI1)), V))*dX\
          - (1.-penalty) * dolfin.inner(dolfin.grad(U),
                                        dolfin.grad(V))*dX
    else:
        assert (0), "images_dimension must be 2 or 3. Aborting."

    # linear system
    A = None
    B = None
    if (use_I0_tangent):
        A = dolfin.assemble(a, tensor=A)

    print "Checking number of frames…"
    n_frames = len(glob.glob(images_folder+"/"+images_basename+"_*.vti"))
    print "n_frames = " + str(n_frames)

    assert (abs(k_ref) < n_frames)
    k_ref = k_ref%n_frames

    n_iter_tot = 0
    for forward_or_backward in ["forward", "backward"]: # not sure if this still works…
        print "forward_or_backward = " + forward_or_backward

        if (forward_or_backward == "forward"):
            k_next_frames = range(k_ref+1, n_frames, +1)
        elif (forward_or_backward == "backward"):
            k_next_frames = range(k_ref-1, -1, -1)
        print "k_next_frames = " + str(k_next_frames)

        for k_next_frame in k_next_frames:
            print "k_next_frame = " + str(k_next_frame)

            if (print_iterations):
                if (os.path.exists(working_folder+"/"+working_basename+"-frame="+str(k_next_frame).zfill(2)+".pdf")):
                    os.remove(working_folder+"/"+working_basename+"-frame="+str(k_next_frame).zfill(2)+".pdf")
                file_dat_iter = open(working_folder+"/"+working_basename+"-frame="+str(k_next_frame).zfill(2)+".dat", "w")

                file_pvd_iter = dolfin.File(working_folder+"/"+working_basename+"-frame="+str(k_next_frame).zfill(2)+"_.pvd")
                for vtu in glob.glob(working_folder+"/"+working_basename+"-frame="+str(k_next_frame).zfill(2)+"_*.vtu"):
                    os.remove(vtu)
                file_pvd_iter << (U, 0.)

            print "Loading image and image gradient…"
            if (images_dimension == 2):
                I1.init_image(  filename=images_folder+"/"+images_basename+"_"+str(k_next_frame).zfill(2)+".vti")
                DXI1.init_image(filename=images_folder+"/"+images_basename+"_"+str(k_next_frame).zfill(2)+".vti")
                DYI1.init_image(filename=images_folder+"/"+images_basename+"_"+str(k_next_frame).zfill(2)+".vti")
            elif (images_dimension == 3):
                I1.init_image(  filename=images_folder+"/"+images_basename+"_"+str(k_next_frame).zfill(2)+".vti")
                DXI1.init_image(filename=images_folder+"/"+images_basename+"_"+str(k_next_frame).zfill(2)+".vti")
                DYI1.init_image(filename=images_folder+"/"+images_basename+"_"+str(k_next_frame).zfill(2)+".vti")
                DZI1.init_image(filename=images_folder+"/"+images_basename+"_"+str(k_next_frame).zfill(2)+".vti")
            else:
                assert (0), "images_dimension must be 2 or 3. Aborting."

            print "Running registration…"
            k_iter = 0
            relax = relax_init
            while (True):
                print "k_iter = " + str(k_iter)
                n_iter_tot += 1

                # linear system
                if not (use_I0_tangent):
                    A = dolfin.assemble(a, tensor=A)
                #print "A = " + str(A.array())
                if (k_iter > 0):
                    if (k_iter == 1):
                        B_old = B.copy()
                    elif (k_iter > 1):
                        B_old[:] = B[:]
                    #print "B_old = " + str(B_old[0])
                B = dolfin.assemble(b, tensor=B)
                #print "B = " + str(B[0])
                dolfin.solve(A, dU.vector(), B)
                #print "dU = " + str(dU.vector().array())

                # relaxation
                if (relax_type == "aitken") and (k_iter > 0):
                    if (k_iter == 1):
                        dB = B - B_old
                    elif (k_iter > 1):
                        dB[:] = B[:] - B_old[:]
                    relax *= (-1.) * B_old.inner(dB) / dB.inner(dB)
                print "relax = " + str(relax)

                # solution update
                U.vector()[:] += relax * dU.vector()[:]
                #U.vector().axpy(relax, dU.vector())

                if (print_iterations):
                    #print "U = " + str(U.vector().array())
                    file_pvd_iter << (U, float(k_iter+1))

                # displacement error
                dU_max = numpy.linalg.norm(dU.vector().array(), ord=numpy.Inf)
                U_max = numpy.linalg.norm(U.vector().array(), ord=numpy.Inf)
                err_disp_abs = max(dU_max, U_max)
                U_norm = numpy.linalg.norm(U.vector().array())
                err_disp_rel = dU_max/U_norm
                print "err_disp_abs = " + str(err_disp_abs)
                print "err_disp_rel = " + str(err_disp_rel)

                # image error
                I1I0_norm = dolfin.assemble((I1-I0)**2 * dX)**(1./2)
                err_im = I1I0_norm/I0_norm
                print "err_im = " + str(err_im)

                if (print_iterations):
                    file_dat_iter.write(" ".join([str(val) for val in [k_iter, err_disp_abs, err_disp_rel, err_im]])+"\n")

                # exit test
                if (err_disp_rel < tol_disp) or (err_disp_abs < tol_disp):
                    print "Nonlinear solver converged…"
                    success = True
                    break

                if (k_iter >= n_iter_max-1):
                    print "Warning! Nonlinear solver failed to converge…"
                    success = False
                    break

                # increment counter
                k_iter += 1

            if (print_iterations):
                os.remove(working_folder+"/"+working_basename+"-frame="+str(k_next_frame).zfill(2)+"_.pvd")
                file_dat_iter.close()
                os.system("gnuplot -e \"set terminal pdf; set output '"+working_folder+"/"+working_basename+"-frame="+str(k_next_frame).zfill(2)+".pdf'; set logscale y; plot '"+working_folder+"/"+working_basename+"-frame="+str(k_next_frame).zfill(2)+".dat' using 1:3\"")

            if not (success) and not (continue_after_fail):
                break

            print "Printing solution…"
            file_pvd << (U, float(k_next_frame))

        if not (success) and not (continue_after_fail):
            break

    print "n_iter_tot = " + str(n_iter_tot)

    os.remove(working_folder+"/"+working_basename+"_.pvd")

    return success
