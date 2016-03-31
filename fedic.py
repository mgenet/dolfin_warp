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
        images_k_ref=0,
        images_zfill=2,
        mesh_degree=1,
        penalty_type="const", # const, optim
        penalty_init=0.9,
        penalty_n_iter_max=10,
        tangent_type="Idef", # Idef, Iref, Idef_old
        relax_type="const", # const, aitken, manual
        relax_init=1.0,
        relax_n_iter=10,
        tol_res=None,
        tol_disp=None,
        n_iter_max=100,
        print_iterations=0,
        continue_after_fail=0,
        verbose=1):

    print "Loading mesh…"
    mesh_filename = mesh_folder+"/"+mesh_basename+".xml"
    assert os.path.exists(mesh_filename), "No mesh in "+mesh_filename+". Aborting."
    mesh = dolfin.Mesh(mesh_filename)
    dX = dolfin.dx(mesh)
    mesh_volume = dolfin.assemble(dolfin.Constant(1)*dX)

    print "Computing quadrature degree for images…"
    ref_image_filename = images_folder+"/"+images_basename+"_"+str(images_k_ref).zfill(images_zfill)+".vti"
    images_quadrature = myFEniCS.compute_quadrature_degree(
        image_filename=ref_image_filename,
        mesh=mesh)
    #images_quadrature = 1
    print "images_quadrature = " + str(images_quadrature)

    print "Loading reference image…"
    images_dimension = myVTK.computeImageDimensionality(
        filename=ref_image_filename,
        verbose=0)
    assert (images_dimension in (2,3)), "images_dimension must be 2 or 3. Aborting."
    fe = dolfin.FiniteElement(
        family="Quadrature",
        cell=mesh.ufl_cell(),
        degree=images_quadrature,
        quad_scheme="default")
    ve = dolfin.VectorElement(
        family="Quadrature",
        cell=mesh.ufl_cell(),
        degree=images_quadrature,
        quad_scheme="default")
    if (images_dimension == 2):
        Iref = myFEniCS.ExprIm2(
            filename=ref_image_filename,
            element=fe)
        DIref = myFEniCS.ExprGradIm2(
            filename=ref_image_filename,
            element=ve)
    elif (images_dimension == 3):
        Iref = myFEniCS.ExprIm3(
            filename=ref_image_filename,
            element=fe)
        DIref = myFEniCS.ExprGradIm3(
            filename=ref_image_filename,
            element=ve)
    Iref_norm = (dolfin.assemble(Iref**2 * dX, form_compiler_parameters={'quadrature_degree':images_quadrature})/mesh_volume)**(1./2)
    assert (Iref_norm > 0.), "Iref_norm = "+str(Iref_norm)+" <= 0. Aborting."
    print "Iref_norm = " + str(Iref_norm)

    print "Defining functions…"
    vfs = dolfin.VectorFunctionSpace(
        mesh=mesh,
        family="Lagrange",
        degree=mesh_degree)
    U = dolfin.Function(
        vfs,
        name="displacement")
    U_old = dolfin.Function(
        vfs,
        name="previous displacement")
    dU = dolfin.Function(
        vfs,
        name="displacement correction")
    dU_ = dolfin.TrialFunction(vfs)
    dV_ = dolfin.TestFunction(vfs)

    print "Defining variational forms…"
    penalty = dolfin.Constant(penalty)
    if (images_dimension == 2):
        Idef = myFEniCS.ExprDefIm2(
            U=U,
            filename=ref_image_filename,
            element=fe)
        DIdef = myFEniCS.ExprGradDefIm2(
            U=U,
            filename=ref_image_filename,
            element=ve)
    elif (images_dimension == 3):
        Idef = myFEniCS.ExprDefIm3(
            U=U,
            filename=ref_image_filename,
            element=fe)
        DIdef = myFEniCS.ExprGradDefIm3(
            U=U,
            filename=ref_image_filename,
            element=ve)
    if (tangent_type == "Iref"):
        a =     penalty  * dolfin.inner(dolfin.dot(DIref, dU_),
                                        dolfin.dot(DIref, dV_))*dX
    elif (tangent_type == "Idef") or (tangent_type == "Idef_old"):
        a =     penalty  * dolfin.inner(dolfin.dot(DIdef, dU_),
                                        dolfin.dot(DIdef, dV_))*dX
    a +=    (1.-penalty) * dolfin.inner(dolfin.grad(dU_),
                                        dolfin.grad(dV_))*dX
    b =         penalty  * dolfin.inner(Iref-Idef,
                                        dolfin.dot(DIdef, dV_))*dX\
          - (1.-penalty) * dolfin.inner(dolfin.grad(U),
                                        dolfin.grad(dV_))*dX
    b0 =                   dolfin.inner(Iref,
                                        dolfin.dot(DIref, dV_))*dX
    B0 = dolfin.assemble(b0, form_compiler_parameters={'quadrature_degree':images_quadrature})
    B0_norm = numpy.linalg.norm(B0)
    assert (B0_norm > 0.), "B0_norm = "+str(B0_norm)+" <= 0. Aborting."
    print "B0_norm = " + str(B0_norm)

    # linear system
    A = None
    if (tangent_type == "Iref"):
        A = dolfin.assemble(a, tensor=A, form_compiler_parameters={'quadrature_degree':images_quadrature})
    B = None

    print "Checking number of frames…"
    n_frames = len(glob.glob(images_folder+"/"+images_basename+"_*.vti"))
    #n_frames = 2
    assert (n_frames > 1), "n_frames = "+str(n_frames)+" <= 1. Aborting."
    print "n_frames = " + str(n_frames)

    assert (abs(images_k_ref) < n_frames), "abs(images_k_ref) = "+str(images_k_ref)+" >= n_frames. Aborting."
    images_k_ref = images_k_ref%n_frames
    print "images_k_ref = " + str(images_k_ref)

    print "Printing initial solution…"
    if not os.path.exists(working_folder):
        os.mkdir(working_folder)
    pvd_basename = working_folder+"/"+working_basename+"_"
    file_pvd = dolfin.File(pvd_basename+".pvd")
    for vtu_filename in glob.glob(pvd_basename+"*.vtu"):
        os.remove(vtu_filename)
    file_pvd << (U, float(images_k_ref))

    n_iter_tot = 0
    global_success = True
    for forward_or_backward in ["forward", "backward"]: # not sure if this still works…
        print "forward_or_backward = " + forward_or_backward

        if (forward_or_backward == "forward"):
            k_frames = range(images_k_ref+1, n_frames, +1)
        elif (forward_or_backward == "backward"):
            k_frames = range(images_k_ref-1, -1, -1)
        print "k_frames = " + str(k_frames)

        if (forward_or_backward == "backward"):
            U.vector()[:] = 0.
            Idef.init_image( filename=ref_image_filename)
            DIdef.init_image(filename=ref_image_filename)

        for k_frame in k_frames:
            print "k_frame = " + str(k_frame)

            if (print_iterations):
                frame_basename = working_folder+"/"+working_basename+"-frame="+str(k_frame).zfill(images_zfill)
                if (os.path.exists(frame_basename+".pdf")):
                    os.remove(frame_basename+".pdf")
                file_dat_frame = open(frame_basename+".dat", "w")
                file_dat_frame.write("#k_iter B_norm err_res relax dU_norm U_norm err_disp dI_norm err_im err_im_rel\n")

                file_pvd_frame = dolfin.File(frame_basename+"_.pvd")
                for vtu_filename in glob.glob(frame_basename+"_*.vtu"):
                    os.remove(vtu_filename)
                file_pvd_frame << (U, 0.)

            # linear system: matrix
            if (tangent_type == "Idef_old"):
                A = dolfin.assemble(a, tensor=A, form_compiler_parameters={'quadrature_degree':images_quadrature})
                #print "A = " + str(A.array())
                #A_norm = numpy.linalg.norm(A.array())
                #print "A_norm = " + str(A_norm)

            print "Loading image and image gradient…"
            image_filename = images_folder+"/"+images_basename+"_"+str(k_frame).zfill(images_zfill)+".vti"
            Idef.init_image( filename=image_filename)
            DIdef.init_image(filename=image_filename)

            if (print_iterations):
                U_old.vector()[:] = U.vector()[:]
                U.vector()[:] = 0.
                dI_norm = (dolfin.assemble((Idef-Iref)**2 * dX, form_compiler_parameters={'quadrature_degree':images_quadrature})/mesh_volume)**(1./2)
                err_im = dI_norm/Iref_norm
                file_dat_frame.write(" ".join([str(val) for val in [-2, None, None, None, None, None, None, dI_norm, err_im, None]])+"\n")
                U.vector()[:] = U_old.vector()[:]
                dI_norm = (dolfin.assemble((Idef-Iref)**2 * dX, form_compiler_parameters={'quadrature_degree':images_quadrature})/mesh_volume)**(1./2)
                err_im = dI_norm/Iref_norm
                file_dat_frame.write(" ".join([str(val) for val in [-1, None, None, None, None, None, None, dI_norm, err_im, None]])+"\n")

            print "Running registration…"
            k_iter = 0
            success = False
            while (True):
                print "k_iter = " + str(k_iter)
                n_iter_tot += 1

                # linear system: matrix
                if (tangent_type == "Idef"):
                    A = dolfin.assemble(a, tensor=A, form_compiler_parameters={'quadrature_degree':images_quadrature})
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
                B = dolfin.assemble(b, tensor=B, form_compiler_parameters={'quadrature_degree':images_quadrature})
                #print "B = " + str(B.array())

                # residual error
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
                elif (relax_type == "manual"):
                    B_relax = numpy.empty(relax_n_iter)
                    for relax_k in xrange(relax_n_iter):
                        #print "relax_k = " + str(relax_k)
                        relax = float(relax_k+1)/relax_n_iter
                        U.vector()[:] = U_old.vector()[:] + relax * dU.vector()[:]
                        B = dolfin.assemble(b, tensor=B, form_compiler_parameters={'quadrature_degree':images_quadrature})
                        B_relax[relax_k] = numpy.linalg.norm(B)
                        #print "B_relax = " + str(B_relax[relax_k])
                    #print "B_relax = " + str(B_relax)
                    if (print_iterations):
                        iter_basename = frame_basename+"-iter="+str(k_iter).zfill(3)
                        open(iter_basename+".dat", "w").write("\n".join([" ".join([str(val) for val in [float(relax_k+1)/relax_n_iter, B_relax[relax_k]]]) for relax_k in xrange(relax_n_iter)]))
                        os.system("gnuplot -e \"set terminal pdf; set output '"+iter_basename+".pdf'; plot '"+iter_basename+".dat' u 1:2 w l notitle\"")
                    relax_k = numpy.argmin(B_relax)
                    relax = float(relax_k+1)/relax_n_iter
                    print "relax = " + str(relax)
                else:
                    assert (0), "relax_type must be \"constant\", \"aitken\" or \"manual\". Aborting."

                # solution update
                U.vector()[:] = U_old.vector()[:] + relax * dU.vector()[:]

                if (print_iterations):
                    #print "U = " + str(U.vector().array())
                    file_pvd_frame << (U, float(k_iter+1))

                # displacement error
                dU_norm = numpy.linalg.norm(dU.vector().array())
                print "dU_norm = " + str(dU_norm)
                U_norm = numpy.linalg.norm(U.vector().array())
                assert (U_norm > 0.), "U_norm = "+str(U_norm)+" <= 0. Aborting."
                print "U_norm = " + str(U_norm)
                err_disp = dU_norm/U_norm
                print "err_disp = " + str(err_disp)

                # image error
                if (k_iter > 0):
                    dI_norm_old = dI_norm
                dI_norm = (dolfin.assemble((Idef-Iref)**2 * dX, form_compiler_parameters={'quadrature_degree':images_quadrature})/mesh_volume)**(1./2)
                #print "dI_norm = " + str(dI_norm)
                err_im = dI_norm/Iref_norm
                print "err_im = " + str(err_im)
                if (k_iter == 0):
                    err_im_rel = 1.
                else:
                    err_im_rel = abs(dI_norm-dI_norm_old)/dI_norm_old
                    print "err_im_rel = " + str(err_im_rel)

                if (print_iterations):
                    file_dat_frame.write(" ".join([str(val) for val in [k_iter, B_norm, err_res, relax, dU_norm, U_norm, err_disp, dI_norm, err_im, err_im_rel]])+"\n")

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
                os.remove(frame_basename+"_.pvd")
                file_dat_frame.close()
                os.system("gnuplot -e \"set terminal pdf; set output '"+frame_basename+".pdf'; set key box textcolor variable; set grid; set logscale y; set yrange [1e-4:1e1]; plot '"+frame_basename+".dat' u 1:3 pt 1 lw 3 title 'err_res', '' u 1:4 pt 1 lw 3 title 'relax', '' u 1:7 pt 1 lw 3 title 'err_disp', '' using 1:9 pt 1 lw 3 title 'err_im', '' using 1:10 pt 1 lw 3 title 'err_im_rel', "+str(tol_res)+" lt -1 notitle\"")

            if not (success) and not (continue_after_fail):
                break

            print "Printing solution…"
            file_pvd << (U, float(k_frame))

        if not (success) and not (continue_after_fail):
            break

    print "n_iter_tot = " + str(n_iter_tot)

    os.remove(pvd_basename+".pvd")

    return global_success
