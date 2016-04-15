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

def print_str(tab, string):
    print  " | "*tab + string

def print_var(tab, name, val):
    print " | "*tab + name + " = " + str(val)

def print_sci(tab, name, val):
    print " | "*tab + name.ljust(12) + " = " + format(val,".4e")

def fedic(
        working_folder,
        working_basename,
        images_folder,
        images_basename,
        mesh_folder,
        mesh_basename,
        images_k_ref=0,
        images_zfill=2,
        images_quadrature=None,
        mesh_degree=1,
        penalty=0.9,
        tangent_type="Idef", # Idef, Idef-wHess, Idef_old, Idef_old-wHess, Iref
        relax_type="const", # const, aitken, manual
        relax_init=1.0,
        relax_n_iter=10,
        tol_res=None,
        tol_disp=None,
        n_iter_max=100,
        print_iterations=0,
        continue_after_fail=0):

    tab = 0
    print_str(tab,"Loading mesh…")
    mesh_filename = mesh_folder+"/"+mesh_basename+".xml"
    assert os.path.exists(mesh_filename), "No mesh in "+mesh_filename+". Aborting."
    mesh = dolfin.Mesh(mesh_filename)
    dX = dolfin.dx(mesh)
    mesh_volume = dolfin.assemble(dolfin.Constant(1)*dX)

    print_str(tab,"Computing quadrature degree for images…")
    ref_image_filename = images_folder+"/"+images_basename+"_"+str(images_k_ref).zfill(images_zfill)+".vti"
    if (images_quadrature is None):
        images_quadrature = myFEniCS.compute_quadrature_degree(
            image_filename=ref_image_filename,
            mesh=mesh,
            verbose=0)
    print_var(tab+1,"images_quadrature",images_quadrature)

    print_str(tab,"Loading reference image…")
    images_dimension = myVTK.computeImageDimensionality(
        image_filename=ref_image_filename,
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
    te = dolfin.TensorElement(
        family="Quadrature",
        cell=mesh.ufl_cell(),
        degree=images_quadrature,
        quad_scheme="default")
    te._quad_scheme = "default"
    for k in xrange(images_dimension**2):
        te.sub_elements()[k]._quad_scheme = "default"
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
    print_var(tab+1,"Iref_norm",Iref_norm)

    print_str(tab,"Defining functions…")
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
    DU_old = dolfin.Function(
        vfs,
        name="previous displacement increment")
    dU = dolfin.Function(
        vfs,
        name="displacement correction")
    dU_ = dolfin.TrialFunction(vfs)
    dV_ = dolfin.TestFunction(vfs)

    print_str(tab,"Defining variational forms…")
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
        DDIdef = myFEniCS.ExprHessDefIm2(
            U=U,
            filename=ref_image_filename,
            element=te)
    elif (images_dimension == 3):
        Idef = myFEniCS.ExprDefIm3(
            U=U,
            filename=ref_image_filename,
            element=fe)
        DIdef = myFEniCS.ExprGradDefIm3(
            U=U,
            filename=ref_image_filename,
            element=ve)
        DDIdef = myFEniCS.ExprHessDefIm3(
            U=U,
            filename=ref_image_filename,
            element=te)
    if (tangent_type == "Iref"):
        a =     penalty  * dolfin.inner(dolfin.dot(DIref, dU_),
                                        dolfin.dot(DIref, dV_)) * dX
    elif (tangent_type.startswith("Idef")):
        a =     penalty  * dolfin.inner(dolfin.dot(DIdef, dU_),
                                        dolfin.dot(DIdef, dV_)) * dX
        if ("-wHess" in tangent_type):
           a += penalty  * (Idef-Iref) * dolfin.inner(dolfin.dot(DDIdef, dU_),
                                                                         dV_) * dX
    b =   -     penalty  * (Idef-Iref) * dolfin.dot(DIdef, dV_) * dX
    a +=    (1.-penalty) * dolfin.inner(dolfin.grad(dU_),
                                        dolfin.grad(dV_)) * dX
    b +=  - (1.-penalty) * dolfin.inner(dolfin.grad( U ),
                                        dolfin.grad(dV_)) * dX
    b0 = dolfin.inner(Iref, dolfin.dot(DIref, dV_)) * dX
    B0 = dolfin.assemble(b0, form_compiler_parameters={'quadrature_degree':images_quadrature})
    B0_norm = numpy.linalg.norm(B0)
    assert (B0_norm > 0.), "B0_norm = "+str(B0_norm)+" <= 0. Aborting."
    print_var(tab+1,"B0_norm",B0_norm)

    # linear system
    A = None
    if (tangent_type == "Iref"):
        A = dolfin.assemble(a, tensor=A, form_compiler_parameters={'quadrature_degree':images_quadrature})
    B = None

    print_str(tab,"Checking number of frames…")
    n_frames = len(glob.glob(images_folder+"/"+images_basename+"_"+"?"*images_zfill+".vti"))
    #n_frames = 2
    assert (n_frames > 1), "n_frames = "+str(n_frames)+" <= 1. Aborting."
    print_var(tab+1,"n_frames",n_frames)

    assert (abs(images_k_ref) < n_frames), "abs(images_k_ref) = "+str(images_k_ref)+" >= n_frames. Aborting."
    images_k_ref = images_k_ref%n_frames
    print_var(tab+1,"images_k_ref",images_k_ref)

    print_str(tab,"Printing initial solution…")
    if not os.path.exists(working_folder):
        os.mkdir(working_folder)
    pvd_basename = working_folder+"/"+working_basename+"_"
    file_pvd = dolfin.File(pvd_basename+".pvd")
    for vtu_filename in glob.glob(pvd_basename+"*.vtu"):
        os.remove(vtu_filename)
    file_pvd << (U, float(images_k_ref))

    print_str(tab,"Looping over frames…")
    n_iter_tot = 0
    global_success = True
    for forward_or_backward in ["forward", "backward"]: # not sure if this still works…
        print_var(tab,"forward_or_backward",forward_or_backward)

        if (forward_or_backward == "forward"):
            k_frames = range(images_k_ref+1, n_frames, +1)
        elif (forward_or_backward == "backward"):
            k_frames = range(images_k_ref-1, -1, -1)
        print_var(tab,"k_frames",k_frames)

        if (forward_or_backward == "backward"):
            U.vector()[:] = 0.
            U_old.vector()[:] = 0.
            DU_old.vector()[:] = 0.
            Idef.init_image( filename=ref_image_filename)
            DIdef.init_image(filename=ref_image_filename)
            DDIdef.init_image(filename=ref_image_filename)

        tab += 1
        for k_frame in k_frames:
            print_var(tab-1,"k_frame",k_frame)

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
            if (tangent_type.startswith("Idef_old")):
                A = dolfin.assemble(a, tensor=A, form_compiler_parameters={'quadrature_degree':images_quadrature})
                #print_var(tab,"A",A.array())
                #A_norm = numpy.linalg.norm(A.array())
                #print_var(tab,"A_norm",A_norm)

            print_str(tab,"Loading image and image gradient…")
            image_filename = images_folder+"/"+images_basename+"_"+str(k_frame).zfill(images_zfill)+".vti"
            Idef.init_image( filename=image_filename)
            DIdef.init_image(filename=image_filename)
            DDIdef.init_image(filename=image_filename)

            if (print_iterations):
                U.vector()[:] = 0.
                dI_norm = (dolfin.assemble((Idef-Iref)**2 * dX, form_compiler_parameters={'quadrature_degree':images_quadrature})/mesh_volume)**(1./2)
                err_im = dI_norm/Iref_norm
                file_dat_frame.write(" ".join([str(val) for val in [-2, None, None, None, None, None, None, dI_norm, err_im, None]])+"\n")
                U.vector()[:] = U_old.vector()[:]
                dI_norm = (dolfin.assemble((Idef-Iref)**2 * dX, form_compiler_parameters={'quadrature_degree':images_quadrature})/mesh_volume)**(1./2)
                err_im = dI_norm/Iref_norm
                file_dat_frame.write(" ".join([str(val) for val in [-1, None, None, None, None, None, None, dI_norm, err_im, None]])+"\n")

            print_str(tab,"Running registration…")
            tab += 1
            k_iter = 0
            success = False
            while (True):
                print_var(tab-1,"k_iter",k_iter)
                n_iter_tot += 1

                # linear system: matrix
                if (tangent_type.startswith("Idef")):
                    A = dolfin.assemble(a, tensor=A, form_compiler_parameters={'quadrature_degree':images_quadrature})
                    #print_var(tab,"A",A.array())
                    #A_norm = numpy.linalg.norm(A.array())
                    #print_sci(tab,"A_norm",A_norm)

                # linear system: residual
                if (relax_type == "aitken"):
                    if (k_iter == 1):
                        B_old = B.copy()
                    elif (k_iter > 1):
                        B_old[:] = B[:]
                B = dolfin.assemble(b, tensor=B, form_compiler_parameters={'quadrature_degree':images_quadrature})
                #print_var(tab,"B",B.array())

                # residual error
                B_norm = numpy.linalg.norm(B.array())
                #print_sci(tab,"B_norm",B_norm)
                err_res = B_norm/B0_norm
                print_sci(tab,"err_res",err_res)

                # linear system: solve
                dolfin.solve(A, dU.vector(), B)
                #print_var(tab,"dU",dU.vector().array())

                # relaxation
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
                    print_var(tab,"relax",relax)
                elif (relax_type == "manual"):
                    B_norm_relax = numpy.empty(relax_n_iter)
                    tab += 1
                    for relax_k in xrange(relax_n_iter):
                        print_var(tab-1,"relax_k",relax_k)
                        U.vector()[:] += (1./relax_n_iter) * dU.vector()[:]
                        B = dolfin.assemble(b, tensor=B, form_compiler_parameters={'quadrature_degree':images_quadrature})
                        B_norm_relax[relax_k] = numpy.linalg.norm(B)
                        print_sci(tab,"B_norm_relax",B_norm_relax[relax_k])
                    U.vector()[:] -= dU.vector()[:]
                    #print_var(tab,"B_norm_relax",B_norm_relax)
                    if (print_iterations):
                        iter_basename = frame_basename+"-iter="+str(k_iter).zfill(3)
                        open(iter_basename+".dat", "w").write("\n".join([" ".join([str(val) for val in [float(relax_k+1)/relax_n_iter, B_norm_relax[relax_k]]]) for relax_k in xrange(relax_n_iter)]))
                        os.system("gnuplot -e \"set terminal pdf; set output '"+iter_basename+".pdf'; plot '"+iter_basename+".dat' u 1:2 w l notitle\"")
                    relax_k = numpy.argmin(B_norm_relax)
                    relax = float(relax_k+1)/relax_n_iter
                    print_var(tab,"relax",relax)
                else:
                    assert (0), "relax_type must be \"constant\", \"aitken\" or \"manual\". Aborting."

                # solution update
                DU_old.vector()[:] += relax * dU.vector()[:]
                U.vector()[:] += relax * dU.vector()[:]

                if (print_iterations):
                    #print_var(tab,"U",U.vector().array())
                    file_pvd_frame << (U, float(k_iter+1))

                # displacement error
                dU_norm = numpy.linalg.norm(dU.vector().array())
                print_sci(tab,"dU_norm",dU_norm)
                U_norm = numpy.linalg.norm(U.vector().array())
                assert (U_norm > 0.), "U_norm = "+str(U_norm)+" <= 0. Aborting."
                print_sci(tab,"U_norm",U_norm)
                err_disp = dU_norm/U_norm
                print_sci(tab,"err_disp",err_disp)

                # image error
                if (k_iter > 0):
                    dI_norm_old = dI_norm
                dI_norm = (dolfin.assemble((Idef-Iref)**2 * dX, form_compiler_parameters={'quadrature_degree':images_quadrature})/mesh_volume)**(1./2)
                #print_sci(tab,"dI_norm",dI_norm)
                err_im = dI_norm/Iref_norm
                print_sci(tab,"err_im",err_im)
                if (k_iter == 0):
                    err_im_rel = 1.
                else:
                    err_im_rel = abs(dI_norm-dI_norm_old)/dI_norm_old
                    print_sci(tab,"err_im_rel",err_im_rel)

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
                    print_str(tab,"Nonlinear solver converged…")
                    break

                if (k_iter == n_iter_max-1):
                    global_success = False
                    print_str(tab,"Warning! Nonlinear solver failed to converge… (k_frame = "+str(k_frame)+")")
                    break

                # increment counter
                k_iter += 1

            tab -= 1

            if (print_iterations):
                os.remove(frame_basename+"_.pvd")
                file_dat_frame.close()
                os.system("gnuplot -e \"set terminal pdf; set output '"+frame_basename+".pdf'; set key box textcolor variable; set grid; set logscale y; set yrange [1e-4:1e1]; plot '"+frame_basename+".dat' u 1:3 pt 1 lw 3 title 'err_res', '' u 1:4 pt 1 lw 3 title 'relax', '' u 1:7 pt 1 lw 3 title 'err_disp', '' using 1:9 pt 1 lw 3 title 'err_im', '' using 1:10 pt 1 lw 3 title 'err_im_rel', "+str(tol_res)+" lt -1 notitle\"")

            if not (success) and not (continue_after_fail):
                break

            # solution update
            U_old.vector()[:] += DU_old.vector()[:]
            DU_old.vector()[:] = 0.

            print_str(tab,"Printing solution…")
            file_pvd << (U, float(k_frame))

        tab -= 1

        if not (success) and not (continue_after_fail):
            break

    print_var(tab,"n_iter_tot",n_iter_tot)

    os.remove(pvd_basename+".pvd")

    return global_success
