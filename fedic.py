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

dolfin.parameters['form_compiler']['cpp_optimize_flags'] = '-O3'
dolfin.parameters["form_compiler"]["cpp_optimize"] = True
dolfin.parameters["form_compiler"]["optimize"] = True

def print_str(tab, string):
    print  " | "*tab + string

def print_var(tab, name, val):
    print " | "*tab + name + " = " + str(val)

def print_sci(tab, name, val):
    print " | "*tab + name.ljust(13) + " = " + format(val,".4e")

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
        images_quadrature=None,
        mesh_degree=1,
        penalty=0.9,
        tangent_type="Idef", # Idef, Idef-wHess, Iold, Iref
        residual_type="Iref", # Iref, Iold, Iref-then-Iold
        relax_type="const", # const, aitken, manual
        relax_init=1.0,
        relax_n_iter=10,
        tol_res=None,
        tol_dU=None,
        tol_im=None,
        n_iter_max=100,
        continue_after_fail=0,
        print_iterations=0):

    tab = 0
    print_str(tab,"Loading mesh…")
    mesh_filename = mesh_folder+"/"+mesh_basename+".xml"
    assert os.path.exists(mesh_filename), "No mesh in "+mesh_filename+". Aborting."
    mesh = dolfin.Mesh(mesh_filename)
    dX = dolfin.dx(mesh)
    mesh_volume = dolfin.assemble(dolfin.Constant(1)*dX)

    print_str(tab,"Checking number of frames…")
    n_frames = len(glob.glob(images_folder+"/"+images_basename+"_"+"[0-9]"*images_zfill+".vti"))
    #n_frames = 2
    assert (n_frames > 1), "n_frames = "+str(n_frames)+" <= 1. Aborting."
    print_var(tab+1,"n_frames",n_frames)

    assert (abs(images_k_ref) < n_frames), "abs(images_k_ref) = "+str(images_k_ref)+" >= n_frames. Aborting."
    images_k_ref = images_k_ref%n_frames
    print_var(tab+1,"images_k_ref",images_k_ref)

    print_str(tab,"Computing quadrature degree for images…")
    ref_image_filename = images_folder+"/"+images_basename+"_"+str(images_k_ref).zfill(images_zfill)+".vti"
    if (images_quadrature is None):
        images_quadrature = myFEniCS.compute_quadrature_degree(
            image_filename=ref_image_filename,
            mesh=mesh,
            verbose=0)
    print_var(tab+1,"images_quadrature",images_quadrature)
    dolfin.parameters["form_compiler"]["quadrature_degree"] = images_quadrature

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
    Iref_norm = (dolfin.assemble(Iref**2 * dX)/mesh_volume)**(1./2)
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
    U_norm = 0.
    Uold = dolfin.Function(
        vfs,
        name="previous displacement")
    Uold_norm = 0.
    dU = dolfin.Function(
        vfs,
        name="displacement correction")
    dU_ = dolfin.TrialFunction(vfs)
    dV_ = dolfin.TestFunction(vfs)

    print_str(tab,"Printing initial solution…")
    if not os.path.exists(working_folder):
        os.mkdir(working_folder)
    pvd_basename = working_folder+"/"+working_basename+"_"
    file_pvd = dolfin.File(pvd_basename+".pvd")
    for vtu_filename in glob.glob(pvd_basename+"*.vtu"):
        os.remove(vtu_filename)
    file_pvd << (U, float(images_k_ref))

    print_str(tab,"Defining variational forms…")
    penalty = dolfin.Constant(penalty)
    if (images_dimension == 2):
        Idef = myFEniCS.ExprDefIm2(
            U=U,
            #filename=ref_image_filename,
            element=fe)
        DIdef = myFEniCS.ExprGradDefIm2(
            U=U,
            #filename=ref_image_filename,
            element=ve)
        if ("-wHess" in tangent_type):
            DDIdef = myFEniCS.ExprHessDefIm2(
                U=U,
                #filename=ref_image_filename,
                element=te)
        Iold = myFEniCS.ExprDefIm2(
            U=Uold,
            #filename=ref_image_filename,
            element=fe)
        DIold = myFEniCS.ExprGradDefIm2(
            U=Uold,
            #filename=ref_image_filename,
            element=ve)
    elif (images_dimension == 3):
        Idef = myFEniCS.ExprDefIm3(
            U=U,
            #filename=ref_image_filename,
            element=fe)
        DIdef = myFEniCS.ExprGradDefIm3(
            U=U,
            #filename=ref_image_filename,
            element=ve)
        if ("-wHess" in tangent_type):
            DDIdef = myFEniCS.ExprHessDefIm3(
                U=U,
                filename=ref_image_filename,
                element=te)
        Iold = myFEniCS.ExprDefIm3(
            U=Uold,
            #filename=ref_image_filename,
            element=fe)
        DIold = myFEniCS.ExprGradDefIm3(
            U=Uold,
            #filename=ref_image_filename,
            element=ve)
    if (tangent_type.startswith("Idef")):
        a =       penalty  * dolfin.inner(dolfin.dot(DIdef, dU_),
                                          dolfin.dot(DIdef, dV_)) * dX
        if ("-wHess" in tangent_type):
            a +=  penalty  * (Iref-Idef) * dolfin.inner(dolfin.dot(DDIdef, dU_),
                                                                           dV_) * dX
    elif (tangent_type == "Iold"):
        a =       penalty  * dolfin.inner(dolfin.dot(DIold, dU_),
                                          dolfin.dot(DIold, dV_)) * dX
    elif (tangent_type == "Iref"):
        a =       penalty  * dolfin.inner(dolfin.dot(DIref, dU_),
                                          dolfin.dot(DIref, dV_)) * dX
    bdef =  -     penalty  * (Idef-Iref) * dolfin.dot(DIdef, dV_) * dX
    bold =  -     penalty  * (Idef-Iold) * dolfin.dot(DIdef, dV_) * dX
    a +=      (1.-penalty) * dolfin.inner(dolfin.grad(dU_),
                                          dolfin.grad(dV_)) * dX
    bdef += - (1.-penalty) * dolfin.inner(dolfin.grad( U ),
                                          dolfin.grad(dV_)) * dX
    bold += - (1.-penalty) * dolfin.inner(dolfin.grad( U ),
                                          dolfin.grad(dV_)) * dX
    b0 = dolfin.inner(Iref, dolfin.dot(DIref, dV_)) * dX
    B0 = dolfin.assemble(b0)
    res_norm0 = numpy.linalg.norm(B0)
    assert (res_norm0 > 0.), "res_norm0 = "+str(res_norm0)+" <= 0. Aborting."
    print_var(tab+1,"res_norm0",res_norm0)

    # linear system
    A = None
    if (tangent_type == "Iref"):
        A = dolfin.assemble(a, tensor=A)
    B = None

    print_str(tab,"Looping over frames…")
    n_iter_tot = 0
    global_success = True
    for forward_or_backward in ["forward", "backward"]:
        print_var(tab,"forward_or_backward",forward_or_backward)

        if (forward_or_backward == "forward"):
            k_frames_old = range(images_k_ref  , n_frames-1, +1)
            k_frames     = range(images_k_ref+1, n_frames  , +1)
        elif (forward_or_backward == "backward"):
            k_frames_old = range(images_k_ref  ,  0, -1)
            k_frames     = range(images_k_ref-1, -1, -1)
        print_var(tab,"k_frames",k_frames)

        if (forward_or_backward == "backward"):
            U.vector()[:] = 0.
            U_norm = 0.
            Uold.vector()[:] = 0.
            Uold_norm = 0.

        tab += 1
        success = True
        for (k_frame,k_frame_old) in zip(k_frames,k_frames_old):
            print_var(tab-1,"k_frame",k_frame)

            if (print_iterations):
                frame_basename = working_folder+"/"+working_basename+"-frame="+str(k_frame).zfill(images_zfill)
                if (os.path.exists(frame_basename+".pdf")):
                    os.remove(frame_basename+".pdf")
                file_dat_frame = open(frame_basename+".dat", "w")
                file_dat_frame.write("#k_iter res_norm res_err relax dU_norm U_norm dU_err im_diff im_err im_err_rel\n")

                file_pvd_frame = dolfin.File(frame_basename+"_.pvd")
                for vtu_filename in glob.glob(frame_basename+"_*.vtu"):
                    os.remove(vtu_filename)
                file_pvd_frame << (U, 0.)

            print_str(tab,"Loading image, image gradient and image hessian…")
            image_filename = images_folder+"/"+images_basename+"_"+str(k_frame).zfill(images_zfill)+".vti"
            Idef.init_image( filename=image_filename)
            DIdef.init_image(filename=image_filename)
            if ("-wHess" in tangent_type):
                DDIdef.init_image(filename=image_filename)
            image_filename = images_folder+"/"+images_basename+"_"+str(k_frame_old).zfill(images_zfill)+".vti"
            Iold.init_image( filename=image_filename)
            DIold.init_image(filename=image_filename)

            # linear system: matrix
            if (tangent_type == "Iold"):
                A = dolfin.assemble(a, tensor=A)
                #print_var(tab,"A",A.array())
                #A_norm = numpy.linalg.norm(A.array())
                #print_var(tab,"A_norm",A_norm)

            if (print_iterations):
                U.vector()[:] = 0.
                im_diff = (dolfin.assemble((Idef-Iref)**2 * dX)/mesh_volume)**(1./2)
                im_err = im_diff/Iref_norm
                file_dat_frame.write(" ".join([str(val) for val in [-2, None, None, None, None, None, None, im_diff, im_err, None]])+"\n")
                U.vector()[:] = Uold.vector()[:]
                im_diff = (dolfin.assemble((Idef-Iref)**2 * dX)/mesh_volume)**(1./2)
                im_err = im_diff/Iref_norm
                file_dat_frame.write(" ".join([str(val) for val in [-1, None, None, None, None, None, None, im_diff, im_err, None]])+"\n")

            print_str(tab,"Running registration…")
            tab += 1
            k_iter = 0
            if   (residual_type.startswith("Iref")):
                using_Iold_residual = False
            elif (residual_type.startswith("Iold")):
                using_Iold_residual = True
            while (True):
                print_var(tab-1,"k_iter",k_iter)
                n_iter_tot += 1

                # linear system: matrix
                if (tangent_type.startswith("Idef")):
                    A = dolfin.assemble(a, tensor=A)
                    #print_var(tab,"A",A.array())
                    #A_norm = numpy.linalg.norm(A.array())
                    #print_sci(tab,"A_norm",A_norm)

                # linear system: residual
                if (relax_type == "aitken"):
                    if (k_iter == 1):
                        B_old = B.copy()
                    elif (k_iter > 1):
                        B_old[:] = B[:]
                if (using_Iold_residual):
                    B = dolfin.assemble(bold, tensor=B)
                else:
                    B = dolfin.assemble(bdef, tensor=B)
                #print_var(tab,"B",B.array())

                # residual error
                res_norm = numpy.linalg.norm(B.array())
                #print_sci(tab,"res_norm",res_norm)
                res_err = res_norm/res_norm0
                print_sci(tab,"res_err",res_err)

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
                    print_sci(tab,"relax",relax)
                elif (relax_type == "manual"):
                    res_norm_relax = numpy.empty(relax_n_iter)
                    tab += 1
                    for relax_k in xrange(relax_n_iter):
                        print_var(tab-1,"relax_k",relax_k)
                        U.vector()[:] += (1./relax_n_iter) * dU.vector()[:]
                        B = dolfin.assemble(b, tensor=B)
                        res_norm_relax[relax_k] = numpy.linalg.norm(B)
                        print_sci(tab,"res_norm_relax",res_norm_relax[relax_k])
                    U.vector()[:] -= dU.vector()[:]
                    #print_var(tab,"res_norm_relax",res_norm_relax)
                    if (print_iterations):
                        iter_basename = frame_basename+"-iter="+str(k_iter).zfill(3)
                        open(iter_basename+".dat", "w").write("\n".join([" ".join([str(val) for val in [float(relax_k+1)/relax_n_iter, res_norm_relax[relax_k]]]) for relax_k in xrange(relax_n_iter)]))
                        os.system("gnuplot -e \"set terminal pdf; set output '"+iter_basename+".pdf'; plot '"+iter_basename+".dat' u 1:2 w l notitle\"")
                    relax_k = numpy.argmin(res_norm_relax)
                    relax = float(relax_k+1)/relax_n_iter
                    print_sci(tab,"relax",relax)
                else:
                    assert (0), "relax_type must be \"constant\", \"aitken\" or \"manual\". Aborting."

                # solution update
                U.vector()[:] += relax * dU.vector()[:]
                U_norm = numpy.linalg.norm(U.vector().array())

                if (print_iterations):
                    #print_var(tab,"U",U.vector().array())
                    file_pvd_frame << (U, float(k_iter+1))

                # displacement error
                dU_norm = numpy.linalg.norm(dU.vector().array())
                dU_err = dU_norm/max(Uold_norm,U_norm)
                print_sci(tab,"dU_err",dU_err)

                # image error
                if (k_iter > 0):
                    im_diff_old = im_diff
                im_diff = (dolfin.assemble((Idef-Iref)**2 * dX)/mesh_volume)**(1./2)
                #print_sci(tab,"im_diff",im_diff)
                im_err = im_diff/Iref_norm
                print_sci(tab,"im_err",im_err)
                if (k_iter == 0):
                    im_err_rel = 1.
                else:
                    im_err_rel = abs(im_diff-im_diff_old)/im_diff_old
                    print_sci(tab,"im_err_rel",im_err_rel)

                if (print_iterations):
                    file_dat_frame.write(" ".join([str(val) for val in [k_iter, res_norm, res_err, relax, dU_norm, U_norm, dU_err, im_diff, im_err, im_err_rel]])+"\n")

                # exit test
                success = True
                if (tol_res is not None) and (res_err > tol_res):
                    success = False
                if (tol_dU is not None) and (dU_err > tol_dU):
                    success = False
                if (tol_im is not None) and (im_err_rel > tol_im):
                    success = False

                # exit
                if (success):
                    print_str(tab,"Nonlinear solver converged…")
                    break

                if (k_iter == n_iter_max-1):
                    if (residual_type=="Iref-then-Iold") and not (using_Iold_residual):
                        print_str(tab,"Warning! Nonlinear solver failed to converge…using Iold instead of Iref. (k_frame = "+str(k_frame)+")")
                        using_Iold_residual = True
                        U.vector()[:] = Uold.vector()[:]
                        U_norm = Uold_norm
                        k_iter = 0
                        continue
                    else:
                        print_str(tab,"Warning! Nonlinear solver failed to converge… (k_frame = "+str(k_frame)+")")
                        global_success = False
                        break

                # increment counter
                k_iter += 1

            tab -= 1

            if (print_iterations):
                #os.remove(frame_basename+"_.pvd")
                file_dat_frame.close()
                os.system("gnuplot -e \"set terminal pdf; set output '"+frame_basename+".pdf'; set key box textcolor variable; set grid; set logscale y; set yrange [1e-4:1e1]; set ytics nomirror; set y2tics nomirror; plot '"+frame_basename+".dat' u 1:3 pt 1 lw 3 title 'res_err', '' u 1:7 pt 1 lw 3 title 'dU_err', '' using 1:9 pt 1 lw 3 title 'im_err', '' using 1:10 pt 1 lw 3 title 'im_err_rel', '' u 1:4 axis x1y2 pt 2 lw 3 title 'relax', "+str(tol_res or tol_dU or tol_im)+" lt -1 notitle\"")

            if not (success) and not (continue_after_fail):
                break

            # solution update
            Uold.vector()[:] = U.vector()[:]
            Uold_norm = U_norm

            print_str(tab,"Printing solution…")
            file_pvd << (U, float(k_frame))

        tab -= 1

        if not (success) and not (continue_after_fail):
            break

    print_var(tab,"n_iter_tot",n_iter_tot)

    #os.remove(pvd_basename+".pvd")

    return global_success
