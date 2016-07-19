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
import time
import numpy
import os

import myVTKPythonLibrary as myVTK

import myFEniCSPythonLibrary as myFEniCS
from print_tools import *

########################################################################

dolfin.parameters['form_compiler']['cpp_optimize_flags'] = '-O3'
dolfin.parameters["form_compiler"]["cpp_optimize"] = True
dolfin.parameters["form_compiler"]["optimize"] = False # can't use that for "complex" mechanical models…

########################################################################

def fedic(
        working_folder,
        working_basename,
        images_folder,
        images_basename,
        images_zfill=2,
        images_n_frames=None,
        images_ref_frame=0,
        images_quadrature=None,
        images_expressions_type="cpp", # cpp, py
        images_dynamic_scaling=0,
        mesh=None,
        mesh_folder=None,
        mesh_basename=None,
        mesh_degree=1,
        regul_type="neo-hookean",
        regul=0.1,
        tangent_type="Idef", # Idef, Idef-wHess, Iold, Iref
        residual_type="Iref", # Iref, Iold, Iref-then-Iold
        relax_type="constant", # constant, aitken, manual
        relax_init=1.0,
        tol_res=None,
        tol_dU=None,
        tol_im=None,
        n_iter_max=100,
        continue_after_fail=0,
        print_iterations=0):

    tab = 0

    print_str(tab,"Checking number of frames…")
    if (images_n_frames is None):
        images_n_frames = len(glob.glob(images_folder+"/"+images_basename+"_"+"[0-9]"*images_zfill+".vti"))
    assert (images_n_frames > 1), "images_n_frames = "+str(images_n_frames)+" <= 1. Aborting."
    print_var(tab+1,"images_n_frames",images_n_frames)

    assert (abs(images_ref_frame) < images_n_frames), "abs(images_ref_frame) = "+str(images_ref_frame)+" >= images_n_frames. Aborting."
    images_ref_frame = images_ref_frame%images_n_frames
    print_var(tab+1,"images_ref_frame",images_ref_frame)

    print_str(tab,"Loading mesh…")
    assert (mesh is not None or ((mesh_folder is not None) and (mesh_basename is not None))), "Must provide a mesh (mesh = "+str(mesh)+") or a mesh file (mesh_folder = "+str(mesh_folder)+", mesh_basename = "+str(mesh_basename)+"). Aborting."
    if (mesh is None):
        mesh_filename = mesh_folder+"/"+mesh_basename+".xml"
        assert os.path.exists(mesh_filename), "No mesh in "+mesh_filename+". Aborting."
        mesh = dolfin.Mesh(mesh_filename)
    dX = dolfin.dx(mesh)
    mesh_volume = dolfin.assemble(dolfin.Constant(1)*dX)

    print_str(tab,"Computing quadrature degree for images…")
    ref_image_filename = images_folder+"/"+images_basename+"_"+str(images_ref_frame).zfill(images_zfill)+".vti"
    if (images_quadrature is None):
        images_quadrature = myFEniCS.compute_quadrature_degree_from_points_count(
            image_filename=ref_image_filename,
            mesh_filebasename=mesh_filebasename,
            verbose=1)
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
    te._quad_scheme = "default"                       # shouldn't be needed
    for k in xrange(images_dimension**2):             # shouldn't be needed
        te.sub_elements()[k]._quad_scheme = "default" # shouldn't be needed
    if (images_expressions_type == "cpp"):
        Iref = dolfin.Expression(
            cppcode=myFEniCS.get_ExprIm_cpp(
                im_dim=images_dimension,
                im_type="im",
                im_is_def=0),
            element=fe)
        Iref.init()
        Iref.init_image(ref_image_filename)
        DIref = dolfin.Expression(
            cppcode=myFEniCS.get_ExprIm_cpp(
                im_dim=images_dimension,
                im_type="grad",
                im_is_def=0),
            element=ve)
        DIref.init()
        DIref.init_image(ref_image_filename)
    elif (images_expressions_type == "py"):
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
    else:
        assert (0), "\"images_expressions_type\" (="+str(images_expressions_type)+") must be \"cpp\" or \"py\". Aborting."
    Iref_int = dolfin.assemble(Iref * dX)/mesh_volume
    Iref_norm = (dolfin.assemble(Iref**2 * dX)/mesh_volume)**(1./2)
    assert (Iref_norm > 0.), "Iref_norm = "+str(Iref_norm)+" <= 0. Aborting."
    print_var(tab+1,"Iref_int",Iref_int)
    print_var(tab+1,"Iref_norm",Iref_norm)

    print_str(tab,"Defining functions…")
    vfs = dolfin.VectorFunctionSpace(
        mesh=mesh,
        family="Lagrange",
        degree=mesh_degree)
    U = dolfin.Function(
        vfs,
        name="displacement")
    U.vector().zero()
    U_norm = 0.
    Uold = dolfin.Function(
        vfs,
        name="previous displacement")
    Uold.vector().zero()
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
    file_pvd << (U, float(images_ref_frame))

    if (print_iterations):
        for filename in glob.glob(working_folder+"/"+working_basename+"-frame=*.*"):
            os.remove(filename)

    print_str(tab,"Defining mechanical energy…")
    E     = dolfin.Constant(1.0)
    nu    = dolfin.Constant(0.0)
    kappa = E/3/(1-2*nu)         # = E/3 if nu = 0
    lmbda = E*nu/(1+nu)/(1-2*nu) # = 0   if nu = 0
    mu    = E/2/(1+nu)           # = E/2 if nu = 0
    C1    = mu/2
    C2    = mu/2
    D1    = kappa/2

    if   (regul_type == "laplacian"): # <- super bad
        e     = dolfin.sym(dolfin.grad(U))
        psi_m = (lmbda * dolfin.tr(e)**2 + 2*mu * dolfin.tr(e*e))/2
    elif (regul_type == "kirchhoff"): # <- pretty bad too
        I     = dolfin.Identity(images_dimension)
        F     = I + dolfin.grad(U)
        C     = F.T * F
        E     = (C - I)/2
        psi_m = (lmbda * dolfin.tr(E)**2 + 2*mu * dolfin.tr(E*E))/2
    elif (regul_type == "neo-hookean"):
        I     = dolfin.Identity(images_dimension)
        F     = I + dolfin.grad(U)
        J     = dolfin.det(F)
        C     = F.T * F
        Ic    = dolfin.tr(C)
        Ic0   = dolfin.tr(I)
        psi_m = C1 * (Ic - Ic0 - 2*dolfin.ln(J)) + D1 * (J**2 - 1 - 2*dolfin.ln(J))
    elif (regul_type == "mooney-rivlin"):
        I     = dolfin.Identity(images_dimension)
        F     = I + dolfin.grad(U)
        J     = dolfin.det(F)
        C     = F.T * F
        Ic    = dolfin.tr(C)
        Ic0   = dolfin.tr(I)
        IIc   = (dolfin.tr(C)**2 - dolfin.tr(C*C))/2
        IIc0  = (dolfin.tr(I)**2 - dolfin.tr(I*I))/2
        psi_m = (C1/2) * (Ic - Ic0 - 2*dolfin.ln(J)) + (C2/2) * (IIc - IIc0 - 4*dolfin.ln(J)) + D1 * (J**2 - 1 - 2*dolfin.ln(J))
    else:
        assert (0), "\"regul_type\" must be \"laplacian\", \"kirchhoff\", \"neo-hookean\", or \"mooney-rivlin\". Aborting."

    Dpsi_m  = dolfin.derivative( psi_m, U, dV_)
    DDpsi_m = dolfin.derivative(Dpsi_m, U, dU_)

    print_str(tab,"Defining deformed image…")
    scaling = [1.,0.]
    if (images_expressions_type == "cpp"):
        Idef = dolfin.Expression(
            cppcode=myFEniCS.get_ExprIm_cpp(
                im_dim=images_dimension,
                im_type="im",
                im_is_def=1),
            element=fe)
        Idef.init(U)
        DIdef = dolfin.Expression(
            cppcode=myFEniCS.get_ExprIm_cpp(
                im_dim=images_dimension,
                im_type="grad",
                im_is_def=1),
            element=ve)
        DIdef.init(U)
        if ("-wHess" in tangent_type):
            assert (0), "ToDo"
        Iold = dolfin.Expression(
            cppcode=myFEniCS.get_ExprIm_cpp(
                im_dim=images_dimension,
                im_type="im",
                im_is_def=1),
            element=fe)
        Iold.init(Uold)
        DIold = dolfin.Expression(
            cppcode=myFEniCS.get_ExprIm_cpp(
                im_dim=images_dimension,
                im_type="grad",
                im_is_def=1),
            element=ve)
        DIold.init(Uold)
    elif (images_expressions_type == "py"):
        if (images_dimension == 2):
            Idef = myFEniCS.ExprDefIm2(
                U=U,
                scaling=scaling,
                element=fe)
            DIdef = myFEniCS.ExprGradDefIm2(
                U=U,
                scaling=scaling,
                element=ve)
            if ("-wHess" in tangent_type):
                DDIdef = myFEniCS.ExprHessDefIm2(
                    U=U,
                    scaling=scaling,
                    element=te)
            Iold = myFEniCS.ExprDefIm2(
                U=Uold,
                scaling=scaling,
                element=fe)
            DIold = myFEniCS.ExprGradDefIm2(
                U=Uold,
                scaling=scaling,
                element=ve)
        elif (images_dimension == 3):
            Idef = myFEniCS.ExprDefIm3(
                U=U,
                scaling=scaling,
                element=fe)
            DIdef = myFEniCS.ExprGradDefIm3(
                U=U,
                scaling=scaling,
                element=ve)
            if ("-wHess" in tangent_type):
                DDIdef = myFEniCS.ExprHessDefIm3(
                    U=U,
                    scaling=scaling,
                    element=te)
            Iold = myFEniCS.ExprDefIm3(
                U=Uold,
                scaling=scaling,
                element=fe)
            DIold = myFEniCS.ExprGradDefIm3(
                U=Uold,
                scaling=scaling,
                element=ve)
    else:
        assert (0), "\"images_expressions_type\" (="+str(images_expressions_type)+") must be \"cpp\" or \"py\". Aborting."

    print_str(tab,"Defining correlation energy…")
    psi_c  = (Idef - Iref)**2/2
    Dpsi_c = (Idef - Iref) * dolfin.dot(DIdef, dV_)
    if (tangent_type.startswith("Idef")):
        DDpsi_c = dolfin.dot(DIdef, dU_) * dolfin.dot(DIdef, dV_)
        if ("-wHess" in tangent_type):
            DDpsi_c += (Idef - Iref) * dolfin.inner(dolfin.dot(DDIdef, dU_), dV_)
    elif (tangent_type == "Iold"):
        DDpsi_c = dolfin.dot(DIold, dU_) * dolfin.dot(DIold, dV_)
    elif (tangent_type == "Iref"):
        DDpsi_c = dolfin.dot(DIref, dU_) * dolfin.dot(DIref, dV_)

    print_str(tab,"Defining variational forms…")
    regul = dolfin.Constant(regul)
    psi =   (1.-regul) *   psi_c      + (regul) * psi_m

    a   =   (1.-regul) * DDpsi_c * dX + (regul) * DDpsi_m * dX
    b   = - (1.-regul) *  Dpsi_c * dX - (regul) *  Dpsi_m * dX

    psi_c_old  = (Idef - Iold)**2/2
    Dpsi_c_old = (Idef - Iold) * dolfin.dot(DIdef, dV_)
    b_old = - (1.-regul) *  Dpsi_c_old * dX + (regul) *  Dpsi_m * dX

    b0 = Iref * dolfin.dot(DIref, dV_) * dX
    B0 = dolfin.assemble(b0)
    res_norm0 = B0.norm("l2")
    assert (res_norm0 > 0.), "res_norm0 = "+str(res_norm0)+" <= 0. Aborting."
    print_var(tab+1,"res_norm0",res_norm0)

    A = None
    if (tangent_type == "Iref"):
        print_str(tab,"Assembly…")
        A = dolfin.assemble(a, tensor=A)
    B = None

    print_str(tab,"Looping over frames…")
    n_iter_tot = 0
    global_success = True
    for forward_or_backward in ["forward", "backward"]:
        print_var(tab,"forward_or_backward",forward_or_backward)

        if (forward_or_backward == "forward"):
            k_frames_old = range(images_ref_frame  , images_n_frames-1, +1)
            k_frames     = range(images_ref_frame+1, images_n_frames  , +1)
        elif (forward_or_backward == "backward"):
            k_frames_old = range(images_ref_frame  ,  0, -1)
            k_frames     = range(images_ref_frame-1, -1, -1)
        print_var(tab,"k_frames",k_frames)

        if (forward_or_backward == "backward"):
            U.vector().zero()
            U_norm = 0.
            Uold.vector().zero()
            Uold_norm = 0.

        tab += 1
        success = True
        for (k_frame,k_frame_old) in zip(k_frames,k_frames_old):
            print_var(tab-1,"k_frame",k_frame)

            if (print_iterations):
                frame_basename = working_folder+"/"+working_basename+"-frame="+str(k_frame).zfill(images_zfill)
                file_dat_frame = open(frame_basename+".dat", "w")
                file_dat_frame.write("#k_iter res_norm res_err relax dU_norm U_norm dU_err im_diff im_err\n")

                file_pvd_frame = dolfin.File(frame_basename+"_.pvd")
                file_pvd_frame << (U, 0.)

            print_str(tab,"Loading image, image gradient and image hessian…")
            image_filename = images_folder+"/"+images_basename+"_"+str(k_frame).zfill(images_zfill)+".vti"
            Idef.init_image(image_filename)
            DIdef.init_image(image_filename)
            if ("-wHess" in tangent_type):
                DDIdef.init_image(image_filename)
            image_filename = images_folder+"/"+images_basename+"_"+str(k_frame_old).zfill(images_zfill)+".vti"
            Iold.init_image(image_filename)
            DIold.init_image(image_filename)

            # linear system: matrix
            if (tangent_type == "Iold"):
                A = dolfin.assemble(a, tensor=A)
                #print_var(tab,"A",A.array())
                #A_norm = A.norm("l2")
                #print_var(tab,"A_norm",A_norm)

            if (print_iterations):
                U.vector().zero()
                im_diff = (dolfin.assemble(psi_c * dX)/mesh_volume)**(1./2)
                im_err = im_diff/Iref_norm
                file_dat_frame.write(" ".join([str(val) for val in [-2, None, None, None, None, None, None, im_diff, im_err, None]])+"\n")
                U.vector()[:] = Uold.vector()[:]
                im_diff = (dolfin.assemble(psi_c * dX)/mesh_volume)**(1./2)
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
                    #A_norm = A.norm("l2")
                    #print_sci(tab,"A_norm",A_norm)

                # linear system: residual
                if (relax_type == "aitken"):
                    if (k_iter == 1):
                        B_old = B.copy()
                    elif (k_iter > 1):
                        B_old[:] = B[:]
                if (using_Iold_residual):
                    B = dolfin.assemble(b_old, tensor=B)
                else:
                    B = dolfin.assemble(b, tensor=B)
                #print_var(tab,"B",B.array())

                # residual error
                res_norm = B.norm("l2")
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
                    if (print_iterations):
                        iter_basename = frame_basename+"-iter="+str(k_iter).zfill(3)
                        file_dat_iter = open(iter_basename+".dat", "w")

                    relax_list = numpy.arange(-1.0, 2.1, 0.1)
                    #relax_list = numpy.arange(0.1, 1.1, 0.1)
                    relax_vals = numpy.empty(len(relax_list))
                    tab += 1
                    for relax_k in xrange(len(relax_list)):
                        #print_var(tab-1,"relax_k",relax_k)
                        #print_sci(tab,"relax",relax_list[relax_k])

                        U.vector().axpy(+relax_list[relax_k], dU.vector())
                        B = dolfin.assemble(b, tensor=B)
                        relax_res_norm = B.norm("l2")
                        relax_res_dU = B.inner(dU.vector())
                        relax_res_dU = abs(relax_res_dU)

                        psi_c_int = dolfin.assemble(psi_c * dX)
                        psi_m_int = dolfin.assemble(psi_m * dX)
                        psi_int   = dolfin.assemble(psi   * dX)
                        #print_sci(tab,"psi_c_int",psi_c_int)
                        #print_sci(tab,"psi_m_int",psi_m_int)
                        #print_sci(tab,"psi_int"  ,psi_int  )

                        if (print_iterations):
                            file_dat_iter.write(" ".join([str(val) for val in [relax_list[relax_k],
                                                                               psi_c_int,
                                                                               psi_m_int,
                                                                               psi_int,
                                                                               relax_res_norm,
                                                                               relax_res_dU]])+"\n")

                        relax_vals[relax_k] = psi_int
                        U.vector().axpy(-relax_list[relax_k], dU.vector())
                    tab -= 1
                    #print_var(tab,"relax_vals",relax_vals)

                    if (print_iterations):
                        file_dat_iter.close()
                        os.system("gnuplot -e \"set terminal pdf; set output '"+iter_basename+".pdf'; plot '"+iter_basename+".dat' u 1:2 w l title 'psi_c_int'; plot '' u 1:3 w l title 'psi_m_int'; plot '' u 1:4 w l title 'psi_int'; plot '' u 1:5 w l title 'relax_res_norm'; plot '' u 1:6 w l title 'relax_res_dU'\"")

                    relax = relax_list[numpy.argmin(relax_vals)]
                    print_sci(tab,"relax",relax)
                else:
                    assert (0), "relax_type must be \"constant\", \"aitken\" or \"manual\". Aborting."

                # solution update
                U.vector().axpy(relax, dU.vector())
                U_norm = U.vector().norm("l2")

                if (print_iterations):
                    #print_var(tab,"U",U.vector().array())
                    file_pvd_frame << (U, float(k_iter+1))

                # displacement error
                dU_norm = dU.vector().norm("l2")
                if (dU_norm == 0.) and (Uold_norm == 0.) and (U_norm == 0.):
                    dU_err = 0.
                elif (Uold_norm == 0.):
                    dU_err = dU_norm/U_norm
                else:
                    dU_err = dU_norm/Uold_norm
                print_sci(tab,"dU_err",dU_err)

                # image error
                if (k_iter > 0):
                    im_diff_old = im_diff
                im_diff = (dolfin.assemble(psi_c * dX)/mesh_volume)**(1./2)
                #print_sci(tab,"im_diff",im_diff)
                im_err = im_diff/Iref_norm
                print_sci(tab,"im_err",im_err)

                if (print_iterations):
                    file_dat_frame.write(" ".join([str(val) for val in [k_iter,
                                                                        res_norm,
                                                                        res_err,
                                                                        relax,
                                                                        dU_norm,
                                                                        U_norm,
                                                                        dU_err,
                                                                        im_diff,
                                                                        im_err]])+"\n")

                # exit test
                success = True
                if (tol_res is not None) and (res_err > tol_res):
                    success = False
                if (tol_dU is not None) and (dU_err > tol_dU):
                    success = False
                if (tol_im is not None) and (im_err > tol_im):
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
                os.system("gnuplot -e \"set terminal pdf; set output '"+frame_basename+".pdf'; set key box textcolor variable; set grid; set logscale y; set yrange [1e-4:1e1]; plot '"+frame_basename+".dat' u 1:3 pt 1 lw 3 title 'res_err', '' u 1:7 pt 1 lw 3 title 'dU_err', '' using 1:9 pt 1 lw 3 title 'im_err', "+str(tol_res or tol_dU or tol_im)+" lt -1 notitle; unset logscale y; set yrange [*:*]; plot '' u 1:4 pt 1 lw 3 title 'relax'\"")

            if not (success) and not (continue_after_fail):
                break

            # solution update
            Uold.vector()[:] = U.vector()[:]
            Uold_norm = U_norm

            print_str(tab,"Printing solution…")
            file_pvd << (U, float(k_frame))

            if (images_dynamic_scaling):
                p = numpy.empty((2,2))
                q = numpy.empty(2)
                p[0,0] = dolfin.assemble(Idef**2 * dX)
                p[0,1] = dolfin.assemble(Idef * dX)
                p[1,0] = p[0,1]
                p[1,1] = 1.
                q[0] = dolfin.assemble(Idef*Iref * dX)
                q[1] = dolfin.assemble(Iref * dX)
                scaling[:] = numpy.linalg.solve(p,q)
                print_var(tab,"scaling", scaling)

                im_diff = (dolfin.assemble(psi_c * dX)/mesh_volume)**(1./2)
                im_err = im_diff/Iref_norm
                print_sci(tab,"im_err",im_err)

        tab -= 1

        if not (success) and not (continue_after_fail):
            break

    print_var(tab,"n_iter_tot",n_iter_tot)

    #os.remove(pvd_basename+".pvd")

    return global_success
