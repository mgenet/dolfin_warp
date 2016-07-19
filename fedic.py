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
import math
import numpy
import os
import time

import myVTKPythonLibrary as myVTK

import myFEniCSPythonLibrary as myFEniCS
from print_tools import *

########################################################################

dolfin.parameters["form_compiler"]["optimize"] = False # can't use that for "complex" mechanical models…
dolfin.parameters["form_compiler"]["cpp_optimize"] = True
# dolfin.parameters["num_threads"] = 8 # 2016-07-07: doesn't seem to work…

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
        relax_type="constant", # constant, aitken, gss
        relax_init=1.0,
        tol_res=None,
        tol_res_rel=None,
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
        mesh_filebasename = mesh_folder+"/"+mesh_basename
        mesh_filename = mesh_filebasename+"."+"xml"
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

    print_str(tab,"Loading reference image…")
    images_dimension = myVTK.computeImageDimensionality(
        image_filename=ref_image_filename,
        verbose=0)
    print_var(tab+1,"images_dimension",images_dimension)
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
        Iref.init_image(ref_image_filename)
        DIref = dolfin.Expression(
            cppcode=myFEniCS.get_ExprIm_cpp(
                im_dim=images_dimension,
                im_type="grad",
                im_is_def=0),
            element=ve)
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
    Iref_int = dolfin.assemble(Iref * dX, form_compiler_parameters={'quadrature_degree':images_quadrature})/mesh_V0
    Iref_norm = (dolfin.assemble(Iref**2 * dX, form_compiler_parameters={'quadrature_degree':images_quadrature})/mesh_V0)**(1./2)
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
    scaling = numpy.array([1.,0.])
    #scaling = dolfin.Constant([1.,0.])
    if (images_expressions_type == "cpp"):
        Idef = dolfin.Expression(
            cppcode=myFEniCS.get_ExprIm_cpp(
                im_dim=images_dimension,
                im_type="im",
                im_is_def=1),
            element=fe)
        Idef.init_disp(U)
        Idef.init_dynamic_scaling(scaling)
        DIdef = dolfin.Expression(
            cppcode=myFEniCS.get_ExprIm_cpp(
                im_dim=images_dimension,
                im_type="grad",
                im_is_def=1),
            element=ve)
        DIdef.init_disp(U)
        DIdef.init_dynamic_scaling(scaling)
        if ("-wHess" in tangent_type):
            assert (0), "ToDo"
        Iold = dolfin.Expression(
            cppcode=myFEniCS.get_ExprIm_cpp(
                im_dim=images_dimension,
                im_type="im",
                im_is_def=1),
            element=fe)
        Iold.init_disp(Uold)
        Iold.init_dynamic_scaling(scaling)
        DIold = dolfin.Expression(
            cppcode=myFEniCS.get_ExprIm_cpp(
                im_dim=images_dimension,
                im_type="grad",
                im_is_def=1),
            element=ve)
        DIold.init_disp(Uold)
        DIold.init_dynamic_scaling(scaling)
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
    psi_c   = (Idef - Iref)**2/2
    Dpsi_c  = (Idef - Iref) * dolfin.dot(DIdef, dV_)
    DDpsi_c = dolfin.dot(DIdef, dU_) * dolfin.dot(DIdef, dV_)
    if ("-wHess" in tangent_type):
        DDpsi_c += (Idef - Iref) * dolfin.inner(dolfin.dot(DDIdef, dU_), dV_)

    Dpsi_c_old  = (Idef - Iold) * dolfin.dot(DIdef, dV_)

    DDpsi_c_old = dolfin.dot(DIold, dU_) * dolfin.dot(DIold, dV_)
    DDpsi_c_ref = dolfin.dot(DIref, dU_) * dolfin.dot(DIref, dV_)

    b0 = Iref * dolfin.dot(DIref, dV_) * dX
    B0 = dolfin.assemble(b0, form_compiler_parameters={'quadrature_degree':images_quadrature})
    res_norm0 = B0.norm("l2")
    assert (res_norm0 > 0.), "res_norm0 = "+str(res_norm0)+" <= 0. Aborting."
    print_var(tab+1,"res_norm0",res_norm0)

    A = None
    if (tangent_type == "Iref"):
        print_str(tab,"Assembly…")
        A = dolfin.assemble((1.-regul_level) * DDpsi_c_ref * dX, tensor=A, form_compiler_parameters={'quadrature_degree':images_quadrature})
        A = dolfin.assemble((   regul_level) * DDpsi_m     * dX, tensor=A, add_values=True)
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
                file_dat_frame.write("#k_iter res_norm err_res err_res_rel relax dU_norm U_norm err_dU im_diff err_im\n")

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
                print_str(tab,"Assembly…")
                A = dolfin.assemble((1.-regul_level) * DDpsi_c_old * dX, tensor=A, form_compiler_parameters={'quadrature_degree':images_quadrature})
                A = dolfin.assemble((   regul_level) * DDpsi_m     * dX, tensor=A, add_values=True)
                #print_var(tab,"A",A.array())
                #A_norm = A.norm("l2")
                #print_var(tab,"A_norm",A_norm)

            if (print_iterations):
                U.vector().zero()
                im_diff = (dolfin.assemble(psi_c * dX, form_compiler_parameters={'quadrature_degree':images_quadrature})/mesh_V0)**(1./2)
                err_im = im_diff/Iref_norm
                file_dat_frame.write(" ".join([str(val) for val in [-2, None, None, None, None, None, None, im_diff, err_im, None]])+"\n")
                U.vector()[:] = Uold.vector()[:]
                im_diff = (dolfin.assemble(psi_c * dX, form_compiler_parameters={'quadrature_degree':images_quadrature})/mesh_V0)**(1./2)
                err_im = im_diff/Iref_norm
                file_dat_frame.write(" ".join([str(val) for val in [-1, None, None, None, None, None, None, im_diff, err_im, None]])+"\n")

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

                # linear system: matrix assembly
                if (tangent_type.startswith("Idef")):
                    print_str(tab,"Matrix assembly…")
                    A = dolfin.assemble((1.-regul_level) * DDpsi_c * dX, tensor=A, form_compiler_parameters={'quadrature_degree':images_quadrature})
                    A = dolfin.assemble((   regul_level) * DDpsi_m * dX, tensor=A, add_values=True)
                    #print_var(tab,"A",A.array())
                    #A_norm = A.norm("l2")
                    #print_sci(tab,"A_norm",A_norm)

                # linear system: residual assembly
                if (k_iter > 0):
                    if (k_iter == 1):
                        B_old = B.copy()
                    elif (k_iter > 1):
                        B_old[:] = B[:]
                    res_old_norm = res_norm
                print_str(tab,"Residual assembly…")
                if (using_Iold_residual):
                    B = dolfin.assemble(- (1.-regul_level) * Dpsi_c_old * dX, tensor=B, form_compiler_parameters={'quadrature_degree':images_quadrature})
                    B = dolfin.assemble(- (   regul_level) * Dpsi_m     * dX, tensor=B, add_values=True)
                else:
                    B = dolfin.assemble(- (1.-regul_level) * Dpsi_c * dX, tensor=B, form_compiler_parameters={'quadrature_degree':images_quadrature})
                    B = dolfin.assemble(- (   regul_level) * Dpsi_m * dX, tensor=B, add_values=True)
                #print_var(tab,"B",B.array())

                # residual error
                res_norm = B.norm("l2")
                #print_sci(tab,"res_norm",res_norm)
                err_res = res_norm/res_norm0
                print_sci(tab,"err_res",err_res)

                if (k_iter == 0):
                    err_res_rel = 1.
                else:
                    if (k_iter == 1):
                        dB = B - B_old
                    elif (k_iter > 1):
                        dB[:] = B[:] - B_old[:]
                    dres_norm = dB.norm("l2")
                    err_res_rel = dres_norm / res_old_norm
                    print_sci(tab,"err_res_rel",err_res_rel)

                # linear system: solve
                print_str(tab,"Solve…")
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
                        relax *= (-1.) * B_old.inner(dB) / dres_norm**2
                    print_sci(tab,"relax",relax)
                elif (relax_type == "gss"):
                    phi = (1 + math.sqrt(5)) / 2
                    relax_a = (1-phi)/(2-phi)
                    relax_b = 1/(2-phi)
                    need_update_c = True
                    need_update_d = True
                    relax_cur = 0.
                    relax_list = []
                    relax_vals = []
                    tab += 1
                    relax_k = 0
                    while (True):
                        print_var(tab-1,"relax_k",relax_k)
                        print_sci(tab,"relax_a",relax_a)
                        print_sci(tab,"relax_b",relax_b)
                        if (need_update_c):
                            relax_c = relax_b - (relax_b - relax_a) / phi
                            relax_list.append(relax_c)
                            print_sci(tab,"relax_c",relax_c)
                            U.vector().axpy(relax_c-relax_cur, dU.vector())
                            relax_cur = relax_c
                            relax_fc  = dolfin.assemble((1.-regul_level) * psi_c * dX, form_compiler_parameters={'quadrature_degree':images_quadrature})
                            print_sci(tab,"relax_fc",relax_fc)
                            relax_fc += dolfin.assemble((   regul_level) * psi_m * dX)
                            print_sci(tab,"relax_fc",relax_fc)
                            #print_sci(tab,"relax_fc",relax_fc)
                            if (numpy.isnan(relax_fc)):
                                relax_fc = float('+inf')
                                print_sci(tab,"relax_fc",relax_fc)
                            relax_vals.append(relax_fc)
                            #print_var(tab,"relax_list",relax_list)
                            #print_var(tab,"relax_vals",relax_vals)
                        if (need_update_d):
                            relax_d = relax_a + (relax_b - relax_a) / phi
                            relax_list.append(relax_d)
                            print_sci(tab,"relax_d",relax_d)
                            U.vector().axpy(relax_d-relax_cur, dU.vector())
                            relax_cur = relax_d
                            relax_fd  = dolfin.assemble((1.-regul_level) * psi_c * dX, form_compiler_parameters={'quadrature_degree':images_quadrature})
                            print_sci(tab,"relax_fd",relax_fd)
                            relax_fd += dolfin.assemble((   regul_level) * psi_m * dX)
                            print_sci(tab,"relax_fd",relax_fd)
                            #print_sci(tab,"relax_fd",relax_fd)
                            if (numpy.isnan(relax_fd)):
                                relax_fd = float('+inf')
                                print_sci(tab,"relax_fd",relax_fd)
                            relax_vals.append(relax_fd)
                            #print_var(tab,"relax_list",relax_list)
                            #print_var(tab,"relax_vals",relax_vals)
                        if (relax_fc < relax_fd):
                            relax_b = relax_d
                            relax_d = relax_c
                            relax_fd = relax_fc
                            need_update_c = True
                            need_update_d = False
                        elif (relax_fc > relax_fd):
                            relax_a = relax_c
                            relax_c = relax_d
                            relax_fc = relax_fd
                            need_update_c = False
                            need_update_d = True
                        else: assert(0)
                        if (relax_k >= 5) and (numpy.argmin(relax_vals) > 0):
                            break
                        relax_k += 1
                    tab -= 1
                    U.vector().axpy(-relax_cur, dU.vector())
                    #print_var(tab,"relax_vals",relax_vals)

                    if (print_iterations):
                        iter_basename = frame_basename+"-iter="+str(k_iter).zfill(3)
                        file_dat_iter = open(iter_basename+".dat","w")
                        file_dat_iter.write("\n".join([" ".join([str(val) for val in [relax_list[relax_k], relax_vals[relax_k]]]) for relax_k in xrange(len(relax_list))]))
                        file_dat_iter.close()
                        os.system("gnuplot -e \"set terminal pdf; set output '"+iter_basename+".pdf'; plot '"+iter_basename+".dat' u 1:2 w p title 'psi_int'\"")

                    relax = relax_list[numpy.argmin(relax_vals)]
                    print_sci(tab,"relax",relax)
                else:
                    assert (0), "relax_type must be \"constant\", \"aitken\" or \"gss\". Aborting."

                # solution update
                U.vector().axpy(relax, dU.vector())
                U_norm = U.vector().norm("l2")

                if (print_iterations):
                    #print_var(tab,"U",U.vector().array())
                    file_pvd_frame << (U, float(k_iter+1))

                # displacement error
                dU_norm = abs(relax) * dU.vector().norm("l2")
                if (dU_norm == 0.) and (Uold_norm == 0.) and (U_norm == 0.):
                    err_dU = 0.
                elif (Uold_norm == 0.):
                    err_dU = dU_norm/U_norm
                else:
                    err_dU = dU_norm/Uold_norm
                print_sci(tab,"err_dU",err_dU)

                # image error
                if (k_iter > 0):
                    im_diff_old = im_diff
                im_diff = (dolfin.assemble(psi_c * dX, form_compiler_parameters={'quadrature_degree':images_quadrature})/mesh_V0)**(1./2)
                #print_sci(tab,"im_diff",im_diff)
                err_im = im_diff/Iref_norm
                print_sci(tab,"err_im",err_im)

                if (print_iterations):
                    file_dat_frame.write(" ".join([str(val) for val in [k_iter, res_norm, err_res, err_res_rel, relax, dU_norm, U_norm, err_dU, im_diff, err_im]])+"\n")

                # exit test
                success = True
                if (tol_res is not None) and (err_res > tol_res):
                    success = False
                if (tol_res_rel is not None) and (err_res_rel > tol_res_rel):
                    success = False
                if (tol_dU is not None) and (err_dU > tol_dU):
                    success = False
                if (tol_im is not None) and (err_im > tol_im):
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
                os.system("gnuplot -e \"set terminal pdf; set output '"+frame_basename+".pdf'; set key box textcolor variable; set grid; set logscale y; set yrange [1e-3:1e0]; plot '"+frame_basename+".dat' u 1:3 pt 1 lw 3 title 'err_res', '' u 1:7 pt 1 lw 3 title 'err_dU', '' using 1:9 pt 1 lw 3 title 'err_im', "+str(tol_res or tol_dU or tol_im)+" lt -1 notitle; unset logscale y; set yrange [*:*]; plot '' u 1:4 pt 1 lw 3 title 'relax'\"")

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
                p[0,0] = dolfin.assemble(Idef**2 * dX, form_compiler_parameters={'quadrature_degree':images_quadrature})
                p[0,1] = dolfin.assemble(Idef * dX, form_compiler_parameters={'quadrature_degree':images_quadrature})
                p[1,0] = p[0,1]
                p[1,1] = 1.
                q[0] = dolfin.assemble(Idef*Iref * dX, form_compiler_parameters={'quadrature_degree':images_quadrature})
                q[1] = dolfin.assemble(Iref * dX, form_compiler_parameters={'quadrature_degree':images_quadrature})
                scaling[:] = numpy.linalg.solve(p,q)
                print_var(tab,"scaling", scaling)

                im_diff = (dolfin.assemble(psi_c * dX, form_compiler_parameters={'quadrature_degree':images_quadrature})/mesh_V0)**(1./2)
                err_im = im_diff/Iref_norm
                print_sci(tab,"err_im",err_im)

        tab -= 1

        if not (success) and not (continue_after_fail):
            break

    print_var(tab,"n_iter_tot",n_iter_tot)

    #os.remove(pvd_basename+".pvd")

    return global_success
