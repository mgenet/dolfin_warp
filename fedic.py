#coding=utf8

########################################################################
###                                                                  ###
### Created by Martin Genet, 2016-2017                               ###
###                                                                  ###
### École Polytechnique, Palaiseau, France                           ###
###                                                                  ###
########################################################################

import dolfin
import glob
import math
import numpy
import os
import shutil
import time

import myPythonLibrary as mypy
import myVTKPythonLibrary as myvtk

import dolfin_dic as ddic

########################################################################

dolfin.parameters["form_compiler"]["optimize"] = False # can't use that for "complex" mechanical models…
dolfin.parameters["form_compiler"]["cpp_optimize"] = True
# dolfin.parameters["num_threads"] = 8 # 2016-07-07: doesn't seem to work…

linear_solver = "default"
#linear_solver = "mumps"
#linear_solver = "petsc"
#linear_solver = "umfpack"

########################################################################

def fedic(
        working_folder,
        working_basename,
        images_folder,
        images_basename,
        images_ext="vti", # vti, vtk
        images_n_frames=None,
        images_ref_frame=0,
        images_quadrature=None,
        images_quadrature_from="points_count", # points_count, integral
        images_expressions_type="cpp", # cpp, py
        images_dynamic_scaling=0,
        mesh=None,
        mesh_folder=None,
        mesh_basename=None,
        mesh_degree=1,
        regul_type="neo-hookean",
        regul_level=0.1,
        regul_poisson=0.0,
        tangent_type="Idef", # Idef, Idef-wHess, Iold, Iref
        residual_type="Iref", # Iref, Iold, Iref-then-Iold
        relax_type="gss", # constant, aitken, gss
        relax_init=1.0,
        initialize_DU_with_DUold=0,
        tol_res=None,
        tol_res_rel=None,
        tol_dU=None,
        tol_im=None,
        n_iter_max=100,
        continue_after_fail=0,
        print_refined_mesh=0,
        print_iterations=0):

    tab = 0

    if not os.path.exists(working_folder):
        os.mkdir(working_folder)

    mypy.print_str(tab,"Checking number of frames…")
    image_filenames = glob.glob(images_folder+"/"+images_basename+"_[0-9]*"+"."+images_ext)
    images_zfill = len(image_filenames[0].rsplit("_",1)[-1].split(".",1)[0])
    #mypy.print_var(tab+1,"images_zfill",images_zfill)
    if (images_n_frames is None):
        images_n_frames = len(image_filenames)
    assert (images_n_frames > 1), "images_n_frames = "+str(images_n_frames)+" <= 1. Aborting."
    mypy.print_var(tab+1,"images_n_frames",images_n_frames)

    assert (abs(images_ref_frame) < images_n_frames), "abs(images_ref_frame) = "+str(images_ref_frame)+" >= images_n_frames. Aborting."
    images_ref_frame = images_ref_frame%images_n_frames
    mypy.print_var(tab+1,"images_ref_frame",images_ref_frame)

    mypy.print_str(tab,"Loading mesh…")
    assert (mesh is not None or ((mesh_folder is not None) and (mesh_basename is not None))), "Must provide a mesh (mesh = "+str(mesh)+") or a mesh file (mesh_folder = "+str(mesh_folder)+", mesh_basename = "+str(mesh_basename)+"). Aborting."
    if (mesh is None):
        mesh_filebasename = mesh_folder+"/"+mesh_basename
        mesh_filename = mesh_filebasename+"."+"xml"
        assert os.path.exists(mesh_filename), "No mesh in "+mesh_filename+". Aborting."
        mesh = dolfin.Mesh(mesh_filename)
    dX = dolfin.dx(mesh)
    mesh_V0 = dolfin.assemble(dolfin.Constant(1)*dX)
    mypy.print_var(tab+1,"mesh_n_cells",len(mesh.cells()))
    mypy.print_sci(tab+1,"mesh_V0",mesh_V0)

    if (print_refined_mesh):
        mesh_for_plot = dolfin.refine(mesh)
        V_for_plot = dolfin.VectorFunctionSpace(mesh_for_plot, "Lagrange", 1)

    mypy.print_str(tab,"Computing quadrature degree for images…")
    ref_image_filename = images_folder+"/"+images_basename+"_"+str(images_ref_frame).zfill(images_zfill)+"."+images_ext
    if (images_quadrature is None):
        if (images_quadrature_from == "points_count"):
            images_quadrature = ddic.compute_quadrature_degree_from_points_count(
                image_filename=ref_image_filename,
                mesh_filebasename=mesh_filebasename,
                verbose=1)
        elif (images_quadrature_from == "integral"):
            images_quadrature = ddic.compute_quadrature_degree_from_integral(
                image_filename=ref_image_filename,
                mesh=mesh,
                deg_min=1,
                deg_max=10,
                tol=1e-2,
                verbose=1)
    mypy.print_var(tab+1,"images_quadrature",images_quadrature)

    mypy.print_str(tab,"Loading reference image…")
    ref_image = myvtk.readImage(
        filename=ref_image_filename,
        verbose=0)
    images_dimension = myvtk.getImageDimensionality(
        image=ref_image,
        verbose=0)
    mypy.print_var(tab+1,"images_dimension",images_dimension)
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
    te._quad_scheme = "default"                       # should not be needed
    for k in xrange(images_dimension**2):             # should not be needed
        te.sub_elements()[k]._quad_scheme = "default" # should not be needed
    if (images_expressions_type == "cpp"):
        Iref = dolfin.Expression(
            cppcode=ddic.get_ExprIm_cpp(
                im_dim=images_dimension,
                im_type="im",
                im_is_def=0),
            element=fe)
        Iref.init_image(ref_image_filename)
        DIref = dolfin.Expression(
            cppcode=ddic.get_ExprIm_cpp(
                im_dim=images_dimension,
                im_type="grad",
                im_is_def=0),
            element=ve)
        DIref.init_image(ref_image_filename)
    elif (images_expressions_type == "py"):
        if (images_dimension == 2):
            Iref = ddic.ExprIm2(
                filename=ref_image_filename,
                element=fe)
            DIref = ddic.ExprGradIm2(
                filename=ref_image_filename,
                element=ve)
        elif (images_dimension == 3):
            Iref = ddic.ExprIm3(
                filename=ref_image_filename,
                element=fe)
            DIref = ddic.ExprGradIm3(
                filename=ref_image_filename,
                element=ve)
    else:
        assert (0), "\"images_expressions_type\" (="+str(images_expressions_type)+") must be \"cpp\" or \"py\". Aborting."
    Iref_int = dolfin.assemble(Iref * dX, form_compiler_parameters={'quadrature_degree':images_quadrature})/mesh_V0
    Iref_norm = (dolfin.assemble(Iref**2 * dX, form_compiler_parameters={'quadrature_degree':images_quadrature})/mesh_V0)**(1./2)
    assert (Iref_norm > 0.), "Iref_norm = "+str(Iref_norm)+" <= 0. Aborting."
    mypy.print_var(tab+1,"Iref_int",Iref_int)
    mypy.print_var(tab+1,"Iref_norm",Iref_norm)

    file_error_basename = working_folder+"/"+working_basename+"-error"
    file_error = open(file_error_basename+".dat", "w")
    file_error.write("#k_frame err_im"+"\n")
    file_error.write(" ".join([str(val) for val in [images_ref_frame, 0.]])+"\n")

    mypy.print_str(tab,"Defining functions…")
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
    DUold = dolfin.Function(
        vfs,
        name="previous displacement increment")
    dU = dolfin.Function(
        vfs,
        name="displacement correction")
    dU_ = dolfin.TrialFunction(vfs)
    dV_ = dolfin.TestFunction(vfs)

    mypy.print_str(tab,"Printing initial solution…")
    if not os.path.exists(working_folder):
        os.mkdir(working_folder)
    pvd_basename = working_folder+"/"+working_basename
    for vtu_filename in glob.glob(pvd_basename+"_[0-9]*.vtu"):
        os.remove(vtu_filename)
    file_pvd = dolfin.File(pvd_basename+"__.pvd")
    file_pvd << (U, float(images_ref_frame))
    os.remove(
        pvd_basename+"__.pvd")
    shutil.move(
        pvd_basename+"__"+"".zfill(6)+".vtu",
        pvd_basename+"_"+str(images_ref_frame).zfill(6)+".vtu")

    if (print_refined_mesh):
        U.set_allow_extrapolation(True)
        U_for_plot = dolfin.interpolate(U, V_for_plot)
        U_for_plot.rename("displacement", "a Function")
        file_pvd = dolfin.File(pvd_basename+"-refined__.pvd")
        file_pvd << (U_for_plot, float(images_ref_frame))
        os.remove(
            pvd_basename+"-refined__.pvd")
        shutil.move(
            pvd_basename+"-refined__"+"".zfill(6)+".vtu",
            pvd_basename+"-refined_"+str(images_ref_frame).zfill(6)+".vtu")

    if (print_iterations):
        for filename in glob.glob(working_folder+"/"+working_basename+"-frame=[0-9]*.*"):
            os.remove(filename)

    mypy.print_str(tab,"Defining regularization energy…")
    E     = dolfin.Constant(1.0)
    nu    = dolfin.Constant(regul_poisson)
    kappa = E/3/(1-2*nu)         # = E/3 if nu = 0
    lmbda = E*nu/(1+nu)/(1-2*nu) # = 0   if nu = 0
    mu    = E/2/(1+nu)           # = E/2 if nu = 0
    C1    = mu/2
    C2    = mu/2
    D1    = kappa/2

    I = dolfin.Identity(images_dimension)
    F = I + dolfin.grad(U)
    J = dolfin.det(F)
    if   (regul_type == "laplacian"): # <- super bad
        e     = dolfin.sym(dolfin.grad(U))
        psi_m = (lmbda * dolfin.tr(e)**2 + 2*mu * dolfin.tr(e*e))/2
    elif (regul_type == "kirchhoff"): # <- pretty bad too
        C     = F.T * F
        E     = (C - I)/2
        psi_m = (lmbda * dolfin.tr(E)**2 + 2*mu * dolfin.tr(E*E))/2
    elif (regul_type == "neo-hookean"):
        C     = F.T * F
        Ic    = dolfin.tr(C)
        Ic0   = dolfin.tr(I)
        psi_m = C1 * (Ic - Ic0 - 2*dolfin.ln(J)) + D1 * (J**2 - 1 - 2*dolfin.ln(J))
    elif (regul_type == "mooney-rivlin"):
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

    mesh_V = dolfin.assemble(J*dX)
    mypy.print_sci(tab+1,"mesh_V",mesh_V)

    regularization_quadrature = 2*mesh_degree+1
    #regularization_quadrature = 2*mesh_degree
    #regularization_quadrature = mesh_degree+1
    #regularization_quadrature = mesh_degree
    #regularization_quadrature = 1
    #regularization_quadrature = None

    file_volume_basename = working_folder+"/"+working_basename+"-volume"
    file_volume = open(file_volume_basename+".dat","w")
    file_volume.write("#k_frame mesh_V"+"\n")
    file_volume.write(" ".join([str(val) for val in [images_ref_frame, mesh_V]])+"\n")

    mypy.print_str(tab,"Defining deformed image…")
    scaling = numpy.array([1.,0.])
    if (images_expressions_type == "cpp"):
        Idef = dolfin.Expression(
            cppcode=ddic.get_ExprIm_cpp(
                im_dim=images_dimension,
                im_type="im",
                im_is_def=1),
            element=fe)
        Idef.init_dynamic_scaling(scaling)
        Idef.init_disp(U)
        DIdef = dolfin.Expression(
            cppcode=ddic.get_ExprIm_cpp(
                im_dim=images_dimension,
                im_type="grad",
                im_is_def=1),
            element=ve)
        Idef.init_dynamic_scaling(scaling)
        DIdef.init_disp(U)
        if ("-wHess" in tangent_type):
            assert (0), "ToDo"
        Iold = dolfin.Expression(
            cppcode=ddic.get_ExprIm_cpp(
                im_dim=images_dimension,
                im_type="im",
                im_is_def=1),
            element=fe)
        Iold.init_dynamic_scaling(scaling) # 2016/07/25: ok, same scaling must apply to Idef & Iold…
        Iold.init_disp(Uold)
        DIold = dolfin.Expression(
            cppcode=ddic.get_ExprIm_cpp(
                im_dim=images_dimension,
                im_type="grad",
                im_is_def=1),
            element=ve)
        DIold.init_dynamic_scaling(scaling) # 2016/07/25: ok, same scaling must apply to Idef & Iold…
        DIold.init_disp(Uold)
    elif (images_expressions_type == "py"):
        if (images_dimension == 2):
            Idef = ddic.ExprDefIm2(
                U=U,
                scaling=scaling,
                element=fe)
            DIdef = ddic.ExprGradDefIm2(
                U=U,
                scaling=scaling,
                element=ve)
            if ("-wHess" in tangent_type):
                DDIdef = ddic.ExprHessDefIm2(
                    U=U,
                    scaling=scaling,
                    element=te)
            Iold = ddic.ExprDefIm2(
                U=Uold,
                scaling=scaling, # 2016/07/25: ok, same scaling must apply to Idef & Iold…
                element=fe)
            DIold = ddic.ExprGradDefIm2(
                U=Uold,
                scaling=scaling, # 2016/07/25: ok, same scaling must apply to Idef & Iold…
                element=ve)
        elif (images_dimension == 3):
            Idef = ddic.ExprDefIm3(
                U=U,
                scaling=scaling,
                element=fe)
            DIdef = ddic.ExprGradDefIm3(
                U=U,
                scaling=scaling,
                element=ve)
            if ("-wHess" in tangent_type):
                DDIdef = ddic.ExprHessDefIm3(
                    U=U,
                    scaling=scaling, # 2016/07/25: ok, same scaling must apply to Idef & Iold…
                    element=te)
            Iold = ddic.ExprDefIm3(
                U=Uold,
                scaling=scaling,
                element=fe)
            DIold = ddic.ExprGradDefIm3(
                U=Uold,
                scaling=scaling, # 2016/07/25: ok, same scaling must apply to Idef & Iold…
                element=ve)
    else:
        assert (0), "\"images_expressions_type\" (="+str(images_expressions_type)+") must be \"cpp\" or \"py\". Aborting."

    mypy.print_str(tab,"Defining correlation energy…")
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
    mypy.print_var(tab+1,"res_norm0",res_norm0)

    #regul_level = dolfin.Constant(regul_level) * 5*(1+nu)*(1-2*nu)/(5-4*nu)
    regul_level = dolfin.Constant(regul_level)

    A = None
    if (tangent_type == "Iref"):
        mypy.print_str(tab,"Matrix assembly… (image term)")
        A = dolfin.assemble((1.-regul_level) * DDpsi_c_ref * dX, tensor=A, form_compiler_parameters={'quadrature_degree':images_quadrature})
        mypy.print_str(tab,"Matrix assembly… (regularization term)")
        if (regularization_quadrature is not None):
            A = dolfin.assemble((   regul_level) * DDpsi_m     * dX, tensor=A, form_compiler_parameters={'quadrature_degree':regularization_quadrature}, add_values=True)
        else:
            A = dolfin.assemble((   regul_level) * DDpsi_m     * dX, tensor=A, add_values=True)
    B = None

    mypy.print_str(tab,"Looping over frames…")
    n_iter_tot = 0
    global_success = True
    for forward_or_backward in ["forward", "backward"]:
        mypy.print_var(tab,"forward_or_backward",forward_or_backward)

        if (forward_or_backward == "forward"):
            k_frames_old = range(images_ref_frame  , images_n_frames-1, +1)
            k_frames     = range(images_ref_frame+1, images_n_frames  , +1)
        elif (forward_or_backward == "backward"):
            k_frames_old = range(images_ref_frame  ,  0, -1)
            k_frames     = range(images_ref_frame-1, -1, -1)
        mypy.print_var(tab,"k_frames",k_frames)

        if (forward_or_backward == "backward"):
            U.vector().zero()
            U_norm = 0.
            Uold.vector().zero()
            Uold_norm = 0.
            DUold.vector().zero()
            scaling[:] = [1.,0.]

        tab += 1
        success = True
        for (k_frame,k_frame_old) in zip(k_frames,k_frames_old):
            mypy.print_var(tab-1,"k_frame",k_frame)

            if (print_iterations):
                frame_basename = working_folder+"/"+working_basename+"-frame="+str(k_frame).zfill(images_zfill)
                file_dat_frame = open(frame_basename+".dat", "w")
                file_dat_frame.write("#k_iter res_norm err_res err_res_rel relax dU_norm U_norm err_dU im_diff err_im\n")

                file_pvd_frame = dolfin.File(frame_basename+"_.pvd")
                file_pvd_frame << (U, 0.)

            mypy.print_str(tab,"Loading image, image gradient and image hessian…")
            image_filename = images_folder+"/"+images_basename+"_"+str(k_frame).zfill(images_zfill)+"."+images_ext
            Idef.init_image(image_filename)
            DIdef.init_image(image_filename)
            if ("-wHess" in tangent_type):
                DDIdef.init_image(image_filename)
            image_filename = images_folder+"/"+images_basename+"_"+str(k_frame_old).zfill(images_zfill)+"."+images_ext
            Iold.init_image(image_filename)
            DIold.init_image(image_filename)

            # linear system: matrix
            if (tangent_type == "Iold"):
                mypy.print_str(tab,"Matrix assembly… (image term)")
                A = dolfin.assemble((1.-regul_level) * DDpsi_c_old * dX, tensor=A, form_compiler_parameters={'quadrature_degree':images_quadrature})
                mypy.print_str(tab,"Matrix assembly… (regularization term)")
                if (regularization_quadrature is not None):
                    A = dolfin.assemble((   regul_level) * DDpsi_m     * dX, tensor=A, form_compiler_parameters={'quadrature_degree':regularization_quadrature}, add_values=True)
                else:
                    A = dolfin.assemble((   regul_level) * DDpsi_m     * dX, tensor=A, add_values=True)
                #mypy.print_var(tab,"A",A.array())
                #A_norm = A.norm("l2")
                #mypy.print_var(tab,"A_norm",A_norm)

            if (print_iterations):
                U.vector().zero()
                im_diff = (dolfin.assemble(psi_c * dX, form_compiler_parameters={'quadrature_degree':images_quadrature})/mesh_V0)**(1./2)
                err_im = im_diff/Iref_norm
                file_dat_frame.write(" ".join([str(val) for val in [-2, None, None, None, None, None, None, im_diff, err_im, None]])+"\n")
                U.vector()[:] = Uold.vector()[:]
                im_diff = (dolfin.assemble(psi_c * dX, form_compiler_parameters={'quadrature_degree':images_quadrature})/mesh_V0)**(1./2)
                err_im = im_diff/Iref_norm
                file_dat_frame.write(" ".join([str(val) for val in [-1, None, None, None, None, None, None, im_diff, err_im, None]])+"\n")

            if (initialize_DU_with_DUold):
                U.vector().axpy(1., DUold.vector())

            mypy.print_str(tab,"Running registration…")
            tab += 1
            k_iter = 0
            if   (residual_type.startswith("Iref")):
                using_Iold_residual = False
            elif (residual_type.startswith("Iold")):
                using_Iold_residual = True
            while (True):
                mypy.print_var(tab-1,"k_iter",k_iter)
                n_iter_tot += 1

                # linear system: matrix assembly
                if (tangent_type.startswith("Idef")):
                    mypy.print_str(tab,"Matrix assembly… (image term)")
                    A = dolfin.assemble((1.-regul_level) * DDpsi_c * dX, tensor=A, form_compiler_parameters={'quadrature_degree':images_quadrature})
                    mypy.print_str(tab,"Matrix assembly… (regularization term)")
                    if (regularization_quadrature is not None):
                        A = dolfin.assemble((   regul_level) * DDpsi_m * dX, tensor=A, form_compiler_parameters={'quadrature_degree':regularization_quadrature}, add_values=True)
                    else:
                        A = dolfin.assemble((   regul_level) * DDpsi_m * dX, tensor=A, add_values=True)
                    #mypy.print_var(tab,"A",A.array())
                    #A_norm = A.norm("l2")
                    #mypy.print_sci(tab,"A_norm",A_norm)

                # linear system: residual assembly
                if (k_iter > 0):
                    if (k_iter == 1):
                        B_old = B.copy()
                    elif (k_iter > 1):
                        B_old[:] = B[:]
                    res_old_norm = res_norm
                if (using_Iold_residual):
                    mypy.print_str(tab,"Residual assembly… (image term)")
                    B = dolfin.assemble(- (1.-regul_level) * Dpsi_c_old * dX, tensor=B, form_compiler_parameters={'quadrature_degree':images_quadrature})
                    mypy.print_str(tab,"Residual assembly… (regularization term)")
                    if (regularization_quadrature is not None):
                        B = dolfin.assemble(- (   regul_level) * Dpsi_m     * dX, tensor=B, form_compiler_parameters={'quadrature_degree':regularization_quadrature}, add_values=True)
                    else:
                        B = dolfin.assemble(- (   regul_level) * Dpsi_m     * dX, tensor=B, add_values=True)
                else:
                    mypy.print_str(tab,"Residual assembly… (image term)")
                    B = dolfin.assemble(- (1.-regul_level) * Dpsi_c * dX, tensor=B, form_compiler_parameters={'quadrature_degree':images_quadrature})
                    mypy.print_str(tab,"Residual assembly… (regularization term)")
                    if (regularization_quadrature is not None):
                        B = dolfin.assemble(- (   regul_level) * Dpsi_m * dX, tensor=B, form_compiler_parameters={'quadrature_degree':regularization_quadrature}, add_values=True)
                    else:
                        B = dolfin.assemble(- (   regul_level) * Dpsi_m * dX, tensor=B, add_values=True)
                #mypy.print_var(tab,"B",B.array())

                # residual error
                res_norm = B.norm("l2")
                #mypy.print_sci(tab,"res_norm",res_norm)
                err_res = res_norm/res_norm0
                mypy.print_sci(tab,"err_res",err_res)

                if (k_iter == 0):
                    err_res_rel = 1.
                else:
                    if (k_iter == 1):
                        dB = B - B_old
                    elif (k_iter > 1):
                        dB[:] = B[:] - B_old[:]
                    dres_norm = dB.norm("l2")
                    err_res_rel = dres_norm / res_old_norm
                    mypy.print_sci(tab,"err_res_rel",err_res_rel)

                # linear system: solve
                mypy.print_str(tab,"Solve…")
                dolfin.solve(A, dU.vector(), B,
                             linear_solver)
                #mypy.print_var(tab,"dU",dU.vector().array())

                # relaxation
                if (relax_type == "constant"):
                    if (k_iter == 0):
                        relax = relax_init
                elif (relax_type == "aitken"):
                    if (k_iter == 0):
                        relax = relax_init
                    else:
                        relax *= (-1.) * B_old.inner(dB) / dres_norm**2
                    mypy.print_sci(tab,"relax",relax)
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
                        mypy.print_var(tab-1,"relax_k",relax_k)
                        mypy.print_sci(tab,"relax_a",relax_a)
                        mypy.print_sci(tab,"relax_b",relax_b)
                        if (need_update_c):
                            relax_c = relax_b - (relax_b - relax_a) / phi
                            relax_list.append(relax_c)
                            mypy.print_sci(tab,"relax_c",relax_c)
                            U.vector().axpy(relax_c-relax_cur, dU.vector())
                            relax_cur = relax_c
                            relax_fc  = dolfin.assemble((1.-regul_level) * psi_c * dX, form_compiler_parameters={'quadrature_degree':images_quadrature})
                            #mypy.print_sci(tab,"relax_fc",relax_fc)
                            if (regularization_quadrature is not None):
                                relax_fc += dolfin.assemble((   regul_level) * psi_m * dX, form_compiler_parameters={'quadrature_degree':regularization_quadrature})
                            else:
                                relax_fc += dolfin.assemble((   regul_level) * psi_m * dX)
                            #mypy.print_sci(tab,"relax_fc",relax_fc)
                            if (numpy.isnan(relax_fc)):
                                relax_fc = float('+inf')
                                #mypy.print_sci(tab,"relax_fc",relax_fc)
                            mypy.print_sci(tab,"relax_fc",relax_fc)
                            relax_vals.append(relax_fc)
                            #mypy.print_var(tab,"relax_list",relax_list)
                            #mypy.print_var(tab,"relax_vals",relax_vals)
                        if (need_update_d):
                            relax_d = relax_a + (relax_b - relax_a) / phi
                            relax_list.append(relax_d)
                            mypy.print_sci(tab,"relax_d",relax_d)
                            U.vector().axpy(relax_d-relax_cur, dU.vector())
                            relax_cur = relax_d
                            relax_fd  = dolfin.assemble((1.-regul_level) * psi_c * dX, form_compiler_parameters={'quadrature_degree':images_quadrature})
                            #mypy.print_sci(tab,"relax_fd",relax_fd)
                            if (regularization_quadrature is not None):
                                relax_fd += dolfin.assemble((   regul_level) * psi_m * dX, form_compiler_parameters={'quadrature_degree':regularization_quadrature})
                            else:
                                relax_fd += dolfin.assemble((   regul_level) * psi_m * dX)
                            #mypy.print_sci(tab,"relax_fd",relax_fd)
                            if (numpy.isnan(relax_fd)):
                                relax_fd = float('+inf')
                                #mypy.print_sci(tab,"relax_fd",relax_fd)
                            mypy.print_sci(tab,"relax_fd",relax_fd)
                            relax_vals.append(relax_fd)
                            #mypy.print_var(tab,"relax_list",relax_list)
                            #mypy.print_var(tab,"relax_vals",relax_vals)
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
                    #mypy.print_var(tab,"relax_vals",relax_vals)

                    if (print_iterations):
                        iter_basename = frame_basename+"-iter="+str(k_iter).zfill(3)
                        file_dat_iter = open(iter_basename+".dat","w")
                        file_dat_iter.write("\n".join([" ".join([str(val) for val in [relax_list[relax_k], relax_vals[relax_k]]]) for relax_k in xrange(len(relax_list))]))
                        file_dat_iter.close()
                        os.system("gnuplot -e \"set terminal pdf; set output '"+iter_basename+".pdf'; plot '"+iter_basename+".dat' using 1:2 with points title 'psi_int'; plot '"+iter_basename+".dat' using (\$2=='inf'?\$1:1/0):(GPVAL_Y_MIN+(0.8)*(GPVAL_Y_MAX-GPVAL_Y_MIN)):(0):((0.2)*(GPVAL_Y_MAX-GPVAL_Y_MIN)) with vectors notitle, '"+iter_basename+".dat' u 1:2 with points title 'psi_int'\"")

                    relax = relax_list[numpy.argmin(relax_vals)]
                    mypy.print_sci(tab,"relax",relax)
                else:
                    assert (0), "relax_type must be \"constant\", \"aitken\" or \"gss\". Aborting."

                # solution update
                U.vector().axpy(relax, dU.vector())
                U_norm = U.vector().norm("l2")

                if (print_iterations):
                    #mypy.print_var(tab,"U",U.vector().array())
                    file_pvd_frame << (U, float(k_iter+1))

                # displacement error
                dU_norm = abs(relax) * dU.vector().norm("l2")
                if (dU_norm == 0.) and (Uold_norm == 0.) and (U_norm == 0.):
                    err_dU = 0.
                elif (Uold_norm == 0.):
                    err_dU = dU_norm/U_norm
                else:
                    err_dU = dU_norm/Uold_norm
                mypy.print_sci(tab,"err_dU",err_dU)

                # image error
                if (k_iter > 0):
                    im_diff_old = im_diff
                im_diff = (dolfin.assemble(psi_c * dX, form_compiler_parameters={'quadrature_degree':images_quadrature})/mesh_V0)**(1./2)
                #mypy.print_sci(tab,"im_diff",im_diff)
                err_im = im_diff/Iref_norm
                mypy.print_sci(tab,"err_im",err_im)

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
                    mypy.print_str(tab,"Nonlinear solver converged…")
                    break

                if (k_iter == n_iter_max-1):
                    if (residual_type=="Iref-then-Iold") and not (using_Iold_residual):
                        mypy.print_str(tab,"Warning! Nonlinear solver failed to converge…using Iold instead of Iref. (k_frame = "+str(k_frame)+")")
                        using_Iold_residual = True
                        U.vector()[:] = Uold.vector()[:]
                        U_norm = Uold_norm
                        k_iter = 0
                        continue
                    else:
                        mypy.print_str(tab,"Warning! Nonlinear solver failed to converge… (k_frame = "+str(k_frame)+")")
                        global_success = False
                        break

                # increment counter
                k_iter += 1

            tab -= 1

            if (print_iterations):
                os.remove(frame_basename+"_.pvd")
                file_dat_frame.close()
                os.system("gnuplot -e \"set terminal pdf; set output '"+frame_basename+".pdf'; set key box textcolor variable; set grid; set logscale y; set yrange [1e-3:1e0]; plot '"+frame_basename+".dat' u 1:3 pt 1 lw 3 title 'err_res', '' u 1:7 pt 1 lw 3 title 'err_dU', '' using 1:9 pt 1 lw 3 title 'err_im', "+str(tol_res or tol_dU or tol_im)+" lt -1 notitle; unset logscale y; set yrange [*:*]; plot '' u 1:4 pt 1 lw 3 title 'relax'\"")

            if not (success) and not (continue_after_fail):
                break

            # solution update
            DUold.vector()[:] = U.vector()[:] - Uold.vector()[:]
            Uold.vector()[:] = U.vector()[:]
            Uold_norm = U_norm

            mesh_V = dolfin.assemble(J*dX)
            mypy.print_sci(tab+1,"mesh_V",mesh_V)
            file_volume.write(" ".join([str(val) for val in [k_frame, mesh_V]])+"\n")

            mypy.print_str(tab,"Printing solution…")
            file_pvd = dolfin.File(pvd_basename+"__.pvd")
            file_pvd << (U, float(k_frame))
            os.remove(
                pvd_basename+"__.pvd")
            shutil.move(
                pvd_basename+"__"+"".zfill(6)+".vtu",
                pvd_basename+"_"+str(k_frame).zfill(6)+".vtu")

            if (print_refined_mesh):
                U_for_plot = dolfin.interpolate(U, V_for_plot)
                U_for_plot.rename("displacement", "a Function")
                file_pvd = dolfin.File(pvd_basename+"-refined__.pvd")
                file_pvd << (U_for_plot, float(k_frame))
                os.remove(
                    pvd_basename+"-refined__.pvd")
                shutil.move(
                    pvd_basename+"-refined__"+"".zfill(6)+".vtu",
                    pvd_basename+"-refined_"+str(k_frame).zfill(6)+".vtu")

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
                mypy.print_var(tab,"scaling", scaling)

                if (images_expressions_type == "cpp"):         # should not be needed
                    Idef.update_dynamic_scaling(scaling)       # should not be needed
                    DIdef.update_dynamic_scaling(scaling)      # should not be needed
                    if ("-wHess" in tangent_type):             # should not be needed
                        DDIdef.update_dynamic_scaling(scaling) # should not be needed
                    Iold.update_dynamic_scaling(scaling)       # should not be needed
                    DIold.update_dynamic_scaling(scaling)      # should not be needed

                im_diff = (dolfin.assemble(psi_c * dX, form_compiler_parameters={'quadrature_degree':images_quadrature})/mesh_V0)**(1./2)
                err_im = im_diff/Iref_norm
                mypy.print_sci(tab,"err_im",err_im)

            file_error.write(" ".join([str(val) for val in [k_frame, err_im]])+"\n")

        tab -= 1

        if not (success) and not (continue_after_fail):
            break

    mypy.print_var(tab,"n_iter_tot",n_iter_tot)

    file_error.close()
    os.system("gnuplot -e \"set terminal pdf; set output '"+file_error_basename+".pdf'; set grid; set yrange [0:1]; plot '"+file_error_basename+".dat' u 1:2 lw 3 notitle\"")

    file_volume.close()
    os.system("gnuplot -e \"set terminal pdf; set output '"+file_volume_basename+".pdf'; set grid; set yrange [0:*]; plot '"+file_volume_basename+".dat' u 1:2 lw 3 notitle\"")

    return global_success
