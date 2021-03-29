#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2016-2021                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

from builtins import range

import dolfin
import glob
import math
import numpy
import os
import shutil
import time

import myPythonLibrary    as mypy
import myVTKPythonLibrary as myvtk

import dolfin_warp as dwarp

################################################################################

#dolfin.set_log_level(dolfin.CRITICAL)
#dolfin.set_log_level(dolfin.ERROR)
#dolfin.set_log_level(dolfin.WARNING)
#dolfin.set_log_level(dolfin.INFO)
#dolfin.set_log_level(dolfin.PROGRESS)
#dolfin.set_log_level(dolfin.TRACE)
#dolfin.set_log_level(dolfin.DBG)

dolfin.parameters["form_compiler"]["representation"] = "uflacs"
dolfin.parameters["form_compiler"]["optimize"] = True
#dolfin.parameters["form_compiler"]["cpp_optimize"] = False
#dolfin.parameters["form_compiler"]["cpp_optimize_flags"] = '-O0'

dolfin.parameters['allow_extrapolation'] = True # 2017-04-10: why do I need to do that?

# dolfin.parameters["num_threads"] = 8 # 2016-07-07: doesn't seem to work…

linear_solver = "default"
#linear_solver = "mumps"
#linear_solver = "petsc"
#linear_solver = "umfpack"

################################################################################

def fedic(
        working_folder,
        working_basename,
        images_folder,
        images_basename,
        images_grad_basename=None,
        images_ext="vti", # vti, vtk
        images_n_frames=None,
        images_ref_frame=0,
        images_quadrature=None,
        images_quadrature_from="points_count", # points_count, integral
        images_expressions_type="cpp", # cpp, py
        images_dynamic_scaling=1,
        mesh=None,
        mesh_folder=None,
        mesh_basename=None,
        mesh_degree=1,
        regul_type="equilibrated", # hyperelastic, equilibrated
        regul_model="neohookean", # linear, kirchhoff, neohookean, mooneyrivlin
        regul_quadrature=None,
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

    print("WARNING! This function is deprecated. Use the warp function instead.")

    tab = 0

    if not os.path.exists(working_folder):
        os.mkdir(working_folder)

    mypy.print_str("Checking number of frames…",tab)
    image_filenames = glob.glob(images_folder+"/"+images_basename+"_[0-9]*"+"."+images_ext)
    assert (len(image_filenames) > 1),\
        "Not enough images ("+images_folder+"/"+images_basename+"_[0-9]*"+"."+images_ext+"). Aborting."
    images_zfill = len(image_filenames[0].rsplit("_",1)[-1].split(".",1)[0])
    #mypy.print_var("images_zfill",images_zfill,tab+1)
    if (images_n_frames is None):
        images_n_frames = len(image_filenames)
    assert (images_n_frames > 1),\
        "images_n_frames = "+str(images_n_frames)+" <= 1. Aborting."
    mypy.print_var("images_n_frames",images_n_frames,tab+1)
    assert (abs(images_ref_frame) < images_n_frames),\
        "abs(images_ref_frame) = "+str(abs(images_ref_frame))+" >= images_n_frames. Aborting."
    images_ref_frame = images_ref_frame%images_n_frames
    mypy.print_var("images_ref_frame",images_ref_frame,tab+1)

    mypy.print_str("Loading mesh…",tab)
    assert ((mesh is not None) or ((mesh_folder is not None) and (mesh_basename is not None))),\
        "Must provide a mesh (mesh = "+str(mesh)+") or a mesh file (mesh_folder = "+str(mesh_folder)+", mesh_basename = "+str(mesh_basename)+"). Aborting."
    if (mesh is None):
        mesh_filebasename = mesh_folder+"/"+mesh_basename
        mesh_filename = mesh_filebasename+"."+"xml"
        assert (os.path.exists(mesh_filename)),\
            "No mesh in "+mesh_filename+". Aborting."
        mesh = dolfin.Mesh(mesh_filename)
    mesh_dimension = mesh.ufl_domain().geometric_dimension()
    assert (mesh_dimension in (2,3)),\
        "mesh_dimension ("+str(mesh_dimension)+") must be 2 or 3. Aborting."
    mypy.print_var("mesh_dimension",mesh_dimension,tab+1)
    mypy.print_var("mesh_n_cells",len(mesh.cells()),tab+1)
    dV = dolfin.Measure(
        "dx",
        domain=mesh)
    dS = dolfin.Measure(
        "ds",
        domain=mesh)
    dF = dolfin.Measure(
        "dS",
        domain=mesh)
    mesh_V0 = dolfin.assemble(dolfin.Constant(1) * dV)
    mypy.print_sci("mesh_V0",mesh_V0,tab+1)
    mesh_h0 = mesh_V0**(1./mesh_dimension)
    mypy.print_sci("mesh_h0",mesh_h0,tab+1)
    mesh_h0 = dolfin.Constant(mesh_h0)

    if (print_refined_mesh):
        mesh_for_plot = dolfin.refine(mesh)
        function_space_for_plot = dolfin.VectorFunctionSpace(mesh_for_plot, "Lagrange", 1)

    mypy.print_str("Computing quadrature degree for images…",tab)
    ref_image_filename = images_folder+"/"+images_basename+"_"+str(images_ref_frame).zfill(images_zfill)+"."+images_ext
    if (images_grad_basename is not None):
        ref_image_grad_filename = images_folder+"/"+images_grad_basename+"_"+str(images_ref_frame).zfill(images_zfill)+"."+images_ext
    if (images_quadrature is None):
        if (images_quadrature_from == "points_count"):
            images_quadrature = dwarp.compute_quadrature_degree_from_points_count(
                image_filename=ref_image_filename,
                mesh=mesh,
                verbose=1)
        elif (images_quadrature_from == "integral"):
            images_quadrature = dwarp.compute_quadrature_degree_from_integral(
                image_filename=ref_image_filename,
                mesh=mesh,
                deg_min=1,
                deg_max=10,
                tol=1e-2,
                verbose=1)
    mypy.print_var("images_quadrature",images_quadrature,tab+1)

    form_compiler_parameters_for_images = {}
    form_compiler_parameters_for_images["quadrature_degree"] = images_quadrature

    mypy.print_str("Loading reference image…",tab)
    ref_image = myvtk.readImage(
        filename=ref_image_filename,
        verbose=0)
    images_dimension = myvtk.getImageDimensionality(
        image=ref_image,
        verbose=0)
    assert (images_dimension == mesh_dimension),\
        "images_dimension ("+str(images_dimension)+") ≠ mesh_dimension ("+str(mesh_dimension)+"). Aborting."
    mypy.print_var("images_dimension",images_dimension,tab+1)
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
    te._quad_scheme = "default"              # should not be needed
    for sub_element in te.sub_elements():    # should not be needed
        sub_element._quad_scheme = "default" # should not be needed
    if (images_expressions_type == "cpp"):
        Iref = dolfin.Expression(
            cppcode=dwarp.get_ExprIm_cpp(
                im_dim=images_dimension,
                im_type="im",
                im_is_def=0),
            element=fe)
        Iref.init_image(ref_image_filename)
        DIref = dolfin.Expression(
            cppcode=dwarp.get_ExprIm_cpp(
                im_dim=images_dimension,
                im_type="grad" if (images_grad_basename is None) else "grad_no_deriv",
                im_is_def=0),
            element=ve)
        DIref.init_image(ref_image_filename if (images_grad_basename is None) else ref_image_grad_filename)
    elif (images_expressions_type == "py"):
        if (images_dimension == 2):
            Iref = dwarp.ExprIm2(
                filename=ref_image_filename,
                element=fe)
            DIref = dwarp.ExprGradIm2(
                filename=ref_image_filename,
                element=ve)
        elif (images_dimension == 3):
            Iref = dwarp.ExprIm3(
                filename=ref_image_filename,
                element=fe)
            DIref = dwarp.ExprGradIm3(
                filename=ref_image_filename,
                element=ve)
    else:
        assert (0), "\"images_expressions_type\" (="+str(images_expressions_type)+") must be \"cpp\" or \"py\". Aborting."
    Iref_int = dolfin.assemble(Iref * dV, form_compiler_parameters=form_compiler_parameters_for_images)/mesh_V0
    Iref_norm = (dolfin.assemble(Iref**2 * dV, form_compiler_parameters=form_compiler_parameters_for_images)/mesh_V0)**(1./2)
    assert (Iref_norm > 0.),\
        "Iref_norm = "+str(Iref_norm)+" <= 0. Aborting."
    mypy.print_var("Iref_int",Iref_int,tab+1)
    mypy.print_var("Iref_norm",Iref_norm,tab+1)

    mypy.print_str("Initializing image error file…",tab)
    file_error_basename = working_folder+"/"+working_basename+"-error"
    file_error = open(file_error_basename+".dat", "w")
    file_error.write("#k_frame im_err"+"\n")
    file_error.write(" ".join([str(val) for val in [images_ref_frame, 0., 0., 0.]])+"\n")

    mypy.print_str("Defining functions…",tab)
    function_space = dolfin.VectorFunctionSpace(
        mesh=mesh,
        family="Lagrange",
        degree=mesh_degree)
    U = dolfin.Function(
        function_space,
        name="displacement")
    U.vector().zero()
    U_norm = 0.
    Uold = dolfin.Function(
        function_space,
        name="previous displacement")
    Uold.vector().zero()
    Uold_norm = 0.
    DUold = dolfin.Function(
        function_space,
        name="previous displacement increment")
    dU = dolfin.Function(
        function_space,
        name="displacement correction")
    dU_trial = dolfin.TrialFunction(function_space)
    dU_test = dolfin.TestFunction(function_space)

    mypy.print_str("Printing initial solution…",tab)
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
        U_for_plot = dolfin.interpolate(U, function_space_for_plot)
        U_for_plot.rename("displacement", "displacement")
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

    mypy.print_str("Initializing volume file…",tab)
    I = dolfin.Identity(mesh_dimension)
    F = I + dolfin.grad(U)
    J = dolfin.det(F)

    mesh_V = dolfin.assemble(J * dV)
    mypy.print_sci("mesh_V",mesh_V,tab+1)

    file_volume_basename = working_folder+"/"+working_basename+"-volume"
    file_volume = open(file_volume_basename+".dat","w")
    file_volume.write("#k_frame mesh_V"+"\n")
    file_volume.write(" ".join([str(val) for val in [images_ref_frame, mesh_V]])+"\n")

    if (regul_level > 0):
        mypy.print_str("Defining regularization energy…",tab)
        E     = dolfin.Constant(1.0)
        nu    = dolfin.Constant(regul_poisson)
        lmbda = E*nu/(1+nu)/(1-2*nu)                        # = 0   if nu = 0
        mu    = E/2/(1+nu)                                  # = E/2 if nu = 0
        kappa = (mesh_dimension*lmbda+2*mu)/mesh_dimension  # = E/3 (3D) or E/2 (2D) if nu = 0
        C1    = mu/2                                        # = E/4 if nu = 0
        C2    = mu/2                                        # = E/4 if nu = 0
        D1    = kappa/2                                     # = E/6 (3D) or E/4 (2D) if nu = 0

        if (regul_model == "linear"): # <- super bad
            e = dolfin.sym(dolfin.grad(U))
            psi_m = (lmbda * dolfin.tr(e)**2 + 2*mu * dolfin.tr(e*e))/2
            if (int(dolfin.dolfin_version().split('.')[0]) >= 2018): # 2018-03-16: does not seem to work…
                e   = dolfin.variable(e)
                S_m = dolfin.diff(psi_m, e)
            else:
                S_m = lmbda * dolfin.tr(e) * I + 2*mu * e
            P_m = S_m
        elif (regul_model in ("kirchhoff", "neohookean", "mooneyrivlin")):
            C = F.T * F
            E = (C - I)/2
            if (regul_model == "kirchhoff"): # <- pretty bad too
                psi_m = (lmbda * dolfin.tr(E)**2 + 2*mu * dolfin.tr(E*E))/2
                if (int(dolfin.dolfin_version().split('.')[0]) < 2018): # 2018-03-16: does not seem to work…
                    S_m = lmbda * dolfin.tr(E) * I + 2*mu * E
            elif (regul_model in ("neohookean", "mooneyrivlin")):
                Ic    = dolfin.tr(C)
                Ic0   = dolfin.tr(I)
                if (int(dolfin.dolfin_version().split('.')[0]) < 2018): # 2018-03-16: does not seem to work…
                    Cinv = dolfin.inv(C)
                if (regul_model == "neohookean"):
                    psi_m = D1 * (J**2 - 1 - 2*dolfin.ln(J)) + C1 * (Ic - Ic0 - 2*dolfin.ln(J))
                    if (int(dolfin.dolfin_version().split('.')[0]) < 2018): # 2018-03-16: does not seem to work…
                        S_m = 2*D1 * (J**2 - 1) * Cinv + 2*C1 * (I - Cinv)
                elif (regul_model == "mooneyrivlin"):
                    IIc   = (dolfin.tr(C)**2 - dolfin.tr(C*C))/2
                    IIc0  = (dolfin.tr(I)**2 - dolfin.tr(I*I))/2
                    psi_m = D1 * (J**2 - 1 - 2*dolfin.ln(J)) + C1 * (Ic - Ic0 - 2*dolfin.ln(J)) + C2 * (IIc - IIc0 - 4*dolfin.ln(J))
                    if (int(dolfin.dolfin_version().split('.')[0]) < 2018): # 2018-03-16: does not seem to work…
                        S_m = 2*D1 * (J**2 - 1) * Cinv + 2*C1 * (I - Cinv) + 2*C2 * (Ic * I - C - 2*Cinv)
            if (int(dolfin.dolfin_version().split('.')[0]) >= 2018): # 2018-03-16: does not seem to work…
                E   = dolfin.variable(E)
                S_m = dolfin.diff(psi_m, E)
            P_m = F * S_m
        else:
            assert (0), "\"regul_model\" must be \"linear\", \"kirchhoff\", \"neohookean\", or \"mooneyrivlin\". Aborting."

        if (regul_type == "hyperelastic"):
            psi_m_V = psi_m
            psi_m_F = dolfin.Constant(0)
            psi_m_S = dolfin.Constant(0)
        elif (regul_type == "equilibrated"):
            Div_P = dolfin.div(P_m)
            psi_m_V = dolfin.inner(Div_P,
                                   Div_P)
            N = dolfin.FacetNormal(mesh)
            Jump_P_N = dolfin.jump(P_m, N)
            cell_h = dolfin.Constant(mesh.hmin())
            psi_m_F = dolfin.inner(Jump_P_N,
                                   Jump_P_N)/cell_h
            #P_N = P_m * N
            #P_N_N = dolfin.dot(N, P_N)
            #P_N_T = P_N - P_N_N * N
            #psi_m_S  = dolfin.inner(P_N_T,
                                  #P_N_T)/cell_h
            #psi_m_S  = dolfin.inner(P_N,
                                  #P_N)/cell_h
            psi_m_S = dolfin.Constant(0)
        else:
            assert (0), "\"regul_type\" must be \"hyperelastic\" or \"equilibrated\". Aborting."
        Dpsi_m_V  = dolfin.derivative( psi_m_V, U, dU_test)
        DDpsi_m_V = dolfin.derivative(Dpsi_m_V, U, dU_trial)
        Dpsi_m_F  = dolfin.derivative( psi_m_F, U, dU_test)
        DDpsi_m_F = dolfin.derivative(Dpsi_m_F, U, dU_trial)
        Dpsi_m_S  = dolfin.derivative( psi_m_S, U, dU_test)
        DDpsi_m_S = dolfin.derivative(Dpsi_m_S, U, dU_trial)

        #regul_quadrature = 2*mesh_degree+1
        #regul_quadrature = 2*mesh_degree
        #regul_quadrature = mesh_degree+1
        #regul_quadrature = mesh_degree
        #regul_quadrature = 1
        regul_quadrature = None

        form_compiler_parameters_for_regul = {}
        form_compiler_parameters_for_regul["representation"] = "uflacs" # MG20180327: Is that needed?
        if (regul_quadrature is not None):
            form_compiler_parameters_for_regul["quadrature_degree"] = regul_quadrature

    mypy.print_str("Defining deformed image…",tab)
    scaling = numpy.array([1.,0.])
    if (images_expressions_type == "cpp"):
        Idef = dolfin.Expression(
            cppcode=dwarp.get_ExprIm_cpp(
                im_dim=images_dimension,
                im_type="im",
                im_is_def=1),
            element=fe)
        Idef.init_dynamic_scaling(scaling)
        Idef.init_disp(U)
        DIdef = dolfin.Expression(
            cppcode=dwarp.get_ExprIm_cpp(
                im_dim=images_dimension,
                im_type="grad" if (images_grad_basename is None) else "grad_no_deriv",
                im_is_def=1),
            element=ve)
        DIdef.init_dynamic_scaling(scaling)
        DIdef.init_disp(U)
        if ("-wHess" in tangent_type):
            assert (0), "ToDo"
        Iold = dolfin.Expression(
            cppcode=dwarp.get_ExprIm_cpp(
                im_dim=images_dimension,
                im_type="im",
                im_is_def=1),
            element=fe)
        Iold.init_dynamic_scaling(scaling) # 2016/07/25: ok, same scaling must apply to Idef & Iold…
        Iold.init_disp(Uold)
        DIold = dolfin.Expression(
            cppcode=dwarp.get_ExprIm_cpp(
                im_dim=images_dimension,
                im_type="grad" if (images_grad_basename is None) else "grad_no_deriv",
                im_is_def=1),
            element=ve)
        DIold.init_dynamic_scaling(scaling) # 2016/07/25: ok, same scaling must apply to Idef & Iold…
        DIold.init_disp(Uold)
    elif (images_expressions_type == "py"):
        if (images_dimension == 2):
            Idef = dwarp.ExprDefIm2(
                U=U,
                scaling=scaling,
                element=fe)
            DIdef = dwarp.ExprGradDefIm2(
                U=U,
                scaling=scaling,
                element=ve)
            if ("-wHess" in tangent_type):
                DDIdef = dwarp.ExprHessDefIm2(
                    U=U,
                    scaling=scaling,
                    element=te)
            Iold = dwarp.ExprDefIm2(
                U=Uold,
                scaling=scaling, # 2016/07/25: ok, same scaling must apply to Idef & Iold…
                element=fe)
            DIold = dwarp.ExprGradDefIm2(
                U=Uold,
                scaling=scaling, # 2016/07/25: ok, same scaling must apply to Idef & Iold…
                element=ve)
        elif (images_dimension == 3):
            Idef = dwarp.ExprDefIm3(
                U=U,
                scaling=scaling,
                element=fe)
            DIdef = dwarp.ExprGradDefIm3(
                U=U,
                scaling=scaling,
                element=ve)
            if ("-wHess" in tangent_type):
                DDIdef = dwarp.ExprHessDefIm3(
                    U=U,
                    scaling=scaling, # 2016/07/25: ok, same scaling must apply to Idef & Iold…
                    element=te)
            Iold = dwarp.ExprDefIm3(
                U=Uold,
                scaling=scaling,
                element=fe)
            DIold = dwarp.ExprGradDefIm3(
                U=Uold,
                scaling=scaling, # 2016/07/25: ok, same scaling must apply to Idef & Iold…
                element=ve)
    else:
        assert (0), "\"images_expressions_type\" (="+str(images_expressions_type)+") must be \"cpp\" or \"py\". Aborting."

    mypy.print_str("Defining correlation energy…",tab)
    phi_Iref = dolfin.Expression(
        cppcode=dwarp.get_ExprCharFuncIm_cpp(
            im_dim=images_dimension),
        element=fe)
    phi_Iref.init_image(ref_image_filename)
    psi_c   = phi_Iref * (Idef - Iref)**2/2
    Dpsi_c  = phi_Iref * (Idef - Iref) * dolfin.dot(DIdef, dU_test)
    DDpsi_c = phi_Iref * dolfin.dot(DIdef, dU_trial) * dolfin.dot(DIdef, dU_test)
    if ("-wHess" in tangent_type):
        DDpsi_c += (Idef - Iref) * dolfin.dot(dolfin.dot(DDIdef, dU_trial), dU_test)

    psi_c_old  = (Idef - Iold)**2/2
    Dpsi_c_old = (Idef - Iold) * dolfin.dot(DIdef, dU_test)
    DDpsi_c_old = dolfin.dot(DIold, dU_trial) * dolfin.dot(DIold, dU_test)

    DDpsi_c_ref = dolfin.dot(DIref, dU_trial) * dolfin.dot(DIref, dU_test)

    b0 = Iref * dolfin.dot(DIref, dU_test)
    B0 = dolfin.assemble(b0 * dV, form_compiler_parameters=form_compiler_parameters_for_images)
    res_norm0 = B0.norm("l2")
    assert (res_norm0 > 0.),\
        "res_norm0 = "+str(res_norm0)+" <= 0. Aborting."
    mypy.print_var("res_norm0",res_norm0,tab+1)

    A_c = None
    A_m = None
    A   = None
    if (tangent_type.startswith("Iref")):
        mypy.print_str("Matrix assembly (image term)…",tab,newline=False)
        t = time.time()
        A_c = dolfin.assemble(DDpsi_c_ref * dV, tensor=A_c, form_compiler_parameters=form_compiler_parameters_for_images)
        t = time.time() - t
        mypy.print_str(" "+str(t)+" s")
    if (regul_level > 0) and (regul_model == "linear"):
        mypy.print_str("Matrix assembly (regularization term)…",tab,newline=False)
        t = time.time()
        A_m = dolfin.assemble(DDpsi_m_V * dV + DDpsi_m_F * dF + DDpsi_m_S * dS, tensor=A_m, form_compiler_parameters=form_compiler_parameters_for_regul)
        t = time.time() - t
        mypy.print_str(" "+str(t)+" s")
    if (tangent_type.startswith("Iref")) and (regul_model == "linear"):
        if (regul_level > 0):
            mypy.print_str("Matrix assembly (combination)…",tab,newline=False)
            t = time.time()
            A = (1.-regul_level) * A_c + regul_level * A_m
            t = time.time() - t
            mypy.print_str(" "+str(t)+" s")
        else:
            A = A_c
    B_c = None
    B_m = None
    B   = None

    mypy.print_str("Looping over frames…",tab)
    n_iter_tot = 0
    global_success = True
    for forward_or_backward in ["forward", "backward"]:
        mypy.print_var("forward_or_backward",forward_or_backward,tab)

        if (forward_or_backward == "forward"):
            k_frames_old = range(images_ref_frame  , images_n_frames-1, +1)
            k_frames     = range(images_ref_frame+1, images_n_frames  , +1)
        elif (forward_or_backward == "backward"):
            k_frames_old = range(images_ref_frame  ,  0, -1)
            k_frames     = range(images_ref_frame-1, -1, -1)
        mypy.print_var("k_frames",k_frames,tab)

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
            mypy.print_var("k_frame",k_frame,tab-1)

            if (print_iterations):
                frame_basename = working_folder+"/"+working_basename+"-frame="+str(k_frame).zfill(images_zfill)
                file_dat_frame = open(frame_basename+".dat", "w")
                file_dat_frame.write("#k_iter res_norm res_err res_err_rel relax dU_norm U_norm dU_err im_diff im_err\n")

                file_pvd_frame = dolfin.File(frame_basename+"_.pvd")
                file_pvd_frame << (U, 0.)

            mypy.print_str("Loading image, image gradient and image hessian…",tab)
            image_filename = images_folder+"/"+images_basename+"_"+str(k_frame).zfill(images_zfill)+"."+images_ext
            Idef.init_image(image_filename)
            if (images_grad_basename is None):
                DIdef.init_image(image_filename)
            else:
                image_grad_filename = images_folder+"/"+images_grad_basename+"_"+str(k_frame).zfill(images_zfill)+"."+images_ext
                DIdef.init_image(image_grad_filename)
            if ("-wHess" in tangent_type):
                DDIdef.init_image(image_filename)
            image_filename = images_folder+"/"+images_basename+"_"+str(k_frame_old).zfill(images_zfill)+"."+images_ext
            Iold.init_image(image_filename)
            if (images_grad_basename is None):
                DIold.init_image(image_filename)
            else:
                image_grad_filename = images_folder+"/"+images_grad_basename+"_"+str(k_frame_old).zfill(images_zfill)+"."+images_ext
                DIold.init_image(image_grad_filename)

            U.vector().zero()
            im_diff_0 = (dolfin.assemble(psi_c * dV, form_compiler_parameters=form_compiler_parameters_for_images)/mesh_V0)**(1./2)
            im_err_0 = im_diff_0/Iref_norm
            mypy.print_sci("im_err_0",im_err_0,tab)
            U.vector()[:] = Uold.vector()[:]
            im_diff_m1 = (dolfin.assemble(psi_c * dV, form_compiler_parameters=form_compiler_parameters_for_images)/mesh_V0)**(1./2)
            im_err_m1 = im_diff_m1/Iref_norm
            mypy.print_sci("im_err_m1",im_err_m1,tab)

            if (print_iterations):
                file_dat_frame.write(" ".join([str(val) for val in [-2, None, None, None, None, None, None, None, im_diff_0, im_err_0]])+"\n")
                file_dat_frame.write(" ".join([str(val) for val in [-1, None, None, None, None, None, None, None, im_diff_m1, im_err_m1]])+"\n")

            # linear system: matrix
            if (tangent_type.startswith("Iold")):
                mypy.print_str("Matrix assembly (image term)…",tab,newline=False)
                t = time.time()
                A_c = dolfin.assemble(DDpsi_c_old * dV, tensor=A_c, form_compiler_parameters=form_compiler_parameters_for_images)
                t = time.time() - t
                mypy.print_str(" "+str(t)+" s")
            if (tangent_type.startswith("Iold")) and (regul_model == "linear"):
                mypy.print_str("Matrix assembly (combination)…",tab,newline=False)
                t = time.time()
                if (regul_level > 0):
                    if (A is None):
                        A = (1.-regul_level) * A_c + regul_level * A_m
                    else:
                        A.zero()
                        A.axpy(1.-regul_level, A_c, False)
                        A.axpy(   regul_level, A_m, False)
                else:
                    if (A is None):
                        A = A_c
                t = time.time() - t
                mypy.print_str(" "+str(t)+" s")

            if (initialize_DU_with_DUold):
                U.vector().axpy(1., DUold.vector())

            mypy.print_str("Running registration…",tab)
            tab += 1
            k_iter = 0
            if   (residual_type.startswith("Iref")):
                using_Iold_residual = False
            elif (residual_type.startswith("Iold")):
                using_Iold_residual = True
            while (True):
                mypy.print_var("k_iter",k_iter,tab-1)
                n_iter_tot += 1

                # linear system: matrix assembly
                if (tangent_type.startswith("Idef")):
                    mypy.print_str("Matrix assembly (image term)…",tab,newline=False)
                    t = time.time()
                    A_c = dolfin.assemble(DDpsi_c * dV, tensor=A_c, form_compiler_parameters=form_compiler_parameters_for_images)
                    #mypy.print_var("A_c",A_c.array(),tab)
                    t = time.time() - t
                    mypy.print_str(" "+str(t)+" s")
                if (regul_level > 0) and (regul_model != "linear"):
                    mypy.print_str("Matrix assembly (regularization term)…",tab,newline=False)
                    t = time.time()
                    A_m = dolfin.assemble(DDpsi_m_V * dV + DDpsi_m_F * dF + DDpsi_m_S * dS, tensor=A_m, form_compiler_parameters=form_compiler_parameters_for_regul)
                    #mypy.print_var("A_m",A_m.array(),tab)
                    t = time.time() - t
                    mypy.print_str(" "+str(t)+" s")
                if (tangent_type.startswith("Idef")) or (regul_model != "linear"):
                    mypy.print_str("Matrix assembly (combination)…",tab,newline=False)
                    t = time.time()
                    if (regul_level > 0):
                        if (A is None):
                            A = (1.-regul_level) * A_c + regul_level * A_m
                        else:
                            A.zero()
                            A.axpy(1.-regul_level, A_c, False)
                            A.axpy(   regul_level, A_m, False)
                    else:
                        if (A is None):
                            A = A_c
                    t = time.time() - t
                    mypy.print_str(" "+str(t)+" s")
                    #mypy.print_var("A",A.array(),tab)

                # linear system: residual assembly
                if (k_iter > 0):
                    if (k_iter == 1):
                        B_old = B.copy()
                    elif (k_iter > 1):
                        B_old[:] = B[:]
                    res_old_norm = res_norm
                mypy.print_str("Residual assembly (image term)…",tab,newline=False)
                t = time.time()
                if (using_Iold_residual):
                    B_c = dolfin.assemble(Dpsi_c_old * dV, tensor=B_c, form_compiler_parameters=form_compiler_parameters_for_images)
                else:
                    B_c = dolfin.assemble(Dpsi_c * dV, tensor=B_c, form_compiler_parameters=form_compiler_parameters_for_images)
                t = time.time() - t
                mypy.print_str(" "+str(t)+" s")
                if (regul_level > 0):
                    mypy.print_str("Residual assembly (regularization term)…",tab,newline=False)
                    t = time.time()
                    B_m = dolfin.assemble(Dpsi_m_V * dV + Dpsi_m_F * dF + Dpsi_m_S * dS, tensor=B_m, form_compiler_parameters=form_compiler_parameters_for_regul)
                    t = time.time() - t
                    mypy.print_str(" "+str(t)+" s")
                mypy.print_str("Residual assembly (combination)…",tab,newline=False)
                t = time.time()
                if (regul_level > 0):
                    if (B is None):
                        B = -(1.-regul_level) * B_c - regul_level * B_m
                    else:
                        B.zero()
                        B.axpy(-(1.-regul_level), B_c)
                        B.axpy(-    regul_level , B_m)
                else:
                    if (B is None):
                        B = - B_c
                    else:
                        B.zero()
                        B.axpy(-1., B_c)
                t = time.time() - t
                mypy.print_str(" "+str(t)+" s")
                #mypy.print_var("B",B.array(),tab)

                # residual error
                res_norm = B.norm("l2")
                #mypy.print_sci("res_norm",res_norm,tab)
                res_err = res_norm/res_norm0
                mypy.print_sci("res_err",res_err,tab)

                if (k_iter == 0):
                    res_err_rel = 1.
                else:
                    if (k_iter == 1):
                        dB = B - B_old
                    elif (k_iter > 1):
                        dB[:] = B[:] - B_old[:]
                    dres_norm = dB.norm("l2")
                    res_err_rel = dres_norm / res_old_norm
                    mypy.print_sci("res_err_rel",res_err_rel,tab)

                # linear system: solve
                mypy.print_str("Solve…",tab,newline=False)
                t = time.time()
                dolfin.solve(A, dU.vector(), B,
                             linear_solver)
                t = time.time() - t
                mypy.print_str(" "+str(t)+" s")
                #mypy.print_var("dU",dU.vector().array(),tab)

                dU_norm = dU.vector().norm("l2")
                #mypy.print_sci("dU_norm",dU_norm,tab)
                if not (numpy.isfinite(dU_norm)):
                    dU.vector().zero()

                # relaxation
                if (relax_type == "constant"):
                    if (k_iter == 0):
                        relax = relax_init
                elif (relax_type == "aitken"):
                    if (k_iter == 0):
                        relax = relax_init
                    else:
                        relax *= (-1.) * B_old.inner(dB) / dres_norm**2
                    mypy.print_sci("relax",relax,tab)
                elif (relax_type == "gss"):
                    phi = (1 + math.sqrt(5)) / 2
                    relax_a = (1-phi)/(2-phi)
                    relax_b = 1./(2-phi)
                    need_update_c = True
                    need_update_d = True
                    relax_cur = 0.
                    relax_list = []
                    relax_vals = []
                    tab += 1
                    relax_k = 0
                    while (True):
                        mypy.print_var("relax_k",relax_k,tab-1)
                        mypy.print_sci("relax_a",relax_a,tab)
                        mypy.print_sci("relax_b",relax_b,tab)
                        if (need_update_c):
                            relax_c = relax_b - (relax_b - relax_a) / phi
                            relax_list.append(relax_c)
                            mypy.print_sci("relax_c",relax_c,tab)
                            U.vector().axpy(relax_c-relax_cur, dU.vector())
                            relax_cur = relax_c
                            if (using_Iold_residual):
                                relax_fc  = (1.-regul_level) * dolfin.assemble(psi_c_old * dV, form_compiler_parameters=form_compiler_parameters_for_images)
                            else:
                                relax_fc  = (1.-regul_level) * dolfin.assemble(psi_c * dV, form_compiler_parameters=form_compiler_parameters_for_images)
                            #mypy.print_sci("relax_fc",relax_fc,tab)
                            if (regul_level > 0):
                                relax_fc += regul_level * dolfin.assemble(psi_m_V * dV + psi_m_F * dF + psi_m_S * dS, form_compiler_parameters=form_compiler_parameters_for_regul)
                                #mypy.print_sci("relax_fc",relax_fc,tab)
                            if (numpy.isnan(relax_fc)):
                                relax_fc = float('+inf')
                                #mypy.print_sci("relax_fc",relax_fc,tab)
                            mypy.print_sci("relax_fc",relax_fc,tab)
                            relax_vals.append(relax_fc)
                            #mypy.print_var("relax_list",relax_list,tab)
                            #mypy.print_var("relax_vals",relax_vals,tab)
                        if (need_update_d):
                            relax_d = relax_a + (relax_b - relax_a) / phi
                            relax_list.append(relax_d)
                            mypy.print_sci("relax_d",relax_d,tab)
                            U.vector().axpy(relax_d-relax_cur, dU.vector())
                            relax_cur = relax_d
                            if (using_Iold_residual):
                                relax_fd  = (1.-regul_level) * dolfin.assemble(psi_c_old * dV, form_compiler_parameters=form_compiler_parameters_for_images)
                            else:
                                relax_fd  = (1.-regul_level) * dolfin.assemble(psi_c * dV, form_compiler_parameters=form_compiler_parameters_for_images)
                            #mypy.print_sci("relax_fd",relax_fd,tab)
                            if (regul_level > 0):
                                relax_fd += regul_level * dolfin.assemble(psi_m_V * dV + psi_m_F * dF + psi_m_S * dS, form_compiler_parameters=form_compiler_parameters_for_regul)
                                #mypy.print_sci("relax_fd",relax_fd,tab)
                            if (numpy.isnan(relax_fd)):
                                relax_fd = float('+inf')
                                #mypy.print_sci("relax_fd",relax_fd,tab)
                            mypy.print_sci("relax_fd",relax_fd,tab)
                            relax_vals.append(relax_fd)
                            #mypy.print_var("relax_list",relax_list,tab)
                            #mypy.print_var("relax_vals",relax_vals,tab)
                        #if ((relax_fc < 1e-12) and (relax_fd < 1e-12)):
                            #break
                        if (relax_fc < relax_fd):
                            relax_b = relax_d
                            relax_d = relax_c
                            relax_fd = relax_fc
                            need_update_c = True
                            need_update_d = False
                        elif (relax_fc >= relax_fd):
                            relax_a = relax_c
                            relax_c = relax_d
                            relax_fc = relax_fd
                            need_update_c = False
                            need_update_d = True
                        else: assert(0)
                        if (relax_k >= 9):
                        #if (relax_k >= 9) and (numpy.argmin(relax_vals) > 0):
                            break
                        relax_k += 1
                    tab -= 1
                    U.vector().axpy(-relax_cur, dU.vector())
                    #mypy.print_var("relax_vals",relax_vals,tab)

                    if (print_iterations):
                        iter_basename = frame_basename+"-iter="+str(k_iter).zfill(3)
                        file_dat_iter = open(iter_basename+".dat","w")
                        file_dat_iter.write("\n".join([" ".join([str(val) for val in [relax_list[relax_k], relax_vals[relax_k]]]) for relax_k in range(len(relax_list))]))
                        file_dat_iter.close()
                        os.system("gnuplot -e \"set terminal pdf; set output '"+iter_basename+".pdf'; plot '"+iter_basename+".dat' using 1:2 with points title 'psi_int'; plot '"+iter_basename+".dat' using (\$2=='inf'?\$1:1/0):(GPVAL_Y_MIN+(0.8)*(GPVAL_Y_MAX-GPVAL_Y_MIN)):(0):((0.2)*(GPVAL_Y_MAX-GPVAL_Y_MIN)) with vectors notitle, '"+iter_basename+".dat' u 1:2 with points title 'psi_int'\"")

                    relax = relax_list[numpy.argmin(relax_vals)]
                    mypy.print_sci("relax",relax,tab)
                    if (relax == 0.):
                        mypy.print_str("Warning! Optimal relaxation is null…",tab)
                else:
                    assert (0), "relax_type must be \"constant\", \"aitken\" or \"gss\". Aborting."

                # solution update
                U.vector().axpy(relax, dU.vector())
                U_norm = U.vector().norm("l2")
                #mypy.print_sci("U_norm",U_norm,tab)

                if (print_iterations):
                    #mypy.print_var("U",U.vector().array(),tab)
                    file_pvd_frame << (U, float(k_iter+1))

                # displacement error
                dU_norm *= abs(relax)
                #mypy.print_sci("dU_norm",dU_norm,tab)
                if (dU_norm == 0.) and (Uold_norm == 0.) and (U_norm == 0.):
                    dU_err = 0.
                elif (Uold_norm == 0.):
                    dU_err = dU_norm/U_norm
                else:
                    dU_err = dU_norm/Uold_norm
                mypy.print_sci("dU_err",dU_err,tab)

                # image error
                if (k_iter > 0):
                    im_diff_old = im_diff
                im_diff = (dolfin.assemble(psi_c * dV, form_compiler_parameters=form_compiler_parameters_for_images)/mesh_V0)**(1./2)
                #mypy.print_sci("im_diff",im_diff,tab)
                im_err = im_diff/Iref_norm
                mypy.print_sci("im_err",im_err,tab)

                if (print_iterations):
                    file_dat_frame.write(" ".join([str(val) for val in [k_iter, res_norm, res_err, res_err_rel, relax, dU_norm, U_norm, dU_err, im_diff, im_err]])+"\n")

                # exit test
                success = True
                if (tol_res is not None) and (res_err > tol_res):
                    success = False
                if (tol_res_rel is not None) and (res_err_rel > tol_res_rel):
                    success = False
                if (tol_dU is not None) and (dU_err > tol_dU):
                    success = False
                if (tol_im is not None) and (im_err > tol_im):
                    success = False

                # exit
                if (success):
                    mypy.print_str("Nonlinear solver converged…",tab)
                    break

                if (k_iter == n_iter_max-1):
                    if (residual_type=="Iref-then-Iold") and not (using_Iold_residual):
                        mypy.print_str("Warning! Nonlinear solver failed to converge…using Iold instead of Iref. (k_frame = "+str(k_frame)+")",tab)
                        using_Iold_residual = True
                        U.vector()[:] = Uold.vector()[:]
                        U_norm = Uold_norm
                        k_iter = 0
                        continue
                    else:
                        mypy.print_str("Warning! Nonlinear solver failed to converge… (k_frame = "+str(k_frame)+")",tab)
                        global_success = False
                        break

                # increment counter
                k_iter += 1

            tab -= 1

            if (print_iterations):
                os.remove(frame_basename+"_.pvd")
                file_dat_frame.close()
                os.system("gnuplot -e \"set terminal pdf noenhanced; set output '"+frame_basename+".pdf'; set key box textcolor variable; set grid; set logscale y; set yrange [1e-3:1e0]; plot '"+frame_basename+".dat' u 1:3 pt 1 lw 3 title 'res_err', '' u 1:8 pt 1 lw 3 title 'dU_err', '' using 1:10 pt 1 lw 3 title 'im_err', "+str(tol_res or tol_dU or tol_im)+" lt -1 notitle; unset logscale y; set yrange [*:*]; plot '' u 1:4 pt 1 lw 3 title 'relax'\"")

            if not (success) and not (continue_after_fail):
                break

            # solution update
            DUold.vector()[:] = U.vector()[:] - Uold.vector()[:]
            Uold.vector()[:] = U.vector()[:]
            Uold_norm = U_norm

            mesh_V = dolfin.assemble(J * dV)
            mypy.print_sci("mesh_V",mesh_V,tab+1)
            file_volume.write(" ".join([str(val) for val in [k_frame, mesh_V]])+"\n")

            mypy.print_str("Printing solution…",tab)
            file_pvd = dolfin.File(pvd_basename+"__.pvd")
            file_pvd << (U, float(k_frame))
            os.remove(
                pvd_basename+"__.pvd")
            shutil.move(
                pvd_basename+"__"+"".zfill(6)+".vtu",
                pvd_basename+"_"+str(k_frame).zfill(6)+".vtu")

            if (print_refined_mesh):
                U_for_plot = dolfin.interpolate(U, function_space_for_plot)
                U_for_plot.rename("displacement", "displacement")
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
                p[0,0] = dolfin.assemble(Idef**2 * dV, form_compiler_parameters=form_compiler_parameters_for_images)
                p[0,1] = dolfin.assemble(Idef * dV, form_compiler_parameters=form_compiler_parameters_for_images)
                p[1,0] = p[0,1]
                p[1,1] = 1.
                q[0] = dolfin.assemble(Idef*Iref * dV, form_compiler_parameters=form_compiler_parameters_for_images)
                q[1] = dolfin.assemble(Iref * dV, form_compiler_parameters=form_compiler_parameters_for_images)
                scaling[:] = numpy.linalg.solve(p,q)
                mypy.print_var("scaling",scaling,tab)

                if (images_expressions_type == "cpp"):         # should not be needed
                    Idef.update_dynamic_scaling(scaling)       # should not be needed
                    DIdef.update_dynamic_scaling(scaling)      # should not be needed
                    if ("-wHess" in tangent_type):             # should not be needed
                        DDIdef.update_dynamic_scaling(scaling) # should not be needed
                    Iold.update_dynamic_scaling(scaling)       # should not be needed
                    DIold.update_dynamic_scaling(scaling)      # should not be needed

                im_diff = (dolfin.assemble(psi_c * dV, form_compiler_parameters=form_compiler_parameters_for_images)/mesh_V0)**(1./2)
                im_err = im_diff/Iref_norm
                mypy.print_sci("im_err",im_err,tab)

            file_error.write(" ".join([str(val) for val in [k_frame, im_err_0, im_err_m1, im_err]])+"\n")

        tab -= 1

        if not (success) and not (continue_after_fail):
            break

    mypy.print_var("n_iter_tot",n_iter_tot,tab)

    file_error.close()
    os.system("gnuplot -e \"set terminal pdf; set output '"+file_error_basename+".pdf'; set key box textcolor variable; set grid; plot '"+file_error_basename+".dat' u 1:2 pt 1 lw 3 title 'correlation energy for U=0', '' u 1:3 pt 1 lw 3 title 'correlation energy for U=U_old', '' u 1:4 pt 1 lw 3 title 'correlation energy for U=U'\"")

    file_volume.close()
    os.system("gnuplot -e \"set terminal pdf; set output '"+file_volume_basename+".pdf'; set grid; set yrange [0:*]; plot '"+file_volume_basename+".dat' u 1:2 lw 3 notitle\"")

    return global_success
