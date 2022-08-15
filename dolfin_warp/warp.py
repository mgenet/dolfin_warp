#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2016-2022                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

import dolfin_warp as dwarp

################################################################################

def warp(
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
        images_expressions_type="cpp", # cpp
        images_static_scaling=0,
        images_dynamic_scaling=0,
        images_char_func=1,
        images_is_cone=0,
        mesh=None,
        mesh_folder=None,
        mesh_basename=None,
        mesh_degree=1,
        regul_type="continuous-equilibrated", # continuous-equilibrated, continuous-elastic, continuous-hyperelastic, discrete-linear-equilibrated, discrete-linear-elastic, discrete-equilibrated, discrete-tractions, discrete-tractions-normal, discrete-tractions-tangential, discrete-tractions-normal-tangential
        regul_types=None,
        regul_model="ciarletgeymonatneohookean", # hooke, kirchhoff, ciarletgeymonatneohookean, ciarletgeymonatneohookeanmooneyrivlin
        regul_models=None,
        regul_quadrature=None,
        regul_level=0.,
        regul_levels=None,
        regul_poisson=0.,
        tangent_type="Idef", # Idef
        residual_type="Iref", # Iref
        relax_type=None, # constant, aitken, backtracking, gss
        relax_init=1., # 1.
        relax_backtracking_factor=None,
        relax_tol=None,
        relax_n_iter_max=None,
        relax_must_advance=None,
        normalize_energies=0,
        initialize_U_from_file=0,
        initialize_U_folder=None,
        initialize_U_basename=None,
        initialize_U_ext="vtu",
        initialize_U_array_name="displacement",
        initialize_DU_with_DUold=0,
        register_ref_frame=0,
        iteration_mode="normal", # normal, loop
        gimic=0,
        gimic_texture="no",
        gimic_resample=1,
        nonlinearsolver="newton", # newton, CMA
        tol_res=None, # None
        tol_res_rel=None,
        tol_dU=None,
        tol_im=None, # None
        n_iter_max=100,
        continue_after_fail=0,
        write_qois_limited_precision=0,
        write_VTU_file=1,
        write_XML_file=0,
        print_refined_mesh=0, # False
        print_iterations=0):

    assert (images_expressions_type == "cpp"),\
        "Python image expression are deprecated. Aborting."
    assert (tangent_type == "Idef"),\
        "tangent_type must be \"Idef\". Aborting."
    assert (residual_type == "Iref"),\
        "residual_type must be \"Iref\". Aborting."
    assert (relax_init == 1.),\
        "relax_init must be 1. Aborting."
    assert (not ((initialize_U_from_file) and (initialize_DU_with_DUold))),\
        "Cannot initialize U from file and DU with DUold together. Aborting."
    assert (tol_res is None),\
        "tol_res is deprecated. Aborting."
    assert (tol_im is None),\
        "tol_im is deprecated. Aborting."
    assert (continue_after_fail == 0),\
        "continue_after_fail is deprecated. Aborting."
    assert (print_refined_mesh == 0),\
        "print_refined_mesh is deprecated. Aborting."

    # assert (regul_type is not None) or (regul_types is not None),\
    #     "Must provide \"regul_type\" or \"regul_types\". Aborting."
    # assert (regul_model is not None) or (regul_models is not None),\
    #     "Must provide \"regul_model\" or \"regul_models\". Aborting."
    # assert (regul_level is not None) or (regul_levels is not None),\
    #     "Must provide \"regul_level\" or \"regul_levels\". Aborting."

    # assert (regul_type is None) or (regul_types is None),\
    #     "Cannot provide both \"regul_type\" and \"regul_types\". Aborting."
    # assert (regul_model is None) or (regul_models is None),\
    #     "Cannot provide both \"regul_model\" and \"regul_models\". Aborting."
    # assert (regul_level is None) or (regul_levels is None),\
    #     "Cannot provide both \"regul_level\" and \"regul_levels\". Aborting."

    if (regul_types is not None):
        if (regul_models is not None):
            assert (len(regul_models) == len(regul_types))
        else:
            regul_models = [regul_model]*len(regul_types)
        if (regul_levels is not None):
            assert (len(regul_levels) == len(regul_types))
        else:
            regul_levels = [regul_level]*len(regul_types)
    else:
        if (regul_models is not None) and (regul_levels is not None):
            assert (len(regul_models) == len(regul_levels))
            regul_types = [regul_type]*len(regul_models)
        elif (regul_models is not None):
            regul_types  = [regul_type ]*len(regul_models)
            regul_levels = [regul_level]*len(regul_models)
        elif (regul_levels is not None):
            regul_types  = [regul_type ]*len(regul_levels)
            regul_models = [regul_model]*len(regul_levels)
        else:
            regul_types  = [regul_type ]
            regul_models = [regul_model]
            regul_levels = [regul_level]

    problem = dwarp.WarpingProblem(
        mesh=mesh,
        mesh_folder=mesh_folder,
        mesh_basename=mesh_basename,
        U_degree=mesh_degree)

    images_series = dwarp.ImagesSeries(
        folder=images_folder,
        basename=images_basename,
        grad_basename=images_grad_basename,
        n_frames=images_n_frames,
        ext=images_ext,
        printer=problem.printer)

    if (images_quadrature is None):
        problem.printer.print_str("Computing quadrature degree…")
        problem.printer.inc()
        if (images_quadrature_from == "points_count"):
            images_quadrature = dwarp.compute_quadrature_degree_from_points_count(
                image_filename=images_series.get_image_filename(k_frame=images_ref_frame),
                mesh=problem.mesh,
                verbose=1)
        elif (images_quadrature_from == "integral"):
            images_quadrature = dwarp.compute_quadrature_degree_from_integral(
                image_filename=images_series.get_image_filename(k_frame=images_ref_frame),
                mesh=problem.mesh,
                verbose=1)
        else:
            assert (0), "\"images_quadrature_from\" (="+str(images_quadrature_from)+") must be \"points_count\" or \"integral\". Aborting."
        problem.printer.print_var("images_quadrature",images_quadrature)
        problem.printer.dec()

    image_w = 1.-sum(regul_levels)
    assert (image_w > 0.),\
        "1.-sum(regul_levels) must be positive. Aborting."

    if (gimic):
        generated_image_energy = dwarp.GeneratedImageContinuousEnergy(
            problem=problem,
            images_series=images_series,
            quadrature_degree=images_quadrature,
            texture=gimic_texture,
            w=image_w,
            ref_frame=images_ref_frame,
            resample=gimic_resample)
        problem.add_image_energy(generated_image_energy)
    else:
        warped_image_energy = dwarp.WarpedImageContinuousEnergy(
            problem=problem,
            images_series=images_series,
            quadrature_degree=images_quadrature,
            w=image_w,
            ref_frame=images_ref_frame,
            w_char_func=images_char_func,
            im_is_cone=images_is_cone,
            static_scaling=images_static_scaling,
            dynamic_scaling=images_dynamic_scaling)
        problem.add_image_energy(warped_image_energy)

    for regul_type, regul_model, regul_level in zip(regul_types, regul_models, regul_levels):
        if (regul_level>0):
            name_suffix  = ""
            name_suffix += ("_"+    regul_type  )*(len(regul_types )>1)
            name_suffix += ("_"+    regul_model )*(len(regul_models)>1)
            name_suffix += ("_"+str(regul_level))*(len(regul_levels)>1)
            if (regul_type in ("continuous-equilibrated", "continuous-elastic", "continuous-hyperelastic")):
                regularization_energy = dwarp.RegularizationContinuousEnergy(
                    name="reg"+name_suffix,
                    problem=problem,
                    w=regul_level,
                    type=regul_type.split("-")[1],
                    model=regul_model,
                    poisson=regul_poisson,
                    quadrature_degree=regul_quadrature)
                problem.add_regul_energy(regularization_energy)
            elif (regul_type in ("discrete-linear-equilibrated", "discrete-linear-elastic")):
                regularization_energy = dwarp.LinearRegularizationDiscreteEnergy(
                    name="reg"+name_suffix,
                    problem=problem,
                    w=regul_level,
                    type=regul_type.split("-")[2],
                    model="hooke",
                    poisson=regul_poisson,
                    quadrature_degree=regul_quadrature)
                problem.add_regul_energy(regularization_energy)
            elif (regul_type in ("discrete-equilibrated")):
                regularization_energy = dwarp.VolumeRegularizationDiscreteEnergy(
                    name="reg"+name_suffix,
                    problem=problem,
                    w=regul_level,
                    type=regul_type.split("-")[1],
                    model=regul_model,
                    poisson=regul_poisson,
                    quadrature_degree=regul_quadrature)
                problem.add_regul_energy(regularization_energy)
            elif (regul_type in ("discrete-tractions", "discrete-tractions-normal", "discrete-tractions-tangential", "discrete-tractions-normal-tangential")):
                regularization_energy = dwarp.SurfaceRegularizationDiscreteEnergy(
                    name="reg"+name_suffix,
                    problem=problem,
                    w=regul_level,
                    type=regul_type.split("-",1)[1],
                    model=regul_model,
                    poisson=regul_poisson,
                    quadrature_degree=regul_quadrature)
                problem.add_regul_energy(regularization_energy)
            else:
                assert (0), "\"regul_type\" (="+str(regul_type)+") must be \"continuous-equilibrated\", \"continuous-elastic\", \"continuous-hyperelastic\", \"discrete-linear-equilibrated\", \"discrete-linear-elastic\" or \"discrete-equilibrated\". Aborting."

    if (normalize_energies):
        dwarp.compute_energies_normalization(
            problem=problem)

    if (nonlinearsolver == "newton"):
        solver = dwarp.NewtonNonlinearSolver(
            problem=problem,
            parameters={
                "working_folder":working_folder,
                "working_basename":working_basename,
                "relax_type":relax_type,
                "relax_backtracking_factor":relax_backtracking_factor,
                "relax_tol":relax_tol,
                "relax_n_iter_max":relax_n_iter_max,
                "relax_must_advance":relax_must_advance,
                "tol_res_rel":tol_res_rel,
                "tol_dU":tol_dU,
                "n_iter_max":n_iter_max,
                "write_iterations":print_iterations})
    elif (nonlinearsolver == "reduced_kinematic_newton"):
        motion = dwarp.MotionModel(
            problem=problem,
            type="translation_and_scaling")
        solver = dwarp.ReducedKinematicsNewtonNonlinearSolver(
            problem=problem,
            motion_model=motion,
            parameters={
                "working_folder":working_folder,
                "working_basename":working_basename,
                "relax_type":relax_type,
                "relax_backtracking_factor":relax_backtracking_factor,
                "relax_tol":relax_tol,
                "relax_n_iter_max":relax_n_iter_max,
                "relax_must_advance":relax_must_advance,
                "tol_res_rel":tol_res_rel,
                "tol_dU":tol_dU,
                "n_iter_max":n_iter_max,
                "write_iterations":print_iterations})
    elif (nonlinearsolver == "cma"):
        solver = dwarp.CMANonlinearSolver(
            problem=problem,
            parameters={
                "working_folder":working_folder,
                "working_basename":working_basename,
                "write_iterations":print_iterations})

    image_iterator = dwarp.ImageIterator(
        problem=problem,
        solver=solver,
        parameters={
            "working_folder":working_folder,
            "working_basename":working_basename,
            "register_ref_frame":register_ref_frame,
            "initialize_U_from_file":initialize_U_from_file,
            "initialize_U_folder":initialize_U_folder,
            "initialize_U_basename":initialize_U_basename,
            "initialize_U_ext":initialize_U_ext,
            "initialize_U_array_name":initialize_U_array_name,
            "initialize_DU_with_DUold":initialize_DU_with_DUold,
            "write_qois_limited_precision":write_qois_limited_precision,
            "write_VTU_file":write_VTU_file,
            "write_XML_file":write_XML_file,
            "iteration_mode":iteration_mode})

    image_iterator.iterate()

    problem.close()

fedic2 = warp
