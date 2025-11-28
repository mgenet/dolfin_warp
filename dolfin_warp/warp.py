#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2016-2025                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

import dolfin

import dolfin_warp as dwarp

################################################################################

def warp(
        working_folder                              : str,
        working_basename                            : str,
        images_folder                               : str,
        images_basename                             : str,
        images_grad_basename                        : str         = None                                ,
        images_ext                                  : str         = "vti"                               , # vti, vtk
        images_n_frames                             : int         = None                                ,
        images_ref_frame                            : int         = 0                                   ,
        images_static_scaling                       : bool        = False                               ,
        images_dynamic_scaling                      : bool        = False                               ,
        images_char_func                            : bool        = True                                ,
        images_is_cone                              : bool        = False                               ,
        mesh                                        : dolfin.Mesh = None                                ,
        mesh_folder                                 : str         = None                                ,
        mesh_basename                               : str         = None                                ,
        mesh_degree                                 : int         = 1                                   ,
        kinematics_type                             : str         = "full"                              , # full, reduced
        reduced_kinematics_model                    : str         = "translation+rotation+scaling+shear", # translation, rotation, scaling, shear, translation+rotation+scaling+shear, etc.
        image_energy_type                           : str         = "warped"                            , # warped, generated 
        image_energy_quadrature                     : int         = None                                ,
        image_energy_quadrature_from                : str         = "points_count"                      , # points_count, integral
        generated_image_energy_texture              : str         = "tagging"                           , # no, tagging
        generated_image_energy_resample             : bool        = True                                ,
        generated_image_energy_resampling_factor    : int         = 1                                   ,
        generated_image_energy_type                 : str         = "image"                             , # image, fourier
        regul_type                                  : str         = "continuous-equilibrated"           , # continuous-linear-equilibrated, continuous-linear-elastic, continuous-equilibrated, continuous-elastic, continuous-hyperelastic, discrete-simple-equilibrated, discrete-simple-elastic, discrete-linear-equilibrated, discrete-linear-equilibrated-tractions, discrete-linear-equilibrated-tractions-normal, discrete-linear-equilibrated-tractions-tangential, discrete-linear-equilibrated-tractions-normal-tangential, discrete-equilibrated, discrete-equilibrated-tractions, discrete-equilibrated-tractions-normal, discrete-equilibrated-tractions-tangential, discrete-equilibrated-tractions-normal-tangential
        regul_types                                 : list        = None                                ,
        regul_model                                 : str         = "ogdenciarletgeymonatneohookean"    , # hooke, kirchhoff, ogdenciarletgeymonatneohookean, ogdenciarletgeymonatneohookeanmooneyrivlin
        regul_models                                : list        = None                                ,
        regul_quadrature                            : int         = None                                ,
        regul_level                                 : float       = 0.                                  ,
        regul_levels                                : list        = None                                ,
        regul_poisson                               : float       = 0.                                  ,
        regul_body_force                            : float       = None                                ,
        regul_volume_subdomain_data                               = None                                ,
        regul_volume_subdomain_id                                 = None                                ,
        regul_surface_subdomain_data                              = None                                ,
        regul_surface_subdomain_id                                = None                                ,
        normalize_energies                          : bool        = False                               ,
        nonlinear_solver_type                       : str         = "newton"                            , # None, newton, cma/CMA, gradient-free-cma, gradient-free-minimize, gradient-free-differential_evolution
        nonlinear_solver_print_iterations           : bool        = False                               ,
        newton_tol_res_rel                          : float       = None                                ,
        newton_tol_dU                               : float       = None                                ,
        newton_tol_dU_rel                           : float       = None                                ,
        newton_n_iter_max                           : int         = 100                                 ,
        relax_type                                  : str         = None                                , # None, constant, aitken, backtracking, gss
        relax_backtracking_factor                   : float       = None                                ,
        relax_tol                                   : float       = None                                ,
        relax_n_iter_max                            : int         = None                                ,
        relax_must_advance                          : bool        = None                                ,
        register_ref_frame                          : bool        = False                               ,
        initialize_U_from_file                      : bool        = False                               ,
        initialize_U_folder                         : str         = None                                ,
        initialize_U_basename                       : str         = None                                ,
        initialize_U_ext                            : str         = "vtu"                               ,
        initialize_U_array_name                     : str         = "displacement"                      ,
        initialize_U_method                         : str         = "dofs_transfer"                     , # dofs_transfer, interpolation, projection
        initialize_reduced_U_from_file              : bool        = False                               ,
        initialize_reduced_U_filename               : str         = None                                ,
        write_qois_limited_precision                : bool        = False                               ,
        write_VTU_files                             : bool        = True                                ,
        write_VTU_files_with_preserved_connectivity : bool        = False                               ,
        write_XML_files                             : bool        = False                               ,
        iteration_mode                              : str         = "normal"                            , # normal, loop
        continue_after_fail                         : bool        = False                               ,
        print_out                                   : bool        = True                                ):

################################################################# kinematics ###

    if (kinematics_type == "full"):
        assert (initialize_reduced_U_from_file is False),\
            "Should use \"initialize_U_from_file\", not \"initialize_reduced_U_from_file\". Aborting."
        problem = dwarp.FullKinematicsWarpingProblem(
            working_folder=working_folder,
            working_basename=working_basename,
            mesh=mesh,
            mesh_folder=mesh_folder,
            mesh_basename=mesh_basename,
            U_degree=mesh_degree,
            print_out=print_out)
    elif (kinematics_type == "reduced"):
        assert (initialize_U_from_file is False),\
            "Should use \"initialize_reduced_U_from_file\", not \"initialize_U_from_file\". Aborting."
        problem = dwarp.ReducedKinematicsWarpingProblem(
            working_folder=working_folder,
            working_basename=working_basename,
            mesh=mesh,
            mesh_folder=mesh_folder,
            mesh_basename=mesh_basename,
            kinematics_model=reduced_kinematics_model,
            print_out=print_out)
    else:
        assert (0), "\"kinematics_type\" (="+str(kinematics_type)+") must be \"full\" or \"reduced\". Aborting."

############################################################### image series ###

    image_series = dwarp.ImageSeries(
        folder=images_folder,
        basename=images_basename,
        grad_basename=images_grad_basename,
        n_frames=images_n_frames,
        ext=images_ext,
        printer=problem.printer)

#################################################### image quadrature degree ###

    if (image_energy_quadrature is None):
        problem.printer.print_str("Computing quadrature degree…")
        problem.printer.inc()
        if (image_energy_quadrature_from == "points_count"):
            image_energy_quadrature = dwarp.compute_quadrature_degree_from_points_count(
                image_filename=image_series.get_image_filename(k_frame=images_ref_frame),
                mesh=problem.mesh,
                verbose=1)
        elif (image_energy_quadrature_from == "integral"):
            image_energy_quadrature = dwarp.compute_quadrature_degree_from_integral(
                image_filename=image_series.get_image_filename(k_frame=images_ref_frame),
                mesh=problem.mesh,
                verbose=1)
        else:
            assert (0), "\"image_energy_quadrature_from\" (="+str(image_energy_quadrature_from)+") must be \"points_count\" or \"integral\". Aborting."
        problem.printer.print_var("image_energy_quadrature",image_energy_quadrature)
        problem.printer.dec()

############################################################### image weight ###

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
        assert (regul_type is not None)
        if ("tractions" in regul_type):
            if (regul_type.startswith("discrete-linear-equilibrated-")):
                regul_types = ["discrete-linear-equilibrated"]
                if (regul_type == "discrete-linear-equilibrated-tractions"):
                    regul_types += ["discrete-linear-tractions"]
                elif (regul_type == "discrete-linear-equilibrated-tractions-normal"):
                    regul_types += ["discrete-linear-tractions-normal"]
                elif (regul_type == "discrete-linear-equilibrated-tractions-tangential"):
                    regul_types += ["discrete-linear-tractions-tangential"]
                elif (regul_type == "discrete-linear-equilibrated-tractions-normal-tangential"):
                    regul_types += ["discrete-linear-tractions-normal-tangential"]
            elif (regul_type.startswith("discrete-equilibrated-")):
                regul_types = ["discrete-equilibrated"]
                if (regul_type == "discrete-equilibrated-tractions"):
                    regul_types += ["discrete-tractions"]
                elif (regul_type == "discrete-equilibrated-tractions-normal"):
                    regul_types += ["discrete-tractions-normal"]
                elif (regul_type == "discrete-equilibrated-tractions-tangential"):
                    regul_types += ["discrete-tractions-tangential"]
                elif (regul_type == "discrete-equilibrated-tractions-normal-tangential"):
                    regul_types += ["discrete-tractions-normal-tangential"]
            else: assert (0), "Unknown regul_type ("+str(regul_type)+"). Aborting."
            regul_models = [regul_model  ]*2
            regul_levels = [regul_level/2]*2
        else:
            regul_types  = [regul_type ]
            regul_models = [regul_model]
            regul_levels = [regul_level]
    # print (regul_types)
    # print (regul_models)
    # print (regul_levels)

    image_w = 1.-sum(regul_levels)
    assert (image_w > 0.),\
        "1.-sum(regul_levels) must be positive. Aborting."

############################################################### image energy ###

    if (image_energy_type == "warped"):
        warped_image_energy = dwarp.WarpedImageContinuousEnergy(
            problem=problem,
            image_series=image_series,
            quadrature_degree=image_energy_quadrature,
            w=image_w,
            ref_frame=images_ref_frame,
            w_char_func=images_char_func,
            im_is_cone=images_is_cone,
            static_scaling=images_static_scaling,
            dynamic_scaling=images_dynamic_scaling)
        problem.add_image_energy(warped_image_energy)
    elif (image_energy_type == "generated"):
        generated_image_energy = dwarp.GeneratedImageDiscreteEnergy(
            problem=problem,
            image_series=image_series,
            quadrature_degree=image_energy_quadrature,
            texture=generated_image_energy_texture,
            w=image_w,
            ref_frame=images_ref_frame,
            resample=generated_image_energy_resample,
            resampling_factor=generated_image_energy_resampling_factor,
            ener_type=generated_image_energy_type)
        problem.add_image_energy(generated_image_energy)
    else:
        assert (0), "\"image_energy_type\" (="+str(image_energy_type)+") must be \"warped\" or \"generated\". Aborting."

############################################################# regularization ###

    for regul_type, regul_model, regul_level in zip(regul_types, regul_models, regul_levels):
        if (regul_level>0):
            name_suffix  = ""
            name_suffix += ("_"+    regul_type  )*(len(regul_types )>1)
            name_suffix += ("_"+    regul_model )*(len(regul_models)>1)
            name_suffix += ("_"+str(regul_level))*(len(regul_levels)>1)
            regul_body_force_ = None
            if regul_type.startswith("continuous"):
                regularization_energy_type = dwarp.RegularizationContinuousEnergy
                if regul_type.startswith("continuous-linear"):
                    regul_type_ = regul_type.split("-",2)[2]
                else:
                    regul_type_ = regul_type.split("-",1)[1]
            elif regul_type.startswith("discrete-simple"):
                regularization_energy_type = dwarp.SimpleRegularizationDiscreteEnergy
                regul_type_ = regul_type.split("-",2)[2]
            elif regul_type.startswith("discrete"):
                if ("equilibrated" in regul_type):
                    regularization_energy_type = dwarp.VolumeRegularizationDiscreteEnergy
                    regul_body_force_ = regul_body_force
                elif ("tractions" in regul_type):
                    regularization_energy_type = dwarp.SurfaceRegularizationDiscreteEnergy
                else: assert (0), "regul_type (= "+str(regul_type)+") unknown. Aborting."
                if regul_type.startswith("discrete-linear"):
                    regul_type_ = regul_type.split("-",2)[2]
                else:
                    regul_type_ = regul_type.split("-",1)[1]
            else: assert (0), "regul_type (= "+str(regul_type)+") unknown. Aborting."
            regularization_energy = regularization_energy_type(
                name="reg"+name_suffix,
                problem=problem,
                w=regul_level,
                type=regul_type_,
                model=regul_model,
                poisson=regul_poisson,
                b_fin=regul_body_force_,
                volume_subdomain_data=regul_volume_subdomain_data,
                volume_subdomain_id=regul_volume_subdomain_id,
                surface_subdomain_data=regul_surface_subdomain_data,
                surface_subdomain_id=regul_surface_subdomain_id,
                quadrature_degree=regul_quadrature)
            problem.add_regul_energy(regularization_energy)

####################################################### energy normalization ###

    if (normalize_energies):
        dwarp.compute_energies_normalization(
            problem=problem,
            verbose=1)

##################################################################### solver ###

    if (nonlinear_solver_type == "newton"):
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
                "tol_res_rel":newton_tol_res_rel,
                "tol_dU":newton_tol_dU,
                "tol_dU_rel":newton_tol_dU_rel,
                "n_iter_max":newton_n_iter_max,
                "write_iterations":nonlinear_solver_print_iterations})
    elif (nonlinear_solver_type in ["cma", "CMA"]):
        assert (relax_type is None),\
            "Not implemented. Aborting."
        solver = dwarp.CMANonlinearSolver(
            problem=problem,
            parameters={
                "working_folder":working_folder,
                "working_basename":working_basename,
                "write_iterations":nonlinear_solver_print_iterations})
    elif (nonlinear_solver_type.startswith("gradient-free")):
        assert (relax_type is None),\
            "Not implemented. Aborting."
        solver = dwarp.GradientFreeNonlinearSolver(
            problem=problem,
            parameters={
                "working_folder":working_folder,
                "working_basename":working_basename,
                "solver_type":nonlinear_solver_type.split("-", 2)[2],
                "write_iterations":nonlinear_solver_print_iterations})
    else:
        assert (0), "\"nonlinear_solver_type\" (="+str(nonlinear_solver_type)+") must be \"newton\", \"CMA\" or \"gradient-free\". Aborting."

############################################################# image iterator ###

    image_iterator = dwarp.ImageIterator(
        problem=problem,
        solver=solver,
        parameters={
            "working_folder":working_folder,
            "working_basename":working_basename,
            "register_ref_frame":register_ref_frame,
            "initialize_reduced_U_from_file":initialize_reduced_U_from_file,
            "initialize_reduced_U_filename":initialize_reduced_U_filename,
            "initialize_U_from_file":initialize_U_from_file,
            "initialize_U_folder":initialize_U_folder,
            "initialize_U_basename":initialize_U_basename,
            "initialize_U_ext":initialize_U_ext,
            "initialize_U_array_name":initialize_U_array_name,
            "initialize_U_method":initialize_U_method,
            "write_qois_limited_precision":write_qois_limited_precision,
            "write_VTU_files":write_VTU_files,
            "write_VTU_files_with_preserved_connectivity":write_VTU_files_with_preserved_connectivity,
            "write_XML_files":write_XML_files,
            "iteration_mode":iteration_mode,
            "continue_after_fail":continue_after_fail})

    success = image_iterator.iterate()

######################################################################## end ###

    problem.close()

    return success

fedic2 = warp

################################################################################

if (__name__ == "__main__"):
    import fire
    fire.Fire(warp)
