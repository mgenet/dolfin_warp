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

def warp_and_refine(
        working_folder                    : str,
        working_basename                  : str,
        images_folder                     : str,
        images_basename                   : str,
        mesh                              : dolfin.Mesh = None                            ,
        refinement_levels                 : list        = [0]                             ,
        meshes                            : list        = None                            ,
        mesh_folder                       : str         = None                            ,
        mesh_basenames                    : list        = None                            ,
        image_energy_quadrature           : int         = None                            ,
        image_energy_quadrature_from      : str         = "points_count"                  , # points_count, integral
        regul_type                        : str         = "continuous-equilibrated"       , # continuous-equilibrated, continuous-elastic, continuous-hyperelastic, discrete-linear-equilibrated, discrete-linear-elastic, discrete-equilibrated, discrete-tractions, discrete-tractions-normal, discrete-tractions-tangential, discrete-tractions-normal-tangential
        regul_types                       : list        = None                            ,
        regul_model                       : str         = "ogdenciarletgeymonatneohookean", # hooke, kirchhoff, ogdenciarletgeymonatneohookean, ogdenciarletgeymonatneohookeanmooneyrivlin
        regul_models                      : list        = None                            ,
        regul_level                       : float       = 0.                              ,
        regul_levels                      : list        = None                            ,
        regul_poisson                     : float       = 0.                              ,
        regul_body_force                  : float       = None                            ,
        regul_volume_subdomain_data                     = None                            ,
        regul_volume_subdomain_id                       = None                            ,
        regul_surface_subdomain_data                    = None                            ,
        regul_surface_subdomain_id                      = None                            ,
        normalize_energies                : bool        = False                           ,
        nonlinear_solver_print_iterations : bool        = False                           ,
        newton_tol_dU                     : float       = None                            ,
        newton_n_iter_max                 : int         = 100                             ,
        relax_type                        : str         = None                            , # constant, aitken, backtracking, gss
        relax_tol                         : float       = None                            ,
        relax_n_iter_max                  : int         = None                            ,
        write_qois_limited_precision      : bool        = False                           ,
        continue_after_fail               : bool        = False                           ,
        print_out                         : bool        = True                            ):

    if (meshes is None):
        meshes = []
        if (mesh is not None) and (refinement_levels is not None):
            for refinement_level in refinement_levels:
                mesh_for_warp = dolfin.Mesh(mesh)
                for _ in range(refinement_level):
                    mesh_for_warp = dolfin.refine(mesh_for_warp)
                meshes += [mesh_for_warp]
        elif (mesh_folder is not None) and (mesh_basenames is not None):
            for mesh_basename in mesh_basenames:
                mesh_filename = mesh_folder+"/"+mesh_basename+".xml"
                meshes += [dolfin.Mesh(mesh_filename)]
    
    regul_volume_subdomain_data_lst = regul_volume_subdomain_data
    regul_volume_subdomain_id_lst   = regul_volume_subdomain_id
    regul_surface_subdomain_data_lst = regul_surface_subdomain_data
    regul_surface_subdomain_id_lst   = regul_surface_subdomain_id
    
    for k_mesh, mesh_for_warp in enumerate(meshes):
        working_basename_for_warp  = working_basename
        working_basename_for_warp += "-refine="+str(k_mesh)

        if (k_mesh == 0):
            initialize_U_from_file = False

            working_basename_for_init = None
 
        else:
            initialize_U_from_file = True

            working_basename_for_init  = working_basename
            working_basename_for_init += "-refine="+str(k_mesh-1)

        if (regul_volume_subdomain_data):
            regul_volume_subdomain_data = regul_volume_subdomain_data_lst[k_mesh]
            regul_volume_subdomain_id   = regul_volume_subdomain_id_lst[k_mesh]
        if (regul_surface_subdomain_data):
            regul_surface_subdomain_data = regul_surface_subdomain_data_lst[k_mesh]
            regul_surface_subdomain_id   = regul_surface_subdomain_id_lst[k_mesh]

        success = dwarp.warp(
            working_folder                              = working_folder                   ,
            working_basename                            = working_basename_for_warp        ,
            images_folder                               = images_folder                    ,
            images_basename                             = images_basename                  ,
            mesh                                        = mesh_for_warp                    ,
            image_energy_quadrature                     = image_energy_quadrature          ,
            image_energy_quadrature_from                = image_energy_quadrature_from     ,
            regul_type                                  = regul_type                       ,
            regul_types                                 = regul_types                      ,
            regul_model                                 = regul_model                      ,
            regul_models                                = regul_models                     ,
            regul_level                                 = regul_level                      ,
            regul_levels                                = regul_levels                     ,
            regul_poisson                               = regul_poisson                    ,
            regul_body_force                            = regul_body_force                 ,
            regul_volume_subdomain_data                 = regul_volume_subdomain_data      ,
            regul_volume_subdomain_id                   = regul_volume_subdomain_id        ,
            regul_surface_subdomain_data                = regul_surface_subdomain_data     ,
            regul_surface_subdomain_id                  = regul_surface_subdomain_id       ,
            normalize_energies                          = normalize_energies               ,
            nonlinear_solver_print_iterations           = nonlinear_solver_print_iterations,
            newton_tol_dU                               = newton_tol_dU                    ,
            newton_n_iter_max                           = newton_n_iter_max                ,
            relax_type                                  = relax_type                       ,
            relax_tol                                   = relax_tol                        ,
            relax_n_iter_max                            = relax_n_iter_max                 ,
            initialize_U_from_file                      = initialize_U_from_file           ,
            initialize_U_folder                         = working_folder                   ,
            initialize_U_basename                       = working_basename_for_init        ,
            initialize_U_ext                            = "vtu"                            ,
            initialize_U_array_name                     = "displacement"                   ,
            initialize_U_method                         = "projection"                     , # dofs_transfer, interpolation, projection
            write_qois_limited_precision                = write_qois_limited_precision     ,
            write_VTU_files                             = True                             ,
            write_VTU_files_with_preserved_connectivity = True                             ,
            write_XML_files                             = True                             ,
            continue_after_fail                         = continue_after_fail              ,
            print_out                                   = print_out                        )

        if not (success) and not (continue_after_fail):
            break

    return success

########################################################################

if (__name__ == "__main__"):
    import fire
    fire.Fire(warp_and_refine)
