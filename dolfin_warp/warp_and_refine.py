#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2016-2023                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

import dolfin

import dolfin_warp as dwarp

################################################################################

def warp_and_refine(
        working_folder               : str,
        working_basename             : str,
        images_folder                : str,
        images_basename              : str,
        images_quadrature            : int         = None                       ,
        images_quadrature_from       : str         = "points_count"             , # points_count, integral
        mesh                         : dolfin.Mesh = None                       ,
        refinement_levels            : list        = [0]                        ,
        meshes                       : list        = None                       ,
        mesh_folder                  : str         = None                       ,
        mesh_basenames               : list        = None                       ,
        regul_type                   : str         = "continuous-equilibrated"  , # continuous-equilibrated, continuous-elastic, continuous-hyperelastic, discrete-linear-equilibrated, discrete-linear-elastic, discrete-equilibrated, discrete-tractions, discrete-tractions-normal, discrete-tractions-tan       gential, discrete-tractions-normal-tangential
        regul_types                  : list        = None                       ,
        regul_model                  : str         = "ciarletgeymonatneohookean", # hooke, kirchhoff, ciarletgeymonatneohookean, ciarletgeymonatneohookeanmooneyrivlin
        regul_models                 : list        = None                       ,
        regul_level                  : float       = 0.                         ,
        regul_levels                 : list        = None                       ,
        regul_poisson                : float       = 0.                         ,
        relax_type                   : str         = None                       , # constant, aitken, backtracking, gss
        relax_tol                    : float       = None                       ,
        relax_n_iter_max             : int         = None                       ,
        normalize_energies           : bool        = False                      ,
        tol_dU                       : float       = None                       ,
        n_iter_max                   : int         = 100                        ,
        continue_after_fail          : bool        = False                      ,
        write_qois_limited_precision : bool        = False                      ,
        print_iterations             : bool        = False                      ):

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

        success = dwarp.warp(
            working_folder                              = working_folder,
            working_basename                            = working_basename_for_warp,
            images_folder                               = images_folder,
            images_basename                             = images_basename,
            images_quadrature                           = images_quadrature,
            images_quadrature_from                      = images_quadrature_from,
            mesh                                        = mesh_for_warp,
            regul_type                                  = regul_type,
            regul_types                                 = regul_types,
            regul_model                                 = regul_model,
            regul_models                                = regul_models,
            regul_level                                 = regul_level,
            regul_levels                                = regul_levels,
            regul_poisson                               = regul_poisson,
            relax_type                                  = relax_type,
            relax_tol                                   = relax_tol,
            relax_n_iter_max                            = relax_n_iter_max,
            normalize_energies                          = normalize_energies,
            tol_dU                                      = tol_dU,
            n_iter_max                                  = n_iter_max,
            continue_after_fail                         = continue_after_fail,
            initialize_U_from_file                      = initialize_U_from_file,
            initialize_U_folder                         = working_folder,
            initialize_U_basename                       = working_basename_for_init,
            initialize_U_ext                            = "vtu",
            initialize_U_array_name                     = "displacement",
            initialize_U_method                         = "projection", # dofs_transfer, interpolation, projection
            write_qois_limited_precision                = write_qois_limited_precision,
            write_VTU_files                             = True,
            write_VTU_files_with_preserved_connectivity = True,
            write_XML_files                             = True,
            print_iterations                            = print_iterations)

        if not (success) and not (continue_after_fail):
            break

    return success

########################################################################

if (__name__ == "__main__"):
    import fire
    fire.Fire(warp_and_refine)
