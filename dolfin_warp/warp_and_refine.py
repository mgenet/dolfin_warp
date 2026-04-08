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
        working_folder      : str                ,
        working_basename    : str                ,
        meshes              : list        = None ,
        mesh                : dolfin.Mesh = None ,
        refinement_levels   : list        = None ,
        mesh_folder         : str         = None ,
        mesh_basenames      : list        = None ,
        continue_after_fail : bool        = False,
        **kwargs                                 ):

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

        kwargs.setdefault("write_VTU_files"                            , True)
        kwargs.setdefault("write_VTU_files_with_preserved_connectivity", True)
        kwargs.setdefault("write_XML_files"                            , True)
        
        success = dwarp.warp(
            working_folder          = working_folder           ,
            working_basename        = working_basename_for_warp,
            mesh                    = mesh_for_warp            ,
            initialize_U_from_file  = initialize_U_from_file   ,
            initialize_U_folder     = working_folder           ,
            initialize_U_basename   = working_basename_for_init,
            initialize_U_ext        = "vtu"                    ,
            initialize_U_array_name = "displacement"           ,
            initialize_U_method     = "projection"             ,
            continue_after_fail     = continue_after_fail      ,
            **kwargs                                           )

        if not (success) and not (continue_after_fail):
            break

    return success

########################################################################

if (__name__ == "__main__"):
    import fire
    fire.Fire(warp_and_refine)
