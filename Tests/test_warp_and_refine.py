#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2016-2023                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

import dolfin
import sys

import myPythonLibrary as mypy
import dolfin_warp     as dwarp

################################################################################

res_folder = sys.argv[0][:-3]

# test = mypy.Test(
#     res_folder=res_folder,
#     perform_tests=1,
#     stop_at_failure=1,
#     clean_after_tests=1,
#     qois_suffix="-strains")

n_dim_lst  = [ ]
n_dim_lst += [2]
# n_dim_lst += [3]

for n_dim in n_dim_lst:

    deformation_type_lst = []
    # deformation_type_lst += ["translation"]
    deformation_type_lst += ["rotation"   ]

    for deformation_type in deformation_type_lst:

        if (n_dim == 2):
            images_basename = "square"
        elif (n_dim == 3):
            images_basename = "cube"
        images_basename += "-"+deformation_type

        images = {
            "n_dim":n_dim,
            "L":[1.]*n_dim,
            "n_voxels": [99]*n_dim,
            "T":1.,
            "n_frames":11,
            "data_type":"float",
            "folder":res_folder,
            "basename":images_basename}

        if (deformation_type == "translation"):
            structure_Xmin = [0.1]+[0.3]*(n_dim-1)
            structure_Xmax = [0.5]+[0.7]*(n_dim-1)
        elif (deformation_type == "rotation"):
            structure_Xmin = [0.3]+[0.3]*(n_dim-1)
            structure_Xmax = [0.7]+[0.7]*(n_dim-1)
        structure = {
            "type":"box",
            "Xmin":structure_Xmin,
            "Xmax":structure_Xmax}

        texture = {
            "type":"tagging",
            "s":0.1}

        noise = {
            "type":"no"}

        if (deformation_type == "translation"):
            deformation = {"type":"translation", "Dx":0.4, "Dy":0.}
        elif (deformation_type == "rotation"):
            deformation = {"type":"rotation", "Cx":0.5, "Cy":0.5, "Rz":45.}

        evolution = {
            "type":"linear"}

        if (1): dwarp.generate_images(
            images=images,
            structure=structure,
            texture=texture,
            noise=noise,
            deformation=deformation,
            evolution=evolution,
            verbose=1)

        n_cells = 4
        if (n_dim == 2):
            mesh = dolfin.RectangleMesh(
                dolfin.Point(structure_Xmin),
                dolfin.Point(structure_Xmax),
                n_cells, n_cells,
                "crossed")
        elif (n_dim == 3):
            mesh = dolfin.BoxMesh(
                dolfin.Point(structure_Xmin),
                dolfin.Point(structure_Xmax),
                n_cells, n_cells, n_cells)

        regul_type    = "discrete-equilibrated-tractions-normal-tangential"
        regul_model   = "ogdenciarletgeymonatneohookean"
        regul_level   = 0.1
        regul_poisson = 0.

        if (1): dwarp.warp_and_refine(
                working_folder=res_folder,
                working_basename=images_basename,
                images_folder=res_folder,
                images_basename=images_basename,
                mesh=mesh,
                refinement_levels=[0,1],
                regul_type=regul_type,
                regul_model=regul_model,
                regul_level=regul_level,
                regul_poisson=regul_poisson,
                normalize_energies=1,
                relax_type="gss",
                relax_tol=1e-3,
                relax_n_iter_max=100,
                tol_dU=1e-2,
                continue_after_fail=1,
                print_iterations=1,
                write_qois_limited_precision=1)

        n_cells_lst  = []
        n_cells_lst += [4]
        n_cells_lst += [4*2**1]

        mesh_basenames = []
        for n_cells in n_cells_lst:
            if (n_dim == 2):
                mesh = dolfin.RectangleMesh(
                    dolfin.Point(structure_Xmin),
                    dolfin.Point(structure_Xmax),
                    n_cells, n_cells,
                    "crossed")
            elif (n_dim == 3):
                mesh = dolfin.BoxMesh(
                    dolfin.Point(structure_Xmin),
                    dolfin.Point(structure_Xmax),
                    n_cells, n_cells, n_cells)

            mesh_basename  = images_basename
            mesh_basename += "-n_cells="+str(n_cells)

            mesh_filename = res_folder+"/"+mesh_basename+".xml"
            dolfin.File(mesh_filename) << mesh

            mesh_basenames += [mesh_basename]

        if (1): dwarp.warp_and_refine(
                working_folder=res_folder,
                working_basename=images_basename,
                images_folder=res_folder,
                images_basename=images_basename,
                mesh_folder=res_folder,
                mesh_basenames=mesh_basenames,
                regul_type=regul_type,
                regul_model=regul_model,
                regul_level=regul_level,
                regul_poisson=regul_poisson,
                normalize_energies=1,
                relax_type="gss",
                relax_tol=1e-3,
                relax_n_iter_max=100,
                tol_dU=1e-2,
                continue_after_fail=1,
                print_iterations=1,
                write_qois_limited_precision=1)
                
        # test.test(images_basename)
