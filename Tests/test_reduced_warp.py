#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2016-2025                                       ###
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

test = mypy.Test(
    res_folder=res_folder,
    perform_tests=1,
    stop_at_failure=1,
    clean_after_tests=1,
    qois_suffix="-strains")

n_dim_lst  = [ ]
n_dim_lst += [2]
# n_dim_lst += [3]

for n_dim in n_dim_lst:

    deformation_type_lst  = [             ]
    deformation_type_lst += ["translation"]
    deformation_type_lst += ["rotation"   ]
    deformation_type_lst += ["scaling"    ]
    deformation_type_lst += ["shear"      ]

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
        else:
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
        elif (deformation_type == "scaling"):
            deformation = {"type":"homogeneous", "X0":0.5, "Y0":0.5, "Z0":0.5, "Exx":+0.20}
        elif (deformation_type == "shear"):
            deformation = {"type":"homogeneous", "X0":0.5, "Y0":0.5, "Z0":0.5, "Fxy":+0.20}

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

        n_cells = 1
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

        res_basename  = images_basename
        res_basename += "-"+deformation_type

        print (n_dim)
        print (deformation_type)
        print (deformation_type)

        if (1): dwarp.warp(
            working_folder=res_folder,
            working_basename=res_basename,
            images_folder=res_folder,
            images_basename=images_basename,
            mesh=mesh,
            kinematics_type="reduced",
            reduced_kinematics_model=deformation_type,
            normalize_energies=1,
            relax_type="backtracking",
            tol_dU=1e-2,
            write_qois_limited_precision=1)

        if (1): dwarp.compute_strains(
            working_folder=res_folder,
            working_basename=res_basename,
            verbose=1)

        test.test(res_basename)

        res_basename  = images_basename
        res_basename += "-"+"all"

        print (n_dim)
        print (deformation_type)
        print ("all")

        if (1): dwarp.warp(
            working_folder=res_folder,
            working_basename=res_basename,
            images_folder=res_folder,
            images_basename=images_basename,
            mesh=mesh,
            kinematics_type="reduced",
            reduced_kinematics_model="translation+rotation+scaling+shear",
            normalize_energies=1,
            relax_type="backtracking",
            tol_dU=1e-2,
            write_qois_limited_precision=1)

        if (1): dwarp.compute_strains(
            working_folder=res_folder,
            working_basename=res_basename,
            verbose=1)

        test.test(res_basename)
