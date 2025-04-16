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

    if (n_dim == 2):
        images_basename = "square"
    elif (n_dim == 3):
        images_basename = "cube"

    images = {
        "n_dim":n_dim,
        "L":[1.]*n_dim,
        "n_voxels": [99]*n_dim,
        "T":1.,
        "n_frames":11,
        "data_type":"float",
        "folder":res_folder,
        "basename":images_basename}

    structure_Xmin = [0.1]+[0.3]*(n_dim-1)
    structure_Xmax = [0.5]+[0.7]*(n_dim-1)
    structure = {
        "type":"box",
        "Xmin":structure_Xmin,
        "Xmax":structure_Xmax}

    texture = {
        "type":"tagging",
        "s":0.1}

    noise = {
        "type":"no"}

    deformation = {
        "type":"translation",
        "Dx":0.4,
        "Dy":0.0}

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

    regul_type_lst  = []
    regul_type_lst += ["continuous-linear-elastic"                               ]
    regul_type_lst += ["continuous-linear-equilibrated"                          ]
    regul_type_lst += ["continuous-elastic"                                      ]
    regul_type_lst += ["continuous-equilibrated"                                 ]
    regul_type_lst += ["discrete-simple-elastic"                                 ]
    regul_type_lst += ["discrete-simple-equilibrated"                            ]
    regul_type_lst += ["discrete-linear-equilibrated"                            ]
    regul_type_lst += ["discrete-linear-equilibrated-tractions-normal"           ]
    regul_type_lst += ["discrete-linear-equilibrated-tractions-tangential"       ]
    regul_type_lst += ["discrete-linear-equilibrated-tractions-normal-tangential"]
    regul_type_lst += ["discrete-equilibrated"                                   ]
    regul_type_lst += ["discrete-equilibrated-tractions-normal"                  ]
    regul_type_lst += ["discrete-equilibrated-tractions-tangential"              ]
    regul_type_lst += ["discrete-equilibrated-tractions-normal-tangential"       ]

    for regul_type in regul_type_lst:

        res_basename = images_basename
        res_basename += "-"+regul_type

        if any([_ in regul_type for _ in ["linear", "simple"]]):
            regul_model = "hooke"
        else:
            regul_model = "ogdenciarletgeymonatneohookean"

        regul_level = 0.1

        print (n_dim)
        print (regul_type)

        if (1): dwarp.warp(
            working_folder=res_folder,
            working_basename=res_basename,
            images_folder=res_folder,
            images_basename=images_basename,
            mesh=mesh,
            regul_type=regul_type,
            regul_model=regul_model,
            regul_level=regul_level,
            normalize_energies=1,
            relax_type="backtracking",
            tol_dU=1e-2,
            write_qois_limited_precision=1)

        if (1): dwarp.compute_strains(
            working_folder=res_folder,
            working_basename=res_basename,
            verbose=1)

        test.test(res_basename)
