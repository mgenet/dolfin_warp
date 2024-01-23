#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2016-2024                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

import os
import shutil
import sys

import myPythonLibrary as mypy
import dolfin_mech     as dmech
import dolfin_warp     as dwarp

################################################################################

res_folder = sys.argv[0][:-3]
if not os.path.exists(res_folder): os.mkdir(res_folder)

# test = mypy.Test(
#     res_folder=res_folder,
#     perform_tests=0,
#     stop_at_failure=0,
#     clean_after_tests=0,
#     qois_suffix="")

images = {
    "T":1.,
    "n_frames":2}

X0 = [0.5,0.5]
R = 0.35
structure = {"type":"disc", "X0":X0, "R":R}

# deformation = {"type":"homogeneous", "X0":X0[0], "Y0":X0[1], "Fxx":-0.9}
deformation = {"type":"homogeneous", "X0":X0[0], "Y0":X0[1], "Exx":+0.1, "Eyy":+0.1}

evolution = {"type":"linear"}

N = 10
mesh_basename = "mesh"
mesh, _, _, _, _, _, _, _ = dmech.run_Disc_Mesh(params={"X0":X0[0], "Y0":X0[1], "R":R, "l":R/N, "mesh_filebasename":res_folder+"/"+mesh_basename})

dwarp.compute_warped_mesh(
    working_folder=res_folder,
    working_basename=mesh_basename,
    images=images,
    structure=structure,
    deformation=deformation,
    evolution=evolution,
    mesh=mesh,
    mesh_ext="vtu",
    verbose=1)            

regul_type_lst  = []
regul_type_lst += ["continuous-linear-elastic"           ]
regul_type_lst += ["continuous-linear-equilibrated"      ]
regul_type_lst += ["continuous-elastic"                  ]
regul_type_lst += ["continuous-equilibrated"             ]
regul_type_lst += ["discrete-simple-elastic"             ]
regul_type_lst += ["discrete-simple-equilibrated"        ]
regul_type_lst += ["discrete-linear-equilibrated"        ]
regul_type_lst += ["discrete-linear-tractions"           ]
regul_type_lst += ["discrete-linear-tractions-normal"    ]
regul_type_lst += ["discrete-linear-tractions-tangential"]
regul_type_lst += ["discrete-equilibrated"               ]
regul_type_lst += ["discrete-tractions"                  ]
regul_type_lst += ["discrete-tractions-normal"           ]
regul_type_lst += ["discrete-tractions-tangential"       ]

dwarp.compute_regularization_energies(
    dim = 2,
    working_folder = res_folder,
    working_basename = mesh_basename,
    working_ext = "vtu",
    working_displacement_array_name = "displacement",
    noise_type = None,
    noise_level = 0.,
    regul_types = regul_type_lst,
    regul_model_for_lin = "hooke",
    regul_model_for_nl = "ogdenciarletgeymonatneohookean",
    regul_poisson = 0.,
    normalize_energies = True,
    write_regularization_energy_file = True,
    plot_regularization_energy = False,
    verbose = 1)

# test.test(mesh_basename+"-regul_ener")

shutil.rmtree(res_folder, ignore_errors=1)
