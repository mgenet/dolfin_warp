#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2016-2026                                       ###
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

    deformation = {
        "type":"homogeneous",
        "X0":0.5, "Y0":0.5, "Z0":0.5,
        "Exx":+0.20}

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

    ######################################## full kinematics, Newton variants ###

    solver_lst  = [] # name, solver type, solver options, print iterations
    solver_lst += [["newton-backtracking"         , "newton", {"relax_type":"backtracking"                         }, 1]] # also writes the iterations
    solver_lst += [["newton-backtracking-max_dU"  , "newton", {"relax_type":"backtracking", "relax_max_dU_inf":0.01}, 0]]
    solver_lst += [["newton-constant"             , "newton", {"relax_type":"constant", "relax":1.                 }, 0]]
    solver_lst += [["newton-aitken"               , "newton", {"relax_type":"aitken"                               }, 0]]
    # solver_lst += [["newton-gss"                , "newton", {"relax_type":"gss"                                  }, 1]] # MG20261010: Does not converge on this case (the golden-section bracket allows negative steps, and the first evaluation is excluded from the argmin), to be checked
    solver_lst += [["newton-bfgs"                 , "newton", {"relax_type":"backtracking", "use_bfgs":True        }, 0]]
    solver_lst += [["newton-damping_elastic"      , "newton", {"relax_type":"backtracking", "damping_type":"elastic"}, 0]]
    solver_lst += [["newton-damping_mass"         , "newton", {"relax_type":"backtracking", "damping_type":"mass"   }, 0]]
    solver_lst += [["scipy-L-BFGS-B"              , "scipy" , {"method":"L-BFGS-B"                                 }, 0]]

    for solver_name, solver_type, solver_options, solver_print_iterations in solver_lst:

        res_basename  = images_basename
        res_basename += "-full"
        res_basename += "-"+solver_name

        print (n_dim)
        print (solver_name)

        if (1): dwarp.warp(
            working_folder=res_folder,
            working_basename=res_basename,
            images_folder=res_folder,
            images_basename=images_basename,
            mesh=mesh,
            regul_type="continuous-elastic",
            regul_model="ogdenciarletgeymonatneohookean",
            regul_level=0.01,
            normalize_energies=1,
            nonlinear_solver_type=solver_type,
            nonlinear_solver_options=dict(solver_options, **({"tol_dU_rel_U":1e-2} if (solver_type == "newton") else {})),
            nonlinear_solver_print_iterations=solver_print_iterations,
            write_qois_limited_precision=1)

        if (1): dwarp.compute_strains(
            working_folder=res_folder,
            working_basename=res_basename,
            verbose=1)

        test.test(res_basename)

    ######################## reduced kinematics, derivative-free solvers ###

    solver_lst  = []
    solver_lst += [["scipy-Nelder-Mead", "scipy", {"method":"Nelder-Mead", "maxiter":2000}]]
    # solver_lst += [["cma", "cma", {}]] # MG20261010: stochastic (and needs the cma package), so not compared to a reference

    for solver_name, solver_type, solver_options in solver_lst:

        res_basename  = images_basename
        res_basename += "-reduced"
        res_basename += "-"+solver_name

        print (n_dim)
        print (solver_name)

        if (1): dwarp.warp(
            working_folder=res_folder,
            working_basename=res_basename,
            images_folder=res_folder,
            images_basename=images_basename,
            mesh=mesh,
            kinematics_type="reduced",
            reduced_kinematics_model="translation+rotation+scaling+shear",
            normalize_energies=1,
            nonlinear_solver_type=solver_type,
            nonlinear_solver_options=solver_options,
            write_qois_limited_precision=1)

        if (1): dwarp.compute_strains(
            working_folder=res_folder,
            working_basename=res_basename,
            verbose=1)

        test.test(res_basename)
