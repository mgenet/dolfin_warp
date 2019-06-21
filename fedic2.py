#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2016-2019                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

from builtins import *

import myPythonLibrary    as mypy
import myVTKPythonLibrary as myvtk

import dolfin_dic as ddic

################################################################################

def fedic2(
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
        images_expressions_type="cpp", # cpp, py
        images_dynamic_scaling=1,
        images_is_cone=0,
        mesh=None,
        mesh_folder=None,
        mesh_basename=None,
        mesh_degree=1,
        regul_type="equilibrated", # hyperelastic, equilibrated
        regul_model="neohookean", # linear, kirchhoff, neohookean, mooneyrivlin
        regul_quadrature=None,
        regul_level=0.1,
        regul_poisson=0.0,
        tangent_type="Idef", # Idef, Idef-wHess, Iold, Iref
        residual_type="Iref", # Iref, Iold, Iref-then-Iold
        relax_type="gss", # constant, aitken, gss
        relax_init=1.0,
        initialize_U_from_file=0,
        initialize_U_folder=None,
        initialize_U_basename=None,
        initialize_U_ext="vtu",
        initialize_U_array_name="displacement",
        initialize_DU_with_DUold=0,
        tol_res=None,
        tol_res_rel=None,
        tol_dU=None,
        tol_im=None,
        n_iter_max=100,
        continue_after_fail=0,
        print_refined_mesh=0,
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

    problem = ddic.ImageRegistrationProblem(
        mesh=mesh,
        mesh_folder=mesh_folder,
        mesh_basename=mesh_basename,
        U_degree=mesh_degree)

    image_series = ddic.ImageSeries(
        problem=problem,
        folder=images_folder,
        basename=images_basename,
        grad_basename=images_grad_basename,
        n_frames=images_n_frames,
        ext=images_ext)

    if (images_quadrature is None):
        problem.printer.print_str("Computing quadrature degree…")
        problem.printer.inc()
        if (images_quadrature_from == "points_count"):
            images_quadrature = ddic.compute_quadrature_degree_from_points_count(
                image_filename=image_series.get_image_filename(images_ref_frame),
                mesh=mesh,
                verbose=1)
        elif (method == "integral"):
            images_quadrature = ddic.compute_quadrature_degree_from_integral(
                image_filename=self.get_image_filename(images_ref_frame),
                mesh=mesh,
                verbose=1)
        else:
            assert (0), "\"images_quadrature_from\" (="+str(images_quadrature_from)+") must be \"points_count\" or \"integral\". Aborting."
        problem.printer.print_var("images_quadrature",images_quadrature)
        problem.printer.dec()

    warped_image_energy = ddic.WarpedImageEnergy(
        problem=problem,
        image_series=image_series,
        quadrature_degree=images_quadrature,
        w=1.-regul_level,
        ref_frame=images_ref_frame,
        dynamic_scaling=images_dynamic_scaling,
        im_is_cone=images_is_cone)
    problem.add_image_energy(warped_image_energy)

    regularization_energy = ddic.RegularizationEnergy(
        problem=problem,
        w=regul_level,
        type=regul_type,
        model=regul_model,
        poisson=regul_poisson,
        quadrature_degree=regul_quadrature)
    problem.add_regul_energy(regularization_energy)

    solver = ddic.NonlinearSolver(
        problem=problem,
        parameters={
            "working_folder":working_folder,
            "working_basename":working_basename,
            "relax_type":relax_type,
            "tol_res_rel":tol_res_rel,
            "tol_dU":tol_dU,
            "n_iter_max":n_iter_max,
            "write_iterations":print_iterations})

    image_iterator = ddic.ImageIterator(
        problem=problem,
        solver=solver,
        parameters={
            "working_folder":working_folder,
            "working_basename":working_basename,
            "initialize_U_from_file":initialize_U_from_file,
            "initialize_U_folder":initialize_U_folder,
            "initialize_U_basename":initialize_U_basename,
            "initialize_U_ext":initialize_U_ext,
            "initialize_U_array_name":initialize_U_array_name,
            "initialize_DU_with_DUold":initialize_DU_with_DUold})

    image_iterator.iterate()
