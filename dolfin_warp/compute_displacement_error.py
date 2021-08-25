#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2016-2021                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

from builtins import range

import dolfin
import glob
import numpy

import myPythonLibrary    as mypy
import myVTKPythonLibrary as myvtk

import dolfin_warp as dwarp

################################################################################

def compute_displacement_error(
        working_folder,
        working_basename,
        ref_folder,
        ref_basename,
        working_ext="vtu",
        ref_ext="vtu",
        working_disp_array_name="displacement",
        ref_disp_array_name="displacement",
        verbose=1,
        sort_mesh=0):

    working_filenames = glob.glob(working_folder+"/"+working_basename+"_[0-9]*."+working_ext)
    ref_filenames = glob.glob(ref_folder+"/"+ref_basename+"_[0-9]*."+ref_ext)

    working_zfill = len(working_filenames[0].rsplit("_",1)[-1].split(".",1)[0])
    ref_zfill = len(ref_filenames[0].rsplit("_",1)[-1].split(".",1)[0])
    if (verbose): print("ref_zfill = " + str(ref_zfill))
    if (verbose): print("working_zfill = " + str(working_zfill))

    n_frames = len(working_filenames)
    assert (len(ref_filenames) == n_frames)
    if (verbose): print("n_frames = " + str(n_frames))

    error_file = open(working_folder+"/"+working_basename+"-displacement_error.dat", "w")
    error_file.write("#t e\n")

    err_int = numpy.empty(n_frames)
    ref_int = numpy.empty(n_frames)
    ref_max = float("-Inf")
    for k_frame in range(n_frames):
        ref = myvtk.readUGrid(
            filename=ref_folder+"/"+ref_basename+"_"+str(k_frame).zfill(ref_zfill)+"."+ref_ext,
            verbose=verbose-1)
        n_points = ref.GetNumberOfPoints()
        n_cells = ref.GetNumberOfCells()
        sol = myvtk.readUGrid(
            filename=working_folder+"/"+working_basename+"_"+str(k_frame).zfill(working_zfill)+"."+working_ext,
            verbose=verbose-1)
        assert (sol.GetNumberOfPoints() == n_points)
        assert (sol.GetNumberOfCells() == n_cells)

        ref_disp = ref.GetPointData().GetArray(ref_disp_array_name)
        working_disp = sol.GetPointData().GetArray(working_disp_array_name)

        if (sort_mesh):
            # FA20200311: sort_ref and sort_working are created because enumeration is not the same in the meshes of ref and sol
            from vtk.util import numpy_support
            coords_ref     = numpy_support.vtk_to_numpy(ref.GetPoints().GetData())
            coords_working = numpy_support.vtk_to_numpy(sol.GetPoints().GetData())

            sort_ref     = [i_sort[0] for i_sort in sorted(enumerate(coords_ref.tolist()), key=lambda k: [k[1],k[0]])]
            sort_working = [i_sort[0] for i_sort in sorted(enumerate(coords_working.tolist()), key=lambda k: [k[1],k[0]])]

            err_int[k_frame] = numpy.sqrt(numpy.mean([numpy.sum([(working_disp.GetTuple(sort_working[k_point])[k_dim]-ref_disp.GetTuple(sort_ref[k_point])[k_dim])**2 for k_dim in range(3)]) for k_point in range(n_points)]))

        else:
            err_int[k_frame] = numpy.sqrt(numpy.mean([numpy.sum([(working_disp.GetTuple(k_point)[k_dim]-ref_disp.GetTuple(k_point)[k_dim])**2 for k_dim in range(3)]) for k_point in range(n_points)]))

        ref_int[k_frame] = numpy.sqrt(numpy.mean([numpy.sum([(ref_disp.GetTuple(k_point)[k_dim])**2 for k_dim in range(3)]) for k_point in range(n_points)]))
        ref_max = max(ref_max, numpy.max([numpy.sum([((ref_disp.GetTuple(k_point)[k_dim])**2)**0.5 for k_dim in range(3)]) for k_point in range(n_points)]))

    error_file.write("\n".join([" ".join([str(val) for val in [k_frame, err_int[k_frame], ref_int[k_frame]]]) for k_frame in range(n_frames)]))

    err_int_int = numpy.mean(err_int**2)**(0.5)
    ref_int_int = numpy.mean(ref_int**2)**(0.5)

    # print (err_int_int)
    # print (ref_int_int)
    print (err_int_int/ref_int_int)

    # error_file.write("\n\n")
    # error_file.write(" ".join([str(val) for val in [err_int_int, ref_int_int]]))

    error_file.close()
