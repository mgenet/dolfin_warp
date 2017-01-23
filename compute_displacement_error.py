#coding=utf8

########################################################################
###                                                                  ###
### Created by Martin Genet, 2016-2017                               ###
###                                                                  ###
### École Polytechnique, Palaiseau, France                           ###
###                                                                  ###
########################################################################

import dolfin
import glob
import numpy

import myPythonLibrary as mypy
import myVTKPythonLibrary as myvtk

import dolfin_dic as ddic

########################################################################

def compute_displacement_error(
        sol_folder,
        sol_basename,
        ref_folder,
        ref_basename,
        sol_ext="vtu",
        ref_ext="vtu",
        sol_disp_array_name="displacement",
        ref_disp_array_name="displacement",
        verbose=1):

    sol_filenames = glob.glob(sol_folder+"/"+sol_basename+"_[0-9]*."+sol_ext)
    ref_filenames = glob.glob(ref_folder+"/"+ref_basename+"_[0-9]*."+ref_ext)

    sol_zfill = len(sol_filenames[0].rsplit("_",1)[-1].split(".",1)[0])
    ref_zfill = len(ref_filenames[0].rsplit("_",1)[-1].split(".",1)[0])
    if (verbose): print "ref_zfill = " + str(ref_zfill)
    if (verbose): print "sol_zfill = " + str(sol_zfill)

    n_frames = len(sol_filenames)
    assert (len(ref_filenames) == n_frames)
    if (verbose): print "n_frames = " + str(n_frames)

    error_file = open(sol_folder+"/"+sol_basename+"-displacement_error.dat", "w")
    error_file.write("#t e\n")

    err_int = numpy.empty(n_frames)
    ref_int = numpy.empty(n_frames)
    for k_frame in xrange(n_frames):
        ref = myvtk.readUGrid(
            filename=ref_folder+"/"+ref_basename+"_"+str(k_frame).zfill(ref_zfill)+"."+ref_ext,
            verbose=0)
        n_points = ref.GetNumberOfPoints()
        n_cells = ref.GetNumberOfCells()
        sol = myvtk.readUGrid(
            filename=sol_folder+"/"+sol_basename+"_"+str(k_frame).zfill(sol_zfill)+"."+sol_ext,
            verbose=0)
        assert (sol.GetNumberOfPoints() == n_points)
        assert (sol.GetNumberOfCells() == n_cells)

        ref_disp = ref.GetPointData().GetArray(ref_disp_array_name)
        sol_disp = sol.GetPointData().GetArray(sol_disp_array_name)

        err_int[k_frame] = numpy.sqrt(numpy.mean([numpy.sum([(sol_disp.GetTuple(k_point)[k_dim]-ref_disp.GetTuple(k_point)[k_dim])**2 for k_dim in xrange(3)]) for k_point in xrange(n_points)]))
        ref_int[k_frame] = numpy.sqrt(numpy.mean([numpy.sum([(ref_disp.GetTuple(k_point)[k_dim])**2 for k_dim in xrange(3)]) for k_point in xrange(n_points)]))

    ref_int_int = numpy.mean(ref_int)
    err_int_rel = err_int/ref_int_int

    error_file.write("\n".join([" ".join([str(val) for val in [k_frame, err_int[k_frame], ref_int[k_frame], err_int_rel[k_frame]]]) for k_frame in xrange(n_frames)]))

    error_file.close()
