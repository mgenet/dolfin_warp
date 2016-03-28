#coding=utf8

########################################################################
###                                                                  ###
### Created by Martin Genet, 2016                                    ###
###                                                                  ###
### École Polytechnique, Palaiseau, France                           ###
###                                                                  ###
########################################################################

import glob
import numpy

import myVTKPythonLibrary as myVTK

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

    n_frames = len(glob.glob(ref_folder+"/"+ref_basename+"_*."+ref_ext))
    assert (len(glob.glob(sol_folder+"/"+sol_basename+"_*."+sol_ext)) == n_frames)
    if (verbose): print "n_frames = " + str(n_frames)

    ref_zfill = len(glob.glob(ref_folder+"/"+ref_basename+"_*."+ref_ext)[0].rsplit("_")[-1].split(".")[0])
    sol_zfill = len(glob.glob(sol_folder+"/"+sol_basename+"_*."+sol_ext)[0].rsplit("_")[-1].split(".")[0])
    if (verbose): print "ref_zfill = " + str(ref_zfill)
    if (verbose): print "sol_zfill = " + str(sol_zfill)

    error_file = open(sol_folder+"/"+sol_basename+"-displacement_error.dat", "w")
    error_file.write("#t e\n")

    err_int = numpy.empty(n_frames)
    ref_int = numpy.empty(n_frames)
    for k_frame in xrange(n_frames):
        ref = myVTK.readUGrid(
            filename=ref_folder+"/"+ref_basename+"_"+str(k_frame).zfill(ref_zfill)+"."+ref_ext,
            verbose=0)
        n_points = ref.GetNumberOfPoints()
        n_cells = ref.GetNumberOfCells()
        sol = myVTK.readUGrid(
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
