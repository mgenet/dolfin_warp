#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2016-2022                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

import numpy

import dolfin_warp as dwarp

################################################################################

def compute_displacement_error(
        working_folder          : str                  ,
        working_basename        : str                  ,
        ref_folder              : str                  ,
        ref_basename            : str                  ,
        working_ext             : str  = "vtu"         ,
        ref_ext                 : str  = "vtu"         ,
        working_disp_array_name : str  = "displacement",
        ref_disp_array_name     : str  = "displacement",
        sort_mesh               : bool = False         ,
        verbose                 : bool = True          ):

    working_series = dwarp.MeshesSeries(
        folder   = working_folder  ,
        basename = working_basename,
        ext      = working_ext     ,
        verbose  = verbose         )

    ref_series = dwarp.MeshesSeries(
        folder   = ref_folder  ,
        basename = ref_basename,
        ext      = ref_ext     ,
        verbose  = verbose     )

    if (ref_series.n_frames != working_series.n_frames):
        print ("Warning, ref_series.n_frames = "+str(ref_series.n_frames)+" != "+str(working_series.n_frames)+" = working_series.n_frames.")
    if (verbose): print("n_frames = " + str(working_series.n_frames))

    if (verbose): print("working_zfill = " + str(working_series.zfill))
    if (verbose): print("ref_zfill = " + str(ref_series.zfill))

    error_file = open(working_folder+"/"+working_basename+"-displacement_error.dat", "w")
    error_file.write("#k_frame e\n")

    err_int = numpy.empty(working_series.n_frames)
    ref_int = numpy.empty(working_series.n_frames)
    ref_max = float("-Inf")
    for k_frame in range(working_series.n_frames):
        ref = ref_series.get_mesh(k_frame)
        n_points = ref.GetNumberOfPoints()
        n_cells = ref.GetNumberOfCells()
        sol = working_series.get_mesh(k_frame)
        assert (sol.GetNumberOfPoints() == n_points)
        assert (sol.GetNumberOfCells() == n_cells)

        assert(ref.GetPointData().HasArray(ref_disp_array_name))
        ref_disp = ref.GetPointData().GetArray(ref_disp_array_name)
        assert(sol.GetPointData().HasArray(working_disp_array_name))
        working_disp = sol.GetPointData().GetArray(working_disp_array_name)

        if (sort_mesh):
            # FA20200311: sort_ref and sort_working are created because enumeration is not the same in the meshes of ref and sol
            import vtk
            coords_ref     = vtk.util.numpy_support.vtk_to_numpy(ref.GetPoints().GetData())
            coords_working = vtk.util.numpy_support.vtk_to_numpy(sol.GetPoints().GetData())

            sort_ref     = [i_sort[0] for i_sort in sorted(enumerate(coords_ref.tolist()), key=lambda k: [k[1],k[0]])]
            sort_working = [i_sort[0] for i_sort in sorted(enumerate(coords_working.tolist()), key=lambda k: [k[1],k[0]])]

            err_int[k_frame] = numpy.sqrt(numpy.mean([numpy.sum([numpy.square(working_disp.GetTuple(sort_working[k_point])[k_dim]-ref_disp.GetTuple(sort_ref[k_point])[k_dim]) for k_dim in range(3)]) for k_point in range(n_points)]))
        else:
            err_int[k_frame] = numpy.sqrt(numpy.mean([numpy.sum([numpy.square(working_disp.GetTuple(k_point)[k_dim]-ref_disp.GetTuple(k_point)[k_dim]) for k_dim in range(3)]) for k_point in range(n_points)]))

        ref_int[k_frame] = numpy.sqrt(numpy.mean([numpy.sum([numpy.square(ref_disp.GetTuple(k_point)[k_dim]) for k_dim in range(3)]) for k_point in range(n_points)]))
        ref_max = max(ref_max, numpy.max([numpy.sum([numpy.sqrt(numpy.square(ref_disp.GetTuple(k_point)[k_dim])) for k_dim in range(3)]) for k_point in range(n_points)]))

    error_file.write("\n".join([" ".join([str(val) for val in [k_frame, err_int[k_frame], ref_int[k_frame]]]) for k_frame in range(working_series.n_frames)]))

    err_int_int = numpy.sqrt(numpy.mean(numpy.square(err_int)))
    ref_int_int = numpy.sqrt(numpy.mean(numpy.square(ref_int)))
    # print(err_int_int)
    # print(ref_int_int)

    # error_file.write("\n\n")
    # error_file.write(" ".join([str(val) for val in [err_int_int, ref_int_int]]))

    error_file.close()

    err = err_int_int/ref_int_int
    if (verbose): print(err)
    return err
