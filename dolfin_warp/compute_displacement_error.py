#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2016-2023                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

import dolfin
import numpy
import vtk

import dolfin_warp as dwarp

################################################################################

def compute_displacement_error_with_fenics(
        working_folder          : str                       ,
        working_basename        : str                       ,
        ref_folder              : str                       ,
        ref_basename            : str                       ,
        working_ext             : str  = "vtu"              ,
        ref_ext                 : str  = "vtu"              ,
        working_disp_array_name : str  = "displacement"     ,
        ref_disp_array_name     : str  = "displacement"     ,
        suffix                  : str = "displacement_error",
        sort_mesh               : bool = False              ,
        verbose                 : bool = True               ):

    assert (sort_mesh == False), "Not implemented in compute_displacement_error_with_fenics, use compute_displacement_error_with_vtk instead. Aborting."

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
    if (verbose): print("ref_zfill     = " + str(ref_series.zfill)    )

    working_mesh_filename  = working_folder
    working_mesh_filename += "/"+working_basename
    working_mesh_filename += "-"+"mesh"
    working_mesh_filename += "."+"xml"
    working_mesh = dolfin.Mesh(working_mesh_filename)
    working_fe = dolfin.VectorElement(
        family="Lagrange",
        cell=working_mesh.ufl_cell(),
        degree=1)
    working_fs = dolfin.FunctionSpace(
        working_mesh,
        working_fe)
    working_U = dolfin.Function(working_fs)
    working_U.set_allow_extrapolation(True)

    ref_mesh_filename  = ref_folder
    ref_mesh_filename += "/"+ref_basename
    ref_mesh_filename += "-"+"mesh"
    ref_mesh_filename += "."+"xml"
    ref_mesh = dolfin.Mesh(ref_mesh_filename)
    ref_mesh_dimension = ref_mesh.ufl_domain().geometric_dimension()
    ref_mesh_dV = dolfin.Measure("dx", domain=ref_mesh)
    ref_mesh_V0 = dolfin.assemble(dolfin.Constant(1) * ref_mesh_dV)
    ref_fe = dolfin.VectorElement(
        family="Lagrange",
        cell=ref_mesh.ufl_cell(),
        degree=1)
    ref_fs = dolfin.FunctionSpace(
        ref_mesh,
        ref_fe)
    ref_U = dolfin.Function(ref_fs)
    ref_U.set_allow_extrapolation(True)
    pro_U = dolfin.Function(ref_fs)
    pro_U.set_allow_extrapolation(True)
    err = pro_U - ref_U

    error_file = open(working_folder+"/"+working_basename+"-"+suffix+".dat", "w")
    error_file.write("#k_frame err_int ref_int sol_int\n")

    err_int = numpy.empty(working_series.n_frames)
    ref_int = numpy.empty(working_series.n_frames)
    sol_int = numpy.empty(working_series.n_frames)
    for k_frame in range(working_series.n_frames):
        if (verbose): print("k_frame = " + str(k_frame))

        working_ugrid = working_series.get_mesh(k_frame=k_frame)
        working_array_U = working_ugrid.GetPointData().GetArray(working_disp_array_name)
        working_array_U = vtk.util.numpy_support.vtk_to_numpy(working_array_U)
        working_array_U = working_array_U[:,:ref_mesh_dimension]
        working_array_U = numpy.reshape(working_array_U, working_array_U.size)
        working_U.vector()[:] = working_array_U

        sol_int[k_frame] = (dolfin.assemble(dolfin.inner(working_U, working_U) * ref_mesh_dV)/ref_mesh_V0)**(0.5)
        if (verbose): print("sol_int[k_frame] = " + str(sol_int[k_frame]))

        ref_ugrid = ref_series.get_mesh(k_frame=k_frame)
        ref_array_U = ref_ugrid.GetPointData().GetArray(ref_disp_array_name)
        ref_array_U = vtk.util.numpy_support.vtk_to_numpy(ref_array_U)
        ref_array_U = ref_array_U[:,:ref_mesh_dimension]
        ref_array_U = numpy.reshape(ref_array_U, ref_array_U.size)
        ref_U.vector()[:] = ref_array_U

        ref_int[k_frame] = (dolfin.assemble(dolfin.inner(ref_U, ref_U) * ref_mesh_dV)/ref_mesh_V0)**(0.5)
        if (verbose): print("ref_int[k_frame] = " + str(ref_int[k_frame]))

        dolfin.project(
            v=working_U,
            V=ref_fs,
            function=pro_U)
        if (verbose): print("pro_int = " + str((dolfin.assemble(dolfin.inner(pro_U, pro_U) * ref_mesh_dV)/ref_mesh_V0)**(0.5)))

        err_int[k_frame] = (dolfin.assemble(dolfin.inner(err, err) * ref_mesh_dV)/ref_mesh_V0)**(0.5)
        if (verbose): print("err_int[k_frame] = " + str(err_int[k_frame]))

    error_file.write("\n".join([" ".join([str(val) for val in [k_frame, err_int[k_frame], ref_int[k_frame], sol_int[k_frame]]]) for k_frame in range(working_series.n_frames)]))

    err_int_int = numpy.sqrt(numpy.mean(numpy.square(err_int)))
    ref_int_int = numpy.sqrt(numpy.mean(numpy.square(ref_int)))
    if (verbose): print(err_int_int)
    if (verbose): print(ref_int_int)

    # error_file.write("\n\n")
    # error_file.write(" ".join([str(val) for val in [err_int_int, ref_int_int]]))

    error_file.close()

    err = err_int_int/ref_int_int
    if (verbose): print(err)
    return err



def compute_displacement_error_with_numpy(
        working_folder          : str                       ,
        working_basename        : str                       ,
        ref_folder              : str                       ,
        ref_basename            : str                       ,
        working_ext             : str  = "vtu"              ,
        ref_ext                 : str  = "vtu"              ,
        working_disp_array_name : str  = "displacement"     ,
        ref_disp_array_name     : str  = "displacement"     ,
        suffix                  : str = "displacement_error",
        sort_mesh               : bool = False              ,
        verbose                 : bool = True               ):

    assert (sort_mesh == False), "Not implemented in compute_displacement_error_with_numpy, use compute_displacement_error_with_vtk instead. Aborting."

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

    error_file = open(working_folder+"/"+working_basename+"-"+suffix+".dat", "w")
    error_file.write("#k_frame err_int ref_int sol_int\n")

    err_int = numpy.empty(working_series.n_frames)
    ref_int = numpy.empty(working_series.n_frames)
    sol_int = numpy.empty(working_series.n_frames)
    ref_max = float("-Inf")
    for k_frame in range(working_series.n_frames):
        ref = ref_series.get_np_mesh(k_frame)
        sol = working_series.get_np_mesh(k_frame)
        assert (sol.Points.shape == ref.Points.shape)
        assert (sol.Cells.shape  == ref.Cells.shape )

        ref_disp = ref.PointData[ref_disp_array_name    ]
        sol_disp = sol.PointData[working_disp_array_name]

        ref_int[k_frame] = numpy.sqrt(numpy.mean(numpy.sum(numpy.square(ref_disp), axis=1), axis=0))
        sol_int[k_frame] = numpy.sqrt(numpy.mean(numpy.sum(numpy.square(sol_disp), axis=1), axis=0))

        err_int[k_frame] = numpy.sqrt(numpy.mean(numpy.sum(numpy.square(numpy.subtract(sol_disp,ref_disp)), axis=1), axis=0))

        ref_max = max(ref_max, numpy.max(numpy.sqrt(numpy.sum(numpy.square(ref_disp), axis=1))))

    error_file.write("\n".join([" ".join([str(val) for val in [k_frame, err_int[k_frame], ref_int[k_frame], sol_int[k_frame]]]) for k_frame in range(working_series.n_frames)]))

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



def compute_displacement_error_with_vtk(
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



# compute_displacement_error = compute_displacement_error_with_vtk
compute_displacement_error = compute_displacement_error_with_numpy