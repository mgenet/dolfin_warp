#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2016-2025                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

import numpy

import myVTKPythonLibrary as myvtk

import dolfin_warp as dwarp

################################################################################

def compute_strains(
        working_folder,
        working_basename,
        working_ext="vtu",
        ref_frame=None,                             # MG20190612: Reference configuration.
        disp_array_name="displacement",
        defo_grad_array_name="DeformationGradient",
        strain_array_name="Strain",
        jacobian_array_name="Jacobian",
        equiv_dev_strain_array_basename="EquivDeviatoric",
        ref_mesh_folder=None,                       # MG20190612: Mesh with sectors/parts/etc.
        ref_mesh_basename=None,
        ref_mesh_ext="vtk",
        remove_boundary_layer=1,
        threshold_value=0.5,
        threshold_by_upper_or_lower="lower",
        in_place=1,
        write_strains=1,
        temporal_offset=0,
        temporal_resolution=1,
        plot_strains=1,
        plot_regional_strains=0,
        verbose=1):

    if  (ref_mesh_folder   is not None)\
    and (ref_mesh_basename is not None):
        ref_mesh_filename = ref_mesh_folder+"/"+ref_mesh_basename+"."+ref_mesh_ext
        ref_mesh = myvtk.readUGrid(
            filename=ref_mesh_filename,
            verbose=verbose)
        ref_mesh_n_points = ref_mesh.GetNumberOfPoints()
        ref_mesh_n_cells = ref_mesh.GetNumberOfCells()
        if (verbose): print("ref_mesh_n_points = " + str(ref_mesh_n_points))
        if (verbose): print("ref_mesh_n_cells = " + str(ref_mesh_n_cells))

        if (ref_mesh.GetCellData().HasArray("part_id")):
            iarray_ref_part_id = ref_mesh.GetCellData().GetArray("part_id")
            n_part_ids = 0
            for k_cell in range(ref_mesh_n_cells):
                part_id = int(iarray_ref_part_id.GetTuple1(k_cell))
                if (part_id > n_part_ids-1):
                    n_part_ids = part_id+1
            if (verbose): print("n_part_ids = " + str(n_part_ids))
        else:
            iarray_ref_part_id = None
            n_part_ids = 0

        if (ref_mesh.GetCellData().HasArray("sector_id")):
            iarray_ref_sector_id = ref_mesh.GetCellData().GetArray("sector_id")
            n_sector_ids = 0
            for k_cell in range(ref_mesh_n_cells):
                sector_id = int(iarray_ref_sector_id.GetTuple1(k_cell))
                if (sector_id < 0): continue
                if (sector_id > n_sector_ids-1):
                    n_sector_ids = sector_id+1
            if (verbose): print("n_sector_ids = " + str(n_sector_ids))
        else:
            iarray_ref_sector_id = None
            n_sector_ids = 0

    else:
        ref_mesh = None
        n_part_ids = 0
        n_sector_ids = 0

    working_series = dwarp.MeshesSeries(
        folder=working_folder,
        basename=working_basename,
        ext=working_ext)
    if (verbose): print("n_frames = " + str(working_series.n_frames))
    if (verbose): print("zfill = " + str(working_series.zfill))

    if (write_strains):
        strain_file = open(working_folder+"/"+working_basename+"-strains.dat", "w")
        strain_file.write("#k_frame Exx_avg Exx_std Eyy_avg Eyy_std Ezz_avg Ezz_std Exy_avg Exy_std Exz_avg Exz_std Eyz_avg Eyz_std\n")

    if (ref_frame is not None):
        mesh0 = working_series.get_mesh(k_frame=ref_frame)
        myvtk.addDeformationGradients(
            mesh=mesh0,
            disp_array_name=disp_array_name,
            verbose=verbose)
        farray_F0 = mesh0.GetCellData().GetArray(defo_grad_array_name)

    for k_frame in range(working_series.n_frames):
        print("k_frame = "+str(k_frame))

        mesh = working_series.get_mesh(k_frame=k_frame)
        n_points = mesh.GetNumberOfPoints()
        n_cells = mesh.GetNumberOfCells()
        if (ref_mesh is not None):
            assert (n_points == ref_mesh_n_points),\
                "ref_mesh_n_points ("+str(ref_mesh_n_points)+") ≠ n_points ("+str(n_points)+"). Aborting."
            assert (n_cells == ref_mesh_n_cells),\
                "ref_mesh_n_cells ("+str(ref_mesh_n_cells)+") ≠ n_cells ("+str(n_cells)+"). Aborting."
            if (iarray_ref_part_id is not None):
                mesh.GetCellData().AddArray(iarray_ref_part_id)
            if (iarray_ref_sector_id is not None):
                mesh.GetCellData().AddArray(iarray_ref_sector_id)
        myvtk.addDeformationGradients(
            mesh=mesh,
            disp_array_name=disp_array_name,
            defo_grad_array_name=defo_grad_array_name,
            verbose=verbose)
        if (ref_frame is not None):
            farray_F = mesh.GetCellData().GetArray(defo_grad_array_name)
            for k_cell in range(n_cells):
                F  = numpy.reshape(farray_F.GetTuple(k_cell) , (3,3), order='C')
                F0 = numpy.reshape(farray_F0.GetTuple(k_cell), (3,3), order='C')
                F  = numpy.dot(F, numpy.linalg.inv(F0))
                farray_F.SetTuple(k_cell, numpy.reshape(F, 9, order='C'))
        myvtk.addStrainsFromDeformationGradients(
            mesh=mesh,
            defo_grad_array_name=defo_grad_array_name,
            strain_array_name=strain_array_name,
            mesh_w_local_basis=ref_mesh,
            verbose=verbose)
        myvtk.addJacobiansFromDeformationGradients(
            mesh=mesh,
            defo_grad_array_name=defo_grad_array_name,
            jacobian_array_name=jacobian_array_name,
            verbose=verbose)
        myvtk.addEquivDeviatoricStrainsFromDeformationGradients(
            mesh=mesh,
            defo_grad_array_name=defo_grad_array_name,
            equiv_dev_strain_array_basename=equiv_dev_strain_array_basename,
            verbose=verbose)
        if (ref_mesh is not None):
            if  (iarray_ref_part_id is not None)\
            and (remove_boundary_layer         ):
                mesh = myvtk.getThresholdedUGrid(
                    ugrid=mesh,
                    field_support="cells",
                    field_name="part_id",
                    threshold_value=threshold_value,
                    threshold_by_upper_or_lower=threshold_by_upper_or_lower)
                n_points = mesh.GetNumberOfPoints()
                n_cells = mesh.GetNumberOfCells()
                n_part_ids = 0
                if (iarray_ref_sector_id is not None):
                    iarray_sector_id = mesh.GetCellData().GetArray("sector_id")
            else:
                iarray_sector_id = iarray_ref_sector_id
        mesh_filename = working_folder+"/"+working_basename+("-wStrains")*(not in_place)+"_"+str(k_frame).zfill(working_series.zfill)+"."+working_ext
        myvtk.writeUGrid(
            ugrid=mesh,
            filename=mesh_filename,
            verbose=verbose)

        if (write_strains):
            farray_strain = mesh.GetCellData().GetArray(strain_array_name)
            if (n_sector_ids in (0,1)):
                if (n_part_ids == 0):
                    strains_all = [farray_strain.GetTuple(k_cell) for k_cell in range(n_cells)]
                else:
                    strains_all = [farray_strain.GetTuple(k_cell) for k_cell in range(n_cells) if (iarray_ref_part_id.GetTuple1(k_cell) > 0)]
            elif (n_sector_ids > 1):
                strains_all = []
                strains_per_sector = [[] for _ in range(n_sector_ids)]
                if (n_part_ids == 0):
                    for k_cell in range(n_cells):
                        strains_all.append(farray_strain.GetTuple(k_cell))
                        sector_id = int(iarray_sector_id.GetTuple1(k_cell))
                        strains_per_sector[sector_id].append(farray_strain.GetTuple(k_cell))
                else:
                    for k_cell in range(n_cells):
                        part_id = int(iarray_ref_part_id.GetTuple1(k_cell))
                        if (part_id > 0): continue
                        strains_all.append(farray_strain.GetTuple(k_cell))
                        sector_id = int(iarray_sector_id.GetTuple1(k_cell))
                        if (sector_id < 0): continue
                        strains_per_sector[sector_id].append(farray_strain.GetTuple(k_cell))

            strain_file.write(str(temporal_offset+k_frame*temporal_resolution))
            strains_all_avg = numpy.mean(strains_all, 0)
            strains_all_std = numpy.std(strains_all, 0)
            strain_file.write("".join([" " + str(strains_all_avg[k_comp]) + " " + str(strains_all_std[k_comp]) for k_comp in range(6)]))
            if (n_sector_ids > 1):
                for sector_id in range(n_sector_ids):
                    strains_per_sector_avg = numpy.mean(strains_per_sector[sector_id], 0)
                    strains_per_sector_std = numpy.std(strains_per_sector[sector_id], 0)
                    strain_file.write("".join([" " + str(strains_per_sector_avg[k_comp]) + " " + str(strains_per_sector_std[k_comp]) for k_comp in range(6)]))
            strain_file.write("\n")

    if (write_strains):
        strain_file.close()

        if (plot_strains):
            dwarp.plot_strains(
                working_folder=working_folder,
                working_basenames=[working_basename],
                suffix=None,
                verbose=verbose)

        if (plot_regional_strains):
            dwarp.plot_regional_strains(
                working_folder=working_folder,
                working_basename=working_basename,
                suffix=None,
                verbose=verbose)
