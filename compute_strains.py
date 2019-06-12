#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2016-2019                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

import dolfin
import glob
import numpy

import myPythonLibrary as mypy
import myVTKPythonLibrary as myvtk

import dolfin_dic as ddic

################################################################################

def compute_strains(
        working_folder,
        working_basename,
        working_ext="vtu",
        ref_frame=None,                             # MG20190612: Reference configuration.
        disp_array_name="displacement",
        defo_grad_array_name="DeformationGradient",
        strain_array_name="Strain",
        ref_mesh_folder=None,                       # MG20190612: Mesh with sectors/parts/etc.
        ref_mesh_basename=None,
        ref_mesh_ext="vtk",
        CYL_or_PPS="PPS",
        write_strains=1,
        temporal_offset=None,
        temporal_resolution=None,
        plot_strains=1,
        plot_regional_strains=0,
        write_strains_vs_radius=0,
        write_binned_strains_vs_radius=0,
        verbose=1):

    if (ref_mesh_folder is not None) and (ref_mesh_basename is not None):
        ref_mesh_filename = ref_mesh_folder+"/"+ref_mesh_basename+"."+ref_mesh_ext
        ref_mesh = myvtk.readUGrid(
            filename=ref_mesh_filename,
            verbose=verbose)
        ref_mesh_n_cells = ref_mesh.GetNumberOfCells()
        if (verbose): print "ref_mesh_n_cells = " + str(ref_mesh_n_cells)

        if (ref_mesh.GetCellData().HasArray("sector_id")):
            iarray_sector_id = ref_mesh.GetCellData().GetArray("sector_id")
            n_sector_ids = 0
            for k_cell in xrange(ref_mesh_n_cells):
                sector_id = int(iarray_sector_id.GetTuple1(k_cell))
                if (sector_id < 0): continue
                if (sector_id > n_sector_ids-1):
                    n_sector_ids = sector_id+1
            if (verbose): print "n_sector_ids = " + str(n_sector_ids)
        else:
            iarray_sector_id = None
            n_sector_ids = 0

        if (ref_mesh.GetCellData().HasArray("part_id")):
            iarray_part_id = ref_mesh.GetCellData().GetArray("part_id")
        else:
            iarray_part_id = None

    else:
        ref_mesh = None
        n_sector_ids = 0

    working_filenames = glob.glob(working_folder+"/"+working_basename+"_[0-9]*."+working_ext)
    assert (len(working_filenames) > 0), "There is no working file ("+working_folder+"/"+working_basename+"_[0-9]*."+working_ext+"). Aborting."

    working_zfill = len(working_filenames[0].rsplit("_",1)[-1].split(".")[0])
    if (verbose): print "working_zfill = " + str(working_zfill)

    n_frames = len(working_filenames)
    if (verbose): print "n_frames = " + str(n_frames)

    if (write_strains):
        strain_file = open(working_folder+"/"+working_basename+"-strains.dat", "w")
        strain_file.write("#t Err_avg Err_std Ecc_avg Ecc_std Ell_avg Ell_std Erc_avg Erc_std Erl_avg Erl_std Ecl_avg Ecl_std\n")

    if (write_strains_vs_radius):
        strain_vs_radius_file = open(working_folder+"/"+working_basename+"-strains_vs_radius.dat", "w")
        strain_vs_radius_file.write("#t rr Err Ecc Ell Erc Erl Ecl\n")

    if (write_binned_strains_vs_radius):
        binned_strain_vs_radius_file = open(working_folder+"/"+working_basename+"-binned_strains_vs_radius.dat", "w")
        binned_strain_vs_radius_file.write("#t rr Err Ecc Ell Erc Erl Ecl\n")

    if (ref_frame is not None):
        mesh0_filename = working_folder+"/"+working_basename+"_"+str(ref_frame).zfill(working_zfill)+"."+working_ext
        mesh0 = myvtk.readUGrid(
            filename=mesh0_filename,
            verbose=verbose)
        myvtk.addDeformationGradients(
            mesh=mesh0,
            disp_array_name=disp_array_name,
            verbose=verbose)
        farray_F0 = mesh0.GetCellData().GetArray(defo_grad_array_name)

    for k_frame in xrange(n_frames):
        print "k_frame = "+str(k_frame)

        mesh_filename = working_folder+"/"+working_basename+"_"+str(k_frame).zfill(working_zfill)+"."+working_ext
        mesh = myvtk.readUGrid(
            filename=mesh_filename,
            verbose=verbose)
        n_cells = mesh.GetNumberOfCells()
        if (ref_mesh is not None):
            assert (n_cells == ref_mesh_n_cells), "ref_mesh_n_cells ("+str(ref_mesh_n_cells)+") ≠ n_cells ("+str(n_cells)+"). Aborting."
            if (iarray_part_id is not None):
                mesh.GetCellData().AddArray(iarray_part_id)
            if (iarray_sector_id is not None):
                mesh.GetCellData().AddArray(iarray_sector_id)
        myvtk.addDeformationGradients(
            mesh=mesh,
            disp_array_name=disp_array_name,
            defo_grad_array_name=defo_grad_array_name,
            verbose=verbose)
        if (ref_frame is not None):
            farray_F = mesh.GetCellData().GetArray(defo_grad_array_name)
            for k_cell in xrange(n_cells):
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
        myvtk.writeUGrid(
            ugrid=mesh,
            filename=working_folder+"/"+working_basename+"_"+str(k_frame).zfill(working_zfill)+"."+working_ext,
            verbose=verbose)

        if (write_strains) or (write_strains_vs_radius) or (write_binned_strains_vs_radius):
            if (ref_mesh is not None) and (mesh.GetCellData().HasArray(strain_array_name+"_"+CYL_or_PPS)):
                farray_strain = mesh.GetCellData().GetArray(strain_array_name+"_"+CYL_or_PPS)
            else:
                farray_strain = mesh.GetCellData().GetArray(strain_array_name)

        if (write_strains):
            if (n_sector_ids == 0):
                strains_all = []
                for k_cell in xrange(n_cells):
                    strains_all.append(farray_strain.GetTuple(k_cell))
            elif (n_sector_ids == 1):
                strains_all = []
                for k_cell in xrange(n_cells):
                    sector_id = int(iarray_sector_id.GetTuple(k_cell)[0])
                    if (sector_id < 0): continue
                    strains_all.append(farray_strain.GetTuple(k_cell))
            elif (n_sector_ids > 1):
                strains_all = []
                strains_per_sector = [[] for sector_id in xrange(n_sector_ids)]
                for k_cell in xrange(n_cells):
                    sector_id = int(iarray_sector_id.GetTuple(k_cell)[0])
                    if (sector_id < 0): continue
                    strains_all.append(farray_strain.GetTuple(k_cell))
                    strains_per_sector[sector_id].append(farray_strain.GetTuple(k_cell))

            if (temporal_offset is not None) and (temporal_resolution is not None):
                strain_file.write(str(temporal_offset + k_frame*temporal_resolution))
            else:
                strain_file.write(str(k_frame))
            strains_all_avg = numpy.mean(strains_all, 0)
            strains_all_std = numpy.std(strains_all, 0)
            strain_file.write("".join([" " + str(strains_all_avg[k_comp]) + " " + str(strains_all_std[k_comp]) for k_comp in xrange(6)]))
            if (n_sector_ids > 1):
                for sector_id in xrange(n_sector_ids):
                    strains_per_sector_avg = numpy.mean(strains_per_sector[sector_id], 0)
                    strains_per_sector_std = numpy.std(strains_per_sector[sector_id], 0)
                    strain_file.write("".join([" " + str(strains_per_sector_avg[k_comp]) + " " + str(strains_per_sector_std[k_comp]) for k_comp in xrange(6)]))
            strain_file.write("\n")

        if (write_strains_vs_radius):
            assert (ref_mesh.GetCellData().HasArray("rr"))
            farray_rr = ref_mesh.GetCellData().GetArray("rr")
            for k_cell in xrange(n_cells):
                strain_vs_radius_file.write(" ".join([str(val) for val in [k_frame, farray_rr.GetTuple1(k_cell)]+list(farray_strain.GetTuple(k_cell))]) + "\n")
            strain_vs_radius_file.write("\n")
            strain_vs_radius_file.write("\n")

        if (write_binned_strains_vs_radius):
            assert (ref_mesh.GetCellData().HasArray("rr"))
            farray_rr = ref_mesh.GetCellData().GetArray("rr")
            n_r = 10
            binned_strains = [[] for k_r in xrange(n_r)]
            for k_cell in xrange(n_cells):
                k_r = int(farray_rr.GetTuple1(k_cell)*n_r)
                binned_strains[k_r].append(list(farray_strain.GetTuple(k_cell)))
            #print binned_strains
            binned_strains_avg = []
            binned_strains_std = []
            for k_r in xrange(n_r):
                binned_strains_avg.append(numpy.mean(binned_strains[k_r], 0))
                binned_strains_std.append(numpy.std (binned_strains[k_r], 0))
            #print binned_strains_avg
            #print binned_strains_std
            for k_r in xrange(n_r):
                binned_strain_vs_radius_file.write(" ".join([str(val) for val in [k_frame, (k_r+0.5)/n_r]+[val for k_comp in xrange(6) for val in [binned_strains_avg[k_r][k_comp], binned_strains_std[k_r][k_comp]]]]) + "\n")
            binned_strain_vs_radius_file.write("\n")
            binned_strain_vs_radius_file.write("\n")

    if (write_strains):
        strain_file.close()

        if (plot_strains):
            ddic.plot_strains(
                working_folder=working_folder,
                working_basenames=[working_basename],
                suffix=None,
                verbose=verbose)

        if (plot_regional_strains):
            ddic.plot_regional_strains(
                working_folder=working_folder,
                working_basename=working_basename,
                suffix=None,
                verbose=verbose)

    if (write_strains_vs_radius):
        strain_vs_radius_file.close()

    if (write_binned_strains_vs_radius):
        binned_strain_vs_radius_file.close()
