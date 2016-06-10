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

def compute_strains(
        sol_folder,
        sol_basename,
        sol_zfill=6,
        sol_ext="vtu",
        disp_array_name="displacement",
        ref_folder=None,
        ref_basename=None,
        CYL_or_PPS="PPS",
        write_strains=1,
        write_strain_vs_radius=0,
        verbose=1):

    if (ref_folder is not None) and (ref_basename is not None):
        ref_mesh = myVTK.readUGrid(
            filename=ref_folder+"/"+ref_basename+"-WithLocalBasis.vtk",
            verbose=0)
        ref_n_cells = ref_mesh.GetNumberOfCells()

        if (ref_mesh.GetCellData().HasArray("sector_id")):
            iarray_sector_id = ref_mesh.GetCellData().GetArray("sector_id")
            n_sector_ids = 0
            for k_cell in xrange(ref_n_cells):
                sector_id = int(iarray_sector_id.GetTuple1(k_cell))
                if (sector_id < 0): continue
                if (sector_id > n_sector_ids-1):
                    n_sector_ids = sector_id+1
            if (verbose): print "n_sector_ids = " + str(n_sector_ids)
        else:
            n_sector_ids = 0
    else:
        ref_mesh = None
        n_sector_ids = 0

    n_zfill = len(glob.glob(sol_folder+"/"+sol_basename+"_*."+sol_ext)[0].rsplit("_")[-1].split(".")[0])
    if (verbose): print "n_zfill = " + str(n_zfill)

    n_frames = len(glob.glob(sol_folder+"/"+sol_basename+"_"+"[0-9]"*sol_zfill+"."+sol_ext))
    if (verbose): print "n_frames = " + str(n_frames)

    if (write_strains):
        strain_file = open(sol_folder+"/"+sol_basename+"-strains.dat", "w")
        strain_file.write("#t Err_avg Err_std Ecc_avg Ecc_std Ell_avg Ell_std Erc_avg Erc_std Erl_avg Erl_std Ecl_avg Ecl_std\n")

    if (write_strain_vs_radius):
        strain_vs_radius_file = open(sol_folder+"/"+sol_basename+"-strains_vs_radius.dat", "w")
        strain_vs_radius_file.write("#t r Err Ecc Ell Erc Erl Ecl\n")

    for k_frame in xrange(n_frames):
        mesh = myVTK.readUGrid(
            filename=sol_folder+"/"+sol_basename+"_"+str(k_frame).zfill(n_zfill)+"."+sol_ext,
            verbose=0)
        n_cells = mesh.GetNumberOfCells()
        if (ref_mesh is not None):
            assert (n_cells == ref_n_cells)
            mesh.GetCellData().AddArray(ref_mesh.GetCellData().GetArray("part_id"))
            mesh.GetCellData().AddArray(ref_mesh.GetCellData().GetArray("sector_id"))
        myVTK.computeStrainsFromDisplacements(
            mesh=mesh,
            disp_array_name=disp_array_name,
            ref_mesh=ref_mesh,
            verbose=0)
        myVTK.writeUGrid(
            ugrid=mesh,
            filename=sol_folder+"/"+sol_basename+"_"+str(k_frame).zfill(n_zfill)+"."+sol_ext,
            verbose=0)

        if (write_strains) or (write_strain_vs_radius):
            if (ref_mesh is not None):
                assert (mesh.GetCellData().HasArray("Strain_"+CYL_or_PPS))
                farray_strain = mesh.GetCellData().GetArray("Strain_"+CYL_or_PPS)
            else:
                farray_strain = mesh.GetCellData().GetArray("Strain")

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

        if (write_strain_vs_radius):
            assert (ref_mesh.GetCellData().HasArray("r"))
            farray_r = ref_mesh.GetCellData().GetArray("r")
            for k_cell in xrange(n_cells):
                strain_vs_radius_file.write(" ".join([str(val) for val in [k_frame, farray_r.GetTuple1(k_cell)]+list(farray_strain.GetTuple(k_cell))]) + "\n")
            strain_vs_radius_file.write("\n")
            strain_vs_radius_file.write("\n")

    if (write_strains):
        strain_file.close()

    if (write_strain_vs_radius):
        strain_vs_radius_file.close()
