#coding=utf8

################################################################################
###                                                                          ###
### Created by Ezgi Berberoğlu, 2018-2021                                    ###
###                                                                          ###
### Swiss Federal Institute of Technology (ETH), Zurich, Switzerland         ###
###                                                                          ###
### And Martin Genet, 2016-2021                                              ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

import numpy

import myVTKPythonLibrary as myvtk

################################################################################

def get_centroids(mesh):

    assert mesh.GetCellData().HasArray("sector_id"), \
        "There is no field named sector_id. Aborting."
    sector_id = mesh.GetCellData().GetArray("sector_id")
    n_sector_ids = int(sector_id.GetRange()[1]+1)
    n_cells = mesh.GetNumberOfCells()

    sector_centroids = numpy.zeros([n_sector_ids,3])
    n_cells_per_sector = numpy.zeros(n_sector_ids)
    cell_centers = myvtk.getCellCenters(
        mesh=mesh)

    for k_cell in range(n_cells):
        k_cell_sector_id = int(sector_id.GetTuple(k_cell)[0])
        sector_centroids[k_cell_sector_id,:] = numpy.add(
            sector_centroids[k_cell_sector_id,:],
            cell_centers.GetPoint(k_cell))
        n_cells_per_sector[k_cell_sector_id] += 1

    for k_sector in range(n_sector_ids):
        sector_centroids[k_sector,:] = sector_centroids[k_sector,:]/n_cells_per_sector[k_sector]

    return sector_centroids
