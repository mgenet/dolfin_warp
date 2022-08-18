#coding=utf8

################################################################################
###                                                                          ###
### Created by Ezgi Berberoğlu, 2017-2021                                    ###
###                                                                          ###
### Swiss Federal Institute of Technology (ETH), Zurich, Switzerland         ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

import dolfin_warp as dwarp

################################################################################

folders  = []
# folders += ["resolution_1.0x1.0_noNoise"]
# folders += ["resolution_1.5x1.5_noNoise"]
# folders += ["resolution_2.0x2.0_noNoise"]
# folders += ["resolution_2.5x2.5_noNoise"]
folders += ["resolution_3.0x3.0_noNoise"]
# folders += ["resolution_3.5x3.5_noNoise"]

k_frame = 55
noisy = 0

for k_folder in range(len(folders)):
    folder = folders[k_folder]
    print("Strain error computation for: "+folder)
    dwarp.compute_strain_error(
        k_frame,
        folder,
        noisy)
