#coding=utf8

########################################################################
###                                                                  ###
### Created by Ezgi Berberoğlu, 2017-2021                            ###
###                                                                  ###
### Swiss Federal Institute of Technology (ETH), Zurich, Switzerland ###
### École Polytechnique, Palaiseau, France                           ###
###                                                                  ###
########################################################################

import dolfin_warp as dwarp

########################################################################

folders  = []
# folders += ["resolution_1.0x1.0_noNoise"]
# folders += ["resolution_1.5x1.5_noNoise"]
# folders += ["resolution_2.0x2.0_noNoise"]
# folders += ["resolution_2.5x2.5_noNoise"]
folders += ["resolution_3.0x3.0_noNoise"]
# folders += ["resolution_3.5x3.5_noNoise"]

betas = ["0","0.1","0.2","0.3","0.4","0.5","0.025","0.05","0.075"]
noisy = 0
es_tf = 55

for k_folder in range(len(folders)):
    folder = folders[k_folder]
    print "Displacement error computation for: " + folder
    dwarp.compute_best_beta(
        es_tf,
        betas,
        folder,
        noisy)
