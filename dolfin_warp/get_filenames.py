#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2016-2022                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

import glob

################################################################################

def get_filenames(
        folder,
        basename,
        ext,
        separator="_",
        indices="[0-9]*"):

    return glob.glob(folder+"/"+basename+separator+indices+"."+ext)
