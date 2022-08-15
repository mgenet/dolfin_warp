#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2016-2022                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

import dolfin
import os
import shutil

################################################################################

def write_VTU_file(
        filebasename,
        function,
        time=None,
        zfill=3):

    file_pvd = dolfin.File(filebasename+"__.pvd")
    file_pvd << (function, float(time) if (time is not None) else 0.)
    os.remove(
        filebasename+"__.pvd")
    shutil.move(
        filebasename+"__"+"".zfill(6)+".vtu",
        filebasename+("_"+str(time).zfill(zfill) if (time is not None) else "")+".vtu")
