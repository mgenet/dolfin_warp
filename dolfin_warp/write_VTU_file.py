#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2016-2021                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

import dolfin
import os
import shutil

import myPythonLibrary    as mypy
import myVTKPythonLibrary as myvtk

import dolfin_warp as dwarp

################################################################################

def write_VTU_file(
        filebasename,
        function,
        time,
        zfill=3):

    file_pvd = dolfin.File(filebasename+"__.pvd")
    file_pvd << (function, float(time))
    os.remove(
        filebasename+"__.pvd")
    shutil.move(
        filebasename+"__"+"".zfill(6)+".vtu",
        filebasename+"_"+str(time).zfill(zfill)+".vtu")
