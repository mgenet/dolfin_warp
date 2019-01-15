#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2016-2018                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

import dolfin
import os
import shutil

import myPythonLibrary as mypy
import myVTKPythonLibrary as myvtk

import dolfin_dic as ddic

################################################################################

def write_VTU_file(
        filebasename,
        function,
        time,
        zfill=6):

    file_pvd = dolfin.File(filebasename+"__.pvd")
    file_pvd << (function, float(time))
    os.remove(
        filebasename+"__.pvd")
    shutil.move(
        filebasename+"__"+"".zfill(zfill)+".vtu",
        filebasename+"_"+str(time).zfill(zfill)+".vtu")