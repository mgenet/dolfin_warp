#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2016-2021                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

import dolfin

import myPythonLibrary    as mypy
import myVTKPythonLibrary as myvtk

import dolfin_warp as dwarp

################################################################################

class Energy():



    def reinit(self,
            *kargs,
            **kwargs):

        pass



    def call_before_solve(self,
            *kargs,
            **kwargs):

        pass



    def call_before_assembly(self,
            *kargs,
            **kwargs):

        pass



    def call_after_solve(self,
            *kargs,
            **kwargs):

        pass
