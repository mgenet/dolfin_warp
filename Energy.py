#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2016-2019                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

import dolfin

import myPythonLibrary as mypy
import myVTKPythonLibrary as myvtk

import dolfin_dic as ddic

################################################################################

class Energy():



    def reinit(self,
            *args,
            **kwargs):

        pass



    def call_before_assembly(self,
            *args,
            **kwargs):

        pass



    def call_before_solve(self,
            *args,
            **kwargs):

        pass



    def call_after_solve(self,
            *args,
            **kwargs):

        pass



    def get_qoi_names(self):

        return [self.name+"_ener"]



    def get_qoi_values(self):

        self.ener = (dolfin.assemble(self.ener_form)/self.problem.mesh_V0)**(1./2)
        self.printer.print_sci(self.name+"_ener",self.ener)

        return [self.ener]
