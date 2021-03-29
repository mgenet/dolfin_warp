#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2016-2021                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

import dolfin
import numpy

import myPythonLibrary    as mypy
import myVTKPythonLibrary as myvtk

import dolfin_warp as dwarp
from .Energy import Energy

################################################################################

class ContinuousEnergy(Energy):



    def assemble_ener(self):

        return dolfin.assemble(dolfin.Constant(self.w) * self.ener_form)



    def assemble_res(self,
            res_vec,
            add_values=True,
            finalize_tensor=True):

        dolfin.assemble(
            dolfin.Constant(self.w) * self.res_form,
            tensor=res_vec,
            add_values=add_values,
            finalize_tensor=finalize_tensor)



    def assemble_jac(self,
            jac_mat,
            add_values=True,
            finalize_tensor=True):

        dolfin.assemble(
            dolfin.Constant(self.w) * self.jac_form,
            tensor=jac_mat,
            add_values=add_values,
            finalize_tensor=finalize_tensor)



    def get_qoi_names(self):

        return [self.name+"_ener"]



    def get_qoi_values(self):

        self.ener = (dolfin.assemble(self.ener_form)/self.problem.mesh_V0)**(1./2)
        self.printer.print_sci(self.name+"_ener",self.ener)

        return [self.ener]
