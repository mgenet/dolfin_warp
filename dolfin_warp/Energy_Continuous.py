#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2016-2024                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

import dolfin

import dolfin_warp as dwarp

from .Energy import Energy

################################################################################

class ContinuousEnergy(Energy):



    def assemble_ener_w_weight(self,
            w):

        ener = dolfin.assemble(dolfin.Constant(w) * self.ener_form)

        return ener



    def assemble_res_w_weight(self,
            w,
            res_vec,
            add_values=True,
            finalize_tensor=True):

        dolfin.assemble(
            form=dolfin.Constant(w) * self.res_form,
            tensor=res_vec,
            add_values=add_values,
            finalize_tensor=finalize_tensor)



    def assemble_jac_w_weight(self,
            w,
            jac_mat,
            add_values=True,
            finalize_tensor=True):

        dolfin.assemble(
            form=dolfin.Constant(w) * self.jac_form,
            tensor=jac_mat,
            add_values=add_values,
            finalize_tensor=finalize_tensor)
