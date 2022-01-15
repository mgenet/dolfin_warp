#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2016-2021                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

import dolfin

from .Energy import Energy

################################################################################

class ContinuousEnergy(Energy):



    def assemble_ener(self,
        w_weight=True):

        ener = dolfin.assemble(self.ener_form)
        if (w_weight):
            ener *= self.w
        return ener



    def assemble_res(self,
            res_vec,
            add_values=True,
            finalize_tensor=True,
            w_weight=True):

        dolfin.assemble(
            form=self.res_form,
            tensor=res_vec,
            add_values=add_values,
            finalize_tensor=finalize_tensor)
        if (w_weight):
            res_vec *= self.w



    def assemble_jac(self,
            jac_mat,
            add_values=True,
            finalize_tensor=True,
            w_weight=True):

        dolfin.assemble(
            form=self.jac_form,
            tensor=jac_mat,
            add_values=add_values,
            finalize_tensor=finalize_tensor)
        if (w_weight):
            jac_mat *= self.w



    def get_qoi_names(self):

        return [self.name+"_ener"]



    def get_qoi_values(self):

        self.ener = (self.assemble_ener(w_weight=0)/self.problem.mesh_V0)**(1./2)
        self.printer.print_sci(self.name+"_ener",self.ener)

        return [self.ener]
