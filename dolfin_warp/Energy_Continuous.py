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



    def assemble_ener(self,
            w_weight=True):

        if (w_weight):
            w = self.w
            if hasattr(self, "ener0"):
                w /= self.ener0
        else:
            w = 1.

        ener = dolfin.assemble(dolfin.Constant(w) * self.ener_form)

        return ener



    def assemble_res(self,
            res_vec,
            add_values=True,
            finalize_tensor=True,
            w_weight=True):

        if (w_weight):
            w = self.w
            if hasattr(self, "ener0"):
                w /= self.ener0
        else:
            w = 1.

        dolfin.assemble(
            form=dolfin.Constant(w) * self.res_form,
            tensor=res_vec,
            add_values=add_values,
            finalize_tensor=finalize_tensor)



    def assemble_jac(self,
            jac_mat,
            add_values=True,
            finalize_tensor=True,
            w_weight=True):

        if (w_weight):
            w = self.w
            if hasattr(self, "ener0"):
                w /= self.ener0
        else:
            w = 1.

        if ((type(self) == dwarp.RegularizationContinuousEnergy)\
        and (self.type == "equilibrated")\
        and (self.model in ("kirchhoff", "neohookean", "mooneyrivlin", "neohookeanmooneyrivlin", "ciarletgeymonat", "ciarletgeymonatneohookean", "ciarletgeymonatneohookeanmooneyrivlin", "ogdenciarletgeymonat", "ogdenciarletgeymonatneohookean", "ogdenciarletgeymonatneohookeanmooneyrivlin"))):
            # dolfin.assemble(
            #     form=dolfin.Constant(w) * self.DDPsi_m_V * self.dV, # MG20230320: This part fails somehow, cf. https://fenicsproject.discourse.group/t/possible-bug-on-ufl-conditional/6537, but it is zero anyway for P1 elements…
            #     tensor=jac_mat,
            #     add_values=add_values,
            #     finalize_tensor=finalize_tensor)
            dolfin.assemble(
                form=dolfin.Constant(w) * self.DDPsi_m_F * self.dF,
                tensor=jac_mat,
                add_values=add_values,
                finalize_tensor=finalize_tensor)
            dolfin.assemble(
                form=dolfin.Constant(w) * self.DDPsi_m_S * self.dS,
                tensor=jac_mat,
                add_values=add_values,
                finalize_tensor=finalize_tensor)
        else:
            dolfin.assemble(
                form=dolfin.Constant(w) * self.jac_form,
                tensor=jac_mat,
                add_values=add_values,
                finalize_tensor=finalize_tensor)



    def get_qoi_names(self):

        return [self.name+"_ener"]



    def get_qoi_values(self):

        self.ener  = self.assemble_ener(w_weight=0)
        self.ener /= self.problem.mesh_V0
        assert (self.ener >= 0.),\
            "ener (="+str(self.ener)+") should be non negative. Aborting."
        self.ener  = self.ener**(1./2)
        self.printer.print_sci(self.name+"_ener",self.ener)

        return [self.ener]
