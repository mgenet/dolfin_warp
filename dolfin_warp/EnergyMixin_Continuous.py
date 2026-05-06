#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2016-2026                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

import dolfin

import dolfin_warp as dwarp

################################################################################

class ContinuousEnergyMixin():



    def set_measures(self):

        self.printer.print_str("Defining measures…")
        self.printer.inc()

        self.form_compiler_parameters = {
            "quadrature_scheme":"default",
            "quadrature_degree":self.quadrature_degree}
        self.dV = dolfin.Measure(
            "dx",
            domain=self.problem.mesh,
            subdomain_data=self.volume_subdomain_data if hasattr(self, "volume_subdomain_data") else None,
            subdomain_id=self.volume_subdomain_id if ((hasattr(self, "volume_subdomain_id")) and (self.volume_subdomain_id is not None)) else "everywhere",
            metadata=self.form_compiler_parameters)
        self.dF = dolfin.Measure(
            "dS",
            domain=self.problem.mesh,
            subdomain_data=self.volume_subdomain_data if hasattr(self, "volume_subdomain_data") else None,
            subdomain_id=self.volume_subdomain_id if ((hasattr(self, "volume_subdomain_id")) and (self.volume_subdomain_id is not None)) else "everywhere",
            metadata=self.form_compiler_parameters)
        self.dS = dolfin.Measure(
            "ds",
            domain=self.problem.mesh,
            subdomain_data=self.surface_subdomain_data if hasattr(self, "surface_subdomain_data") else None,
            subdomain_id=self.surface_subdomain_id if ((hasattr(self, "surface_subdomain_id")) and (self.surface_subdomain_id is not None)) else "everywhere",
            metadata=self.form_compiler_parameters)

        self.printer.dec()



    def assemble_ener(self,
            w_weight=True):

        w = self.w if (w_weight) else 1.

        return dolfin.assemble(dolfin.Constant(w) * self.ener_form)



    def assemble_res(self,
            res_vec,
            add_values=True,
            finalize_tensor=True,
            w_weight=True):

        w = self.w if (w_weight) else 1.

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

        w = self.w if (w_weight) else 1.

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
