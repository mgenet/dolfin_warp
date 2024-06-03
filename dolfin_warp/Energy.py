#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2016-2024                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

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



    def compute_weigth(self,
            w_weight=True):

        if (w_weight):
            w = self.w
            if hasattr(self, "ener0"):
                w /= self.ener0
        else:
            w = 1.

        return w


    def assemble_ener(self,
            w_weight=True):

        w = self.compute_weigth(
            w_weight=w_weight)

        ener = self.assemble_ener_w_weight(
            w=w)

        return ener



    def assemble_res(self,
            res_vec,
            add_values=True,
            finalize_tensor=True,
            w_weight=True):

        w = self.compute_weigth(
            w_weight=w_weight)

        self.assemble_res_w_weight(
            w=w,
            res_vec=res_vec,
            add_values=add_values,
            finalize_tensor=finalize_tensor)



    def assemble_jac(self,
            jac_mat,
            add_values=True,
            finalize_tensor=True,
            w_weight=True):

        w = self.compute_weigth(
            w_weight=w_weight)

        self.assemble_jac_w_weight(
            w=w,
            jac_mat=jac_mat,
            add_values=add_values,
            finalize_tensor=finalize_tensor)



    def call_after_solve(self,
            *kargs,
            **kwargs):

        pass



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
