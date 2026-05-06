#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2016-2026                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

################################################################################

class DiscreteEnergyMixin():



    def assemble_ener(self,
            w_weight=True):

        w = self.w if (w_weight) else 1.

        return w * self.update_ener()



    def assemble_res(self,
            res_vec,
            add_values=True,
            finalize_tensor=True,
            w_weight=True):

        assert (add_values      == True)
        assert (finalize_tensor == True)

        w = self.w if (w_weight) else 1.

        self.update_res()

        res_vec.axpy(w, self.res_vec)



    def assemble_jac(self,
            jac_mat,
            add_values=True,
            finalize_tensor=True,
            w_weight=True):

        assert (add_values      == True)
        assert (finalize_tensor == True)

        w = self.w if (w_weight) else 1.

        self.update_jac()

        jac_mat.axpy(w, self.jac_mat, False) # MG20220107: cannot provide same_nonzero_pattern as kwarg



    def hessp(self,
            p_vec,
            hvp_vec,
            add_values=True,
            w_weight=True):

        assert (add_values == True)

        w = self.w if (w_weight) else 1.

        if not hasattr(self, "jac_mat"):
             self.update_jac()

        if not hasattr(self, "hvp_part_vec"):
             self.hvp_part_vec = hvp_vec.copy()

        self.jac_mat.mult(p_vec, self.hvp_part_vec)

        hvp_vec.axpy(w, self.hvp_part_vec)
