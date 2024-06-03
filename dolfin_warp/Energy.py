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
