#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2016-2025                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

import dolfin

################################################################################

class ImageEnergyMixin():



    def set_quadrature_finite_elements(self):

        self.printer.print_str("Defining quadrature finite elements…")
        self.printer.inc()

        # fe
        self.fe = dolfin.FiniteElement(
            family="Quadrature",
            cell=self.problem.mesh.ufl_cell(),
            degree=self.quadrature_degree,
            quad_scheme="default")
        self.fe._quad_scheme = "default"              # should not be needed
        for sub_element in self.fe.sub_elements():    # should not be needed
            sub_element._quad_scheme = "default"      # should not be needed

        # ve
        self.ve = dolfin.VectorElement(
            family="Quadrature",
            cell=self.problem.mesh.ufl_cell(),
            degree=self.quadrature_degree,
            quad_scheme="default")
        self.ve._quad_scheme = "default"              # should not be needed
        for sub_element in self.ve.sub_elements():    # should not be needed
            sub_element._quad_scheme = "default"      # should not be needed

        # te
        self.te = dolfin.TensorElement(
            family="Quadrature",
            cell=self.problem.mesh.ufl_cell(),
            degree=self.quadrature_degree,
            quad_scheme="default")
        self.te._quad_scheme = "default"              # should not be needed
        for sub_element in self.te.sub_elements():    # should not be needed
            sub_element._quad_scheme = "default"      # should not be needed

        self.printer.dec()



    def set_reference_frame(self):

        self.printer.print_str("Defining reference frame…")
        self.printer.inc()

        # ref_frame
        assert (abs(self.ref_frame) < self.image_series.n_frames),\
            "abs(self.ref_frame) = "+str(abs(self.ref_frame))+" >= "+str(self.image_series.n_frames)+" = image_series.n_frames. Aborting."
        self.ref_frame = self.ref_frame%self.image_series.n_frames
        self.ref_image_filename = self.image_series.get_image_filename(k_frame=self.ref_frame)
        self.printer.print_var("ref_frame",self.ref_frame)

        self.printer.dec()
