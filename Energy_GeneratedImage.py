#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2016-2018                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

import dolfin
import numpy

import myPythonLibrary    as mypy
import myVTKPythonLibrary as myvtk

import dolfin_dic as ddic
from .Energy import Energy

################################################################################

class WarpedImageEnergy(Energy):



    def __init__(self,
            problem,
            image_series,
            quadrature_degree,
            name="im",
            w=1.,
            ref_frame=0):

        self.problem           = problem
        self.printer           = self.problem.printer
        self.image_series      = image_series
        self.quadrature_degree = quadrature_degree
        self.name              = name
        self.w                 = w
        self.ref_frame         = ref_frame

        self.printer.print_str("Defining warped image correlation energy…")
        self.printer.inc()

        self.printer.print_str("Defining quadrature finite elements…")

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

        self.printer.print_str("Defining measure…")

        # dV
        self.form_compiler_parameters = {
            "quadrature_degree":self.quadrature_degree,
            "quadrature_scheme":"default"}
        self.dV = dolfin.Measure(
            "dx",
            domain=self.problem.mesh,
            metadata=self.form_compiler_parameters)

        # ref_frame
        assert (abs(self.ref_frame) < self.image_series.n_frames),\
            "abs(ref_frame) = "+str(abs(self.ref_frame))+" >= "+str(self.image_series.n_frames)+" = image_series.n_frames. Aborting."
        self.ref_frame = self.ref_frame%self.image_series.n_frames
        self.ref_image_filename = self.image_series.get_image_filename(self.ref_frame)
        self.printer.print_var("ref_frame",self.ref_frame)

        self.printer.dec()
        self.printer.print_str("Defining deformed image…")

        # Igen
        self.Igen = dolfin.Expression(
            ddic.get_ExprGenIm_cpp(
                im_dim=self.image_series.dimension,
                im_type="im"),
            element=self.fe)
        self.Igen.init_image(self.ref_image_filename)
        self.Igen.init_disp(self.problem.U)

        # DIgen
        self.DIgen = dolfin.Expression(
            ddic.get_ExprGenIm_cpp(
                im_dim=self.image_series.dimension,
                im_type="grad"),
            element=self.ve)
        self.Igen.init_image(self.ref_image_filename)
        self.Igen.init_disp(self.problem.U)

        # Idef
        self.Idef = dolfin.Expression(
            ddic.get_ExprIm_cpp(
                im_dim=self.image_series.dimension,
                im_type="im",
                im_is_def=1),
            element=self.fe)
        self.Idef.init_image(self.ref_image_filename)
        self.Idef.init_disp(self.problem.U)

        # DIdef
        self.DIdef = dolfin.Expression(
            ddic.get_ExprIm_cpp(
                im_dim=self.image_series.dimension,
                im_type="grad" if (self.image_series.grad_basename is None) else "grad_no_deriv",
                im_is_def=1),
            element=self.ve)
        self.DIdef.init_image(self.ref_image_filename)
        self.DIdef.init_disp(self.problem.U)

        self.printer.print_str("Defining previous image…")

        self.printer.print_str("Defining correlation energy…")

        # Phi_ref
        self.Phi_Iref = dolfin.Expression(
            ddic.get_ExprCharFuncIm_cpp(
                im_dim=self.image_series.dimension),
            element=self.fe)
        self.Phi_Iref.init_image(self.ref_image_filename)

        # Phi_def
        self.Phi_Idef = dolfin.Expression(
            ddic.get_ExprCharFuncIm_cpp(
                im_dim=self.image_series.dimension,
                im_is_def=1),
            element=self.fe)
        self.Phi_Idef.init_image(self.ref_image_filename)
        self.Phi_Idef.init_disp(self.problem.U)

        # Psi_c
        self.Psi_c   = self.Phi_Idef * self.Phi_Iref * (self.Igen - self.Idef)**2/2
        self.DPsi_c  = self.Phi_Idef * self.Phi_Iref * (self.Igen - self.Idef) * dolfin.dot(self.DIgen - self.DIdef, self.problem.dU_test)
        self.DDPsi_c = self.Phi_Idef * self.Phi_Iref * dolfin.dot(self.DIgen - self.DIdef, self.problem.dU_trial) * dolfin.dot(self.DIgen - self.DIdef, self.problem.dU_test)

        # forms
        self.ener_form = self.Psi_c   * self.dV
        self.res_form  = self.DPsi_c  * self.dV
        self.jac_form  = self.DDPsi_c * self.dV

        self.printer.dec()



    def reinit(self):

        pass



    def call_before_assembly(self):

        self.Igen.generate_image()
        self.DIgen.generate_image()



    def call_before_solve(self,
            k_frame,
            k_frame_old):

        self.printer.print_str("Loading deformed image for correlation energy…")

        # Idef
        self.def_image_filename = self.image_series.get_image_filename(k_frame)
        self.Idef.init_image(self.def_image_filename)

        # DIdef
        self.def_grad_image_filename = self.image_series.get_image_grad_filename(k_frame)
        self.DIdef.init_image(self.def_grad_image_filename)



    def call_after_solve(self):

        pass



    def get_qoi_names(self):

        return [self.name+"_ener"]



    def get_qoi_values(self):

        self.ener = (dolfin.assemble(self.ener_form)/self.problem.mesh_V0)**(1./2)
        self.printer.print_sci(self.name+"_ener",self.ener)

        return [self.ener]
