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

import myPythonLibrary as mypy
import myVTKPythonLibrary as myvtk

import dolfin_dic as ddic
from Energy import Energy

################################################################################

class WarpedImageEnergy(Energy):



    def __init__(self,
            problem,
            image_series,
            quadrature_degree,
            name="im",
            w=1.,
            ref_frame=0,
            dynamic_scaling=False):

        self.problem           = problem
        self.printer           = self.problem.printer
        self.image_series      = image_series
        self.quadrature_degree = quadrature_degree
        self.name              = name
        self.w                 = w
        self.ref_frame         = ref_frame
        self.dynamic_scaling   = dynamic_scaling

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

        self.form_compiler_parameters = {
            "quadrature_degree":self.quadrature_degree,
            "quadrature_scheme":"default"}
        self.dV = dolfin.Measure(
            "dx",
            domain=self.problem.mesh,
            metadata=self.form_compiler_parameters)

        self.printer.print_str("Loading reference image…")
        self.printer.inc()

        # ref_frame
        assert (abs(self.ref_frame) < self.image_series.n_frames),\
            "abs(ref_frame) = "+str(abs(self.ref_frame))+" >= "+str(self.image_series.n_frames)+" = image_series.n_frames. Aborting."
        self.ref_frame = self.ref_frame%self.image_series.n_frames
        self.printer.print_var("ref_frame",self.ref_frame)

        # Iref
        self.Iref = dolfin.Expression(
            cppcode=ddic.get_ExprIm_cpp(
                im_dim=self.image_series.dimension,
                im_type="im",
                im_is_def=0),
            element=self.fe)
        self.ref_image_filename = self.image_series.get_image_filename(self.ref_frame)
        self.Iref.init_image(self.ref_image_filename)

        self.Iref_int = dolfin.assemble(self.Iref * self.dV)/self.problem.mesh_V0
        self.printer.print_var("Iref_int",self.Iref_int)

        self.Iref_norm = (dolfin.assemble(self.Iref**2 * self.dV)/self.problem.mesh_V0)**(1./2)
        assert (self.Iref_norm > 0.),\
            "Iref_norm = "+str(self.Iref_norm)+" <= 0. Aborting."
        self.printer.print_var("Iref_norm",self.Iref_norm)

        # DIref
        self.DIref = dolfin.Expression(
            cppcode=ddic.get_ExprIm_cpp(
                im_dim=self.image_series.dimension,
                im_type="grad" if (self.image_series.grad_basename is None) else "grad_no_deriv",
                im_is_def=0),
            element=self.ve)
        self.ref_image_grad_filename = self.image_series.get_image_grad_filename(self.ref_frame)
        self.DIref.init_image(self.ref_image_grad_filename)

        self.printer.dec()
        self.printer.print_str("Defining deformed image…")

        self.scaling = numpy.array([1.,0.])
        if (self.dynamic_scaling):
            self.p = numpy.empty((2,2))
            self.q = numpy.empty(2)

        # Idef
        self.Idef = dolfin.Expression(
            cppcode=ddic.get_ExprIm_cpp(
                im_dim=self.image_series.dimension,
                im_type="im",
                im_is_def=1),
            element=self.fe)
        self.Idef.init_image(self.ref_image_filename)
        self.Idef.init_disp(self.problem.U)
        self.Idef.init_dynamic_scaling(self.scaling)

        # DIdef
        self.DIdef = dolfin.Expression(
            cppcode=ddic.get_ExprIm_cpp(
                im_dim=self.image_series.dimension,
                im_type="grad" if (self.image_series.grad_basename is None) else "grad_no_deriv",
                im_is_def=1),
            element=self.ve)
        self.DIdef.init_image(self.ref_image_filename)
        self.DIdef.init_disp(self.problem.U)
        self.DIdef.init_dynamic_scaling(self.scaling)

        self.printer.print_str("Defining previous image…")

        # Iold
        self.Iold = dolfin.Expression(
            cppcode=ddic.get_ExprIm_cpp(
                im_dim=self.image_series.dimension,
                im_type="im",
                im_is_def=1),
            element=self.fe)
        self.Iold.init_image(self.ref_image_filename)
        self.Iold.init_disp(self.problem.Uold)
        self.Iold.init_dynamic_scaling(self.scaling) # 2016/07/25: ok, same scaling must apply to Idef & Iold…

        # DIold
        self.DIold = dolfin.Expression(
            cppcode=ddic.get_ExprIm_cpp(
                im_dim=self.image_series.dimension,
                im_type="grad" if (self.image_series.grad_basename is None) else "grad_no_deriv",
                im_is_def=1),
            element=self.ve)
        self.DIold.init_image(self.ref_image_filename)
        self.DIold.init_disp(self.problem.Uold)
        self.DIold.init_dynamic_scaling(self.scaling) # 2016/07/25: ok, same scaling must apply to Idef & Iold…

        self.printer.print_str("Defining correlation energy…")

        # Phi_ref
        self.Phi_Iref = dolfin.Expression(
            cppcode=ddic.get_ExprCharFuncIm_cpp(
                im_dim=self.image_series.dimension),
            element=self.fe)
        self.Phi_Iref.init_image(self.ref_image_filename)

        # Psi_c
        self.Psi_c  = self.Phi_Iref * (self.Idef - self.Iref)**2/2
        self.DPsi_c = self.Phi_Iref * (self.Idef - self.Iref) * dolfin.dot(self.DIdef, self.problem.dU_test)

        self.DDPsi_c     = self.Phi_Iref * dolfin.dot(self.DIdef, self.problem.dU_trial) * dolfin.dot(self.DIdef, self.problem.dU_test)
        self.DDPsi_c_old = self.Phi_Iref * dolfin.dot(self.DIold, self.problem.dU_trial) * dolfin.dot(self.DIold, self.problem.dU_test)
        self.DDPsi_c_ref = self.Phi_Iref * dolfin.dot(self.DIref, self.problem.dU_trial) * dolfin.dot(self.DIref, self.problem.dU_test)

        self.Psi_c_old   = self.Phi_Iref * (self.Idef - self.Iold)**2/2
        self.DPsi_c_old  = self.Phi_Iref * (self.Idef - self.Iold) * dolfin.dot(self.DIdef, self.problem.dU_test)

        # forms
        self.ener_form = self.Psi_c   * self.dV
        self.res_form  = self.DPsi_c  * self.dV
        self.jac_form  = self.DDPsi_c * self.dV

        self.printer.dec()



    def reinit(self):

        self.scaling[:] = [1.,0.]



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

        self.printer.print_str("Loading previous image for correlation energy…")

        # Iold
        self.old_image_filename = self.image_series.get_image_filename(k_frame_old)
        self.Iold.init_image(self.old_image_filename)

        # DIold
        self.old_grad_image_filename = self.image_series.get_image_grad_filename(k_frame_old)
        self.DIold.init_image(self.old_grad_image_filename)



    def call_after_solve(self):

        if (self.dynamic_scaling):
            self.printer.print_str("Updating dynamic scaling…")
            self.printer.inc()

            self.get_qoi_values()

            self.p[0,0] = dolfin.assemble(self.Idef**2 * self.dV)
            self.p[0,1] = dolfin.assemble(self.Idef * self.dV)
            self.p[1,0] = self.p[0,1]
            self.p[1,1] = 1.
            self.q[0] = dolfin.assemble(self.Idef*self.Iref * self.dV)
            self.q[1] = dolfin.assemble(self.Iref * self.dV)
            self.scaling[:] = numpy.linalg.solve(self.p, self.q)
            self.printer.print_var("scaling",self.scaling)

            self.Idef.update_dynamic_scaling(self.scaling)  # should not be needed
            self.DIdef.update_dynamic_scaling(self.scaling) # should not be needed

            self.Iold.update_dynamic_scaling(self.scaling)  # should not be needed
            self.DIold.update_dynamic_scaling(self.scaling) # should not be needed

            self.get_qoi_values()

            self.printer.dec()



    def get_qoi_names(self):

        return [self.name+"_ener", self.name+"_err"]



    def get_qoi_values(self):

        self.ener = (dolfin.assemble(self.ener_form)/self.problem.mesh_V0)**(1./2)
        self.printer.print_sci(self.name+"_ener",self.ener)
        self.err = self.ener/self.Iref_norm
        self.printer.print_sci(self.name+"_err",self.err)

        return [self.ener, self.err]