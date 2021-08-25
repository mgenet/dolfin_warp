#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2016-2021                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

import dolfin
import numpy

import myPythonLibrary    as mypy
import myVTKPythonLibrary as myvtk

import dolfin_warp as dwarp
from .Energy_Continuous import ContinuousEnergy

################################################################################

class WarpedImageContinuousEnergy(ContinuousEnergy):



    def __init__(self,
            problem,
            image_series,
            quadrature_degree,
            name="im",
            w=1.,
            ref_frame=0,
            w_char_func=1,
            im_is_cone=0,
            static_scaling=0,
            dynamic_scaling=0):

        self.problem           = problem
        self.printer           = self.problem.printer
        self.image_series      = image_series
        self.quadrature_degree = quadrature_degree
        self.name              = name
        self.w                 = w
        self.ref_frame         = ref_frame
        self.static_scaling    = static_scaling
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

        # dV
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
        if (int(dolfin.__version__.split('.')[0]) >= 2018):
            name, cpp = dwarp.get_ExprIm_cpp_pybind(
                im_dim=self.image_series.dimension,
                im_type="im",
                im_is_def=0,
                static_scaling_factor=self.static_scaling)
            module = dolfin.compile_cpp_code(cpp)
            expr = getattr(module, name)
            self.Iref = dolfin.CompiledExpression(
                expr(),
                element=self.fe)
        else:
            cpp = dwarp.get_ExprIm_cpp_swig(
                im_dim=self.image_series.dimension,
                im_type="im",
                im_is_def=0,
                static_scaling_factor=self.static_scaling)
            self.Iref = dolfin.Expression(
                cppcode=cpp,
                element=self.fe)
        self.ref_image_filename = self.image_series.get_image_filename(self.ref_frame)
        self.Iref.init_image(self.ref_image_filename)

        self.Iref_int = dolfin.assemble(self.Iref * self.dV)/self.problem.mesh_V0
        self.printer.print_sci("Iref_int",self.Iref_int)

        self.Iref_norm = (dolfin.assemble(self.Iref**2 * self.dV)/self.problem.mesh_V0)**(1./2)
        assert (self.Iref_norm > 0.),\
            "Iref_norm = "+str(self.Iref_norm)+" <= 0. Aborting."
        self.printer.print_sci("Iref_norm",self.Iref_norm)

        # DIref
        if (int(dolfin.__version__.split('.')[0]) >= 2018):
            name, cpp = dwarp.get_ExprIm_cpp_pybind(
                im_dim=self.image_series.dimension,
                im_type="grad" if (self.image_series.grad_basename is None) else "grad_no_deriv",
                im_is_def=0,
                static_scaling_factor=self.static_scaling)
            module = dolfin.compile_cpp_code(cpp)
            expr = getattr(module, name)
            self.DIref = dolfin.CompiledExpression(
                expr(),
                element=self.ve)
        else:
            cpp = dwarp.get_ExprIm_cpp_swig(
                im_dim=self.image_series.dimension,
                im_type="grad" if (self.image_series.grad_basename is None) else "grad_no_deriv",
                im_is_def=0,
                static_scaling_factor=self.static_scaling)
            self.DIref = dolfin.Expression(
                cppcode=cpp,
                element=self.ve)
        self.ref_image_grad_filename = self.image_series.get_image_grad_filename(self.ref_frame)
        self.DIref.init_image(self.ref_image_grad_filename)

        self.printer.dec()
        self.printer.print_str("Defining deformed image…")
        self.printer.inc()

        if (self.dynamic_scaling):
            self.scaling = numpy.array([1.,0.])
            self.p = numpy.empty((2,2))
            self.q = numpy.empty(2)

        # Idef
        if (int(dolfin.__version__.split('.')[0]) >= 2018):
            name, cpp = dwarp.get_ExprIm_cpp_pybind(
                im_dim=self.image_series.dimension,
                im_type="im",
                im_is_def=1,
                static_scaling_factor=self.static_scaling,
                dynamic_scaling=self.dynamic_scaling)
            module = dolfin.compile_cpp_code(cpp)
            expr = getattr(module, name)
            self.Idef = dolfin.CompiledExpression(
                expr(),
                element=self.fe)
            self.Idef.init_disp(self.problem.U.cpp_object())
        else:
            cpp = dwarp.get_ExprIm_cpp_swig(
                im_dim=self.image_series.dimension,
                im_type="im",
                im_is_def=1,
                static_scaling_factor=self.static_scaling)
            self.Idef = dolfin.Expression(
                cppcode=cpp,
                element=self.fe)
            self.Idef.init_disp(self.problem.U)
        self.Idef.init_image(self.ref_image_filename)
        if (self.dynamic_scaling):
            self.Idef.init_dynamic_scaling(self.scaling)

        self.Idef_int = dolfin.assemble(self.Idef * self.dV)/self.problem.mesh_V0
        self.printer.print_sci("Idef_int",self.Idef_int)

        # DIdef
        if (int(dolfin.__version__.split('.')[0]) >= 2018):
            name, cpp = dwarp.get_ExprIm_cpp_pybind(
                im_dim=self.image_series.dimension,
                im_type="grad" if (self.image_series.grad_basename is None) else "grad_no_deriv",
                im_is_def=1,
                static_scaling_factor=self.static_scaling,
                dynamic_scaling=self.dynamic_scaling)
            module = dolfin.compile_cpp_code(cpp)
            expr = getattr(module, name)
            self.DIdef = dolfin.CompiledExpression(
                expr(),
                element=self.ve)
            self.DIdef.init_disp(self.problem.U.cpp_object())
        else:
            cpp = dwarp.get_ExprIm_cpp_swig(
                im_dim=self.image_series.dimension,
                im_type="grad" if (self.image_series.grad_basename is None) else "grad_no_deriv",
                im_is_def=1,
                static_scaling_factor=self.static_scaling)
            self.DIdef = dolfin.Expression(
                cppcode=cpp,
                element=self.ve)
            self.DIdef.init_disp(self.problem.U)
        self.DIdef.init_image(self.ref_image_filename)
        if (self.dynamic_scaling):
            self.DIdef.init_dynamic_scaling(self.scaling)

        # Charactristic functions
        if (w_char_func):

            self.printer.dec()
            self.printer.print_str("Defining characteristic functions…")
            self.printer.inc()

            # Phi_ref
            if (int(dolfin.__version__.split('.')[0]) >= 2018):
                name, cpp = dwarp.get_ExprCharFuncIm_cpp_pybind(
                    im_dim=self.image_series.dimension,
                    im_is_def=0,
                    im_is_cone=im_is_cone)
                module = dolfin.compile_cpp_code(cpp)
                expr = getattr(module, name)
                self.Phi_ref = dolfin.CompiledExpression(
                    expr(),
                    element=self.fe)
            else:
                cpp = dwarp.get_ExprCharFuncIm_cpp_swig(
                    im_dim=self.image_series.dimension,
                    im_is_def=0,
                    im_is_cone=im_is_cone)
                self.Phi_ref = dolfin.Expression(
                    cppcode=cpp,
                    element=self.fe)
            self.Phi_ref.init_image(self.ref_image_filename)

            self.Phi_ref_int = dolfin.assemble(self.Phi_ref * self.dV)/self.problem.mesh_V0
            self.printer.print_sci("Phi_ref_int",self.Phi_ref_int)

            # Phi_def
            if (int(dolfin.__version__.split('.')[0]) >= 2018):
                name, cpp = dwarp.get_ExprCharFuncIm_cpp_pybind(
                    im_dim=self.image_series.dimension,
                    im_is_def=1,
                    im_is_cone=im_is_cone)
                module = dolfin.compile_cpp_code(cpp)
                expr = getattr(module, name)
                self.Phi_def = dolfin.CompiledExpression(
                    expr(),
                    element=self.fe)
                self.Phi_def.init_disp(self.problem.U.cpp_object())
            else:
                cpp = dwarp.get_ExprCharFuncIm_cpp_swig(
                    im_dim=self.image_series.dimension,
                    im_is_def=1,
                    im_is_cone=im_is_cone)
                self.Phi_def = dolfin.Expression(
                    cppcode=cpp,
                    element=self.fe)
                self.Phi_def.init_disp(self.problem.U)
            self.Phi_def.init_image(self.ref_image_filename)

            self.Phi_def_int = dolfin.assemble(self.Phi_def * self.dV)/self.problem.mesh_V0
            self.printer.print_sci("Phi_def_int",self.Phi_def_int)

        self.printer.dec()
        self.printer.print_str("Defining correlation energy…")
        self.printer.inc()

        # Psi_c
        self.Psi_c  = (self.Idef - self.Iref)**2/2
        self.DPsi_c = (self.Idef - self.Iref) * dolfin.dot(self.DIdef, self.problem.dU_test)

        self.DDPsi_c     = dolfin.dot(self.DIdef, self.problem.dU_trial) * dolfin.dot(self.DIdef, self.problem.dU_test)
        self.DDPsi_c_ref = dolfin.dot(self.DIref, self.problem.dU_trial) * dolfin.dot(self.DIref, self.problem.dU_test)

        if (w_char_func):
            self.Psi_c  *= self.Phi_def * self.Phi_ref
            self.DPsi_c *= self.Phi_def * self.Phi_ref

            self.DDPsi_c     *= self.Phi_def * self.Phi_ref
            self.DDPsi_c_ref *= self.Phi_def * self.Phi_ref

        # forms
        self.ener_form = self.Psi_c   * self.dV
        self.res_form  = self.DPsi_c  * self.dV
        self.jac_form  = self.DDPsi_c * self.dV

        self.printer.dec()
        self.printer.dec()



    def reinit(self):

        if (self.dynamic_scaling):
            self.scaling[:] = [1.,0.]



    def call_before_solve(self,
            k_frame,
            **kwargs):

        self.printer.print_str("Loading deformed image for correlation energy…")

        # Idef
        self.def_image_filename = self.image_series.get_image_filename(k_frame)
        self.Idef.init_image(self.def_image_filename)

        # DIdef
        self.def_grad_image_filename = self.image_series.get_image_grad_filename(k_frame)
        self.DIdef.init_image(self.def_grad_image_filename)



    def call_after_solve(self,
            **kwargs):

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

            if (int(dolfin.__version__.split('.')[0]) <= 2017):
                self.Idef.update_dynamic_scaling(self.scaling)  # should not be needed
                self.DIdef.update_dynamic_scaling(self.scaling) # should not be needed

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
