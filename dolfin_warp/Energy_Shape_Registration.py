#coding=utf8

################################################################################
###                                                                          ###
### Created by Alexandre Daby-Seesaram, 2024-2025                            ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

import dolfin
import numpy

import dolfin_warp as dwarp

from .Energy_Continuous  import ContinuousEnergy
from .FilesSeries_Images import ImagesSeries
from .Problem            import Problem

class SignedImageEnergy(ContinuousEnergy):


    def __init__(self,
            problem: Problem,
            images_series: ImagesSeries,
            quadrature_degree: int,
            name: str = "im",
            w: float = 1.,
            ref_frame: int = 0,
            w_char_func: bool = True,
            im_is_cone: bool = False,
            static_scaling: bool = False,
            dynamic_scaling: bool = False):

        self.problem           = problem
        self.printer           = self.problem.printer
        self.images_series     = images_series
        self.quadrature_degree = quadrature_degree
        self.name              = name
        self.w                 = w
        self.ref_frame         = ref_frame
        self.w_char_func       = w_char_func
        self.static_scaling    = static_scaling
        self.dynamic_scaling   = dynamic_scaling

        self.printer.print_str("Defining warped image registration energy…")
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

        # Definition of I \circ \phi(X)
        if (int(dolfin.__version__.split('.')[0]) >= 2018):
            name, cpp = dwarp.get_ExprIm_cpp_pybind(
                im_dim=self.images_series.dimension,
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
                im_dim=self.images_series.dimension,
                im_type="im",
                im_is_def=1,
                static_scaling_factor=self.static_scaling)
            self.Idef = dolfin.Expression(
                cppcode=cpp,
                element=self.fe)
            self.Idef.init_disp(self.problem.U)
        self.ref_image_filename = self.images_series.get_image_filename(k_frame=self.ref_frame)


        self.Idef.init_image(self.ref_image_filename)
        if (self.dynamic_scaling):
            self.Idef.init_dynamic_scaling(self.scaling)

        self.Idef_int = dolfin.assemble(self.Idef * self.dV)/self.problem.mesh_V0
        self.printer.print_sci("Idef_int",self.Idef_int)

        self.Iref_norm = (dolfin.assemble(self.Idef**2 * self.dV)/self.problem.mesh_V0)**(1./2)  #DEBUG: Here normis compared to only image which is IDef
        assert (self.Iref_norm > 0.),\
            "Iref_norm = "+str(self.Iref_norm)+" <= 0. Aborting."
        self.printer.print_sci("Iref_norm",self.Iref_norm)

        # Definition of \grad_x I \circ \phi(X)
        if (int(dolfin.__version__.split('.')[0]) >= 2018):
            name, cpp = dwarp.get_ExprIm_cpp_pybind(
                im_dim=self.images_series.dimension,
                im_type="grad" if (self.images_series.grad_basename is None) else "grad_no_deriv",
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
                im_dim=self.images_series.dimension,
                im_type="grad" if (self.images_series.grad_basename is None) else "grad_no_deriv",
                im_is_def=1,
                static_scaling_factor=self.static_scaling)
            self.DIdef = dolfin.Expression(
                cppcode=cpp,
                element=self.ve)
            self.DIdef.init_disp(self.problem.U)
        self.DIdef.init_image(self.ref_image_filename)
        if (self.dynamic_scaling):
            self.DIdef.init_dynamic_scaling(self.scaling)

        # self.Psi        = self.Idef 
        self.Psi        = dolfin.Constant(1)
        self.Psi       *= self.problem.J
        # self.dPsi       = dolfin.derivative(self.Psi, self.problem.U, self.problem.dU_test) #DEBUG AUTODIFF
        # self.dPsi       = self.problem.J*self.Idef*dolfin.inner(dolfin.inv(self.problem.F).T, dolfin.grad(self.problem.F*self.problem.dU_test)) #DEBUG manual dif
        self.dPsi       = self.problem.J*dolfin.inner(dolfin.inv(self.problem.F).T, dolfin.grad(self.problem.dU_test)) #DEBUG manual dif
        # self.dPsi       = self.problem.J*dolfin.inner(dolfin.inv(self.problem.F).T*dolfin.grad(self.problem.dU_test),dolfin.inv(self.problem.F).T) #DEBUG manual diff + pull back divergence
        # self.dPsi  += self.problem.J*dolfin.inner(self.DIdef, self.problem.dU_test)  
        # self.dPsi      += self.problem.J*dolfin.inner(self.DIdef, self.problem.F*self.problem.dU_test)  #DEBUG Pull-back 
        # self.dPsi      += self.problem.J*dolfin.inner(dolfin.inv(self.problem.F)*self.DIdef, self.problem.dU_test)  #DEBUG Pull-back 




        # forms
        self.ener_form = self.Psi   * self.dV
        self.res_form  = self.dPsi  * self.dV


    def reinit(self):

        if (self.dynamic_scaling):
            self.scaling[:] = [1.,0.]



    def call_before_solve(self,
            k_frame,
            **kwargs):

        self.printer.print_str("Loading deformed image for correlation energy…")

        # Idef
        self.def_image_filename = self.images_series.get_image_filename(k_frame=k_frame)
        self.Idef.init_image(self.def_image_filename)

        if (self.w_char_func):
            self.Phi_def.init_image(self.def_image_filename)

        # DIdef
        self.def_grad_image_filename = self.images_series.get_image_grad_filename(k_frame=k_frame)
        self.DIdef.init_image(self.def_grad_image_filename)



    def call_after_solve(self,
            **kwargs):
        pass #DEBUG
        # if (self.dynamic_scaling):
        #     self.printer.print_str("Updating dynamic scaling…")
        #     self.printer.inc()

        #     self.get_qoi_values()

        #     self.p[0,0] = dolfin.assemble(self.Idef**2 * self.dV)
        #     self.p[0,1] = dolfin.assemble(self.Idef * self.dV)
        #     self.p[1,0] = self.p[0,1]
        #     self.p[1,1] = 1.
        #     self.q[0] = dolfin.assemble(self.Idef*self.Iref * self.dV)
        #     self.q[1] = dolfin.assemble(self.Iref * self.dV)
        #     self.scaling[:] = numpy.linalg.solve(self.p, self.q)
        #     self.printer.print_var("scaling",self.scaling)

        #     if (int(dolfin.__version__.split('.')[0]) <= 2017):
        #         self.Idef.update_dynamic_scaling(self.scaling)  # should not be needed
        #         self.DIdef.update_dynamic_scaling(self.scaling) # should not be needed

        #     self.get_qoi_values()

        #     self.printer.dec()



    def get_qoi_names(self):

        return [self.name+"_ener", self.name+"_err"]



    def get_qoi_values(self):

        self.ener  = self.assemble_ener(w_weight=0)
        self.ener /= self.problem.mesh_V0

        self.ener  = self.ener**(1./2)
        self.printer.print_sci(self.name+"_ener",self.ener)

        self.err = self.ener/self.Iref_norm
        self.printer.print_sci(self.name+"_err",self.err)

        return [self.ener, self.err]
