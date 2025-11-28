#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2016-2025                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

import dolfin

import dolfin_warp as dwarp

from .Energy                 import Energy
from .EnergyMixin_Continuous import ContinuousEnergyMixin
from .EnergyMixin_Image      import ImageEnergyMixin
from .FileSeries_Images      import ImageSeries
from .Problem                import Problem

################################################################################

class GeneratedImageContinuousEnergy(Energy, ContinuousEnergyMixin, ImageEnergyMixin):



    def __init__(self,
            problem           : Problem               ,
            image_series      : ImageSeries           ,
            quadrature_degree : int                   ,
            texture           : str                   ,
            name              : str         = "gen_im",
            w                 : float       = 1.      ,
            ref_frame         : int         = 0       ,
            resample          : bool        = True    ,
            resampling_factor : float       = 1.      ,
            compute_DIgen     : bool        = True    ):

        self.problem           = problem
        self.printer           = self.problem.printer
        self.image_series      = image_series
        self.quadrature_degree = quadrature_degree
        self.texture           = texture
        self.name              = name
        self.w                 = w
        self.ref_frame         = ref_frame
        self.resample          = resample
        self.resampling_factor = resampling_factor
        self.compute_DIgen     = compute_DIgen

        self.printer.print_str("Defining generated image correlation energy…")
        self.printer.inc()

        self.set_quadrature_finite_elements()

        self.set_measures()

        self.set_reference_frame()

        self.printer.print_str("Defining generated image…")
        self.printer.inc()

        # Igen
        name, cpp = dwarp.get_ExprGenIm_cpp_pybind(
            im_dim=self.image_series.dimension,
            im_type="im",
            im_is_def=self.resample,
            im_texture=self.texture,
            verbose=0)
        # print(name)
        # print(cpp)
        module = dolfin.compile_cpp_code(cpp)
        expr = getattr(module, name)
        self.Igen = dolfin.CompiledExpression(
            expr(),
            element=self.fe)
        self.Igen.init_image(
            filename=self.ref_image_filename,
            resampling_factor_=self.resampling_factor)
        self.Igen.init_ugrid(
            mesh_=self.problem.mesh,
            U_=self.problem.U.cpp_object())
        self.Igen.generate_image()
        self.Igen.write_image(
            filename="run_gimic_Igen.vti")

        self.Igen_int0 = dolfin.assemble(self.Igen * self.dV)/self.problem.mesh_V0
        self.printer.print_sci("Igen_int0",self.Igen_int0)

        if (self.compute_DIgen):
            # DIgen
            name, cpp = dwarp.get_ExprGenIm_cpp_pybind(
                im_dim=self.image_series.dimension,
                im_type="grad",
                im_is_def=1,
                im_texture=self.texture,
                verbose=0)
            module = dolfin.compile_cpp_code(cpp)
            expr = getattr(module, name)
            self.DIgen = dolfin.CompiledExpression(
                expr(),
                element=self.ve)
            self.DIgen.init_image(
                filename=self.ref_image_filename)
            self.DIgen.init_ugrid(
                mesh_=self.problem.mesh,
                U_=self.problem.U.cpp_object())
            self.DIgen.generate_image()
            self.Igen.write_image(
                filename="run_gimic_DIgen.vti")

        self.printer.dec()
        self.printer.print_str("Defining deformed image…")
        self.printer.inc()

        # Idef
        name, cpp = dwarp.get_ExprIm_cpp_pybind(
            im_dim=self.image_series.dimension,
            im_type="im",
            im_is_def=1)
        module = dolfin.compile_cpp_code(cpp)
        expr = getattr(module, name)
        self.Idef = dolfin.CompiledExpression(
            expr(),
            element=self.fe)
        self.Idef.init_image(
            filename=self.ref_image_filename)
        self.Idef.init_disp(
            U_=self.problem.U.cpp_object())

        self.Idef_int0 = dolfin.assemble(self.Idef * self.dV)/self.problem.mesh_V0
        self.printer.print_sci("Idef_int0",self.Idef_int0)

        self.Idef_norm0 = (dolfin.assemble(self.Idef**2 * self.dV)/self.problem.mesh_V0)**(1./2)
        self.printer.print_sci("Idef_norm0",self.Idef_norm0)

        # DIdef
        name, cpp = dwarp.get_ExprIm_cpp_pybind(
            im_dim=self.image_series.dimension,
            im_type="grad" if (self.image_series.grad_basename is None) else "grad_no_deriv",
            im_is_def=1)
        module = dolfin.compile_cpp_code(cpp)
        expr = getattr(module, name)
        self.DIdef = dolfin.CompiledExpression(
            expr(),
            element=self.ve)
        self.DIdef.init_image(
            filename=self.ref_image_filename)
        self.DIdef.init_disp(
            U_=self.problem.U.cpp_object())

        self.printer.dec()

        self.printer.print_str("Defining characteristic functions…")
        self.printer.inc()

        # Phi_ref
        if (int(dolfin.__version__.split('.')[0]) >= 2018):
            name, cpp = dwarp.get_ExprCharFuncIm_cpp_pybind(
                im_dim=self.image_series.dimension,
                im_is_def=0)
            module = dolfin.compile_cpp_code(cpp)
            expr = getattr(module, name)
            self.Phi_ref = dolfin.CompiledExpression(
                expr(),
                element=self.fe)
        else:
            cpp = dwarp.get_ExprCharFuncIm_cpp_swig(
                im_dim=self.image_series.dimension,
                im_is_def=0)
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
                im_is_def=1)
            module = dolfin.compile_cpp_code(cpp)
            expr = getattr(module, name)
            self.Phi_def = dolfin.CompiledExpression(
                expr(),
                element=self.fe)
            self.Phi_def.init_disp(self.problem.U.cpp_object())
        else:
            cpp = dwarp.get_ExprCharFuncIm_cpp_swig(
                im_dim=self.image_series.dimension,
                im_is_def=1)
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
        self.Psi_c = self.Phi_def * self.Phi_ref * (self.Igen - self.Idef)**2/2
        if (self.compute_DIgen):
                self.DPsi_c  = self.Phi_def * self.Phi_ref * (self.Igen - self.Idef) * dolfin.dot(self.DIgen - self.DIdef, self.problem.dU_test)
                self.DDPsi_c = self.Phi_def * self.Phi_ref * dolfin.dot(self.DIgen - self.DIdef, self.problem.dU_trial) * dolfin.dot(self.DIgen - self.DIdef, self.problem.dU_test)
        else:
            self.DPsi_c  = - self.Phi_def * self.Phi_ref * (self.Igen - self.Idef) * dolfin.dot(self.DIdef, self.problem.dU_test)
            self.DDPsi_c =   self.Phi_def * self.Phi_ref * dolfin.dot(self.DIdef, self.problem.dU_trial) * dolfin.dot(self.DIdef, self.problem.dU_test)

        # forms
        self.ener_form = self.Psi_c   * self.dV
        self.res_form  = self.DPsi_c  * self.dV
        self.jac_form  = self.DDPsi_c * self.dV

        self.printer.dec()
        self.printer.dec()



    def call_before_solve(self,
            k_frame,
            **kwargs):

        self.printer.print_str("Loading deformed image for correlation energy…")

        # Idef
        self.def_image_filename = self.image_series.get_image_filename(k_frame=k_frame)

        self.Idef.update_image(
            filename=self.def_image_filename)

        # DIdef
        self.def_grad_image_filename = self.image_series.get_image_grad_filename(k_frame=k_frame)
        self.DIdef.update_image(
            filename=self.def_grad_image_filename)



    def call_before_assembly(self,
            write_iterations=False,
            basename=None,
            k_frame=None,
            k_iter=None,
            **kwargs):

        if (self.resample):
            self.Igen.update_disp()
            self.Igen.generate_image()
            if (write_iterations):
                self.Igen.write_image(
                    filename=basename+"_Igen_"+str(k_frame).zfill(3)+"_"+str(k_iter-1).zfill(3)+".vti")

            if (self.compute_DIgen):
                self.DIgen.update_disp()
                self.DIgen.generate_image()
                if (write_iterations):
                    self.DIgen.write_grad_image(
                        filename=basename+"_DIgen_"+str(k_frame).zfill(3)+"_"+str(k_iter-1).zfill(3)+".vti")



    def call_after_solve(self,
            k_frame,
            basename,
            **kwargs):

        self.Igen.write_image(
            filename=basename+"_Igen_"+str(k_frame)+".vti")

        if (self.compute_DIgen):
            self.DIgen.write_image(
                filename=basename+"_DIgen_"+str(k_frame).zfill(3)+".vti")



    def get_qoi_names(self):

        return [self.name+"_ener", self.name+"_ener_norm"]



    def get_qoi_values(self):

        self.ener  = self.assemble_ener(w_weight=0)
        self.ener /= self.problem.mesh_V0
        assert (self.ener >= 0.),\
            "ener (="+str(self.ener)+") should be non negative. Aborting."
        self.ener  = self.ener**(1./2)
        self.printer.print_sci(self.name+"_ener",self.ener)

        self.ener_norm = self.ener/self.Idef_norm0
        self.printer.print_sci(self.name+"_ener_norm",self.ener_norm)

        return [self.ener, self.ener_norm]
