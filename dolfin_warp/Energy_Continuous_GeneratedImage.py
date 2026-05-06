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
            w_char_func       : bool        = True    ,
            resampling_factor : float       = 1.      ): # image, fourier

        self.problem           = problem
        self.printer           = self.problem.printer
        self.image_series      = image_series
        self.quadrature_degree = quadrature_degree
        self.texture           = texture
        self.name              = name
        self.w                 = w
        self.ref_frame         = ref_frame
        self.w_char_func       = w_char_func
        self.resampling_factor = resampling_factor

        self.printer.print_str("Defining generated image correlation energy…")
        self.printer.inc()

        self.set_quadrature_finite_elements()
        self.ve_im_grad = dolfin.VectorElement(
            family="Quadrature",
            cell=self.problem.mesh.ufl_cell(),
            degree=self.quadrature_degree,
            dim=4+2*self.image_series.dimension)
        self.ve_im_grad._quad_scheme = "default"           # should not be needed
        for sub_element in self.ve_im_grad.sub_elements(): # should not be needed
            sub_element._quad_scheme = "default"           # should not be needed

        self.set_measures()

        self.set_reference_frame()

        self.printer.print_str("Defining generated image…")
        self.printer.inc()

        name, cpp = dwarp.get_ExprGenContIm_cpp(
            im_dim=self.image_series.dimension,
            im_texture=self.texture,
            verbose=0)
        # print(name)
        # print(cpp)
        module = dolfin.compile_cpp_code(cpp)
        expr = getattr(module, name)
        self.IDIgen = dolfin.CompiledExpression(
            expr(),
            element=self.ve_im_grad)
        self.IDIgen.init_images(
            filename=self.ref_image_filename,
            resampling_factor_=self.resampling_factor)
        self.IDIgen.init_mesh_and_disp(
            mesh_=self.problem.mesh,
            U_=self.problem.U.cpp_object())
        self.IDIgen.update_generated_image()
        # self.IDIgen.write_image(
        #     image_name="generated",
        #     filename="run_gimic_Igen.vti")
        # self.IDIgen.write_image(
        #     image_name="generated_gradient",
        #     filename="run_gimic_DIgen.vti")

        ### C++ Layout: [ I_gen(x(X)), R(x(X)), R_tilde(x(X)), I_gen0(X), Grad_Igen0(X), grad_Imes(x(X)) ]
        self.Igen       = self.IDIgen[0]
        self.R          = self.IDIgen[1]
        self.R_tilde    = self.IDIgen[2]
        self.Igen0      = self.IDIgen[3]
        self.Grad_Igen0 = dolfin.as_vector([self.IDIgen[i] for i in range(4, 4+self.image_series.dimension)])
        self.grad_Imes  = dolfin.as_vector([self.IDIgen[i] for i in range(4+self.image_series.dimension, 4+2*self.image_series.dimension)])

        self.Igen_int0 = dolfin.assemble(self.Igen * self.dV)/self.problem.mesh_V0
        self.printer.print_sci("Igen_int0",self.Igen_int0)

        self.Igen_norm0 = (dolfin.assemble(self.Igen**2 * self.dV)/self.problem.mesh_V0)**(1./2)
        self.printer.print_sci("Igen_norm0",self.Igen_norm0)

        self.printer.dec()

        if (self.w_char_func):
            self.printer.print_str("Defining characteristic functions…")
            self.printer.inc()

            ### Phi_ref
            name, cpp = dwarp.get_ExprCharFuncIm_cpp(
                im_dim=self.image_series.dimension,
                im_is_def=0)
            module = dolfin.compile_cpp_code(cpp)
            expr = getattr(module, name)
            self.Phi_ref = dolfin.CompiledExpression(
                expr(),
                element=self.fe)
            self.Phi_ref.init_image(self.ref_image_filename)

            self.Phi_ref_int = dolfin.assemble(self.Phi_ref * self.dV)/self.problem.mesh_V0
            self.printer.print_sci("Phi_ref_int",self.Phi_ref_int)

            ### Phi_def
            name, cpp = dwarp.get_ExprCharFuncIm_cpp(
                im_dim=self.image_series.dimension,
                im_is_def=1)
            module = dolfin.compile_cpp_code(cpp)
            expr = getattr(module, name)
            self.Phi_def = dolfin.CompiledExpression(
                expr(),
                element=self.fe)
            self.Phi_def.init_disp(self.problem.U.cpp_object())
            self.Phi_def.init_image(self.ref_image_filename)

            self.Phi_def_int = dolfin.assemble(self.Phi_def * self.dV)/self.problem.mesh_V0
            self.printer.print_sci("Phi_def_int",self.Phi_def_int)

            self.printer.dec()

        self.printer.print_str("Defining correlation energy…")
        self.printer.inc()

        ### Psi_c
        self.Psi_c = (1./2) * (self.R**2)

        ### DPsi_c
        grad_Igen0 = dolfin.dot(dolfin.inv(self.problem.F).T, self.Grad_Igen0)
        f_vol = (self.R_tilde * grad_Igen0) - (self.R * self.grad_Imes)
        self.DPsi_c = dolfin.inner(f_vol, self.problem.dU_test)

        scalar_stress = (1./2) * (self.R**2) + (self.R_tilde * self.Igen0)
        tensor_stress = scalar_stress * dolfin.inv(self.problem.F).T
        self.DPsi_c += dolfin.inner(tensor_stress, dolfin.grad(self.problem.dU_test))

        # f_vol = - self.R_tilde * dolfin.dot(dolfin.inv(self.problem.F).T, self.Grad_Igen0)
        # self.DPsi_c_vol = dolfin.inner(f_vol, self.problem.dU_test)

        # p  = (1./2) * (self.R**2)
        # p += (self.R_tilde * self.Igen0)
        # f_surf = p * dolfin.dot(dolfin.inv(self.problem.F).T, self.problem.N)
        # self.DPsi_c_surf = dolfin.inner(f_surf, self.problem.dU_test)

        ### DDPsi_c
        self.DDPsi_c = dolfin.inner(self.Grad_Igen0, self.problem.dU_trial) * \
                       dolfin.inner(self.Grad_Igen0, self.problem.dU_test )

        if (self.w_char_func):
            self.Psi_c   *= self.Phi_def * self.Phi_ref
            self.DPsi_c  *= self.Phi_def * self.Phi_ref
            self.DDPsi_c *= self.Phi_def * self.Phi_ref

        # forms
        self.ener_form = self.Psi_c   * self.problem.J * self.dV
        self.res_form  = self.DPsi_c  * self.problem.J * self.dV
        self.jac_form  = self.DDPsi_c * self.problem.J * self.dV

        self.printer.dec()
        self.printer.dec()



    def call_before_solve(self,
            k_frame,
            **kwargs):

        self.printer.print_str("Loading deformed image for correlation energy…")

        self.def_image_filename = self.image_series.get_image_filename(
            k_frame=k_frame)
        self.IDIgen.update_measured_image(
            filename=self.def_image_filename)



    def call_before_assembly(self,
            write_iterations=False,
            basename=None,
            k_frame=None,
            k_iter=None,
            **kwargs):

        self.IDIgen.update_disp()
        self.IDIgen.update_generated_image()
        if (write_iterations):
            self.IDIgen.write_image(
                image_name="generated",
                filename=basename+"_Igen_"+str(k_frame).zfill(3)+"_"+str(k_iter).zfill(3)+".vti")
            # self.IDIgen.write_image(
            #     image_name="generated_gradient",
            #     filename=basename+"_DIgen_"+str(k_frame).zfill(3)+"_"+str(k_iter).zfill(3)+".vti")



    def call_after_solve(self,
            k_frame,
            basename,
            **kwargs):

        self.IDIgen.write_image(
            image_name="generated",
            filename=basename+"_Igen_"+str(k_frame).zfill(3)+".vti")
        # self.IDIgen.write_image(
        #     image_name="generated_gradient",
        #     filename=basename+"_DIgen_"+str(k_frame).zfill(3)+".vti")



    def get_qoi_names(self):

        return [self.name+"_ener", self.name+"_ener_norm"]



    def get_qoi_values(self):

        self.ener = self.assemble_ener(w_weight=False)
        assert (self.ener >= 0.),\
            "ener (="+str(self.ener)+") should be non negative. Aborting."
        self.ener /= self.problem.mesh_V0
        self.ener  = self.ener**(1./2)
        self.printer.print_sci(self.name+"_ener",self.ener)

        self.ener_norm = self.ener/self.Igen_norm0
        self.printer.print_sci(self.name+"_ener_norm",self.ener_norm)

        return [self.ener, self.ener_norm]
