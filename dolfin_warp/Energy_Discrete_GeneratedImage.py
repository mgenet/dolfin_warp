#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2016-2026                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

import dolfin
import numpy

import dolfin_warp as dwarp

from .Energy               import Energy
from .EnergyMixin_Discrete import DiscreteEnergyMixin
from .EnergyMixin_Image    import ImageEnergyMixin
from .FileSeries_Images    import ImageSeries
from .Problem              import Problem

################################################################################

class GeneratedImageDiscreteEnergy(Energy, DiscreteEnergyMixin, ImageEnergyMixin):



    def __init__(self,
            problem           : Problem               ,
            image_series      : ImageSeries           ,
            texture           : str                   ,
            name              : str         = "gen_im",
            w                 : float       = 1.      ,
            ref_frame         : int         = 0       ,
            resample          : bool        = True    ,
            resampling_factor : float       = 1.      ,
            ener_type         : str         = "image" ): # image, fourier

        self.problem           = problem
        self.printer           = self.problem.printer
        self.image_series      = image_series
        self.texture           = texture
        self.name              = name
        self.w                 = w
        self.ref_frame         = ref_frame
        self.resample          = resample
        self.resampling_factor = resampling_factor
        self.ener_type         = ener_type

        self.printer.print_str("Defining generated image correlation energy…")
        self.printer.inc()

        self.quadrature_degree = 1
        self.set_quadrature_finite_elements()

        self.set_reference_frame()

        self.printer.print_str("Defining generated image…")
        self.printer.inc()

        # Igen
        name, cpp = dwarp.get_ExprGenDiscIm_cpp(
            im_dim=self.image_series.dimension,
            im_is_def=1,
            im_texture=self.texture,
            im_resample=self.resample,
            verbose=0)
        # print(name)
        # print(cpp)
        module = dolfin.compile_cpp_code(cpp)
        expr = getattr(module, name)
        self.Igen = dolfin.CompiledExpression(
            expr(X0=0., Y0=0., s=0.1),
            element=self.fe)
        if (self.resample):
            self.Igen.init_images(
                filename=self.ref_image_filename,
                resampling_factor_=self.resampling_factor)
        else:
            self.Igen.init_images(
                filename=self.ref_image_filename)
        self.Igen.init_mesh_and_disp(
            mesh_=self.problem.mesh,
            U_=self.problem.U.cpp_object())

        # self.Igen.write_image(
        #     image_name="measured",
        #     filename="test_gimic_Idef.vti")
        # self.Igen.write_image(
        #     image_name="measured_fft",
        #     filename="test_gimic_Idef_fft.vti")
        self.printer.print_sci("self.Igen.compute_measured_image_integral()", self.Igen.compute_image_integral(image_name="measured"))

        self.Igen.update_generated_image()
        # self.Igen.write_image(
        #     image_name="generated",
        #     filename="test_gimic_Igen.vti")
        # self.Igen.write_image(
        #     image_name="generated_fft",
        #     filename="test_gimic_Igen_fft.vti")
        self.printer.print_sci("self.Igen.compute_generated_image_integral()", self.Igen.compute_image_integral(image_name="generated"))

        self.printer.print_sci("self.Igen.compute_image_energy()", self.Igen.compute_image_energy())
        self.printer.print_sci("self.Igen.compute_fourier_energy()", self.Igen.compute_fourier_energy())

        self.printer.dec()
        self.printer.dec()



    def call_before_solve(self,
            k_frame,
            **kwargs):

        self.printer.print_str("Loading measured image…")

        self.def_image_filename = self.image_series.get_image_filename(
            k_frame=k_frame)
        self.Igen.update_measured_image(
            filename=self.def_image_filename)



    def call_before_assembly(self,
            write_iterations=False,
            basename=None,
            k_frame=None,
            k_iter=None,
            **kwargs):

        self.Igen.update_disp()
        self.Igen.update_generated_image()

        if (write_iterations):
            self.IDIgen.write_image(
                image_name="generated",
                filename=basename+"_Igen_"+str(k_frame).zfill(3)+"_"+str(k_iter).zfill(3)+".vti")
            self.IDIgen.write_image(
                image_name="generated_gradient",
                filename=basename+"_DIgen_"+str(k_frame).zfill(3)+"_"+str(k_iter).zfill(3)+".vti")



    def update_ener(self):

        if (self.ener_type == "image"):
            # ener = self.Igen.compute_fourier_energy()
            # self.printer.print_sci("self.Igen.compute_fourier_energy()", ener)
            ener = self.Igen.compute_image_energy()
            # self.printer.print_sci("self.Igen.compute_image_energy()", ener)
        elif (self.ener_type == "fourier"):
            # ener = self.Igen.compute_image_energy()
            # self.printer.print_sci("self.Igen.compute_image_energy()", ener)
            ener = self.Igen.compute_fourier_energy()
            # self.printer.print_sci("self.Igen.compute_fourier_energy()", ener)
        else:
            assert (0),\
                "ener_type (="+str(self.ener_type)+") should be \"image\" or \"fourier\". Aborting."

        return ener



    def call_after_solve(self,
            k_frame,
            basename,
            **kwargs):

        self.Igen.write_image(
            image_name="generated",
            filename=basename+"-Igen_"+str(k_frame).zfill(3)+".vti")
        # self.Igen.write_image(
        #     image_name="probe_filter",
        #     filename=basename+"-Probed"+str(k_frame)+".vti")
        # self.Igen.write_image(
        #     image_name="upsampled",
        #     filename=basename+"-Upsampled"+str(k_frame)+".vti")



    def get_qoi_names(self):

        return [self.name+"_ener"]



    def get_qoi_values(self):

        self.ener = self.update_ener()
        assert (self.ener >= 0.),\
            "ener (="+str(self.ener)+") should be non negative. Aborting."
        self.printer.print_sci(self.name+"_ener", self.ener)

        return [self.ener]
