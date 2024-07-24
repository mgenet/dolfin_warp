#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2016-2024                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

import dolfin
import numpy

import dolfin_warp as dwarp

from .Energy_Discrete    import DiscreteEnergy
from .FilesSeries_Images import ImagesSeries
from .Problem            import Problem

################################################################################

class GeneratedImageDiscreteEnergy(DiscreteEnergy):



    def __init__(self,
            problem           : Problem                ,
            images_series     : ImagesSeries           ,
            quadrature_degree : int                    ,
            texture           : str                    ,
            name              : str          = "gen_im",
            w                 : float        = 1.      ,
            ref_frame         : int          = 0       ,
            resample          : bool         = True    ,
            n_resampling_Igen : int          = 1       ,
            compute_DIgen     : bool         = True    ):

        self.problem           = problem
        self.printer           = self.problem.printer
        self.images_series     = images_series
        self.quadrature_degree = quadrature_degree
        self.texture           = texture
        self.name              = name
        self.w                 = w
        self.ref_frame         = ref_frame
        self.resample          = resample
        self.n_resampling_Igen = n_resampling_Igen
        self.compute_DIgen     = compute_DIgen

        self.printer.print_str("Defining generated image correlation energy…")
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

        self.printer.print_str("Defining measure…")

        # dV
        self.form_compiler_parameters = {
            "quadrature_degree":self.quadrature_degree,
            "quadrature_scheme":"default"}
        self.dV = dolfin.Measure(
            "dx",
            domain=self.problem.mesh,
            metadata=self.form_compiler_parameters)

        self.printer.print_str("Defining generated image…")
        self.printer.inc()

        # ref_frame
        assert (abs(self.ref_frame) < self.images_series.n_frames),\
            "abs(ref_frame) = "+str(abs(self.ref_frame))+" >= "+str(self.images_series.n_frames)+" = images_series.n_frames. Aborting."
        self.ref_frame = self.ref_frame%self.images_series.n_frames
        self.ref_image_filename = self.images_series.get_image_filename(k_frame=self.ref_frame)
        self.printer.print_var("ref_frame", self.ref_frame)

        # Igen
        name, cpp = dwarp.get_ExprGenIm_cpp_pybind(
            im_dim=self.images_series.dimension,
            im_type="im",
            im_is_def=1,
            im_texture=self.texture,
            im_resample=self.resample,
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

        self.printer.print_sci("self.Igen.compute_measured_image_integral()", self.Igen.compute_measured_image_integral())

        self.Igen.generate_image()
        self.Igen.write_image(
            filename="run_gimic.vti")

        self.printer.print_sci("self.Igen.compute_generated_image_integral()", self.Igen.compute_generated_image_integral())

        self.Igen_int0 = dolfin.assemble(self.Igen * self.dV)/self.problem.mesh_V0
        self.printer.print_sci("Igen_int0", self.Igen_int0)

        self.printer.dec()



    def call_before_solve(self,
            k_frame,
            **kwargs):

        self.printer.print_str("Loading deformed image for correlation energy…")

        self.def_image_filename = self.images_series.get_image_filename(k_frame=k_frame)
        self.Idef.init_image(
            filename=self.def_image_filename)
        self.Idef.compute_fft()



    def call_before_assembly(self,
            **kwargs):

        if (self.resample):
            self.Igen.update_disp()
            self.Igen.generate_image()
            self.Igen.compute_fft()



    def assemble_ener(self,
            w_weight=True):

        # ener = numpy.sum(numpy.square(numpy.subtract(self.Igen_fft, self.Idef_fft))) # MG20240523: This is slower than line below
        ener = numpy.linalg.norm(self.Igen.fft - self.Idef.fft)**2                     # MG20240523: This is faster than line above
        ener /= 2

        if (w_weight):
            w = self.w
            if hasattr(self, "ener0"):
                w /= self.ener0
        else:
            w = 1.

        return w*ener



    def call_after_solve(self,
            k_frame,
            **kwargs):

        self.Igen.write_image(
            filename="run_gimic_"+str(k_frame)+".vti")

        pass



    def get_qoi_names(self):

        return [self.name+"_ener", self.name+"_ener_norm"]



    def get_qoi_values(self):

        self.ener  = self.assemble_ener(w_weight=0)
        assert (self.ener >= 0.),\
            "ener (="+str(self.ener)+") should be non negative. Aborting."
        self.ener /= self.problem.mesh_V0
        self.ener  = self.ener**(1./2)
        self.printer.print_sci(self.name+"_ener", self.ener)

        self.ener_norm = self.ener/self.Idef_norm0
        self.printer.print_sci(self.name+"_ener_norm", self.ener_norm)

        return [self.ener, self.ener_norm]
