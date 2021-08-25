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

class GeneratedImageContinuousEnergy(ContinuousEnergy):



    def __init__(self,
            problem,
            image_series,
            quadrature_degree,
            texture,
            name="gen_im",
            w=1.,
            ref_frame=0,
            resample=1,
            compute_DIgen=1,
            n_resampling_Igen=1):

        self.problem           = problem
        self.printer           = self.problem.printer
        self.image_series      = image_series
        self.quadrature_degree = quadrature_degree
        self.texture           = texture
        self.name              = name
        self.w                 = w
        self.ref_frame         = ref_frame
        self.resample          = resample
        self.compute_DIgen     = compute_DIgen
        self.n_resampling_Igen = n_resampling_Igen

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

        self.printer.print_str("Defining generated image…")
        self.printer.inc()

        # ref_frame
        assert (abs(self.ref_frame) < self.image_series.n_frames),\
            "abs(ref_frame) = "+str(abs(self.ref_frame))+" >= "+str(self.image_series.n_frames)+" = image_series.n_frames. Aborting."
        self.ref_frame = self.ref_frame%self.image_series.n_frames
        self.ref_image_filename = self.image_series.get_image_filename(self.ref_frame)
        self.printer.print_var("ref_frame",self.ref_frame)

        # Igen
        name, cpp = dwarp.get_ExprGenIm_cpp_pybind(
            im_dim=self.image_series.dimension,
            im_type="im",
            im_is_def=self.resample,
            im_texture=self.texture,
            verbose=0)
        # print (name)
        # print (cpp)
        module = dolfin.compile_cpp_code(cpp)
        expr = getattr(module, name)
        self.Igen = dolfin.CompiledExpression(
            expr(),
            element=self.fe)
        self.Igen.init_image(
            filename=self.ref_image_filename,
            n_up=self.n_resampling_Igen)
        self.Igen.init_ugrid(
            mesh_=self.problem.mesh,
            U_=self.problem.U.cpp_object())
        self.Igen.generate_image(
            n_down=self.n_resampling_Igen)
        self.Igen.write_image(
            filename="run_gimic.vti")

        self.Igen_int0 = dolfin.assemble(self.Igen * self.dV)/self.problem.mesh_V0
        self.printer.print_sci("Igen_int0",self.Igen_int0)

        ###
        # U_rbm_expr = dolfin.Expression(
        #     ("UX", "UY"),
        #     UX=0.1,
        #     UY=0.1,
        #     element=self.problem.U_fe)
        # U_rbm = dolfin.interpolate(
        #     v=U_rbm_expr,
        #     V=self.problem.U_fs)
        # self.problem.U.vector()[:] = U_rbm.vector()[:]
        # self.Igen.update_disp()
        # self.Igen.generate_image(
        #     n_down=self.n_resampling_Igen)
        # self.Igen.write_image(
        #     filename="run_gimic2.vti")
        # self.Igen_int0 = dolfin.assemble(self.Igen * self.dV)/self.problem.mesh_V0
        # self.printer.print_sci("Igen_int0_b",self.Igen_int0)
        # ###
        #
        # self.Igen_norm0 = (dolfin.assemble(self.Igen**2 * self.dV)/self.problem.mesh_V0)**(1./2)
        # self.printer.print_sci("Igen_norm0",self.Igen_norm0)

        if (self.resample):
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
        name, cpp = dwarp.get_ExprCharFuncIm_cpp_pybind(
            im_dim=self.image_series.dimension,
            im_is_def=0)
        module = dolfin.compile_cpp_code(cpp)
        expr = getattr(module, name)
        self.Phi_ref = dolfin.CompiledExpression(
            expr(),
            element=self.fe)
        self.Phi_ref.init_image(
            filename=self.ref_image_filename)

        self.Phi_ref_int = dolfin.assemble(self.Phi_ref * self.dV)/self.problem.mesh_V0
        self.printer.print_sci("Phi_ref_int",self.Phi_ref_int)

        # Phi_def
        name, cpp = dwarp.get_ExprCharFuncIm_cpp_pybind(
            im_dim=self.image_series.dimension,
            im_is_def=1)
        module = dolfin.compile_cpp_code(cpp)
        expr = getattr(module, name)
        self.Phi_def = dolfin.CompiledExpression(
            expr(),
            element=self.fe)
        self.Phi_def.init_image(
            filename=self.ref_image_filename)
        self.Phi_def.init_disp(
            U_=self.problem.U.cpp_object())

        self.Phi_def_int = dolfin.assemble(self.Phi_def * self.dV)/self.problem.mesh_V0
        self.printer.print_sci("Phi_def_int",self.Phi_def_int)

        self.printer.dec()
        self.printer.print_str("Defining correlation energy…")
        self.printer.inc()

        # Psi_c
        self.Psi_c = self.Phi_def * self.Phi_ref * (self.Igen - self.Idef)**2/2
        if (self.resample) and (self.compute_DIgen):
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
        self.def_image_filename = self.image_series.get_image_filename(k_frame)

        self.Idef.init_image(
            filename=self.def_image_filename)

        # DIdef
        self.def_grad_image_filename = self.image_series.get_image_grad_filename(k_frame)
        self.DIdef.init_image(
            filename=self.def_grad_image_filename)



    def call_before_assembly(self,
            write_iterations=False,
            basename=None,
            k_iter=None,
            **kwargs):

        if (self.resample):
            self.Igen.update_disp()
            self.Igen.generate_image(
                n_down=self.n_resampling_Igen)

            if (self.compute_DIgen):
                self.DIgen.update_disp()
                self.DIgen.generate_image()
                if (write_iterations):
                    self.DIgen.write_grad_image(
                        filename=basename+"_"+str(k_iter-1).zfill(3)+".vti")



    def call_after_solve(self,
            k_frame,
            basename,
            **kwargs):

        if (self.resample):
            if (self.compute_DIgen):
                self.DIgen.write_image(
                    filename=basename+"_"+str(k_frame).zfill(3)+".vti")

        # self.Igen.write_image(
        #     filename="run_gimic_"+str(k_frame)+".vti")



    def get_qoi_names(self):

        return [self.name+"_ener", self.name+"_ener_norm"]



    def get_qoi_values(self):

        self.ener = (dolfin.assemble(self.ener_form)/self.problem.mesh_V0)**(1./2)
        self.printer.print_sci(self.name+"_ener",self.ener)

        self.ener_norm = self.ener/self.Idef_norm0
        self.printer.print_sci(self.name+"_ener_norm",self.ener_norm)

        return [self.ener, self.ener_norm]
