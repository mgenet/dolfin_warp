#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2016-2022                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

import dolfin
import petsc4py
import typing

import dolfin_mech as dmech

from .Energy_Discrete import DiscreteEnergy
from .Problem         import Problem

################################################################################

class SurfaceRegularizationDiscreteEnergy(DiscreteEnergy):



    def __init__(self,
            problem: Problem,
            name: str = "reg",
            w: float = 1.,
            type: str = "tractions",
            model: str = "ciarletgeymonatneohookean",
            young: float = 1.,
            poisson: float = 0.,
            quadrature_degree: typing.Optional[int] = None, # MG20220815: This can be written "int | None" starting with python 3.10, but it is not readily available on the gitlab runners (Ubuntu 20.04)
            scalar_formulation_in_2D: bool = 1):

        self.problem = problem
        self.printer = problem.printer

        self.name = name

        self.w = w

        assert (type in ("tractions", "tractions-normal", "tractions-tangential", "tractions-normal-tangential")),\
            "\"type\" ("+str(type)+") must be \"tractions\", \"tractions-normal\", \"tractions-tangential\" or \"tractions-normal-tangential\", . Aborting."
        self.type = type

        assert (model in ("hooke", "kirchhoff", "neohookean", "mooneyrivlin", "neohookeanmooneyrivlin", "ciarletgeymonat", "ciarletgeymonatneohookean", "ciarletgeymonatneohookeanmooneyrivlin")),\
            "\"model\" ("+str(model)+") must be \"hooke\", \"kirchhoff\", \"neohookean\", \"mooneyrivlin\", \"neohookeanmooneyrivlin\", \"ciarletgeymonat\", \"ciarletgeymonatneohookean\" or \"ciarletgeymonatneohookeanmooneyrivlin\". Aborting."
        self.model = model

        assert (young > 0.),\
            "\"young\" ("+str(young)+") must be > 0. Aborting."
        self.young = young

        assert (poisson > -1.),\
            "\"poisson\" ("+str(poisson)+") must be > -1. Aborting."
        assert (poisson < 0.5),\
            "\"poisson\" ("+str(poisson)+") must be < 0.5. Aborting."
        self.poisson = poisson

        self.printer.print_str("Defining regularization energy…")
        self.printer.inc()

        self.dim = self.problem.mesh_dimension

        self.quadrature_degree = quadrature_degree
        form_compiler_parameters = {
            # "representation":"uflacs", # MG20180327: Is that needed?
            "quadrature_degree":self.quadrature_degree}
        self.dS = dolfin.Measure(
            "ds",
            domain=self.problem.mesh,
            metadata=form_compiler_parameters)
        self.dV = dolfin.Measure(
            "dx",
            domain=self.problem.mesh,
            metadata=form_compiler_parameters)

        self.material = dmech.material_factory(
            kinematics=dmech.Kinematics(
                U=self.problem.U),
            model=self.model,
            parameters={
                "E":self.young,
                "nu":self.poisson})
        self.Psi   = self.material.Psi
        self.Sigma = self.material.Sigma
        self.P     = self.material.P

        self.N = dolfin.FacetNormal(self.problem.mesh)
        self.F = dolfin.dot(self.P, self.N)
        self.Fn = dolfin.inner(self.N, self.F)

        if (self.dim == 2):
            ez = dolfin.as_vector([0, 0, 1])
            N3D = dolfin.as_vector([self.N[0], self.N[1], 0])
            T3D = dolfin.cross(ez, N3D)
            self.T = dolfin.as_vector([T3D[0], T3D[1]])
            self.Ft = dolfin.inner(self.T, self.F)
        elif (self.dim == 3):
            self.Ft = self.F - self.Fn * self.N
            self.Ft = dolfin.inner(self.Ft, self.Ft)
            self.Ft = dolfin.conditional(dolfin.gt(self.Ft, 0.), dolfin.sqrt(self.Ft), 0.)
            # self.Ft = dolfin.sqrt(self.Ft)

        if (self.type == "tractions"):
            self.R_fe = dolfin.TensorElement(
                family="Lagrange",
                cell=self.problem.mesh.ufl_cell(),
                degree=self.problem.U_degree)
        elif (self.type in ("tractions-normal", "tractions-tangential", "tractions-normal-tangential")):
            if (self.dim == 2) and (scalar_formulation_in_2D):
                self.R_fe = dolfin.FiniteElement(
                    family="Lagrange",
                    cell=self.problem.mesh.ufl_cell(),
                    degree=self.problem.U_degree)
            else:
                self.R_fe = dolfin.VectorElement(
                    family="Lagrange",
                    cell=self.problem.mesh.ufl_cell(),
                    degree=self.problem.U_degree)
        self.R_fs = dolfin.FunctionSpace(
            self.problem.mesh,
            self.R_fe)
        self.R = dolfin.Function(
            self.R_fs,
            name="traction gradient projection")
        self.MR = dolfin.Function(
            self.R_fs,
            name="traction gradient projection")
        self.R_vec = self.R.vector()
        self.MR_vec = self.MR.vector()
        self.dR_mat = dolfin.PETScMatrix()
        self.dRMR_vec = self.problem.U.vector().copy()

        self.R_tria = dolfin.TrialFunction(self.R_fs)
        self.R_test = dolfin.TestFunction(self.R_fs)
        self.proj_op = dolfin.Identity(self.dim) - dolfin.outer(self.N, self.N)

        if (self.type == "tractions"):
            # vi = self.R_test[0,:]
            # print(vi)
            # grad_vi = dolfin.grad(vi)
            # print(grad_vi)
            # grads_vi = dolfin.dot(self.proj_op, dolfin.dot(grad_vi, self.proj_op))
            # print(grads_vi)
            # divs_vi = dolfin.tr(grads_vi)
            # print(divs_vi)
            divs_R_test = dolfin.as_vector(
                [dolfin.tr(dolfin.dot(self.proj_op, dolfin.dot(dolfin.grad(self.R_test[i,:]), self.proj_op)))
                 for i in range(self.dim)])
            self.R_form = dolfin.inner(
                self.F,
                divs_R_test) * self.dS
        elif (self.type in ("tractions-normal", "tractions-tangential", "tractions-normal-tangential")):
            if (self.dim == 2) and (scalar_formulation_in_2D):
                divs_R_test = dolfin.inner(self.T, dolfin.grad(self.R_test))
            else:
                divs_R_test = dolfin.tr(dolfin.dot(self.proj_op, dolfin.dot(dolfin.grad(self.R_test), self.proj_op)))
                # divs_R_test = dolfin.tr(dolfin.dot(self.proj_op, dolfin.grad(self.R_test)))
                # divs_R_test = dolfin.tr(dolfin.dot(self.proj_op, dolfin.dot(dolfin.grad(dolfin.inner(self.T, self.R_test) * self.T), self.proj_op)))
                # divs_R_test = dolfin.tr(dolfin.dot(self.proj_op, dolfin.dot(dolfin.sym(dolfin.grad(self.R_test)), self.proj_op)))
                # divs_R_test = dolfin.inner(self.proj_op, dolfin.grad(self.R_test))
                # divs_R_test = dolfin.inner(dolfin.outer(self.T, self.T), dolfin.grad(self.R_test))
            self.R_form = dolfin.Constant(0.) * dolfin.inner(self.R, self.R_test) * self.dS
            if ("-normal" in type):
                self.R_form += dolfin.inner(
                    self.Fn,
                    divs_R_test) * self.dS
                # self.R_form += dolfin.inner(
                #     dolfin.dot(self.proj_op, dolfin.grad(self.Fn)),
                #     self.R_test) * self.dS
            if ("-tangential" in type):
                self.R_form += dolfin.inner(
                    self.Ft,
                    divs_R_test) * self.dS
        self.dR_form = dolfin.derivative(self.R_form, self.problem.U, self.problem.dU_trial)

        # dolfin.assemble(
        #     form=self.R_form,
        #     tensor=self.R_vec)
        # print(f"R_vec.get_local() = {self.R_vec.get_local()}")
        # self.problem.U.vector()[:] = (numpy.random.rand(*self.problem.U.vector().get_local().shape)-0.5)/10
        # dolfin.assemble(
        #     form=self.R_form,
        #     tensor=self.R_vec)
        # print(f"R_vec.get_local() = {self.R_vec.get_local()}")

        # M_form = dolfin.inner(
        #     self.R_tria,
        #     self.R_test) * self.dV
        # E     = 1.0e+6
        # nu    = 0.3
        # lmbda = E*nu/(1+nu)/(1-2*nu) # In 2D: plane strain
        # mu    = E/2/(1+nu)
        # M_form += dolfin.Constant(lmbda) * dolfin.tr(dolfin.sym(dolfin.grad(self.R_tria))) * dolfin.tr(dolfin.sym(dolfin.grad(self.R_test))) * self.dV
        # M_form += 2*dolfin.Constant(mu) * dolfin.inner(dolfin.sym(dolfin.grad(self.R_tria)), dolfin.sym(dolfin.grad(self.R_test))) * self.dV
        # self.M_mat = dolfin.PETScMatrix()
        # dolfin.assemble(
        #     form=M_form,
        #     tensor=self.M_mat)
        # self.linear_solver = dolfin.LUSolver(
        #     self.M_mat,
        #     "mumps")
        # self.linear_solver.parameters['report']    = bool(0)
        # self.linear_solver.parameters['symmetric'] = bool(1)
        # self.linear_solver.parameters['verbose']   = bool(0)

        M_lumped_form = dolfin.inner(
            self.R_tria,
            self.R_test) * dolfin.ds(
                domain=self.problem.mesh,
                scheme="vertex",
                metadata={
                    "degree":1,
                    "representation":"quadrature"})
        M_lumped_form += dolfin.Constant(0.) * dolfin.inner(
            self.R_tria,
            self.R_test) * dolfin.dx(
                domain=self.problem.mesh,
                scheme="vertex",
                metadata={
                    "degree":1,
                    "representation":"quadrature"}) # MG20220114: For some reason this might be needed, cf. https://fenicsproject.discourse.group/t/petsc-error-code-63-argument-out-of-range/1564
        self.M_lumped_mat = dolfin.PETScMatrix()
        dolfin.assemble(
            form=M_lumped_form,
            tensor=self.M_lumped_mat)
        # print(self.M_lumped_mat.array())
        self.M_lumped_vec = self.R_vec.copy()
        self.M_lumped_mat.get_diagonal(
            self.M_lumped_vec)
        # print(self.M_lumped_vec.get_local())
        # print(self.M_lumped_vec.norm("l2"))
        self.M_lumped_inv_vec = self.M_lumped_vec.copy()
        self.M_lumped_inv_vec[:] = 1.
        self.M_lumped_inv_vec.vec().pointwiseDivide(
            self.M_lumped_inv_vec.vec(),
            self.M_lumped_vec.vec()) # MG20220817: Note that VecPointwiseDivide returns 0 if it needs to divide by 0…
        # print(self.M_lumped_inv_vec.get_local())
        # print(self.M_lumped_inv_vec.norm("l2"))
        self.M_lumped_inv_mat = self.M_lumped_mat.copy()
        self.M_lumped_inv_mat.set_diagonal(
            self.M_lumped_inv_vec)
        # print(self.M_lumped_inv_mat.array())

        # self.problem.U.vector()[:] = 0.
        # self.assemble_ener()
        # self.problem.U.vector()[:] = (numpy.random.rand(*self.problem.U.vector().get_local().shape)-0.5)/10
        # self.assemble_ener()

        # self.fe_sca = dolfin.FiniteElement(
        #     family="CG",
        #     cell=self.problem.mesh.ufl_cell(),
        #     degree=1)
        # self.fs_sca = dolfin.FunctionSpace(
        #     self.problem.mesh,
        #     self.fe_sca)
        # self.f_Fn = dolfin.Function(self.fs_sca)
        # self.f_Ft = dolfin.Function(self.fs_sca)
        # self.u_sca = dolfin.TrialFunction(self.fs_sca)
        # self.v_sca = dolfin.TestFunction(self.fs_sca)
        # self.a_sca = dolfin.inner(self.u_sca, self.v_sca) * self.problem.dS
        # self.A_sca = dolfin.assemble(self.a_sca, keep_diagonal=True)
        # self.A_sca.ident_zeros()

        # self.fe_vec = dolfin.VectorElement(
        #     family="CG",
        #     cell=self.problem.mesh.ufl_cell(),
        #     degree=1)
        # self.fs_vec = dolfin.FunctionSpace(
        #     self.problem.mesh,
        #     self.fe_vec)
        # self.f_grad_Fn = dolfin.Function(self.fs_vec)
        # self.f_grads_Fn = dolfin.Function(self.fs_vec)
        # self.u_vec = dolfin.TrialFunction(self.fs_vec)
        # self.v_vec = dolfin.TestFunction(self.fs_vec)
        # self.a_vec = dolfin.inner(self.u_vec, self.v_vec) * self.problem.dS
        # self.A_vec = dolfin.assemble(self.a_vec, keep_diagonal=True)
        # self.A_vec.ident_zeros()

        # self.k_frame = 0

        self.printer.dec()



    def assemble_ener(self,
            w_weight=True):

        # dolfin.plot(self.Fn) # "Don't know how to plot given object"

        # dolfin.project(
        #     v=self.Fn,
        #     V=self.fs,
        #     function=self.f_Fn) # Integral of type cell cannot contain a ReferenceNormal
        # dmech.write_VTU_file("Fn", self.f_Fn, self.k_frame)

        # l = dolfin.inner(self.Fn, self.v_sca) * self.problem.dS
        # L = dolfin.assemble(l)
        # dolfin.solve(self.A_sca, self.f_Fn.vector(), L)
        # dmech.write_VTU_file("Fn", self.f_Fn, self.k_frame)

        # l = dolfin.inner(self.Ft, self.v_sca) * self.problem.dS
        # L = dolfin.assemble(l)
        # dolfin.solve(self.A_sca, self.f_Ft.vector(), L)
        # dmech.write_VTU_file("Ft", self.f_Ft, self.k_frame)

        # grad_Fn = dolfin.grad(self.Fn)
        # l = dolfin.inner(grad_Fn, self.v_vec) * self.problem.dS
        # L = dolfin.assemble(l)
        # dolfin.solve(self.A_vec, self.f_grad_Fn.vector(), L)
        # dmech.write_VTU_file("grad_Fn", self.f_grad_Fn, self.k_frame)

        # grads_Fn = dolfin.dot(self.proj_op, grad_Fn)
        # l = dolfin.inner(grads_Fn, self.v_vec) * self.problem.dS
        # L = dolfin.assemble(l)
        # dolfin.solve(self.A_vec, self.f_grads_Fn.vector(), L)
        # dmech.write_VTU_file("grads_Fn", self.f_grads_Fn, self.k_frame)

        self.R_vec.zero()
        dolfin.assemble(
            form=self.R_form,
            tensor=self.R_vec)
        # print(self.R_vec.get_local())
        # print(self.R_vec.norm("l2"))
        # dmech.write_VTU_file("R", self.R, self.k_frame)

        # l = dolfin.inner(dolfin.inner(self.N, self.R), self.v_sca) * self.problem.dS
        # L = dolfin.assemble(l)
        # dolfin.solve(self.A_sca, self.f_Fn.vector(), L)
        # print(self.f_Fn.vector().norm("l2"))
        # dmech.write_VTU_file("Rn", self.f_Fn, self.k_frame)

        # l = dolfin.inner(dolfin.inner(self.T, self.R), self.v_sca) * self.problem.dS
        # L = dolfin.assemble(l)
        # dolfin.solve(self.A_sca, self.f_Ft.vector(), L)
        # print(self.f_Ft.vector().norm("l2"))
        # dmech.write_VTU_file("Rt", self.f_Ft, self.k_frame)

        self.MR_vec.vec().pointwiseDivide(self.R_vec.vec(), self.M_lumped_vec.vec())
        # self.MR_vec.vec().pointwiseMult(self.R_vec.vec(), self.M_lumped_vec.vec())
        # self.linear_solver.solve(
        #     self.MR_vec,
        #     self.R_vec)
        # print(self.MR_vec.get_local())
        # print(self.MR_vec.norm("l2"))
        # dmech.write_VTU_file("MR", self.MR, self.k_frame)

        # l = dolfin.inner(dolfin.inner(self.N, self.MR), self.v_sca) * self.problem.dS
        # L = dolfin.assemble(l)
        # dolfin.solve(self.A_sca, self.f_Fn.vector(), L)
        # print(self.f_Fn.vector().norm("l2"))
        # dmech.write_VTU_file("MRn", self.f_Fn, self.k_frame)

        # l = dolfin.inner(dolfin.inner(self.T, self.MR), self.v_sca) * self.problem.dS
        # L = dolfin.assemble(l)
        # dolfin.solve(self.A_sca, self.f_Ft.vector(), L)
        # print(self.f_Ft.vector().norm("l2"))
        # dmech.write_VTU_file("MRt", self.f_Ft, self.k_frame)

        ener  = self.R_vec.inner(self.MR_vec)
        ener /= 2
        # print(ener)

        # self.k_frame += 1

        try:
            ener /= self.ener0
        except AttributeError:
            pass

        if (w_weight):
            ener *= self.w
            # print(ener)
        return ener



    def assemble_res(self,
            res_vec,
            add_values=True,
            finalize_tensor=True,
            w_weight=True):

        assert (add_values == True)

        # print(res_vec.get_local())

        dolfin.assemble(
            form=self.R_form,
            tensor=self.R_vec)
        # print(self.R_vec.get_local())

        self.MR_vec.vec().pointwiseDivide(self.R_vec.vec(), self.M_lumped_vec.vec())
        # print(self.MR_vec.get_local())

        dolfin.assemble(
            form=self.dR_form,
            tensor=self.dR_mat)
        # print(self.dR_mat.array())

        self.dR_mat.transpmult(self.MR_vec, self.dRMR_vec)
        # print(self.dRMR_vec.get_local())

        try:
            res_vec /= self.ener0
        except AttributeError:
            pass

        if (w_weight):
            res_vec.axpy(self.w, self.dRMR_vec)
        else:
            res_vec.axpy(     1, self.dRMR_vec)
        # print(res_vec.get_local())



    def assemble_jac(self,
            jac_mat,
            add_values=True,
            finalize_tensor=True,
            w_weight=True):

        assert (add_values == True)

        dolfin.assemble(
            form=self.dR_form,
            tensor=self.dR_mat)
        # print(self.dR_mat.array())

        self.K_mat_mat = petsc4py.PETSc.Mat.PtAP(self.M_lumped_inv_mat.mat(), self.dR_mat.mat())
        self.K_mat = dolfin.PETScMatrix(self.K_mat_mat)

        try:
            jac_mat /= self.ener0
        except AttributeError:
            pass

        if (w_weight):
            jac_mat.axpy(self.w, self.K_mat, False) # MG20220107: cannot provide same_nonzero_pattern as kwarg
        else:
            jac_mat.axpy(     1, self.K_mat, False) # MG20220107: cannot provide same_nonzero_pattern as kwarg



    def get_qoi_names(self):

        return [self.name+"_ener"]



    def get_qoi_values(self):

        self.ener  = self.assemble_ener(w_weight=0)
        self.ener /= self.problem.mesh_V0
        assert (self.ener >= 0.),\
            "ener (="+str(self.ener)+") should be non negative. Aborting."
        self.ener  = self.ener**(1./2)
        self.printer.print_sci(self.name+"_ener",self.ener)

        return [self.ener]
