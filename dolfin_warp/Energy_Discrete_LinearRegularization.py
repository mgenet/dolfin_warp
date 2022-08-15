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

from .Energy_Discrete import DiscreteEnergy
from .Problem         import Problem

################################################################################

class LinearRegularizationDiscreteEnergy(DiscreteEnergy):



    def __init__(self,
            problem: Problem,
            name: str = "reg",
            w: float = 1.,
            type: str = "equilibrated",
            model: str = "hooke",
            young: float = 1.,
            poisson: float = 0.,
            quadrature_degree: typing.Optional[int] = None): # MG20220815: This can be written "int | None" starting with python 3.10, but it is not readily available on the gitlab runners (Ubuntu 20.04)

        self.problem = problem
        self.printer = problem.printer

        self.name = name

        self.w = w

        assert (type in ("equilibrated", "elastic")),\
            "\"type\" ("+str(type)+") must be \"equilibrated\" or \"elastic\". Aborting."
        self.type = type

        assert (model in ("hooke")),\
            "\"model\" ("+str(model)+") must be \"hooke\". Aborting."
        self.model = model

        assert (young > 0.),\
            "\"young\" ("+str(young)+") must be > 0. Aborting."
        self.young = young

        assert (poisson > -1.),\
            "\"poisson\" ("+str(poisson)+") must be > -1. Aborting."
        assert (poisson < 0.5),\
            "\"poisson\" ("+str(poisson)+") must be < 0.5. Aborting."
        self.poisson = poisson

        self.quadrature_degree = quadrature_degree

        self.printer.print_str("Defining regularization energy…")
        self.printer.inc()

        form_compiler_parameters = {
            "representation":"uflacs", # MG20180327: Is that needed?
            "quadrature_degree":self.quadrature_degree}
        dV = dolfin.Measure(
            "dx",
            domain=self.problem.mesh,
            metadata=form_compiler_parameters)

        self.U_vec = self.problem.U.vector()
        self.KU_vec = self.U_vec.copy()

        E  = dolfin.Constant(self.young)
        nu = dolfin.Constant(self.poisson)

        lmbda = E*nu/(1+nu)/(1-2*nu) # Lamé constant (plane strain)
        mu    = E/2/(1+nu)

        epsilon_trial = dolfin.sym(dolfin.grad(self.problem.dU_trial))
        sigma_trial = lmbda * dolfin.tr(epsilon_trial) * dolfin.Identity(self.problem.mesh_dimension) + 2*mu * epsilon_trial

        epsilon_test = dolfin.sym(dolfin.grad(self.problem.dU_test))

        Wint = dolfin.inner(sigma_trial, epsilon_test) * dV

        self.K_mat = dolfin.PETScMatrix()
        dolfin.assemble(Wint, tensor=self.K_mat)

        if (self.type == "equilibrated"):
            sd = dolfin.CompiledSubDomain("on_boundary")
            bc = dolfin.DirichletBC(self.problem.U_fs, [0]*self.problem.mesh_dimension, sd)
            bc.zero(self.K_mat)

            self.K_mat_mat = self.K_mat.mat()
            self.K_mat_mat = petsc4py.PETSc.Mat.transposeMatMult(self.K_mat_mat, self.K_mat_mat)
            self.K_mat = dolfin.PETScMatrix(self.K_mat_mat)

        self.printer.dec()



    def assemble_ener(self,
            w_weight=True):

        self.K_mat.mult(self.U_vec, self.KU_vec)
        ener  = self.U_vec.inner(self.KU_vec)
        ener /= 2

        try:
            ener /= self.ener0
        except AttributeError:
            pass

        if (w_weight):
            ener *= self.w
        return ener



    def assemble_res(self,
            res_vec,
            add_values=True,
            finalize_tensor=True,
            w_weight=True):

        assert (add_values == True)

        self.K_mat.mult(self.U_vec, self.KU_vec)

        try:
            res_vec /= self.ener0
        except AttributeError:
            pass

        if (w_weight):
            res_vec.axpy(self.w, self.KU_vec)
        else:
            res_vec.axpy(     1, self.KU_vec)



    def assemble_jac(self,
            jac_mat,
            add_values=True,
            finalize_tensor=True,
            w_weight=True):

        assert (add_values == True)

        try:
            jac_mat /= self.ener0
        except AttributeError:
            pass

        if (w_weight):
            jac_mat.axpy(self.w, self.K_mat, False)
        else:
            jac_mat.axpy(     1, self.K_mat, False)



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
