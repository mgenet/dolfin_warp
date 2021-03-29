#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2016-2021                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

import dolfin
import petsc4py

import myPythonLibrary    as mypy
import myVTKPythonLibrary as myvtk

import dolfin_mech as dmech

import dolfin_warp as dwarp
from .Energy_Discrete import DiscreteEnergy

################################################################################

class RegularizationDiscreteEnergy(DiscreteEnergy):



    def __init__(self,
            problem,
            name="reg",
            w=1.,
            type="equilibrated",
            model="hooke",
            young=1.,
            poisson=0.,
            quadrature_degree=None):

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



    def assemble_ener(self):

        self.K_mat.mult(self.U_vec, self.KU_vec)
        ener = self.U_vec.inner(self.KU_vec)
        ener *= self.w/2
        return ener



    def assemble_res(self,
            res_vec,
            add_values=True,
            finalize_tensor=True):

        assert (add_values == True)

        self.K_mat.mult(self.U_vec, self.KU_vec)
        res_vec.axpy(self.w, self.KU_vec)



    def assemble_jac(self,
            jac_mat,
            add_values=True,
            finalize_tensor=True):

        assert (add_values == True)

        jac_mat.axpy(self.w, self.K_mat, False)



    def get_qoi_names(self):

        return [self.name+"_ener"]



    def get_qoi_values(self):

        self.K_mat.mult(self.U_vec, self.KU_vec)
        self.ener = self.U_vec.inner(self.KU_vec)
        self.ener /= 2
        self.ener /= self.problem.mesh_V0
        self.ener = self.ener**(1./2)
        self.printer.print_sci(self.name+"_ener",self.ener)

        return [self.ener]
