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
import dolfin_warp as dwarp

from .Energy_Discrete import DiscreteEnergy
from .Problem         import Problem

################################################################################

class VolumeRegularizationDiscreteEnergy(DiscreteEnergy):



    def __init__(self,
            problem: Problem,
            name: str = "reg",
            w: float = 1.,
            type: str = "equilibrated",
            model: str = "ciarletgeymonatneohookean",
            young: float = 1.,
            poisson: float = 0.,
            b: typing.Optional["list[float]"] = None,
            quadrature_degree: typing.Optional[int] = None): # MG20220815: This can be written "int | None" starting with python 3.10, but it is not readily available on the gitlab runners (Ubuntu 20.04)

        self.problem = problem
        self.printer = problem.printer

        self.name = name

        self.w = w

        assert (type in ("equilibrated")),\
            "\"type\" ("+str(type)+") must be \"equilibrated\". Aborting."
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

        self.b = b

        self.printer.print_str("Defining regularization energy…")
        self.printer.inc()

        self.quadrature_degree = quadrature_degree
        form_compiler_parameters = {
            # "representation":"uflacs", # MG20180327: Is that needed?
            "quadrature_degree":self.quadrature_degree}
        dV = dolfin.Measure(
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
        self.Psi = self.material.Psi
        self.Psi = self.Psi * dV
        if (self.b is not None):
            self.Psi += dolfin.inner(dolfin.Constant(self.b), self.problem.U) * dV
        self.Wint  = dolfin.derivative(self.Psi , self.problem.U, self.problem.dU_test )
        self.dWint = dolfin.derivative(self.Wint, self.problem.U, self.problem.dU_trial)

        M_lumped_form = dolfin.inner(
            self.problem.dU_trial,
            self.problem.dU_test) * dolfin.dx(
                domain=self.problem.mesh,
                scheme="vertex",
                metadata={
                    "degree":1,
                    "representation":"quadrature"})
        self.M_lumped_mat = dolfin.PETScMatrix()
        dolfin.assemble(
            form=M_lumped_form,
            tensor=self.M_lumped_mat)
        # print(self.M_lumped_mat.array())
        self.M_lumped_vec = self.problem.U.vector().copy()
        self.M_lumped_mat.get_diagonal(self.M_lumped_vec)
        # print(self.M_lumped_vec.get_local())
        self.M_lumped_inv_vec = self.M_lumped_vec.copy()
        self.M_lumped_inv_vec[:] = 1.
        self.M_lumped_inv_vec.vec().pointwiseDivide(
            self.M_lumped_inv_vec.vec(),
            self.M_lumped_vec.vec())
        # print(self.M_lumped_inv_vec.get_local())
        self.M_lumped_inv_mat = self.M_lumped_mat.copy()
        self.M_lumped_inv_mat.set_diagonal(self.M_lumped_inv_vec)
        # print(self.M_lumped_inv_mat.array())

        self.R_vec = self.problem.U.vector().copy()
        self.MR_vec = self.problem.U.vector().copy()
        self.dRMR_vec = self.problem.U.vector().copy()

        self.dR_mat = dolfin.PETScMatrix()

        sd = dolfin.CompiledSubDomain("on_boundary")
        self.bc = dolfin.DirichletBC(self.problem.U_fs, [0]*self.problem.mesh_dimension, sd)

        # self.assemble_ener()
        # self.problem.U.vector()[:] = (numpy.random.rand(*self.problem.U.vector().get_local().shape)-0.5)/10
        # self.assemble_ener()

        self.printer.dec()



    def assemble_ener(self,
            w_weight=True):

        dolfin.assemble(
            form=self.Wint,
            tensor=self.R_vec)
        # print(self.R_vec.get_local())
        self.bc.apply(self.R_vec)
        # print(self.R_vec.get_local())
        self.MR_vec.vec().pointwiseDivide(self.R_vec.vec(), self.M_lumped_vec.vec())
        # print(self.MR_vec.get_local())
        ener = self.R_vec.inner(self.MR_vec)
        ener /= 2
        # print(ener)

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

        dolfin.assemble(
            form=self.Wint,
            tensor=self.R_vec)
        # print(self.R_vec.get_local())
        self.bc.apply(self.R_vec)
        # print(self.R_vec.get_local())

        self.MR_vec.vec().pointwiseDivide(self.R_vec.vec(), self.M_lumped_vec.vec())
        # print(self.MR_vec.get_local())

        dolfin.assemble(
            form=self.dWint,
            tensor=self.dR_mat)
        # print(self.dR_mat.array())
        self.bc.zero(self.dR_mat)
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



    def assemble_jac(self,
            jac_mat,
            add_values=True,
            finalize_tensor=True,
            w_weight=True):

        assert (add_values == True)

        dolfin.assemble(
            form=self.dWint,
            tensor=self.dR_mat)
        # print(self.dR_mat.array())
        self.bc.zero(self.dR_mat)
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
