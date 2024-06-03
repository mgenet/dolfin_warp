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
import typing

import dolfin_warp as dwarp

from .Energy_Continuous import ContinuousEnergy
from .Problem           import Problem

################################################################################

class MeshVolumeContinuousEnergy(ContinuousEnergy):



    def __init__(self,
            problem: Problem,
            quadrature_degree: typing.Optional[int] = None,
            name: str = "vol",
            w: float = 1.):

        self.problem           = problem
        self.printer           = self.problem.printer
        self.quadrature_degree = quadrature_degree
        self.name              = name
        self.w                 = w

        self.printer.print_str("Defining mesh volume energy…")
        self.printer.inc()

        # J
        self.I = dolfin.Identity(self.problem.mesh_dimension)
        self.F = self.I + dolfin.grad(self.problem.U)
        self.J = dolfin.det(self.F)
        self.DJ_test = dolfin.derivative(self.J, self.problem.U, self.problem.dU_test)
        self.DJ_trial = dolfin.derivative(self.J, self.problem.U, self.problem.dU_trial)
        self.DDJ = dolfin.derivative(self.DJ_test, self.problem.U, self.problem.dU_trial)

        # dV
        self.form_compiler_parameters = {
            "quadrature_degree":self.quadrature_degree}
        self.dV = dolfin.Measure(
            "dx",
            domain=self.problem.mesh,
            metadata=self.form_compiler_parameters)
        
        # forms
        self.mesh_V0 = dolfin.assemble(dolfin.Constant(1.) * self.dV)

        self.printer.dec()



    def assemble_ener_w_weight(self,
            w):

        self.mesh_V = dolfin.assemble(self.J * self.dV)

        ener = self.V0/self.V

        return w*ener



    def assemble_res_w_weight(self,
            w,
            res_vec,
            add_values=True,
            finalize_tensor=True):

        assert (add_values == True)

        self.mesh_V = dolfin.assemble(self.J * self.dV)

        self.res_form = - dolfin.Constant(self.mesh_V0/self.mesh_V**2) * self.DJ_test * self.dV

        dolfin.assemble(
            form=dolfin.Constant(w) * self.res_form,
            tensor=res_vec,
            add_values=add_values,
            finalize_tensor=finalize_tensor)



    def assemble_jac_w_weight(self,
            w,
            jac_mat,
            add_values=True,
            finalize_tensor=True):

        assert (add_values == True)

        self.res_form = - dolfin.Constant(self.mesh_V0/self.mesh_V**2) * self.DJ_test * self.dV

        self.jac_form  = - dolfin.Constant(self.mesh_V0/self.mesh_V**2) * self.DDJ * self.dV
        self.jac_form +=   dolfin.Constant(self.mesh_V0/self.mesh_V**3) * self.DJ_test * self.DJ_trial * self.dV

        dolfin.assemble(
            form=dolfin.Constant(w) * self.jac_form,
            tensor=jac_mat,
            add_values=add_values,
            finalize_tensor=finalize_tensor)
