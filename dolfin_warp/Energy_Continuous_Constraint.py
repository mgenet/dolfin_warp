#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2016-2024                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

import dolfin
import typing

import dolfin_mech as dmech
import dolfin_warp as dwarp

from .Energy_Continuous import ContinuousEnergy
from .Problem           import Problem

################################################################################

class ConstraintContinuousEnergy(ContinuousEnergy):



    def __init__(self,
            problem: Problem,
            name: str = "con",
            w: float = 1.,
            volume_subdomain_data = None,
            volume_subdomain_id = None,
            surface_subdomain_data = None,
            surface_subdomain_id = None,
            quadrature_degree: typing.Optional[int] = None): # MG20220815: This can be written "int | None" starting with python 3.10, but it is not readily available on the gitlab runners (Ubuntu 20.04)

        self.problem = problem
        self.printer = problem.printer

        self.name = name

        self.w = w

        self.quadrature_degree = quadrature_degree

        self.printer.print_str("Defining regularization energy…")
        self.printer.inc()

        self.printer.print_str("Defining measures…")

        self.form_compiler_parameters = {
            "quadrature_degree":self.quadrature_degree}
        self.dV = dolfin.Measure(
            "dx",
            domain=self.problem.mesh,
            subdomain_data=volume_subdomain_data,
            subdomain_id=volume_subdomain_id if volume_subdomain_id is not None else "everywhere",
            metadata=self.form_compiler_parameters)
        self.dF = dolfin.Measure(
            "dS",
            domain=self.problem.mesh,
            subdomain_data=volume_subdomain_data,
            subdomain_id=volume_subdomain_id if volume_subdomain_id is not None else "everywhere",
            metadata=self.form_compiler_parameters)
        self.dS = dolfin.Measure(
            "ds",
            domain=self.problem.mesh,
            subdomain_data=surface_subdomain_data,
            subdomain_id=surface_subdomain_id if volume_subdomain_id is not None else "everywhere",
            metadata=self.form_compiler_parameters)

        self.printer.print_str("Defining constraint energy…")

        self.ener_form = dolfin.inner(self.problem.U, self.problem.U) * self.dS
        self.res_form  = dolfin.derivative(self.ener_form, self.problem.U, self.problem.dU_test)
        self.jac_form  = dolfin.derivative(self.res_form, self.problem.U, self.problem.dU_trial)

        self.printer.dec()
