#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2016-2025                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

import dolfin
import typing

from .Energy                 import Energy
from .EnergyMixin_Continuous import ContinuousEnergyMixin
from .Problem                import Problem

################################################################################

class ConstraintContinuousEnergy(Energy, ContinuousEnergyMixin):



    def __init__(self,
            problem                : Problem                     ,
            name                   : str                  = "con",
            w                      : float                = 1.   ,
            quadrature_degree      : typing.Optional[int] = None , # MG20220815: This can be written "int | None" starting with python 3.10, but it is not readily available on the gitlab runners (Ubuntu 20.04)
            volume_subdomain_data                         = None ,
            volume_subdomain_id                           = None ,
            surface_subdomain_data                        = None ,
            surface_subdomain_id                          = None ):

        self.problem = problem
        self.printer = problem.printer

        self.name = name

        self.w = w

        self.quadrature_degree = quadrature_degree

        self.volume_subdomain_data  = volume_subdomain_data
        self.volume_subdomain_id    = volume_subdomain_id
        self.surface_subdomain_data = surface_subdomain_data
        self.surface_subdomain_id   = surface_subdomain_id

        self.printer.print_str("Defining regularization energy…")
        self.printer.inc()

        self.set_measures()

        self.printer.print_str("Defining constraint energy…")

        self.ener_form = dolfin.inner(self.problem.U, self.problem.U) * self.dS
        self.res_form  = dolfin.derivative(self.ener_form, self.problem.U, self.problem.dU_test)
        self.jac_form  = dolfin.derivative(self.res_form, self.problem.U, self.problem.dU_trial)

        self.printer.dec()
