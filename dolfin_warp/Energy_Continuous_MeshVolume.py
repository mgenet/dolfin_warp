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

class MeshVolumeContinuousEnergy(Energy, ContinuousEnergyMixin):



    def __init__(self,
            problem           : Problem                     ,
            quadrature_degree : typing.Optional[int] = None ,
            name              : str                  = "vol",
            w                 : float                = 1.   ):

        self.problem           = problem
        self.printer           = self.problem.printer
        self.quadrature_degree = quadrature_degree
        self.name              = name
        self.w                 = w

        self.printer.print_str("Defining mesh volume energy…")
        self.printer.inc()

        self.set_measures()

        # J
        self.I = dolfin.Identity(self.problem.mesh_dimension)
        self.F = self.I + dolfin.grad(self.problem.U)
        self.J = dolfin.det(self.F)

        # forms
        self.mesh_V0 = dolfin.assemble(dolfin.Constant(1.) * self.dV)
        self.ener_form = -self.J/self.V0 * self.dV
        self.res_form  = self.DPsi_c  * self.dV
        self.jac_form  = self.DDPsi_c * self.dV

        self.printer.dec()
        self.printer.dec()
