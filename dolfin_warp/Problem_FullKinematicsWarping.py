#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2016-2024                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

import dolfin
import os

import myPythonLibrary as mypy

import dolfin_warp as dwarp

from .Problem_Warping import WarpingProblem

################################################################################

class FullKinematicsWarpingProblem(WarpingProblem):



    def __init__(self,
            mesh=None,
            mesh_folder=None,
            mesh_basename=None,
            U_family="Lagrange",
            U_degree=1,
            silent=False):

        self.printer = mypy.Printer(
            silent=silent)

        self.set_mesh(
            mesh=mesh,
            mesh_folder=mesh_folder,
            mesh_basename=mesh_basename)

        self.set_displacement(
            U_family=U_family,
            U_degree=U_degree)

        self.energies = []

        self.kinematics_type == "full"



    def set_displacement(self,
            U_family="Lagrange",
            U_degree=1):

        self.printer.print_str("Defining functions…")

        self.U_family = U_family
        self.U_degree = U_degree
        self.U_fe = dolfin.VectorElement(
            family=self.U_family,
            cell=self.mesh.ufl_cell(),
            degree=self.U_degree)
        self.U_fs = dolfin.FunctionSpace(
            self.mesh,
            self.U_fe)
        self.U = dolfin.Function(
            self.U_fs,
            name="displacement")
        self.U.vector().zero()
        self.U_norm = 0.
        self.Uold = dolfin.Function(
            self.U_fs,
            name="previous displacement")
        self.Uold.vector().zero()
        self.Uold_norm = 0.
        self.DU = dolfin.Function(
            self.U_fs,
            name="displacement increment")
        self.dU = dolfin.Function(
            self.U_fs,
            name="displacement correction")
        self.dU_trial = dolfin.TrialFunction(self.U_fs)
        self.dU_test = dolfin.TestFunction(self.U_fs)

        # for mesh volume computation
        self.I = dolfin.Identity(self.mesh_dimension)
        self.F = self.I + dolfin.grad(self.U)
        self.J = dolfin.det(self.F)



    def update_displacement(self,
            relax=1):

        self.U.vector().axpy(relax, self.dU.vector())
        self.U_norm = self.U.vector().norm("l2")



    def reinit_displacement(self):

        self.U.vector().zero()
        self.U_norm = 0.
        self.Uold.vector().zero()
        self.Uold_norm = 0.



    def save_old_displacement(self,
            *kargs,
            **kwargs):

        self.Uold.vector()[:] = self.U.vector()
        self.Uold_norm = self.U_norm
