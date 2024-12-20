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

class ReducedKinematicsWarpingProblem(WarpingProblem):



    def __init__(self,
            mesh=None,
            mesh_folder=None,
            mesh_basename=None,
            model="translation+rotation+scaling+shear",
            silent=False):

        self.printer = mypy.Printer(
            silent=silent)

        self.set_mesh(
            mesh=mesh,
            mesh_folder=mesh_folder,
            mesh_basename=mesh_basename)
        
        self.model = model
        self.set_displacement()

        self.energies = []



    def set_displacement(self):

        self.printer.print_str("Defining functions…")

        fe_lst = []; n_reduced_variables = 0
        if ("translation" in self.model):
            fe_lst += [dolfin.VectorElement(
                family="R",
                cell=self.mesh.ufl_cell(),
                degree=0)]; n_reduced_variables += self.mesh_dimension
        if ("rotation" in self.model):
            if   (self.mesh_dimension==2):
                fe_lst += [dolfin.FiniteElement(
                    family="R",
                    cell=self.mesh.ufl_cell(),
                    degree=0)]; n_reduced_variables += 1
            elif (self.mesh_dimension==3):
                fe_lst += [dolfin.VectorElement(
                    family="R",
                    cell=self.mesh.ufl_cell(),
                    degree=0)]; n_reduced_variables += self.mesh_dimension
        if ("scaling" in self.model):
            fe_lst += [dolfin.VectorElement(
                family="R",
                cell=self.mesh.ufl_cell(),
                degree=0)]; n_reduced_variables += self.mesh_dimension
        if ("shear" in self.model):
            if   (self.mesh_dimension==2):
                fe_lst += [dolfin.FiniteElement(
                    family="R",
                    cell=self.mesh.ufl_cell(),
                    degree=0)]; n_reduced_variables += 1
            elif (self.mesh_dimension==3):
                fe_lst += [dolfin.VectorElement(
                    family="R",
                    cell=self.mesh.ufl_cell(),
                    degree=0)]; n_reduced_variables += self.mesh_dimension

        self.reduced_displacement_fe = dolfin.MixedElement(
            fe_lst)
        self.reduced_displacement_fs = dolfin.FunctionSpace(
            self.mesh,
            self.reduced_displacement_fe)
        # self.printer.print_var("reduced_displacement_fs.dim()",self.reduced_displacement_fs.dim())
        self.reduced_displacement = dolfin.Function(
            self.reduced_displacement_fs,
            name="reduced displacement")
        self.reduced_displacement_old = dolfin.Function(
            self.reduced_displacement_fs,
            name="previous reduced displacement")
        self.dreduced_displacement = dolfin.Function(
            self.reduced_displacement_fs,
            name="reduced displacement correction")
        self.reduced_displacement_test = dolfin.TestFunction(self.reduced_displacement_fs)
        self.reduced_displacement_trial = dolfin.TrialFunction(self.reduced_displacement_fs)

        self.reduced_displacement_splitted = dolfin.split(self.reduced_displacement)
        counter = 0
        if ("translation" in self.model):
            reduced_translation = self.reduced_displacement_splitted[counter]; counter += 1
            if (self.mesh_dimension==2):
                T_X = reduced_translation[0]
                T_Y = reduced_translation[1]
            elif (self.mesh_dimension==3):
                T_X = reduced_translation[0]
                T_Y = reduced_translation[1]
                T_Z = reduced_translation[2]
        else:
            if (self.mesh_dimension==2):
                T_X = dolfin.Constant(0)
                T_Y = dolfin.Constant(0)
            elif (self.mesh_dimension==3):
                T_X = dolfin.Constant(0)
                T_Y = dolfin.Constant(0)
                T_Z = dolfin.Constant(0)
        if ("rotation" in self.model):
            reduced_rotation = self.reduced_displacement_splitted[counter]; counter += 1
            if (self.mesh_dimension==2):
                R_Z = reduced_rotation
            elif (self.mesh_dimension==3):
                R_X = reduced_rotation[0]
                R_Y = reduced_rotation[1]
                R_Z = reduced_rotation[2]
        else:
            if (self.mesh_dimension==2):
                R_Z = dolfin.Constant(0)
            elif (self.mesh_dimension==3):
                R_X = dolfin.Constant(0)
                R_Y = dolfin.Constant(0)
                R_Z = dolfin.Constant(0)
        if ("scaling" in self.model):
            reduced_scaling = self.reduced_displacement_splitted[counter]; counter += 1
            if (self.mesh_dimension==2):
                U_XX = 1+reduced_scaling[0]
                U_YY = 1+reduced_scaling[1]
            elif (self.mesh_dimension==3):
                U_XX = 1+reduced_scaling[0]
                U_YY = 1+reduced_scaling[1]
                U_ZZ = 1+reduced_scaling[2]
        else:
            if (self.mesh_dimension==2):
                U_XX = dolfin.Constant(1)
                U_YY = dolfin.Constant(1)
            elif (self.mesh_dimension==3):
                U_XX = dolfin.Constant(1)
                U_YY = dolfin.Constant(1)
                U_ZZ = dolfin.Constant(1)
        if ("shear" in self.model):
            reduced_shear = self.reduced_displacement_splitted[counter]; counter += 1
            if (self.mesh_dimension==2):
                U_XY = reduced_shear
            elif (self.mesh_dimension==3):
                U_XY = reduced_shear[0]
                U_YZ = reduced_shear[1]
                U_ZX = reduced_shear[2]
        else:
            if (self.mesh_dimension==2):
                U_XY = dolfin.Constant(0)
            elif (self.mesh_dimension==3):
                U_XY = dolfin.Constant(0)
                U_YZ = dolfin.Constant(0)
                U_ZX = dolfin.Constant(0)

        if (self.mesh_dimension==2):
            T = dolfin.as_vector([T_X, T_Y])
            R = dolfin.as_matrix([[+dolfin.cos(R_Z), -dolfin.sin(R_Z)],
                                  [+dolfin.sin(R_Z), +dolfin.cos(R_Z)]])
            U = dolfin.as_matrix([[U_XX, U_XY],
                                  [U_XY, U_YY]])
        if (self.mesh_dimension==3):
            T = dolfin.as_vector([T_X, T_Y, T_Z])
            RX = dolfin.as_matrix([[       1        ,        0        ,        0        ],
                                   [       0        , +dolfin.cos(R_X), -dolfin.sin(R_X)],
                                   [       0        , +dolfin.sin(R_X), +dolfin.cos(R_X)]])
            RY = dolfin.as_matrix([[+dolfin.cos(R_Y),        0        , +dolfin.sin(R_Y)],
                                   [       0        ,        1        ,        0        ],
                                   [-dolfin.sin(R_Y),        0        , +dolfin.cos(R_Y)]])
            RZ = dolfin.as_matrix([[+dolfin.cos(R_Z), -dolfin.sin(R_Z),        0        ],
                                   [+dolfin.sin(R_Z), +dolfin.cos(R_Z),        0        ],
                                   [       0        ,        0        ,        1        ]])
            R = dolfin.dot(RZ, dolfin.dot(RY, RX))
            U = dolfin.as_matrix([[U_XX, U_XY, U_ZX],
                                  [U_XY, U_YY, U_YZ],
                                  [U_ZX, U_YZ, U_ZZ]])

        F = dolfin.dot(R, U)
        self.U_expr = T + dolfin.dot(F - dolfin.Identity(self.mesh_dimension), self.X)
        # print(self.U_expr)

        # for compatibility with image expressions and nonlinear solver
        self.U_fe = dolfin.VectorElement(
            family="Lagrange",
            cell=self.mesh.ufl_cell(),
            degree=1)
        self.U_fs = dolfin.FunctionSpace(
            self.mesh,
            self.U_fe)
        self.U = dolfin.Function(
            self.U_fs,
            name="displacement")
        self.U_vec_cp = self.U.vector().copy()
        self.U_norm = 0.
        self.Uold = dolfin.Function(
            self.U_fs,
            name="previous displacement")
        self.Uold_norm = 0.
        self.DU = dolfin.Function(
            self.U_fs,
            name="displacement increment")
        self.dU = dolfin.Function(
            self.U_fs,
            name="displacement correction")
        self.dU_test = dolfin.derivative(self.U_expr, self.reduced_displacement, self.reduced_displacement_test)
        self.dU_trial = dolfin.derivative(self.U_expr, self.reduced_displacement, self.reduced_displacement_trial)

        # for mesh volume computation
        self.I = dolfin.Identity(self.mesh_dimension)
        self.F = self.I + dolfin.grad(self.U)
        self.J = dolfin.det(self.F)



    def update_disp(self):

        # self.U.interpolate(self.U_expr) #MG20241218: Cannot interpolate UFL expression, cf. https://fenicsproject.discourse.group/t/project-works-but-interpolate-does-not/10090/2
        dolfin.project(
            v=self.U_expr,
            V=self.U_fs,
            function=self.U)
        self.U_norm = self.U.vector().norm("l2")



    def update_displacement(self,
            relax=1):

        self.reduced_displacement.vector().axpy(relax, self.dreduced_displacement.vector())
        self.U_vec_cp[:] = self.U.vector()
        self.update_disp()
        self.dU.vector()[:] = self.U.vector() - self.U_vec_cp



    def reinit_displacement(self):

        self.reduced_displacement.vector().zero()
        self.reduced_displacement_old.vector().zero()
        self.U.vector().zero()
        self.U_norm = 0.
        self.Uold.vector().zero()
        self.Uold_norm = 0.



    def save_old_displacement(self,
            *kargs,
            **kwargs):

        self.reduced_displacement_old.vector()[:] = self.reduced_displacement.vector()
        self.Uold.vector()[:] = self.U.vector()
        self.Uold_norm = self.U_norm
