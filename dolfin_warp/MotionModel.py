#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2016-2021                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

# from builtins import range

import dolfin
import numpy

import dolfin_warp as dwarp

################################################################################

class MotionModel():



    def __init__(self,
            problem,
            type):

        self.problem = problem

        if (type == "transX"):
            self.modes = []
            if   (self.problem.mesh_dimension == 2):
                self.modes.append(dolfin.interpolate(
                    v=dolfin.Expression(
                        ("1.", "0."),
                        element=self.problem.U_fe),
                    V=self.problem.U_fs))
            elif (self.problem.mesh_dimension == 3):
                self.modes.append(dolfin.interpolate(
                    v=dolfin.Expression(
                        ("1.", "0.", "0."),
                        element=self.problem.U_fe),
                    V=self.problem.U_fs))
        elif (type == "translations"):
            self.modes = []
            if   (self.problem.mesh_dimension == 2):
                self.modes.append(dolfin.interpolate(
                    v=dolfin.Expression(
                        ("1.", "0."),
                        element=self.problem.U_fe),
                    V=self.problem.U_fs))
                self.modes.append(dolfin.interpolate(
                    v=dolfin.Expression(
                        ("0.", "1."),
                        element=self.problem.U_fe),
                    V=self.problem.U_fs))
            elif (self.problem.mesh_dimension == 3):
                self.modes.append(dolfin.interpolate(
                    v=dolfin.Expression(
                        ("1.", "0.", "0."),
                        element=self.problem.U_fe),
                    V=self.problem.U_fs))
                self.modes.append(dolfin.interpolate(
                    v=dolfin.Expression(
                        ("0.", "1.", "0."),
                        element=self.problem.U_fe),
                    V=self.problem.U_fs))
                self.modes.append(dolfin.interpolate(
                    v=dolfin.Expression(
                        ("0.", "0.", "1."),
                        element=self.problem.U_fe),
                    V=self.problem.U_fs))
        else:
            assert (0),\
                "Not implemented. Aborting."
        self.n_modes = len(self.modes)



    def update_disp(self,
            reduced_disp,
            disp_vec):

            disp_vec.zero()
            for i in range(self.n_modes):
                disp_vec.axpy(reduced_disp[i], self.modes[i].vector())
