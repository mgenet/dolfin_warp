#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2016-2022                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

import dolfin

################################################################################

class MotionModel():



    def __init__(self,
            problem,
            type):

        type_list = ["translation","scaling","translation_and_scaling"]

        self.problem = problem

        self.modes = []
        if (type == "translation" or type == "translation_and_scaling"):
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
        if (type == "scaling" or type == "translation_and_scaling"):
            if   (self.problem.mesh_dimension == 2):
                self.modes.append(dolfin.interpolate(
                    v=dolfin.Expression(
                        ("x[0]", "0."),
                        element=self.problem.U_fe),
                    V=self.problem.U_fs))
                self.modes.append(dolfin.interpolate(
                    v=dolfin.Expression(
                        ("0.", "x[1]"),
                        element=self.problem.U_fe),
                    V=self.problem.U_fs))
            elif (self.problem.mesh_dimension == 3):
                self.modes.append(dolfin.interpolate(
                    v=dolfin.Expression(
                        ("x[0]", "0.", "0."),
                        element=self.problem.U_fe),
                    V=self.problem.U_fs))
                self.modes.append(dolfin.interpolate(
                    v=dolfin.Expression(
                        ("0.", "x[1]", "0."),
                        element=self.problem.U_fe),
                    V=self.problem.U_fs))
                self.modes.append(dolfin.interpolate(
                    v=dolfin.Expression(
                        ("0.", "0.", "x[2]"),
                        element=self.problem.U_fe),
                    V=self.problem.U_fs))
        if (type not in type_list):
            assert (0),\
                "Not implemented. Aborting."
        self.n_modes = len(self.modes)



    def update_disp(self,
            reduced_disp,
            disp_vec):

            disp_vec.zero()
            for i in range(self.n_modes):
                disp_vec.axpy(reduced_disp[i], self.modes[i].vector())
