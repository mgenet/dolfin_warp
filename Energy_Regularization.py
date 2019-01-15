#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2016-2018                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

import dolfin

import myPythonLibrary as mypy
import myVTKPythonLibrary as myvtk

import dolfin_cm as dcm

import dolfin_dic as ddic
from Energy import Energy

################################################################################

class RegularizationEnergy(Energy):



    def __init__(self,
            problem,
            name="reg",
            w=1.,
            type="equilibrated",
            model="neohookean",
            young=1.,
            poisson=0.,
            quadrature_degree=None):

        self.problem           = problem
        self.printer           = problem.printer
        self.name              = name
        self.w                 = w
        self.type              = type
        self.model             = model
        self.young             = young
        self.poisson           = poisson
        self.quadrature_degree = quadrature_degree

        self.printer.print_str("Defining regularization energy…")
        self.printer.inc()

        self.printer.print_str("Defining measures…")

        self.form_compiler_parameters = {
            "representation":"uflacs", # MG20180327: Is that needed?
            "quadrature_degree":self.quadrature_degree}
        self.dV = dolfin.Measure(
            "dx",
            domain=self.problem.mesh,
            metadata=self.form_compiler_parameters)
        self.dF = dolfin.Measure(
            "dS",
            domain=self.problem.mesh,
            metadata=self.form_compiler_parameters)
        self.dS = dolfin.Measure(
            "ds",
            domain=self.problem.mesh,
            metadata=self.form_compiler_parameters)

        self.printer.print_str("Defining mechanical model…")

        self.E  = dolfin.Constant(self.young)
        self.nu = dolfin.Constant(self.poisson)
        self.material_parameters = {
            "E":self.E,
            "nu":self.nu}

        if (self.model == "linear"): # <- super bad
            self.e = dolfin.sym(dolfin.grad(self.problem.U))
            self.material = dcm.HookeElasticMaterial(
                parameters=self.material_parameters)
            self.Psi_m, self.S_m = self.material.get_free_energy(
                epsilon=self.e)
            self.P_m = self.S_m
        elif (self.model in ("kirchhoff", "neohookean")):
            self.dim = self.problem.U.ufl_shape[0]
            self.I = dolfin.Identity(self.dim)
            self.F = self.I + dolfin.grad(self.problem.U)
            self.C = self.F.T * self.F
            if (self.model == "kirchhoff"): # <- pretty bad too
                self.E = (self.C - self.I)/2
                self.material = dcm.KirchhoffElasticMaterial(
                    parameters=self.material_parameters)
                self.Psi_m, self.S_m = self.material.get_free_energy(
                    E=self.E)
            elif (self.model == "neohookean"):
                self.material = dcm.CiarletGeymonatNeoHookeanElasticMaterial(
                    parameters=self.material_parameters)
                self.Psi_m, self.S_m = self.material.get_free_energy(
                    C=self.C)
            self.P_m = self.F * self.S_m
        else:
            assert (0), "\"model\" ("+str(self.model)+") must be \"linear\", \"kirchhoff\" or \"neohookean\". Aborting."

        self.printer.print_str("Defining regularization energy…")

        if (self.type == "hyperelastic"):
            self.Psi_m_V = self.Psi_m
            self.Psi_m_F = dolfin.Constant(0)
            self.Psi_m_S = dolfin.Constant(0)
        elif (self.type == "equilibrated"):
            self.Div_P = dolfin.div(self.P_m)
            self.Psi_m_V = dolfin.inner(self.Div_P,
                                        self.Div_P)
            self.N = dolfin.FacetNormal(self.problem.mesh)
            self.Jump_P_N = dolfin.jump(self.P_m,
                                        self.N)
            self.cell_h = dolfin.Constant(self.problem.mesh.hmin())
            self.Psi_m_F = dolfin.inner(self.Jump_P_N,
                                        self.Jump_P_N)/self.cell_h
            # self.P_N = self.P_m * self.N
            # self.P_N_N = dolfin.dot(self.N,
            #                         self.P_N)
            # self.P_N_T = self.P_N - self.P_N_N * self.N
            # self.Psi_m_S = dolfin.inner(self.P_N_T,
            #                             self.P_N_T)/self.cell_h
            # self.Psi_m_S = dolfin.inner(self.P_N,
            #                             self.P_N)/self.cell_h
            self.Psi_m_S = dolfin.Constant(0)
        else:
            assert (0), "\"type\" ("+str(self.type)+") must be \"hyperelastic\" or \"equilibrated\". Aborting."

        self.DPsi_m_V  = dolfin.derivative( self.Psi_m_V, self.problem.U, self.problem.dU_test )
        self.DPsi_m_F  = dolfin.derivative( self.Psi_m_F, self.problem.U, self.problem.dU_test )
        self.DPsi_m_S  = dolfin.derivative( self.Psi_m_S, self.problem.U, self.problem.dU_test )
        self.DDPsi_m_V = dolfin.derivative(self.DPsi_m_V, self.problem.U, self.problem.dU_trial)
        self.DDPsi_m_F = dolfin.derivative(self.DPsi_m_F, self.problem.U, self.problem.dU_trial)
        self.DDPsi_m_S = dolfin.derivative(self.DPsi_m_S, self.problem.U, self.problem.dU_trial)

        self.ener_form =   self.Psi_m_V * self.dV +   self.Psi_m_F * self.dF +   self.Psi_m_S * self.dS
        self.res_form  =  self.DPsi_m_V * self.dV +  self.DPsi_m_F * self.dF +  self.DPsi_m_S * self.dS
        self.jac_form  = self.DDPsi_m_V * self.dV + self.DDPsi_m_F * self.dF + self.DDPsi_m_S * self.dS

        self.printer.dec()