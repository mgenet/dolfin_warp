#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2016-2020                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

import dolfin

import myPythonLibrary    as mypy
import myVTKPythonLibrary as myvtk

import dolfin_cm as dcm

import dolfin_dic as ddic
from .Energy import Energy

################################################################################

class RegularizationEnergy(Energy):



    def __init__(self,
            problem,
            name="reg",
            w=1.,
            type="equilibrated",
            model="ciarletgeymonatneohookeanmooneyrivlin",
            young=1.,
            poisson=0.,
            quadrature_degree=None):

        self.problem           = problem
        self.dim               = self.problem.U.ufl_shape[0]
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

        if (self.model == "hooke"): # <- super bad
            self.material = dcm.HookeElasticMaterial(
                parameters=self.material_parameters)
            self.Psi_m, self.S_m = self.material.get_free_energy(
                U=self.problem.U)
            self.P_m = self.S_m
        elif (self.model in ("kirchhoff", "neohookean", "mooneyrivlin", "neohookeanmooneyrivlin", "ciarletgeymonat", "ciarletgeymonatneohookean", "ciarletgeymonatneohookeanmooneyrivlin")):
            if (self.model == "kirchhoff"): # <- pretty bad too
                self.material = dcm.KirchhoffElasticMaterial(
                    parameters=self.material_parameters)
            elif (self.model == "neohookean"):
                self.material = dcm.NeoHookeanDevElasticMaterial(
                    parameters=self.material_parameters)
            elif (self.model == "mooneyrivlin"):
                self.material = dcm.MooneyRivlinDevElasticMaterial(
                    parameters=self.material_parameters)
            elif (self.model == "neohookeanmooneyrivlin"):
                self.material = dcm.NeoHookeanMooneyRivlinDevElasticMaterial(
                    parameters=self.material_parameters)
            elif (self.model == "ciarletgeymonat"):
                self.material = dcm.CiarletGeymonatBulkElasticMaterial(
                    parameters=self.material_parameters)
            elif (self.model == "ciarletgeymonatneohookean"):
                self.material = dcm.CiarletGeymonatNeoHookeanElasticMaterial(
                    parameters=self.material_parameters)
            elif (self.model == "ciarletgeymonatneohookeanmooneyrivlin"):
                self.material = dcm.CiarletGeymonatNeoHookeanMooneyRivlinElasticMaterial(
                    parameters=self.material_parameters)
            self.Psi_m, self.S_m = self.material.get_free_energy(
                U=self.problem.U)
            self.I = dolfin.Identity(self.dim)
            self.F = self.I + dolfin.grad(self.problem.U)
            self.P_m = self.F * self.S_m
        else:
            assert (0), "\"model\" ("+str(self.model)+") must be \"hooke\", \"kirchhoff\", \"neohookean\", \"mooneyrivlin\" or \"ciarletgeymonat\". Aborting."

        self.printer.print_str("Defining regularization energy…")

        if (self.type == "hyperelastic"):
            self.Psi_m_V = self.Psi_m
            self.Psi_m_F = dolfin.Constant(0)
            self.Psi_m_S = dolfin.Constant(0)
        elif (self.type == "equilibrated"):
            Div_P = dolfin.div(self.P_m)
            self.Psi_m_V = dolfin.inner(
                Div_P,
                Div_P)/2
            N = dolfin.FacetNormal(self.problem.mesh)
            Jump_P_N = dolfin.jump(
                self.P_m,
                N)
            cell_h = dolfin.Constant(self.problem.mesh.hmin())
            self.Psi_m_F = dolfin.inner(
                Jump_P_N,
                Jump_P_N)/2/cell_h
            # self.P_N = self.P_m * N
            # self.P_N_N = dolfin.dot(N, self.P_N)
            # self.P_N_T = self.P_N - self.P_N_N * N
            # self.Psi_m_S = dolfin.inner(
            #     self.P_N_T,
            #     self.P_N_T)/2/cell_h
            # self.Psi_m_S = dolfin.inner(
            #     self.P_N,
            #     self.P_N)/2/cell_h
            self.Psi_m_S = dolfin.Constant(0)
        elif (self.type == "equilibrated2"):
            # print(self.P_m)
            # print("P_m.ufl_shape = "+str(self.P_m.ufl_shape))
            self.dP_m  = dolfin.diff(self.P_m , self.problem.U)
            self.ddP_m = dolfin.diff(self.dP_m, self.problem.U)
            # print(self.dP_m)
            # print("dP_m.ufl_shape = "+str(self.dP_m.ufl_shape))
            # print(self.ddP_m)
            # print("ddP_m.ufl_shape = "+str(self.ddP_m.ufl_shape))

            self.P_fe = dolfin.TensorElement(
                family="Lagrange",
                cell=self.problem.mesh.ufl_cell(),
                shape=(self.dim,)*2,
                degree=self.problem.U_degree)
            self.P_fs = dolfin.FunctionSpace(
                self.problem.mesh,
                self.P_fe)
            self.P_func = dolfin.Function(
                self.P_fs,
                name="stress")
            # print("P_func.ufl_shape = "+str(self.P_func.ufl_shape))
            self.dP_fe = dolfin.TensorElement(
                family="Lagrange",
                cell=self.problem.mesh.ufl_cell(),
                shape=(self.dim,)*3,
                degree=self.problem.U_degree)
            self.dP_fs = dolfin.FunctionSpace(
                self.problem.mesh,
                self.dP_fe)
            self.dP_func = dolfin.Function(
                self.dP_fs,
                name="dstress")
            # print("dP_func.ufl_shape = "+str(self.dP_func.ufl_shape))
            self.ddP_fe = dolfin.TensorElement(
                family="Lagrange",
                cell=self.problem.mesh.ufl_cell(),
                shape=(self.dim,)*4,
                degree=self.problem.U_degree)
            self.ddP_fs = dolfin.FunctionSpace(
                self.problem.mesh,
                self.ddP_fe)
            self.ddP_func = dolfin.Function(
                self.ddP_fs,
                name="ddstress")
            # print("ddP_func.ufl_shape = "+str(self.ddP_func.ufl_shape))

            Div_P = dolfin.div(self.P_func)
            # print("Div_P.ufl_shape = "+str(self.Div_P.ufl_shape))
            self.Psi_m_V = dolfin.inner(
                Div_P,
                Div_P)/2
            # print(self.Psi_m_V)
            self.Psi_m_F = dolfin.Constant(0)
            self.Psi_m_S = dolfin.Constant(0)
        elif (self.type == "equilibrated3"):
            self.Psi_m_V = dolfin.Constant(0)
            self.Psi_m_F = dolfin.Constant(0)
            self.Psi_m_S = dolfin.Constant(0)
        else:
            assert (0), "\"type\" ("+str(self.type)+") must be \"hyperelastic\" or \"equilibrated\". Aborting."

        if (self.type in ("hyperelastic", "equilibrated")):
            self.DPsi_m_V  = dolfin.derivative( self.Psi_m_V, self.problem.U, self.problem.dU_test )
            self.DPsi_m_F  = dolfin.derivative( self.Psi_m_F, self.problem.U, self.problem.dU_test )
            self.DPsi_m_S  = dolfin.derivative( self.Psi_m_S, self.problem.U, self.problem.dU_test )
            self.DDPsi_m_V = dolfin.derivative(self.DPsi_m_V, self.problem.U, self.problem.dU_trial)
            self.DDPsi_m_F = dolfin.derivative(self.DPsi_m_F, self.problem.U, self.problem.dU_trial)
            self.DDPsi_m_S = dolfin.derivative(self.DPsi_m_S, self.problem.U, self.problem.dU_trial)
        elif (self.type == "equilibrated2"):
            # print(self.dP_func.ufl_shape)
            # print(self.problem.dU_test.ufl_shape)
            Div_dPtest = dolfin.div(dolfin.dot(self.dP_func, self.problem.dU_test))
            # print(Div_dPtest)
            Div_dPtrial = dolfin.div(dolfin.dot(self.dP_func, self.problem.dU_trial))
            # print(Div_dPtrial)
            Div_ddPtesttrial = dolfin.div(dolfin.dot(dolfin.dot(self.ddP_func, self.problem.dU_test), self.problem.dU_trial))
            # print(Div_ddPtesttrial)
            self.DPsi_m_V = dolfin.inner(
                Div_P,
                Div_dPtest)
            # print(self.DPsi_m_V)
            self.DPsi_m_F = dolfin.derivative(dolfin.Constant(0), self.problem.U, self.problem.dU_test) # MG20200225: Needs to be linear in dU_test (arity = 1)
            self.DPsi_m_S = dolfin.derivative(dolfin.Constant(0), self.problem.U, self.problem.dU_test) # MG20200225: Needs to be linear in dU_test (arity = 1)
            self.DDPsi_m_V = dolfin.inner(
                Div_dPtrial,
                Div_dPtest)
            # print(self.DDPsi_m_V)
            self.DDPsi_m_V += dolfin.inner(
                Div_P,
                Div_ddPtesttrial)
            # print(self.DDPsi_m_V)
            self.DDPsi_m_F = dolfin.derivative(self.DPsi_m_F, self.problem.U, self.problem.dU_trial) # MG20200225: Needs to be linear in dU_test and dU_trial (arity = 2)
            self.DDPsi_m_S = dolfin.derivative(self.DPsi_m_S, self.problem.U, self.problem.dU_trial) # MG20200225: Needs to be linear in dU_test and dU_trial (arity = 2)
        elif (self.type == "equilibrated3"):
            self.DPsi_m_V = dolfin.derivative(        self.Psi_m, self.problem.U, self.problem.dU_test)
            self.DPsi_m_F = dolfin.derivative(dolfin.Constant(0), self.problem.U, self.problem.dU_test) # MG20200225: Needs to be linear in dU_test (arity = 1)
            N = dolfin.FacetNormal(self.problem.mesh)
            self.DPsi_m_S = - dolfin.inner(
                self.P_m * N,
                self.problem.dU_test)
            self.DDPsi_m_V = dolfin.derivative(self.DPsi_m_V, self.problem.U, self.problem.dU_trial)
            self.DDPsi_m_F = dolfin.derivative(self.DPsi_m_F, self.problem.U, self.problem.dU_trial) # MG20200225: Needs to be linear in dU_test and dU_trial (arity = 2)
            self.DDPsi_m_S = dolfin.derivative(self.DPsi_m_S, self.problem.U, self.problem.dU_trial)
        else:
            assert (0), "\"type\" ("+str(self.type)+") must be \"hyperelastic\" or \"equilibrated\". Aborting."

        self.ener_form =   self.Psi_m_V * self.dV +   self.Psi_m_F * self.dF +   self.Psi_m_S * self.dS
        self.res_form  =  self.DPsi_m_V * self.dV +  self.DPsi_m_F * self.dF +  self.DPsi_m_S * self.dS
        self.jac_form  = self.DDPsi_m_V * self.dV + self.DDPsi_m_F * self.dF + self.DDPsi_m_S * self.dS

        self.printer.dec()



    def call_before_assembly(self,
            **kwargs):

        if (self.type == "equilibrated2"):
            dolfin.project(
                v=self.P_m,
                V=self.P_fs,
                function=self.P_func)
            dolfin.project(
                v=self.dP_m,
                V=self.dP_fs,
                function=self.dP_func)
            dolfin.project(
                v=self.ddP_m,
                V=self.ddP_fs,
                function=self.ddP_func)
