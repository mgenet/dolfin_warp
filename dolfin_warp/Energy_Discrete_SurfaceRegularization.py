#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2016-2021                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

import dolfin
import numpy
import petsc4py

import myPythonLibrary    as mypy
import myVTKPythonLibrary as myvtk

import dolfin_mech as dmech

import dolfin_warp as dwarp
from .Energy_Discrete import DiscreteEnergy

################################################################################

class SurfaceRegularizationDiscreteEnergy(DiscreteEnergy):



    def __init__(self,
            problem,
            name="reg",
            w=1.,
            type="tractions",
            model="ciarletgeymonatneohookeanmooneyrivlin",
            young=1.,
            poisson=0.,
            quadrature_degree=None):

        self.problem = problem
        self.printer = problem.printer

        self.name = name

        self.w = w

        assert (type in ("tractions")),\
            "\"type\" ("+str(type)+") must be \"tractions\". Aborting."
        self.type = type

        assert (model in ("hooke", "kirchhoff", "neohookean", "mooneyrivlin", "neohookeanmooneyrivlin", "ciarletgeymonat", "ciarletgeymonatneohookean", "ciarletgeymonatneohookeanmooneyrivlin")),\
            "\"model\" ("+str(model)+") must be \"hooke\", \"kirchhoff\", \"neohookean\", \"mooneyrivlin\", \"neohookeanmooneyrivlin\", \"ciarletgeymonat\", \"ciarletgeymonatneohookean\" or \"ciarletgeymonatneohookeanmooneyrivlin\". Aborting."
        self.model = model

        assert (young > 0.),\
            "\"young\" ("+str(young)+") must be > 0. Aborting."
        self.young = young

        assert (poisson > -1.),\
            "\"poisson\" ("+str(poisson)+") must be > -1. Aborting."
        assert (poisson < 0.5),\
            "\"poisson\" ("+str(poisson)+") must be < 0.5. Aborting."
        self.poisson = poisson

        self.printer.print_str("Defining regularization energy…")
        self.printer.inc()

        self.quadrature_degree = quadrature_degree
        form_compiler_parameters = {
            # "representation":"uflacs", # MG20180327: Is that needed?
            "quadrature_degree":self.quadrature_degree}
        self.dS = dolfin.Measure(
            "ds",
            domain=self.problem.mesh,
            metadata=form_compiler_parameters)

        self.material_parameters = {
            "E":self.young,
            "nu":self.poisson}
        self.material = dmech.material(
            model=self.model,
            parameters=self.material_parameters)
        self.Psi, self.S = self.material.get_free_energy(
            U=self.problem.U)
        self.P = self.problem.F * self.S
        self.N = dolfin.FacetNormal(self.problem.mesh)
        self.T = dolfin.dot(self.P, self.N)

        self.R_fe = dolfin.TensorElement(
            family="Lagrange",
            cell=self.problem.mesh.ufl_cell(),
            degree=self.problem.U_degree)
        self.R_fs = dolfin.FunctionSpace(
            self.problem.mesh,
            self.R_fe)
        self.R = dolfin.Function(
            self.R_fs,
            name="tractions gradient projection")
        self.R_vec = self.R.vector()
        self.MR_vec = self.R_vec.copy()
        self.dR_mat = dolfin.PETScMatrix()
        self.dRMR_vec = self.problem.U.vector().copy()

        self.R_tria = dolfin.TrialFunction(self.R_fs)
        self.R_test = dolfin.TestFunction(self.R_fs)
        self.proj_op = dolfin.Identity(self.problem.mesh_dimension) - dolfin.outer(self.N, self.N)

        vi = self.R_test[0,:]
        # print(vi)
        grad_vi = dolfin.grad(vi)
        # print(grad_vi)
        grads_vi = dolfin.dot(self.proj_op, dolfin.dot(grad_vi, self.proj_op))
        # print(grads_vi)
        divs_vi = dolfin.tr(grads_vi)
        # print(divs_vi)

        divs_R_test = dolfin.as_vector(
            [dolfin.tr(dolfin.dot(self.proj_op, dolfin.dot(dolfin.grad(self.R_test[i,:]), self.proj_op)))
             for i in range(self.problem.mesh_dimension)])
        self.R_form = dolfin.inner(
            self.T,
            divs_R_test) * self.dS
        self.dR_form = dolfin.derivative(self.R_form, self.problem.U, self.problem.dU_trial)

        # dolfin.assemble(
        #     form=self.R_form,
        #     tensor=self.R_vec)
        # print(f"R_vec.get_local() = {self.R_vec.get_local()}")
        # self.problem.U.vector()[:] = (numpy.random.rand(*self.problem.U.vector().get_local().shape)-0.5)/10
        # dolfin.assemble(
        #     form=self.R_form,
        #     tensor=self.R_vec)
        # print(f"R_vec.get_local() = {self.R_vec.get_local()}")

        M_lumped_form = dolfin.inner(
            self.R_tria,
            self.R_test) * dolfin.ds(
                domain=self.problem.mesh,
                scheme="vertex",
                metadata={
                    "degree":1,
                    "representation":"quadrature"})
        M_lumped_form += dolfin.Constant(0.) * dolfin.inner(self.R_tria, self.R_test) * dolfin.dx(domain=self.problem.mesh, scheme="vertex", metadata={"degree":1, "representation":"quadrature"}) # MG20220114: For some reason this might be needed, cf. https://fenicsproject.discourse.group/t/petsc-error-code-63-argument-out-of-range/1564
        self.M_lumped_mat = dolfin.PETScMatrix()
        dolfin.assemble(
            form=M_lumped_form,
            tensor=self.M_lumped_mat)
        print(self.M_lumped_mat.size(0))
        print(self.M_lumped_mat.size(1))
        # print(self.M_lumped_mat.array())
        self.M_lumped_vec = self.R_vec.copy()
        self.M_lumped_mat.get_diagonal(self.M_lumped_vec)
        print(self.M_lumped_vec.size())
        # print(self.M_lumped_vec.get_local())
        self.M_lumped_inv_vec = self.M_lumped_vec.copy()
        self.M_lumped_inv_vec[:] = 1.
        self.M_lumped_inv_vec.vec().pointwiseDivide(
            self.M_lumped_inv_vec.vec(),
            self.M_lumped_vec.vec())
        print(self.M_lumped_inv_vec.size())
        # print(self.M_lumped_inv_vec.get_local())
        self.M_lumped_inv_mat = self.M_lumped_mat.copy()
        print(self.M_lumped_inv_mat.size(0))
        print(self.M_lumped_inv_mat.size(1))
        self.M_lumped_inv_mat.set_diagonal(self.M_lumped_inv_vec)
        # print(self.M_lumped_inv_mat.array())

        # self.problem.U.vector()[:] = 0.
        # self.assemble_ener()
        # self.problem.U.vector()[:] = (numpy.random.rand(*self.problem.U.vector().get_local().shape)-0.5)/10
        # self.assemble_ener()

        self.printer.dec()



    def assemble_ener(self,
            w_weight=True):

        dolfin.assemble(
            form=self.R_form,
            tensor=self.R_vec)
        # print(self.R_vec.get_local())
        self.MR_vec.vec().pointwiseDivide(self.R_vec.vec(), self.M_lumped_vec.vec())
        # print(self.MR_vec.get_local())
        ener  = self.R_vec.inner(self.MR_vec)
        ener /= 2
        # print(ener)
        if (w_weight):
            ener *= self.w
            # print(ener)
        return ener



    def assemble_res(self,
            res_vec,
            add_values=True,
            finalize_tensor=True,
            w_weight=True):

        assert (add_values == True)

        dolfin.assemble(
            form=self.R_form,
            tensor=self.R_vec)
        # print(self.R_vec.get_local())

        self.MR_vec.vec().pointwiseDivide(self.R_vec.vec(), self.M_lumped_vec.vec())
        # print(self.MR_vec.get_local())

        dolfin.assemble(
            form=self.dR_form,
            tensor=self.dR_mat)
        # print(self.dR_mat.array())

        self.dR_mat.transpmult(self.MR_vec, self.dRMR_vec)
        # print(self.dRMR_vec.get_local())

        if (w_weight):
            res_vec.axpy(self.w, self.dRMR_vec)
        else:
            res_vec.axpy(     1, self.dRMR_vec)



    def assemble_jac(self,
            jac_mat,
            add_values=True,
            finalize_tensor=True,
            w_weight=True):

        assert (add_values == True)

        dolfin.assemble(
            form=self.dR_form,
            tensor=self.dR_mat)
        # print(self.dR_mat.array())

        self.K_mat_mat = petsc4py.PETSc.Mat.PtAP(self.M_lumped_inv_mat.mat(), self.dR_mat.mat())
        self.K_mat = dolfin.PETScMatrix(self.K_mat_mat)

        if (w_weight):
            jac_mat.axpy(self.w, self.K_mat, False) # MG20220107: cannot provide same_nonzero_pattern as kwarg
        else:
            jac_mat.axpy(     1, self.K_mat, False) # MG20220107: cannot provide same_nonzero_pattern as kwarg



    def get_qoi_names(self):

        return [self.name+"_ener"]



    def get_qoi_values(self):

        self.ener  = self.assemble_ener(w_weight=0)
        self.ener /= self.problem.mesh_V0
        self.ener  = self.ener**(1./2)
        self.printer.print_sci(self.name+"_ener",self.ener)

        return [self.ener]