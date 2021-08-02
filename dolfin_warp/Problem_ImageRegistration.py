#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2016-2021                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

import dolfin
import os

import myPythonLibrary    as mypy
import myVTKPythonLibrary as myvtk

import dolfin_warp as dwarp
from .Problem import Problem

################################################################################

class ImageRegistrationProblem(Problem):



    def __init__(self,
            mesh=None,
            mesh_folder=None,
            mesh_basename=None,
            U_family="Lagrange",
            U_degree=1):

        self.printer = mypy.Printer()

        self.set_mesh(
            mesh=mesh,
            mesh_folder=mesh_folder,
            mesh_basename=mesh_basename)

        self.set_displacement(
            U_family=U_family,
            U_degree=U_degree)

        self.energies = []



    def close(self):

        self.printer.close()



    def set_mesh(self,
            mesh=None,
            mesh_folder=None,
            mesh_basename=None):

        self.printer.print_str("Loading mesh…")
        self.printer.inc()

        assert ((mesh is not None) or ((mesh_folder is not None) and (mesh_basename is not None))),\
            "Must provide a mesh (mesh = "+str(mesh)+") or a mesh file (mesh_folder = "+str(mesh_folder)+", mesh_basename = "+str(mesh_basename)+"). Aborting."

        if (mesh is None):
            self.mesh_folder = mesh_folder
            self.mesh_basename = mesh_basename
            self.mesh_filebasename = self.mesh_folder+"/"+self.mesh_basename
            self.mesh_filename = self.mesh_filebasename+"."+"xml"
            assert (os.path.exists(self.mesh_filename)),\
                "No mesh in "+self.mesh_filename+". Aborting."
            self.mesh = dolfin.Mesh(self.mesh_filename)
        else:
            self.mesh = mesh

        self.mesh_dimension = self.mesh.ufl_domain().geometric_dimension()
        assert (self.mesh_dimension in (2,3)),\
            "mesh_dimension ("+str(self.mesh_dimension)+") must be 2 or 3. Aborting."
        self.printer.print_var("mesh_dimension",self.mesh_dimension)

        self.printer.print_var("mesh_n_vertices",self.mesh.num_vertices())
        self.printer.print_var("mesh_n_cells",self.mesh.num_cells())

        self.dV = dolfin.Measure(
            "dx",
            domain=self.mesh)
        self.dS = dolfin.Measure(
            "ds",
            domain=self.mesh)
        self.dF = dolfin.Measure(
            "dS",
            domain=self.mesh)

        self.mesh_V0 = dolfin.assemble(dolfin.Constant(1) * self.dV)
        self.printer.print_sci("mesh_V0",self.mesh_V0)
        self.mesh_S0 = dolfin.assemble(dolfin.Constant(1) * self.dS)
        self.printer.print_sci("mesh_S0",self.mesh_S0)
        self.mesh_h0 = self.mesh_V0**(1./self.mesh_dimension)
        self.printer.print_sci("mesh_h0",self.mesh_h0)
        self.mesh_h0 = dolfin.Constant(self.mesh_h0)

        self.printer.dec()



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
        self.DUold = dolfin.Function(
            self.U_fs,
            name="previous displacement increment")
        self.dU = dolfin.Function(
            self.U_fs,
            name="displacement correction")
        self.dU_trial = dolfin.TrialFunction(self.U_fs)
        self.dU_test = dolfin.TestFunction(self.U_fs)

        # for mesh volume computation
        self.I = dolfin.Identity(self.mesh_dimension)
        self.F = self.I + dolfin.grad(self.U)
        self.J = dolfin.det(self.F)



    def reinit(self):

        self.U.vector().zero()
        self.U_norm = 0.
        self.Uold.vector().zero()
        self.Uold_norm = 0.
        self.DUold.vector().zero()

        for energy in self.energies:
            energy.reinit()



    def add_image_energy(self,
            energy):

        if (hasattr(self, "images_n_frames")
        and hasattr(self, "images_ref_frame")):
            assert (energy.image_series.n_frames  == self.images_n_frames)
            assert (energy.ref_frame == self.images_ref_frame)
        else:
            self.images_n_frames = energy.image_series.n_frames
            self.images_ref_frame = energy.ref_frame

        self.energies += [energy]



    def add_regul_energy(self,
            energy):

        if isinstance(energy, dwarp.ContinuousEnergy):
            self.energies.insert(0, energy)
        elif isinstance(energy, dwarp.DiscreteEnergy):
            self.energies.append(energy)
        else:
            assert (0), "Wrong energy type \""+str(type(energy))+"\". Aborting."



    def call_before_solve(self,
            *kargs,
            **kwargs):

        for energy in self.energies:
            energy.call_before_solve(
                *kargs,
                **kwargs)



    def call_before_assembly(self,
            *kargs,
            **kwargs):

        for energy in self.energies:
            energy.call_before_assembly(
                *kargs,
                **kwargs)



    def assemble_ener(self):

        ener = 0.
        for energy in self.energies:
            # print (energy.name)
            ener_ = energy.assemble_ener()
            self.printer.print_sci("ener_"+energy.name,ener_)
            ener += ener_
        self.printer.print_sci("ener",ener)
        return ener



    def assemble_res(self,
            res_vec):

        if (res_vec.size() > 0): res_vec.zero()
        for energy in self.energies:
            # print (energy.name)
            energy.assemble_res(
                res_vec=res_vec,
                add_values=True)
        # self.printer.print_var("res_vec",res_vec.array())



    def assemble_jac(self,
            jac_mat):

        if (jac_mat.nnz() > 0): jac_mat.zero()
        for energy in self.energies:
            # print (energy.name)
            # print (jac_mat.nnz())
            # print (jac_mat.array())
            energy.assemble_jac(
                jac_mat=jac_mat,
                add_values=True)
            # print (jac_mat.nnz())
            # print (jac_mat.array())
        # self.printer.print_var("jac_mat",jac_mat.array())



    def call_after_solve(self,
            *kargs,
            **kwargs):

        self.DUold.vector()[:] = self.U.vector()[:] - self.Uold.vector()[:]
        self.Uold.vector()[:] = self.U.vector()[:]
        self.Uold_norm = self.U_norm

        for energy in self.energies:
            energy.call_after_solve(
                *kargs,
                **kwargs)



    def get_qoi_names(self):

        names = ["mesh_V"]

        for energy in self.energies:
            names += energy.get_qoi_names()

        return names



    def get_qoi_values(self):

        self.compute_mesh_volume()
        values = [self.mesh_V]

        for energy in self.energies:
            values += energy.get_qoi_values()

        return values



    def compute_mesh_volume(self):

        self.mesh_V = dolfin.assemble(self.J * self.dV)
        self.printer.print_sci("mesh_V",self.mesh_V)
        return self.mesh_V
