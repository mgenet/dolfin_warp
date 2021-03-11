#coding=utf8

################################################################################
###                                                                          ###
### Created by Felipe Álvarez Barrientos, 2020                               ###
###                                                                          ###
### Pontificia Universidad Católica de Chile, Santiago, Chile                ###
###                                                                          ###
### And Martin Genet, 2016-2020                                              ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

import dolfin
import os
import shutil

class Modal_analysis():


    def __init__(self,
            problem,
            n_mod,
            norm_mod=1):
        """
        INPUT:
            V_fs:     VectorFunctionSpace
            n_mod:    Number of modes selected
            norm_mod: Factor to normalize modes
        """
        self.problem  = problem
        self.V_fs     = self.problem.U_fs
        self.mesh     = self.problem.mesh
        self.n_mod    = n_mod
        self.norm_mod = norm_mod



    def find_modes(self,
            fixed_points,
            mat_params):
        """
        INPUT:
            fixed_points:                Fixed points for boundary conditions
            mat_params=[mu, lmbda, rho]: Material parameters
        """
        # V_fs: Vector Function Space
        u_ = dolfin.TrialFunction(self.V_fs)
        du = dolfin.TestFunction(self.V_fs)

        # dofmap = V_fs.dofmap()
        # dofs   = dofmap.dofs()
        # n_dofs = len(dofs)

        [mu, lmbda, rho] = mat_params

        # Linear elasticity
        def eps(v):
            return dolfin.sym(dolfin.grad(v))

        def sigma(v):
            dim_v = v.geometric_dimension()
            return 2.0*mu*eps(v) + lmbda*dolfin.tr(eps(v))*dolfin.Identity(dim_v)

        # BCs
        bcs = self.set_bc(fixed_points=fixed_points)
        print ("dofs with Dirichlet BCs for modal analysis:")
        for b_cond in bcs:
            print (b_cond.get_boundary_values())

        k_form = dolfin.inner(sigma(du),eps(u_))*dolfin.dx
        l_form = dolfin.Constant(1.)*u_[0]*dolfin.dx
        K = dolfin.PETScMatrix()
        b = dolfin.PETScVector()
        dolfin.assemble_system(k_form, l_form, bcs, A_tensor=K, b_tensor=b)

        m_form = rho*dolfin.dot(du,u_)*dolfin.dx
        M = dolfin.PETScMatrix()
        # dolfin.assemble_system(m_form, l_form, A_tensor=M, b_tensor=b)
        dolfin.assemble(m_form, tensor=M)

        eigensolver = dolfin.SLEPcEigenSolver(K, M)

        eigensolver.parameters['problem_type'] = 'gen_hermitian'
        # eigensolver.parameters["spectrum"] = "smallest real"
        eigensolver.parameters['spectral_transform'] = 'shift-and-invert'
        eigensolver.parameters['spectral_shift'] = 0.

        eigensolver.solve(self.n_mod) # n_dofs

        assert eigensolver.get_number_converged() > 0

        self.eigen_modes = [None]*self.n_mod;
        for ind in range(self.n_mod):
            r, c, rx, cx = eigensolver.get_eigenpair(ind) # n_dofs-1-ind

            eigenmode = dolfin.Function(self.V_fs, name="Eigenvector") # +str(ind)
            eigenmode.vector()[:] = rx
            eigenmode.vector()[:] *= (self.norm_mod/max(abs(eigenmode.vector()[:])))

            self.eigen_modes[ind] = eigenmode

        return self.eigen_modes



    def save_modes(self,
            folder_modes):
        if os.path.exists(folder_modes) and os.path.isdir(folder_modes):
            shutil.rmtree(folder_modes) # Delete the folder, if it exists
        os.makedirs(folder_modes) # Create the (empty) folder

        file_mode = dolfin.File(folder_modes+"modes.pvd")

        for mode_n in range(self.n_mod):
            # Igen.init_ugrid(
            #     mesh_=self.mesh,
            #     U_=self.eigen_modes[mode_n].cpp_object())
            #
            # Igen.update_disp()
            # Igen.generate_image()
            # Igen.write_image(folder_modes+"mode_"+str(mode_n)+".vti")

            file_mode << (self.eigen_modes[mode_n], float(mode_n))



    def set_bc(self, fixed_points):
        """
        INPUT:
            points = [point1, point2]: point 1: Fix displacement in -x and -y directions
                                       point 2: Fix displacement in -y direction
        """
        # BCs
        point1 = fixed_points[0]
        point2 = fixed_points[1]

        class Boundary_p1(dolfin.SubDomain):
            def inside(self, x, on_boundary):
                return dolfin.near(x[0], point1[0]) and dolfin.near(x[1],point1[1])

        class Boundary_p2(dolfin.SubDomain):
            def inside(self, x, on_boundary):
                return dolfin.near(x[0], point2[0]) and dolfin.near(x[1], point2[1])

        bc1 = dolfin.DirichletBC(self.V_fs, dolfin.Constant((0.,0.)), Boundary_p1(), method="pointwise")
        bc2 = dolfin.DirichletBC(self.V_fs.sub(1), dolfin.Constant(0.), Boundary_p2(), method="pointwise")
        bcs = [bc1, bc2]

        return bcs
