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

# from builtins import range

import dolfin
import numpy

try:
    import cma
    has_cma = True
except ImportError:
    has_cma = False

import dolfin_warp as dwarp
from .NonlinearSolver import NonlinearSolver

################################################################################

class CMANonlinearSolver(NonlinearSolver):



    def __init__(self,
            problem,
            parameters={}):

        assert (has_cma),\
            "CMA is needed to use this solver. Aborting."

        self.problem  = problem
        self.mesh     = self.problem.mesh
        self.printer  = self.problem.printer
        self.U_degree = self.problem.U_degree
        self.U_tot = self.problem.U
        self.J     = self.problem.J

        self.fs_J  = dolfin.FunctionSpace(self.mesh, "DG", 0)

        self.working_folder   = parameters["working_folder"]
        self.working_basename = parameters["working_basename"]

        # cma.fmin parameters
        assert (parameters["x_0"] is not None) and (parameters["x_0_range"] is not None)

        self.x_0_real       = parameters["x_0"]       # x_0: Initial guess, not normalized
        self.x_0_real_range = parameters["x_0_range"] # x_0_range = [range for displacement, range for modal coeffs]

        self.sigma0    = parameters["sigma0"]  if ("sigma0" in parameters)  else 2.
        self.bounds    = parameters["bounds"]  if ("bounds" in parameters)  else [0,10]
        self.ftarget   = parameters["ftarget"] if ("ftarget" in parameters) else 1E-6
        self.tolfun    = parameters["tolfun"]  if ("tolfun"  in parameters) else 1E-11

        self.motion_model = parameters["motion_model"] if ("motion_model" in parameters) else "rbm"

        if (self.motion_model == "rbm+eigenmodes"):
            assert ("modal_params" in parameters), "Specify parameters for modal analysis"
            modal_parameters = parameters["modal_params"]

            self.n_modes     = len(self.x_0_real["modal_factors"])

            assert ("modes_fixed_points" in modal_parameters) and ("material_params" in modal_parameters), "Specify boundary conditions and material parameters for modal analysis"
            self.modes_fix_points = modal_parameters["modes_fixed_points"]
            self.modes_mat_par    = modal_parameters["material_params"]

            self.norm_modes       = modal_parameters["norm_modes"] if ("norm_modes" in modal_parameters) else 1.
            self.save_modes       = modal_parameters["save_modes"] if ("save_modes" in modal_parameters) else True
            self.folder_modes     = self.working_folder+"/mesh_modes/"

            Modal_analysis_mesh = dwarp.Modal_analysis(problem=self.problem, n_mod=self.n_modes, norm_mod=self.norm_modes)
            self.eigen_modes = Modal_analysis_mesh.find_modes(fixed_points=self.modes_fix_points, mat_params=self.modes_mat_par)

            if self.save_modes:
                Modal_analysis_mesh.save_modes(self.folder_modes)


        # initial guess for cma.fmin
        if (self.motion_model == "full") or (self.motion_model == "trans") or (self.motion_model == "rbm") or (self.motion_model == "rbm+eigenmodes"):
            # FA20200317: By default, in the "full" case the results in a frame are bounded according to the results in the previous frame
            #             And by default, the range used is previous_result +- 0.07
            self.restrict_x0_range       = parameters["restrict_x0_range"] if ("restrict_x0_range" in parameters) else True
            self.restrict_x0_range_bound = parameters["restrict_x0_range_bound"] if ("restrict_x0_range_bound" in parameters) else 0.07

            if (self.motion_model == "full" and self.restrict_x0_range == False) or self.motion_model != "full":
                # FA20200221: "full" use the same initial guess, and range, as the one used for displacements
                self.range_disp  = self.x_0_real_range["trans"]

        if (self.motion_model == "full"):
            assert (len(self.x_0_real) == 2), "2 values for initial guess required"
            self.n_dofs = len(self.U_tot.vector()[:])
            self.dofs = numpy.zeros(self.n_dofs)
            self.x_0  = numpy.zeros(self.n_dofs)

            if self.restrict_x0_range:
                self.x_real     = numpy.zeros(self.n_dofs)
                self.range_disp = [self.x_0_real_range["trans"]]*(self.n_dofs)
                for dof in range(int(self.n_dofs/2)):
                    self.x_0[2*dof]   = self.real2norm(self.x_0_real["trans_x"], self.range_disp[2*dof][0], self.range_disp[2*dof][1])
                    self.x_0[2*dof+1] = self.real2norm(self.x_0_real["trans_y"], self.range_disp[2*dof+1][0], self.range_disp[2*dof+1][1])

            else:
                for dof in range(int(self.n_dofs/2)):
                    self.x_0[2*dof]   = self.real2norm(self.x_0_real["trans_x"], self.range_disp[0], self.range_disp[1])
                    self.x_0[2*dof+1] = self.real2norm(self.x_0_real["trans_y"], self.range_disp[0], self.range_disp[1])

        elif (self.motion_model == "trans"):
            assert (len(self.x_0_real) == 2), "2 values for initial guess required"
            self.x_0 = numpy.zeros(len(self.x_0_real))
            self.x_0[0] = self.real2norm(self.x_0_real["trans_x"], self.range_disp[0], self.range_disp[1])
            self.x_0[1] = self.real2norm(self.x_0_real["trans_y"], self.range_disp[0], self.range_disp[1])

        elif (self.motion_model == "rbm"):
            assert (len(self.x_0_real) == 3), "3 values for initial guess required"
            self.x_0 = numpy.zeros(len(self.x_0_real))
            self.range_theta = self.x_0_real_range["rot"] if ("rot" in self.x_0_real_range) else [0., 360.]

            self.x_0[0] = self.real2norm(self.x_0_real["trans_x"], self.range_disp[0],  self.range_disp[1])
            self.x_0[1] = self.real2norm(self.x_0_real["trans_y"], self.range_disp[0],  self.range_disp[1])
            self.x_0[2] = self.real2norm(self.x_0_real["rot"], self.range_theta[0], self.range_theta[1])

        elif (self.motion_model == "rbm+eigenmodes"):
            self.range_modal = self.x_0_real_range["modal_factors"]
            self.range_theta = self.x_0_real_range["rot"] if ("rot" in self.x_0_real_range) else [0., 360.]

            self.x_0 = numpy.zeros(3+self.n_modes)
            self.x_0[0] = self.real2norm(self.x_0_real["trans_x"], self.range_disp[0], self.range_disp[1])
            self.x_0[1] = self.real2norm(self.x_0_real["trans_y"], self.range_disp[0], self.range_disp[1])
            self.x_0[2] = self.real2norm(self.x_0_real["rot"], self.range_theta[0], self.range_theta[1])

            for ind in range(self.n_modes):
                self.x_0[3+ind] = self.real2norm(self.x_0_real["modal_factors"][ind], self.range_modal[0], self.range_modal[1])



    def solve(self,
            k_frame=None):
        # print k_frame
        self.k_frame = k_frame

        if self.k_frame is not None:
            self.printer.print_str("k_frame: "+str(k_frame))

        # solve with cma.fmin
        # FA20200218: If objective_function has extra args, in cma.fmin() add them as a tuple: args=(extra_arg1, extra_arg2, ...)
        objective_function = self.compute_corr_energy
        res = cma.fmin(objective_function, self.x_0, self.sigma0, options={'bounds': self.bounds, 'ftarget': self.ftarget, 'tolfun':self.tolfun, 'verb_filenameprefix': self.working_folder + "/outcmaes/"})

        coeffs = res[0]
        self.print_state("CMA obtained values:",coeffs)

        if (self.motion_model == "full") and (self.restrict_x0_range):
            for dof in range(self.n_dofs):
                self.x_real[dof] = self.norm2real(coeffs[dof], self.range_disp[dof][0], self.range_disp[dof][1])

            for dof in range(int(self.n_dofs)):
                self.range_disp[dof] = [self.x_real[dof]-self.restrict_x0_range_bound, self.x_real[dof]+self.restrict_x0_range_bound]

        else:
            self.x_0 = coeffs

        success = True # FA20200218: CHECK: success always true?
        return success, res[4]



    def compute_corr_energy(self,
            coeffs):
        """
        u = t + r + coef_n mode_n
        t: translation
        r: rotation
        """

        self.update_U_tot(coeffs)

        # J = dolfin.det(dolfin.Identity(2) + dolfin.grad(self.U_tot))
        J_p = dolfin.project(self.J, self.fs_J) # FA20200218: TODO: use localproject()
        if min(J_p.vector()[:])<0:
            return numpy.NaN

        # FA20200219: in GeneratedImageEnergy.call_before_assembly():
        #                   Igen.update_disp() and Igen.generate_image()
        #                   It is not computing DIgen
        self.problem.call_before_assembly()

        # FA20200219: CHECK: ener_form includes Phi_def and Phi_ref, is this ok?
        ener = self.problem.assemble_ener()

        return ener



    def update_U_tot(self,
            coeffs):
        """
        INPUT:
            coeffs: normalized coeffs of displacement U_tot
        """

        if (self.motion_model == "full"):
            for dof in range(self.n_dofs):

                if (self.restrict_x0_range):
                    self.dofs[dof] = self.norm2real(coeffs[dof], self.range_disp[dof][0], self.range_disp[dof][1])
                else:
                    self.dofs[dof] = self.norm2real(coeffs[dof], self.range_disp[0], self.range_disp[1])

            self.U_tot.vector()[:] = self.dofs

        else:
            disp_x   = self.norm2real(coeffs[0], self.range_disp[0], self.range_disp[1])
            disp_y   = self.norm2real(coeffs[1], self.range_disp[0], self.range_disp[1])

            if (self.motion_model == "trans"):
                disp_rot = 0.

            elif (self.motion_model == "rbm") or (self.motion_model == "rbm+eigenmodes"):
                disp_rot = self.norm2real(coeffs[2], self.range_theta[0], self.range_theta[1])

            U_rbm = self.U_rbm(disp=[disp_x, disp_y, disp_rot])
            # print(U_rbm.vector()[:])

            self.U_tot.vector()[:] = U_rbm.vector()[:]

            if (self.motion_model == "rbm+eigenmodes"):
                modal_coeffs = coeffs[3:]

                for mod_n in range(self.n_modes):
                    modal_coef = self.norm2real(modal_coeffs[mod_n], self.range_modal[0], self.range_modal[1])
                    self.U_tot.vector()[:] += modal_coef*self.eigen_modes[mod_n].vector()[:]



    def U_rbm(self,
            disp,
            center_rot=[0.5, 0.5]):
        disp_x   = disp[0]
        disp_y   = disp[1]
        disp_rot = disp[2]

        U_rbm_expr = dolfin.Expression(
            ("UX + (x[0]-Cx_THETA)*(cos(THETA)-1) - (x[1]-Cy_THETA)*sin(THETA)", "UY + (x[0]-Cx_THETA)*sin(THETA) + (x[1]-Cy_THETA)*(cos(THETA)-1)"),
            UX=disp_x,
            UY=disp_y,
            THETA=disp_rot*numpy.pi/180,
            Cx_THETA=center_rot[0],
            Cy_THETA=center_rot[1],
            element=self.problem.U_fe)

        # In this case interpolate is the same as project, because the space of rigid body motions
        # is a subspace of the function space we are using for the displacement (problem.U_fs)
        U_rbm = dolfin.interpolate(
            v=U_rbm_expr,
            V=self.problem.U_fs)

        return U_rbm



    def print_state(self,
            title,
            coeffs):

        self.printer.inc()
        self.printer.print_str(title)
        if (self.motion_model == "full"):
            self.printer.print_str("Values of displacement in nodes:")
            for dof in range(int(self.n_dofs/2)):

                if (self.restrict_x0_range):
                    dof_x = self.norm2real(coeffs[2*dof],   self.range_disp[2*dof][0], self.range_disp[2*dof][1])
                    dof_y = self.norm2real(coeffs[2*dof+1], self.range_disp[2*dof+1][0], self.range_disp[2*dof+1][1])
                else:
                    dof_x = self.norm2real(coeffs[2*dof],   self.range_disp[0], self.range_disp[1])
                    dof_y = self.norm2real(coeffs[2*dof+1], self.range_disp[0], self.range_disp[1])

                self.printer.print_str("node "+str(dof)+": "+" "*bool(dof_x>=0)+str(round(dof_x,3))+" "*(5-len(str(round(dof_x%1,3))))+"   "+" "*bool(dof_y>=0)+str(round(dof_y,3)))

        else:
            self.printer.print_str("Values of displacement:")
            self.printer.print_str("u_x   = "+str(round(self.norm2real(coeffs[0], self.range_disp[0], self.range_disp[1]),5)))
            self.printer.print_str("u_y   = "+str(round(self.norm2real(coeffs[1], self.range_disp[0], self.range_disp[1]),5)))

            if (self.motion_model == "rbm") or (self.motion_model == "rbm+eigenmodes"):
                self.printer.print_str("theta = "+str(round(self.norm2real(coeffs[2], self.range_theta[0], self.range_theta[1]),5)) + "°")

            if (self.motion_model == "rbm+eigenmodes"):
                print("Values of modal coefficients:")
                for mod_n in range(self.n_modes):
                    self.printer.print_str("coef_"+str(mod_n)+"  = "+str(round(self.norm2real(coeffs[3+mod_n], self.range_modal[0], self.range_modal[1]),5)))
        self.printer.dec()



    def norm2real(self, norm_val, real_min, real_max, norm_max=10):
        return real_min+((real_max-real_min)*norm_val/norm_max)



    def real2norm(self, real_val, real_min, real_max, norm_max=10):
        return ((real_val-real_min)/(real_max-real_min))*norm_max
