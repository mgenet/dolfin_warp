#coding=utf8

################################################################################
###                                                                          ###
### Created by Felipe Álvarez Barrientos, 2020                               ###
###                                                                          ###
### Pontificia Universidad Católica de Chile, Santiago, Chile                ###
###                                                                          ###
### And Martin Genet, 2016-2025                                              ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

import numpy as np
import dolfin

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
        self.U_tot    = self.problem.U
        self.J        = self.problem.J

        self.fs_J  = dolfin.FunctionSpace(self.mesh, "DG", 0)

        self.working_folder   = parameters.get("working_folder"  , "."  )
        self.working_basename = parameters.get("working_basename", "sol")

        assert (parameters["x0"] is not None) and (parameters["x_bounds"] is not None)
        self.x0_real       = parameters["x0"]  # x0: Initial guess, not normalized
        self.x_bounds = parameters["x_bounds"] # x_bounds = [range for displacement, range for modal coeffs]

        self.sigma0     = parameters.get("sigma0",     2.     )
        self.cma_bounds = parameters.get("cma_bounds", [0, 10])
        self.ftarget    = parameters.get("ftarget",    1e-6   )
        self.tolfun     = parameters.get("tolfun",     1e-11  )
        self.popsize    = parameters.get("popsize",    None   )

        self.motion_model = parameters.get("motion_model", "rbm")

        if (self.motion_model == "rbm+eigenmodes"):
            assert ("modal_params" in parameters),\
                "Specify parameters for modal analysis. Aborting."
            modal_parameters = parameters["modal_params"]

            self.n_modes     = len(self.x0_real["modal_factors"])

            assert ("modes_fixed_points" in modal_parameters) and ("material_params" in modal_parameters),\
                "Specify boundary conditions and material parameters for modal analysis. Aborting."
            self.modes_fix_points = modal_parameters["modes_fixed_points"]
            self.modes_mat_par    = modal_parameters["material_params"]

            self.norm_modes   = modal_parameters.get("norm_modes", 1.  )
            self.save_modes   = modal_parameters.get("save_modes", True)
            self.folder_modes = self.working_folder+"/"+"mesh_modes"+"/"

            ModalAnalysis_mesh = dwarp.ModalAnalysis(
                problem=self.problem,
                n_mod=self.n_modes,
                norm_mod=self.norm_modes)
            self.eigen_modes = ModalAnalysis_mesh.find_modes(
                fixed_points=self.modes_fix_points,
                mat_params=self.modes_mat_par)

            if self.save_modes:
                ModalAnalysis_mesh.save_modes(self.folder_modes)

        
        # x0: initial guess for cma.fmin
        if "full" in self.motion_model:
            assert (len(self.x0_real) == 2), "2 values for initial guess required. Aborting."
            self.n_dofs = len(self.U_tot.vector()[:])
            self.dofs   = np.zeros(self.n_dofs)
            self.x0     = np.zeros(self.n_dofs)

            # FA20200317: By default, in the "full" case the results in a frame are bounded according to the results in the previous frame. 
            # By default, the range used is previous_result +- 0.07
            self.adapt_bounds        = parameters.get("adapt_bounds"       , True)
            self.adapt_bounds_factor = parameters.get("adapt_bounds_factor", 0.07)

            # FA20200221: "full" use the same initial guess, and range, as the one used for displacements.
            if self.adapt_bounds:
                self.x_real     = np.zeros(self.n_dofs)
                self.range_disp = [self.x_bounds["trans"]]*(self.n_dofs)
                for dof in range(int(self.n_dofs/2)):
                    self.x0[2*dof]   = self.real2norm(self.x0_real["trans_x"], self.range_disp[2*dof][0]  , self.range_disp[2*dof  ][1])
                    self.x0[2*dof+1] = self.real2norm(self.x0_real["trans_y"], self.range_disp[2*dof+1][0], self.range_disp[2*dof+1][1])
            else:
                self.range_disp = self.x_bounds["trans"]
                for dof in range(int(self.n_dofs/2)):
                    self.x0[2*dof]   = self.real2norm(self.x0_real["trans_x"], self.range_disp[0], self.range_disp[1])
                    self.x0[2*dof+1] = self.real2norm(self.x0_real["trans_y"], self.range_disp[0], self.range_disp[1])


        if self.motion_model in ["trans", "rbm", "affine", "rbm+eigenmodes"]:
            self.range_disp = self.x_bounds["trans"]

            self.x0    = np.zeros(len(self.x0_real))
            self.x0[0] = self.real2norm(self.x0_real["trans_x"], self.range_disp[0], self.range_disp[1])
            self.x0[1] = self.real2norm(self.x0_real["trans_y"], self.range_disp[0], self.range_disp[1])


        if self.motion_model in ["rbm", "affine", "rbm+eigenmodes"]:
            self.range_theta = self.x_bounds.get("rot", [0., 360.])
            self.x0[2] = self.real2norm(self.x0_real["rot"], self.range_theta[0], self.range_theta[1])


        if (self.motion_model == "trans"):
            assert (len(self.x0_real) == 2), "2 values for initial guess required. Aborting."
            
        elif (self.motion_model == "rbm"):
            assert (len(self.x0_real) == 3), "3 values for initial guess required. Aborting."
        
        elif (self.motion_model == "rbm+eigenmodes"):
            self.range_modal = self.x_bounds["modal_factors"]
            self.x0_modes    = np.zeros(self.n_modes)

            for ind in range(self.n_modes):
                self.x0_modes[ind] = self.real2norm(self.x0_real["modal_factors"][ind], self.range_modal[0], self.range_modal[1])

            self.x0 = np.hstack((self.x0, self.x0_modes))

        elif (self.motion_model == "affine"):
            assert (len(self.x0_real) == 6), "6 values for initial guess required. Aborting."
            self.range_affine_coeffs = self.x_bounds.get("affine_coeffs", [0., 5.])
            self.range_affine_shear  = self.x_bounds.get("affine_shear", [0., 5.])
            
            self.x0[3] = self.real2norm(self.x0_real["affine_comp_x"], self.range_affine_coeffs[0], self.range_affine_coeffs[1])
            self.x0[4] = self.real2norm(self.x0_real["affine_comp_y"], self.range_affine_coeffs[0], self.range_affine_coeffs[1])
            self.x0[5] = self.real2norm(self.x0_real["affine_shear"],  self.range_affine_shear[0],  self.range_affine_shear[1])


        if (self.motion_model == "radial_dependent"):
            self.range_disp  = self.x_bounds["radial"]
            self.range_theta = self.x_bounds.get("theta", [0., 5.])

            self.n_regions = parameters.get("n_regions", 1)
            self.x0   = np.zeros((self.n_regions, 2))

            for region_i in range(self.x0.shape[0]):
                self.x0[region_i, 0] = self.real2norm(self.x0_real["u_radial"], self.range_disp[0], self.range_disp[1])
                self.x0[region_i, 1] = self.real2norm(self.x0_real["u_theta"] , self.range_theta[0], self.range_theta[1])

            self.x0 = self.x0.flatten()

            self.disk_center = parameters["disk_center"]
            self.disk_ri  = parameters["disk_ri"]
            self.disk_re  = parameters["disk_re"]
            self.r_limits = np.linspace(self.disk_ri-1e-2, self.disk_re-1e-2, self.n_regions)



    def solve(self,
            k_frame=None):

        self.k_frame = k_frame

        if (self.k_frame is not None):
            self.printer.print_str("k_frame: "+str(k_frame))

        # Solve with cma
        options_cma = {"bounds"              : [self.cma_bounds[0], self.cma_bounds[1]],
                       "ftarget"             : self.ftarget,
                       "tolfun"              : self.tolfun,
                       "verb_filenameprefix" : self.working_folder+"/outcmaes/",
                       "verb_log"            : 100,
                       "tolflatfitness"      : 20}
        
        if (self.popsize is not None):
            options_cma["popsize"] = self.popsize

        # FA20200218: If the objective_function has extra args, in cma.fmin() add them as a tuple: args=(extra_arg1, extra_arg2, ...)
        res = cma.fmin(
            self.compute_energy,
            self.x0,
            self.sigma0,
            options = options_cma)
        
        coeffs = res[0]

        ##### You may also use differential_evolution #####
        # from scipy import optimize
        # bounds_list = [(self.cma_bounds[0], self.cma_bounds[1])] * len(self.x0)

        # res = optimize.differential_evolution(
        #             func=self.compute_energy,
        #             bounds=bounds_list,
        #             strategy='best1bin',  # best1bin or 'rand1bin' if you want more exploration
        #             maxiter=1000,
        #             popsize=15, # 25
        #             mutation=(0.5, 1.),
        #             recombination=0.9,
        #             atol=1e-6,
        #             disp=True)
        
        # coeffs = res.x
        # print (res)
        ###################################################

        # In case the minimum given by cma.fmin does not correspond to the last evaluation of the objective function
        self.update_U_tot(coeffs)

        self.print_state("CMA obtained values:", coeffs)

        if (self.motion_model == "full") and (self.adapt_bounds):
            for dof in range(self.n_dofs):
                self.x_real[dof] = self.norm2real(coeffs[dof], self.range_disp[dof][0], self.range_disp[dof][1])
            for dof in range(self.n_dofs):
                self.range_disp[dof] = [self.x_real[dof]-self.adapt_bounds_factor, self.x_real[dof]+self.adapt_bounds_factor]
        else:
            self.x0 = coeffs

        success = True # FA20200218: Success always true when using cma.fmin
        
        ##### If using differential_evolution #####
        # return success, res.nit
        ###########################################

        return success, res[4]



    def compute_energy(self,
            coeffs):
        """
        u = t + r + coef_n mode_n
        t: translation
        r: rotation
        """

        self.update_U_tot(coeffs)

        # J = dolfin.det(dolfin.Identity(2) + dolfin.grad(self.U_tot))
        J_p = dolfin.project(self.J, self.fs_J) # FA20200218: TODO: use localproject()

        if (min(J_p.vector()[:]) < 0.):
            return np.NaN

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
                if (self.adapt_bounds):
                    self.dofs[dof] = self.norm2real(coeffs[dof], self.range_disp[dof][0], self.range_disp[dof][1])
                else:
                    self.dofs[dof] = self.norm2real(coeffs[dof], self.range_disp[0], self.range_disp[1])

            self.U_tot.vector()[:] = self.dofs

        elif (self.motion_model == "affine"):
            disp_x      = self.norm2real(coeffs[0], self.range_disp[0], self.range_disp[1])
            disp_y      = self.norm2real(coeffs[1], self.range_disp[0], self.range_disp[1])
            disp_rot    = self.norm2real(coeffs[2], self.range_theta[0], self.range_theta[1])

            affine_comp_x = self.norm2real(coeffs[3], self.range_affine_coeffs[0], self.range_affine_coeffs[1])
            affine_comp_y = self.norm2real(coeffs[4], self.range_affine_coeffs[0], self.range_affine_coeffs[1])
            affine_shear  = self.norm2real(coeffs[5], self.range_affine_shear[0], self.range_affine_shear[1])

            U_affine      = self.U_aff(disp=[disp_x, disp_y, disp_rot, affine_comp_x, affine_comp_y, affine_shear])

            self.U_tot.vector()[:] = U_affine.vector()[:]

        elif (self.motion_model == "radial_dependent"):
            coeffs_array = np.asarray(coeffs).reshape(-1,2)
            disp_array   = np.zeros((self.n_regions, 2))
            disp_array[:,0] = self.norm2real(coeffs_array[:,0], self.range_disp[0], self.range_disp[1])
            disp_array[:,1] = self.norm2real(coeffs_array[:,1], self.range_theta[0], self.range_theta[1])

            self.U_radial_dependent(disp=disp_array)

        else:
            disp_x = self.norm2real(coeffs[0], self.range_disp[0], self.range_disp[1])
            disp_y = self.norm2real(coeffs[1], self.range_disp[0], self.range_disp[1])

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
                ("UX + (x[0]-Cx_THETA)*(cos(THETA)-1) - (x[1]-Cy_THETA)* sin(THETA)   ",
                 "UY + (x[0]-Cx_THETA)* sin(THETA)    + (x[1]-Cy_THETA)*(cos(THETA)-1)"),
            UX=disp_x,
            UY=disp_y,
            THETA=disp_rot*np.pi/180,
            Cx_THETA=center_rot[0],
            Cy_THETA=center_rot[1],
            element=self.problem.U_fe)

        # In this case interpolate is the same as project, because the space of rigid body motions
        # is a subspace of the function space we are using for the displacement (problem.U_fs)
        U_rbm = dolfin.interpolate(
            v=U_rbm_expr,
            V=self.problem.U_fs)

        return U_rbm



    def U_aff(self,
            disp,
            center_rot=[0.5, 0.5]):

        disp_x        = dolfin.Constant(disp[0])
        disp_y        = dolfin.Constant(disp[1])
        disp_rot      = dolfin.Constant(disp[2]*np.pi/180)
        affine_comp_x = dolfin.Constant(disp[3])
        affine_comp_y = dolfin.Constant(disp[4])
        affine_shear  = dolfin.Constant(disp[5])

        T = dolfin.as_vector([disp_x, disp_y])
        R = dolfin.as_matrix([[+dolfin.cos(disp_rot), -dolfin.sin(disp_rot)],
                              [+dolfin.sin(disp_rot), +dolfin.cos(disp_rot)]])
        U = dolfin.as_matrix([[affine_comp_x, affine_shear ],
                              [affine_shear , affine_comp_y]])
        F = dolfin.dot(R, U)

        X  = dolfin.SpatialCoordinate(self.mesh)
        X0 = dolfin.as_vector(center_rot)

        U_aff_expr = T + dolfin.dot(F - dolfin.Identity(2), X-X0)

        U_aff = dolfin.project(
            v=U_aff_expr,
            V=self.problem.U_fs)

        return U_aff
    


    def U_radial_dependent(self,
            disp):
        
        disp_vector = np.asarray(disp).reshape(-1,2)

        space      = self.U_tot.function_space()
        dof_coords = space.tabulate_dof_coordinates()

        dof_i = 0
        while dof_i < len(self.U_tot.vector()[:]):
            x, y = dof_coords[dof_i]
            x = x - self.disk_center[0]
            y = y - self.disk_center[1]

            r     = np.sqrt(x**2 + y**2)
            theta = np.arctan2(y, x)

            # Constant per region
            # i = np.searchsorted(self.r_limits, r, side='right') - 1
            # assert i >= 0, "r value out of bounds"

            # u_r     = disp_vector[i][0]
            # u_theta = disp_vector[i][1]

            # Linear interpolation
            u_r     = np.interp(r, self.r_limits, disp_vector[:,0])
            u_theta = np.interp(r, self.r_limits, disp_vector[:,1])

            u_x     = u_r * np.cos(theta) - u_theta * np.sin(theta)
            u_y     = u_r * np.sin(theta) + u_theta * np.cos(theta)

            self.U_tot.vector()[dof_i]   = u_x
            self.U_tot.vector()[dof_i+1] = u_y

            dof_i += 2



    def print_state(self,
            title,
            coeffs):

        self.printer.inc()
        self.printer.print_str(title)
        if (self.motion_model == "full"):
            self.printer.print_str("Values of displacement in nodes:")
            for dof in range(int(self.n_dofs/2)):

                if (self.adapt_bounds):
                    dof_x = self.norm2real(coeffs[2*dof  ], self.range_disp[2*dof  ][0], self.range_disp[2*dof  ][1])
                    dof_y = self.norm2real(coeffs[2*dof+1], self.range_disp[2*dof+1][0], self.range_disp[2*dof+1][1])
                else:
                    dof_x = self.norm2real(coeffs[2*dof  ], self.range_disp[0], self.range_disp[1])
                    dof_y = self.norm2real(coeffs[2*dof+1], self.range_disp[0], self.range_disp[1])

                self.printer.print_str("node "+str(dof)+": "+" "*bool(dof_x>=0)+str(round(dof_x,3))+" "*(5-len(str(round(dof_x%1,3))))+"   "+" "*bool(dof_y>=0)+str(round(dof_y,3)))
        
        elif (self.motion_model == "radial_dependent"):
            self.printer.print_str("Values of polar displacement:")

            coeffs_array = np.asarray(coeffs).reshape(-1,2)
            disp_array   = np.zeros((self.n_regions, 2))
            disp_array[:,0] = self.norm2real(coeffs_array[:,0], self.range_disp[0], self.range_disp[1])
            disp_array[:,1] = self.norm2real(coeffs_array[:,1], self.range_theta[0], self.range_theta[1])
            
            for region in range(disp_array.shape[0]):
                dof_x = disp_array[region,0]
                dof_y = disp_array[region,1]

                self.printer.print_str("region "+str(region)+": "+" "*bool(dof_x>=0)+str(round(dof_x,3))+" "*(5-len(str(round(dof_x%1,3))))+"   "+" "*bool(dof_y>=0)+str(round(dof_y,3)))

        else:
            self.printer.print_str("Values of displacement:")
            self.printer.print_str("u_x = "+str(round(self.norm2real(coeffs[0], self.range_disp[0], self.range_disp[1]),5)))
            self.printer.print_str("u_y = "+str(round(self.norm2real(coeffs[1], self.range_disp[0], self.range_disp[1]),5)))

            if (self.motion_model == "rbm") or (self.motion_model == "rbm+eigenmodes") or (self.motion_model == "affine"):
                self.printer.print_str("theta = "+str(round(self.norm2real(coeffs[2], self.range_theta[0], self.range_theta[1]),5)) + "°")

            if (self.motion_model == "affine"):
                self.printer.print_str("affine_comp_x = "+str(round(self.norm2real(coeffs[3], self.range_affine_coeffs[0], self.range_affine_coeffs[1]),5)))
                self.printer.print_str("affine_comp_y = "+str(round(self.norm2real(coeffs[4], self.range_affine_coeffs[0], self.range_affine_coeffs[1]),5)))
                self.printer.print_str("affine_shear  = "+str(round(self.norm2real(coeffs[5], self.range_affine_shear[0], self.range_affine_shear[1]),5)))

            if (self.motion_model == "rbm+eigenmodes"):
                print("Values of modal coefficients:")
                for mod_n in range(self.n_modes):
                    self.printer.print_str("coef_"+str(mod_n)+"  = "+str(round(self.norm2real(coeffs[3+mod_n], self.range_modal[0], self.range_modal[1]),5)))
        self.printer.dec()



    def norm2real(self, norm_val, real_min, real_max, norm_min=None, norm_max=None):
        norm_min = self.cma_bounds[0] if norm_min is None else norm_min
        norm_max = self.cma_bounds[1] if norm_max is None else norm_max
        return np.interp(norm_val, [norm_min, norm_max], [real_min, real_max])

    def real2norm(self, real_val, real_min, real_max, norm_min=None, norm_max=None):
        norm_min = self.cma_bounds[0] if norm_min is None else norm_min
        norm_max = self.cma_bounds[1] if norm_max is None else norm_max
        return np.interp(real_val, [real_min, real_max], [norm_min, norm_max])