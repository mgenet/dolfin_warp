#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2016-2026                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

import dolfin
import numpy

class BFGSNonlinearSolverMixin():



    def init_bfgs(self,
            parameters={}):

        self.use_bfgs = parameters.get("use_bfgs", False)
        self.bfgs_memory = parameters.get("bfgs_memory", 5)
        self.bfgs_restart_iter = parameters.get("bfgs_restart_iter", 10)
        self.s_history = []
        self.y_history = []
        self.rho_history = []
        self.s_vec_cur = None



    def compute_bfgs_search_direction(self,
            res_vec,
            dU_vec,
            linear_solver):

        # Two-loop recursion for L-BFGS
        if not hasattr(self, "bfgs_q_vec"):
            self.bfgs_q_vec = res_vec.copy()
            self.bfgs_z_vec = res_vec.copy()

        q = self.bfgs_q_vec
        z = self.bfgs_z_vec
        
        q.zero(); q.axpy(1.0, res_vec)
        
        alpha = []
        # Backward loop
        for i in reversed(range(len(self.s_history))):
            s_i = self.s_history[i]
            y_i = self.y_history[i]
            rho_i = self.rho_history[i]
            
            alpha_i = rho_i * s_i.inner(q)
            alpha.append(alpha_i)
            
            q.axpy(-alpha_i, y_i)
            
        alpha.reverse()
        
        # Apply initial inverse Hessian approximation (solve J_0 * z = q)
        linear_solver.solve(z, q)
        
        # Forward loop
        for i in range(len(self.s_history)):
            s_i = self.s_history[i]
            y_i = self.y_history[i]
            rho_i = self.rho_history[i]
            
            beta_i = rho_i * y_i.inner(z)
            z.axpy(alpha[i] - beta_i, s_i)
            
        # The result z is H_k * res_vec
        # The search direction is -z
        dU_vec.zero(); dU_vec.axpy(1.0, z)
        dU_vec *= -1.0



    def update_bfgs_history(self,
            s_vec,
            y_vec):

        inner_ys = y_vec.inner(s_vec)
        if inner_ys > 1e-14:
            rho = 1.0 / inner_ys
            
            if len(self.s_history) >= self.bfgs_memory:
                s_old = self.s_history.pop(0)
                y_old = self.y_history.pop(0)
                self.rho_history.pop(0)
                
                s_old.zero(); s_old.axpy(1.0, s_vec)
                y_old.zero(); y_old.axpy(1.0, y_vec)
                
                self.s_history.append(s_old)
                self.y_history.append(y_old)
                self.rho_history.append(rho)
            else:
                self.s_history.append(s_vec.copy())
                self.y_history.append(y_vec.copy())
                self.rho_history.append(rho)
        else:
            self.printer.print_str("Warning! BFGS update skipped due to non-positive curvature condition.")



    def reset_bfgs_history(self):

        self.s_history = []
        self.y_history = []
        self.rho_history = []
