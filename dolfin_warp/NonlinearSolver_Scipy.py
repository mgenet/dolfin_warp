#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2016-2026                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

import glob
import numpy
import os
import scipy

import dolfin

import dolfin_mech as dmech
import dolfin_warp as dwarp

from .NonlinearSolver import NonlinearSolver

################################################################################

class ScipyNonlinearSolver(NonlinearSolver):



    def __init__(self,
            problem,
            parameters={}):

        self.problem = problem
        self.printer = self.problem.printer
        
        method = parameters.get("method", "Nelder-Mead")

        if   (method == "Nelder-Mead"):
            default_options = {"disp": False, "maxiter": 100, "xatol": 1e-6, "fatol": 1e-6}
        elif (method == "CG"):
            default_options = {"disp": False, "maxiter": 100, "gtol": 1e-6, "eps":1e-6}
        elif (method == "BFGS"):
            default_options = {"disp": False, "maxiter": 100, "gtol": 1e-6, "eps":1e-6}
        elif (method == "L-BFGS-B"):
            default_options = {"disp": False, "maxiter": 100, "ftol": 1e-6, "gtol": 1e-6, "eps":1e-6}
        elif (method == "Newton-CG"):
            default_options = {"maxiter": 100, "xtol": 1e-6, "fatol": 1e-6, "eps":1e-6}
        options = parameters.get("options", default_options)

        self.scipy_kwargs = {
            "method"   : method               ,
            "options"  : options              ,
            "callback" : self._scipy_callback }

        # Bake the SciPy configuration
        use_finite_difference    = parameters.get("use_finite_difference"   , False    )
        finite_difference_scheme = parameters.get("finite_difference_scheme", "2-point")
        use_combined_jac         = parameters.get("use_combined_jac"        , False    )
        use_exact_hvp            = parameters.get("use_exact_hvp"           , False    )
        
        zero_order_methods   = ["Nelder-Mead"]
        first_order_methods  = ["CG", "BFGS", "L-BFGS-B"]
        second_order_methods = ["Newton-CG"]

        assert (method in zero_order_methods + first_order_methods + second_order_methods),\
            "method ("+str(method)+ ") not implemented. Aborting."

        if (use_finite_difference):
            assert (method in first_order_methods),\
                "Finite difference gradient can only be used with first-order methods. Aborting."
            finite_difference_schemes = ["2-point", "3-point"]
            assert (finite_difference_scheme in finite_difference_schemes),\
                "finite_difference_scheme ("+str(finite_difference_scheme)+ ") not implemented. Aborting."
            assert (not use_combined_jac),\
                "use_combined_jac incompatible with use_finite_difference. Aborting."
            assert (not use_exact_hvp),\
                "use_exact_hvp incompatible with use_finite_difference. Aborting."

        if (use_combined_jac):
            assert (method in first_order_methods + second_order_methods),\
                "use_combined_jac can only be used with first or second-order methods. Aborting."

        if (use_exact_hvp):
            assert (method in second_order_methods),\
                "use_exact_hvp can only be used with second-order methods. Aborting."

        if (method in zero_order_methods):
            self.scipy_fun = self._fun
        else:
            if (use_combined_jac):
                self.scipy_fun = self._fun_and_jac
            else:
                self.scipy_fun = self._fun

        if (method in first_order_methods + second_order_methods):
            if (use_finite_difference):
                self.scipy_kwargs["jac"] = finite_difference_scheme
            elif (use_combined_jac):
                self.scipy_kwargs["jac"] = True
            else:
                self.scipy_kwargs["jac"] = self.jac

            if   (type(self.problem) is dwarp.FullKinematicsWarpingProblem):
                self.res_vec = self.problem.U.vector().copy()
            elif (type(self.problem) is dwarp.ReducedKinematicsWarpingProblem):
                self.res_vec = self.problem.reduced_displacement.vector().copy()

        if (method in second_order_methods):
            if (use_exact_hvp):
                self.scipy_kwargs["hessp"] = self.hessp_exact
            else:
                self.scipy_kwargs["hessp"] = self.hessp_approx

            self.p_func = dolfin.Function(self.problem.U.function_space())
                
        # State & cache trackers
        self._cached_x   = None
        self._cached_fun = None
        self._cached_jac = None

        # write iterations
        self.write_iterations = parameters["write_iterations"] if ("write_iterations" in parameters) and (parameters["write_iterations"] is not None) else False

        if (self.write_iterations):
            self.working_folder   = parameters["working_folder"]
            self.working_basename = parameters["working_basename"]

            for filename in glob.glob(self.working_folder+"/"+self.working_basename+"-frame=[0-9]*.*"):
                os.remove(filename)



    def _update_state(self, x):

        if (x is self._cached_x):
            return False
            
        if (self._cached_x is not None) and numpy.array_equal(x, self._cached_x):
            self._cached_x = x
            self._cached_jac = None
            return False

        if   (type(self.problem) is dwarp.FullKinematicsWarpingProblem):
            self.problem.U.vector()[:] = x
        elif (type(self.problem) is dwarp.ReducedKinematicsWarpingProblem):
            self.problem.reduced_displacement.vector()[:] = x
            self.problem.update_disp()

        self.problem.call_before_assembly()
        
        self._cached_x   = x
        self._cached_fun = None
        self._cached_jac = None
        return True



    def _fun(self, x):

        self._update_state(x)
        
        if (self._cached_fun is None):
            self._cached_fun = self.problem.assemble_ener()
            
        return self._cached_fun



    def _jac(self, x):

        self._update_state(x)
            
        if (self._cached_jac is None):
            self.problem.assemble_res(res_vec=self.res_vec)
            self._cached_jac = self.res_vec.get_local()

        return self._cached_jac



    def _fun_and_jac(self, x): # MG20260410: Not sure this is anywhere faster

        self._update_state(x)

        if (self._cached_fun is None) or (self._cached_jac is None):
            self._cached_fun = self.problem.assemble_ener()
            self.problem.assemble_res(res_vec=self.res_vec)
            self._cached_jac = self.res_vec.get_local()

        return self._cached_fun, self._cached_jac



    def _hessp_approx(self, x, p):

        self._update_state(x)

        self.p_func.vector()[:] = p
        
        hvp_form = dolfin.action(self.energy.jac_form, self.p_func)
        hvp_vec  = dolfin.assemble(hvp_form)

        return hvp_vec.get_local()



    def _hessp_exact(self, x, p):

        self._update_state(x)
        
        # 1. Ask C++ to compute the HVP fields for vector p
        self.energy.IDIgen.update_hvp(p)
        
        # 2. Switch C++ eval mode to export [0, h_vol, q, q_tilde, 0]
        self.energy.IDIgen.set_evaluation_mode(1)
        
        # 3. Assemble the exact UFL action form (defined in your Energy class)
        self.p_func.vector()[:] = p
        
        hvp_vec = dolfin.assemble(self.energy.exact_hvp_form(self.p_func))
                
        # 4. Switch mode back to normal
        self.energy.IDIgen.set_evaluation_mode(0)
        
        return hvp_vec.get_local()



    def _scipy_callback(self, res):
        
        self.k_iter += 1
        self.printer.print_var("k_iter",self.k_iter)

        if (self.write_iterations):
            x = res.x if hasattr(res, "x") else res # MG20260411: res can be x or a scipy.optimize.OptimizeResult object…
            self._update_state(x)

            dmech.write_VTU_file(
                filebasename=self.frame_filebasename,
                function=self.problem.U,
                time=self.k_iter)

            for energy in self.problem.energies:
                if hasattr(energy, "IDIgen"):
                    energy.IDIgen.write_image("generated", f"{self.frame_filebasename}-iter={str(self.k_iter).zfill(3)}.vti")



    def solve(self,
            k_frame=None):

        # Update run-specific parameters for the callback
        self.k_frame = k_frame
        self.k_iter  = 0
        self.frame_filebasename = self.working_folder+"/"+self.working_basename+"-frame="+str(self.k_frame).zfill(len(str(self.problem.images_n_frames)))

        # Initialize Scipy state
        if   (type(self.problem) is dwarp.FullKinematicsWarpingProblem):
            x0 = self.problem.U.vector().get_local()
        elif (type(self.problem) is dwarp.ReducedKinematicsWarpingProblem):
            x0 = self.problem.reduced_displacement.vector().get_local()
        self._cached_x = numpy.zeros_like(x0) * numpy.nan # Force initial update

        # Run optimizer with pre-baked kwargs
        res = scipy.optimize.minimize(
            fun = self.scipy_fun,
            x0  = x0            ,
            **self.scipy_kwargs )
        self.printer.print_var("success", res.success)
        self.printer.print_var("message", res.message)
        self.printer.print_var("nit"    , res.nit    )
        self.printer.print_var("nfev"   , res.nfev   )
        if hasattr(res, "njev"): self.printer.print_var("njev", res.njev)
        if hasattr(res, "nhev"): self.printer.print_var("nhev", res.nhev)
        self.printer.print_var("x"      , res.x      )
        self.printer.print_var("fun"    , res.fun    )

        if (res.success):
            self.printer.print_str("Nonlinear solver converged…")
        else:
            self.printer.print_str("Warning! Nonlinear solver failed to converge… (k_frame = "+str(self.k_frame)+")")

        # Update FEniCS state
        if   (type(self.problem) is dwarp.FullKinematicsWarpingProblem):
            self.problem.U.vector()[:] = res.x
        elif (type(self.problem) is dwarp.ReducedKinematicsWarpingProblem):
            self.problem.reduced_displacement.vector()[:] = res.x
            self.problem.update_disp()
        
        return res.success, res.nit
