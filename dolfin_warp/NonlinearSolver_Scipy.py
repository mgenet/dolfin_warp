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

def sgd( # from https://gist.github.com/jcmgray/e0ab3458a252114beecb1f4b631e19ab
        fun                   ,
        x0                    ,
        jac                   ,
        args          = ()    ,
        learning_rate = 0.001 ,
        mass          = 0.9   ,
        startiter     = 0     ,
        maxiter       = 1000  ,
        callback      = None  ,
        gtol          = None  ,
        ftol          = None  ,
        fatol         = None  ,
        xtol          = None  ,
        xatol         = None  ,
        **kwargs              ):

    x = x0
    velocity = numpy.zeros_like(x)
    f_prev = None
    for i in range(startiter, startiter + maxiter):
        g = jac(x)
        f_curr = fun(x)

        if (callback is not None) and callback(x):
            break
            
        if (gtol is not None) and (numpy.linalg.norm(g, numpy.inf) <= gtol):
            break
            
        if (ftol is not None) and (f_prev is not None) and (abs(f_curr - f_prev) <= ftol * max(1.0, abs(f_curr))):
            break
            
        if (fatol is not None) and (f_prev is not None) and (abs(f_curr - f_prev) <= fatol):
            break

        f_prev = f_curr
        x_prev = x.copy()

        step_dir = mass * velocity - (1.0 - mass) * g
        
        lr = learning_rate
        lr_failed = False
        while True:
            x_new = x + lr * step_dir
            f_new = fun(x_new)
            if numpy.isnan(f_new) or f_new > f_curr:
                lr *= 0.5
                if lr < 1e-12: 
                    lr_failed = True
                    break
            else:
                break
                
        if lr_failed:
            break
                
        if lr == learning_rate:
            learning_rate *= 2.0
        else:
            learning_rate = lr
            
        x = x_new
        velocity = step_dir
        
        if (xtol is not None) and (numpy.linalg.norm(x - x_prev, numpy.inf) <= xtol * max(1.0, numpy.linalg.norm(x, numpy.inf))):
            break
            
        if (xatol is not None) and (numpy.linalg.norm(x - x_prev, numpy.inf) <= xatol):
            break
    else:
        i = startiter + maxiter - 1
        g = jac(x)
        f_curr = fun(x)
    i += 1

    return scipy.optimize.OptimizeResult(x=x, fun=f_curr, jac=g, nit=i, nfev=i, success=True)

def adam( # from https://gist.github.com/jcmgray/e0ab3458a252114beecb1f4b631e19ab
        fun                   ,
        x0                    ,
        jac                   ,
        args          = ()    ,
        learning_rate = 0.001 ,
        beta1         = 0.9   ,
        beta2         = 0.999 ,
        eps           = 1e-8  ,
        startiter     = 0     ,
        maxiter       = 1000  ,
        callback      = None  ,
        gtol          = None  ,
        ftol          = None  ,
        fatol         = None  ,
        xtol          = None  ,
        xatol         = None  ,
        **kwargs              ):

    x = x0
    m = numpy.zeros_like(x)
    v = numpy.zeros_like(x)
    f_prev = None
    for i in range(startiter, startiter + maxiter):
        g = jac(x)
        f_curr = fun(x)

        if (callback is not None) and callback(x):
            break
            
        if (gtol is not None) and (numpy.linalg.norm(g, numpy.inf) <= gtol):
            break
            
        if (ftol is not None) and (f_prev is not None) and (abs(f_curr - f_prev) <= ftol * max(1.0, abs(f_curr))):
            break
            
        if (fatol is not None) and (f_prev is not None) and (abs(f_curr - f_prev) <= fatol):
            break

        f_prev = f_curr
        x_prev = x.copy()

        m_new = (1 - beta1) * g + beta1 * m  # first  moment estimate.
        v_new = (1 - beta2) * (g**2) + beta2 * v  # second moment estimate.
        mhat = m_new / (1 - beta1**(i + 1 - startiter))  # bias correction.
        vhat = v_new / (1 - beta2**(i + 1 - startiter))
        step_dir = - mhat / (numpy.sqrt(vhat) + eps)
        
        lr = learning_rate
        lr_failed = False
        while True:
            x_new = x + lr * step_dir
            f_new = fun(x_new)
            if numpy.isnan(f_new) or f_new > f_curr:
                lr *= 0.5
                if lr < 1e-12: 
                    lr_failed = True
                    break
            else:
                break
                
        if lr_failed:
            break
                
        if lr == learning_rate:
            learning_rate *= 1.1
        else:
            learning_rate = lr
            
        x = x_new
        m = m_new
        v = v_new
        
        if (xtol is not None) and (numpy.linalg.norm(x - x_prev, numpy.inf) <= xtol * max(1.0, numpy.linalg.norm(x, numpy.inf))):
            break
            
        if (xatol is not None) and (numpy.linalg.norm(x - x_prev, numpy.inf) <= xatol):
            break
    else:
        i = startiter + maxiter - 1
        g = jac(x)
        f_curr = fun(x)
    i += 1

    return scipy.optimize.OptimizeResult(x=x, fun=f_curr, jac=g, nit=i, nfev=i, success=True)

################################################################################

class ScipyNonlinearSolver(NonlinearSolver):



    def __init__(self,
            problem,
            parameters={}):

        self.problem = problem
        self.printer = self.problem.printer
        
        self.working_folder   = parameters["working_folder"]
        self.working_basename = parameters["working_basename"]

        user_options = parameters.get("options", {})
        method = user_options.pop("method", "Nelder-Mead")

        if   (method == "Nelder-Mead"):
            default_options = {"xatol": 1e-6, "fatol": 1e-6, "maxiter": 100}
        elif (method == "CG"):
            default_options = {"gtol": 1e-6, "maxiter": 100, "eps":1e-6}
        elif (method == "BFGS"):
            default_options = {"gtol": 1e-6, "maxiter": 100, "eps":1e-6}
        elif (method == "L-BFGS-B"):
            default_options = {"ftol": 1e-6, "gtol": 1e-6, "maxiter": 100, "eps":1e-6}
        elif (method == "Newton-CG"):
            default_options = {"xtol": 1e-6, "fatol": 1e-6, "maxiter": 100, "eps":1e-6}
        elif (method == "SGD"):
            default_options = {"learning_rate": 0.001, "mass": 0.9, "gtol": 1e-6, "fatol": 1e-6, "xatol": 1e-6, "maxiter": 100, "abs_step":1e-6}
        elif (method == "ADAM"):
            default_options = {"learning_rate": 0.001, "beta1": 0.9, "beta2": 0.999, "eps": 1e-8, "gtol": 1e-6, "fatol": 1e-6, "xatol": 1e-6, "maxiter": 100, "abs_step":1e-6}
        else:
            assert (0), "method ("+str(method)+ ") should be Nelder-Mead, CG, BFGS, L-BFGS-B, Newton-CG, SGD or ADAM. Aborting."

        options = default_options.copy()
        options.update(user_options)

        if   (method == "SGD"):
            custom_method = sgd
        elif (method == "ADAM"):
            custom_method = adam
        else:
            custom_method = method

        self.scipy_kwargs = {
            "method"   : custom_method        ,
            "options"  : options              ,
            "callback" : self._scipy_callback }
            
        bounds = options.pop("bounds", None)
        if (bounds is not None):
            self.scipy_kwargs["bounds"] = bounds

        use_finite_difference    = options.pop("use_finite_difference"   , False    )
        finite_difference_scheme = options.pop("finite_difference_scheme", "3-point")
        use_combined_jac         = options.pop("use_combined_jac"        , False    )
        use_exact_hvp            = options.pop("use_exact_hvp"           , False    )
        
        zero_order_methods   = ["Nelder-Mead"]
        first_order_methods  = ["CG", "BFGS", "L-BFGS-B", "SGD", "ADAM"]
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
                if method in ["SGD", "ADAM"]:
                    abs_step = options.get("abs_step", None)
                    self.scipy_kwargs["jac"] = lambda x: self._jac_numdiff(x, abs_step=abs_step)
                    self.finite_difference_scheme = finite_difference_scheme
                else:
                    self.scipy_kwargs["jac"] = finite_difference_scheme
            elif (use_combined_jac):
                self.scipy_kwargs["jac"] = True
            else:
                self.scipy_kwargs["jac"] = self._jac

            if   (type(self.problem) is dwarp.FullKinematicsWarpingProblem):
                self.res_vec = self.problem.U.vector().copy()
            elif (type(self.problem) is dwarp.ReducedKinematicsWarpingProblem):
                self.res_vec = self.problem.reduced_displacement.vector().copy()

        if (method in second_order_methods):
            if (use_exact_hvp):
                self.scipy_kwargs["hessp"] = self._hessp_exact
            else:
                self.scipy_kwargs["hessp"] = self._hessp_approx

            self.p_func = dolfin.Function(self.problem.U.function_space())

        # State & cache trackers
        self._cached_x   = None
        self._cached_fun = None
        self._cached_jac = None

        # write iterations
        self.write_iterations = parameters["write_iterations"] if ("write_iterations" in parameters) and (parameters["write_iterations"] is not None) else False

        if (self.write_iterations):
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



    def _jac_numdiff(self, x, rel_step=None, abs_step=None):

        self._update_state(x)
            
        if (self._cached_jac is None):
            import scipy.optimize._numdiff
            self._cached_jac = scipy.optimize._numdiff.approx_derivative(
                fun=self._fun,
                x0=x,
                method=self.finite_difference_scheme,
                rel_step=rel_step,
                abs_step=abs_step).flatten()

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
                    break

        self.k_iter += 1
        self.printer.print_var("k_iter",self.k_iter)



    def solve(self,
            k_frame=None):

        # Update run-specific parameters for the callback
        self.k_frame = k_frame
        self.frame_filebasename = self.working_folder+"/"+self.working_basename+"-frame="+str(self.k_frame).zfill(len(str(self.problem.images_n_frames)))

        # Initialize Scipy state
        if   (type(self.problem) is dwarp.FullKinematicsWarpingProblem):
            x0 = self.problem.U.vector().get_local()
        elif (type(self.problem) is dwarp.ReducedKinematicsWarpingProblem):
            x0 = self.problem.reduced_displacement.vector().get_local()
        self._cached_x = numpy.zeros_like(x0) * numpy.nan # Force initial update

        # Run optimizer with pre-baked kwargs
        self.k_iter = 1
        self.printer.print_var("k_iter",self.k_iter,-1)
        res = scipy.optimize.minimize(
            fun = self.scipy_fun,
            x0  = x0            ,
            **self.scipy_kwargs )
        if hasattr(res, "success"): self.printer.print_var("success", res.success)
        if hasattr(res, "message"): self.printer.print_var("message", res.message)
        if hasattr(res, "nit"    ): self.printer.print_var("nit"    , res.nit    )
        if hasattr(res, "nfev"   ): self.printer.print_var("nfev"   , res.nfev   )
        if hasattr(res, "njev"   ): self.printer.print_var("njev"   , res.njev   )
        if hasattr(res, "nhev"   ): self.printer.print_var("nhev"   , res.nhev   )
        if hasattr(res, "x"      ): self.printer.print_var("x"      , res.x      )
        if hasattr(res, "fun"    ): self.printer.print_var("fun"    , res.fun    )

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
