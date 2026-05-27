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
        args           = ()   ,
        mass           = 0.5  ,
        learning_rate  = 1e-3 ,
        lr_contraction = 2.0  ,
        lr_min         = 1e-9 ,
        lr_expansion   = 1.1  ,
        lr_max         = None ,
        max_disp_inc   = None ,
        maxiter        = 1000 ,
        callback       = None ,
        xtol           = None ,
        xatol          = None ,
        ftol           = None ,
        fatol          = None ,
        fabstol        = None ,
        gtol           = None ,
        **kwargs              ):

    printer = kwargs.get("printer", None)

    nit  = 0
    nfev = 0
    njev = 0

    x     = x0
    x_old = numpy.zeros_like(x)
    v     = numpy.zeros_like(x)
    f     = fun(x); nfev += 1
    f_old = None
    g     = jac(x); njev += 1

    # printer.print_var("x", x)
    printer.print_sci("f", f)
    # printer.print_sci("g", g)
    
    while (True):
        nit += 1

        if (fabstol is not None) and (f <= fabstol):
            printer.print_str("fabstol reached")
            success = True; break

        # printer.print_sci("numpy.linalg.norm(g) = ", numpy.linalg.norm(g))
        # printer.print_sci("numpy.linalg.norm(g, numpy.inf) = ", numpy.linalg.norm(g, numpy.inf))
        if (gtol is not None) and (numpy.linalg.norm(g, numpy.inf) <= gtol):
            printer.print_str("gtol reached")
            success = True; break

        v[:] = mass * v - (1.0 - mass) * g
        learning_rate_old = learning_rate
        while (True):
            printer.print_sci("learning_rate", learning_rate)
            x_new = x + learning_rate * v
            f_new = fun(x_new); nfev += 1
            # printer.print_var("x_new", x_new)
            printer.print_sci("f_new", f_new)
            if numpy.isnan(f_new) or (f_new >= f) or ((max_disp_inc is not None) and (numpy.linalg.norm(x_new - x, numpy.inf) > max_disp_inc)):
                if numpy.isnan(f_new):
                    printer.print_str("NaNs detected in energy. Halving learning rate.")
                if (f_new >= f):
                    printer.print_str("Energy did not decrease. Halving learning rate.")
                if ((max_disp_inc is not None) and (numpy.linalg.norm(x_new - x, numpy.inf) > max_disp_inc)):
                    printer.print_str("Displacement too large. Halving learning rate.")
                learning_rate /= lr_contraction
                if (learning_rate < lr_min):
                    learning_rate = 0.0
                    break
            else:
                break

        if (learning_rate == 0.0):
            printer.print_str("minimum learning rate reached")
            success = False; break

        if (learning_rate == learning_rate_old):
            learning_rate *= lr_expansion
            if (lr_max is not None) and (learning_rate > lr_max):
                printer.print_str("maximum learning rate reached")
                learning_rate = lr_max

        x_old[:] = x; x = x_new
        f_old = f; f = f_new
        g[:] = jac(x); njev += 1

        # printer.print_sci("numpy.linalg.norm(x - x_old, numpy.inf)/max(1.0, numpy.linalg.norm(x_old, numpy.inf)) = ", numpy.linalg.norm(x - x_old, numpy.inf)/max(1.0, numpy.linalg.norm(x_old, numpy.inf)))
        if (xtol is not None) and (numpy.linalg.norm(x - x_old, numpy.inf)/max(1.0, numpy.linalg.norm(x_old, numpy.inf)) <= xtol):
            printer.print_str("xtol reached")
            success = True; break
            
        # printer.print_sci("numpy.linalg.norm(x - x_old, numpy.inf) = ", numpy.linalg.norm(x - x_old, numpy.inf))
        if (xatol is not None) and (numpy.linalg.norm(x - x_old, numpy.inf) <= xatol):
            printer.print_str("xatol reached")
            success = True; break

        # printer.print_sci("abs(f - f_old)/max(1.0, abs(f_old)) = ", abs(f - f_old)/max(1.0, abs(f_old)))
        if (ftol is not None) and (f_old is not None) and (abs(f - f_old)/max(1.0, abs(f_old)) <= ftol):
            printer.print_str("ftol reached")
            success = True; break
            
        # if (f_old is not None): printer.print_sci("abs(f - f_old) = ", abs(f - f_old))
        if (fatol is not None) and (f_old is not None) and (abs(f - f_old) <= fatol):
            printer.print_str("fatol reached")
            success = True; break

        if (callback is not None):
            callback(x)

        if (nit >= maxiter):
            printer.print_str("maximum number of iterations reached")
            success = False; break

    return scipy.optimize.OptimizeResult(x=x, fun=f, jac=g, nit=nit, nfev=nfev, njev=njev, success=success)

def adam( # from https://gist.github.com/jcmgray/e0ab3458a252114beecb1f4b631e19ab
        fun                    ,
        x0                     ,
        jac                    ,
        args           = ()    ,
        beta1          = 0.9   ,
        beta2          = 0.999 ,
        eps            = 1e-8  ,
        learning_rate  = 1e-3  ,
        lr_contraction = 2.0   ,
        lr_min         = 1e-9  ,
        lr_expansion   = 1.1   ,
        lr_max         = None  ,
        max_disp_inc   = None  ,
        maxiter        = 1000  ,
        callback       = None  ,
        xtol           = None  ,
        xatol          = None  ,
        ftol           = None  ,
        fatol          = None  ,
        fabstol        = None  ,
        gtol           = None  ,
        **kwargs               ):

    printer = kwargs.get("printer", None)

    nit  = 0
    nfev = 0
    njev = 0

    x     = x0
    x_old = numpy.zeros_like(x)
    m     = numpy.zeros_like(x)
    v     = numpy.zeros_like(x)
    f     = fun(x); nfev += 1
    f_old = None
    g     = jac(x); njev += 1

    # printer.print_var("x", x)
    printer.print_sci("f", f)
    # printer.print_sci("g", g)
    
    while (True):
        nit += 1

        if (fabstol is not None) and (f <= fabstol):
            printer.print_str("fabstol reached")
            success = True; break

        # printer.print_sci("numpy.linalg.norm(g) = ", numpy.linalg.norm(g))
        # printer.print_sci("numpy.linalg.norm(g, numpy.inf) = ", numpy.linalg.norm(g, numpy.inf))
        if (gtol is not None) and (numpy.linalg.norm(g, numpy.inf) <= gtol):
            printer.print_str("gtol reached")
            success = True; break

        m[:] = (1 - beta1) *  g     + beta1 * m  # first moment estimate.
        v[:] = (1 - beta2) * (g**2) + beta2 * v  # second moment estimate.
        mhat = m / (1 - beta1**(nit))  # bias correction.
        vhat = v / (1 - beta2**(nit))  # bias correction.
        step_dir = - mhat / (numpy.sqrt(vhat) + eps)
        
        learning_rate_old = learning_rate
        while True:
            printer.print_sci("learning_rate", learning_rate)
            x_new = x + learning_rate * step_dir
            f_new = fun(x_new); nfev += 1
            # printer.print_var("x_new", x_new)
            printer.print_sci("f_new", f_new)
            if numpy.isnan(f_new) or (f_new >= f) or ((max_disp_inc is not None) and (numpy.linalg.norm(x_new - x, numpy.inf) > max_disp_inc)):
                if numpy.isnan(f_new):
                    printer.print_str("NaNs detected in energy. Halving learning rate.")
                if (f_new >= f):
                    printer.print_str("Energy did not decrease. Halving learning rate.")
                if ((max_disp_inc is not None) and (numpy.linalg.norm(x_new - x, numpy.inf) > max_disp_inc)):
                    printer.print_str("Displacement too large. Halving learning rate.")
                learning_rate /= lr_contraction
                if (learning_rate < lr_min):
                    learning_rate = 0.0
                    break
            else:
                break
                
        if (learning_rate == 0.0):
            printer.print_str("minimum learning rate reached")
            success = False; break
                
        if (learning_rate == learning_rate_old):
            learning_rate *= lr_expansion
            if (lr_max is not None) and (learning_rate > lr_max):
                printer.print_str("maximum learning rate reached")
                learning_rate = lr_max
            
        x_old[:] = x; x = x_new
        f_old = f; f = f_new
        g[:] = jac(x); njev += 1
        
        # printer.print_sci("numpy.linalg.norm(x - x_old, numpy.inf) = ", numpy.linalg.norm(x - x_old, numpy.inf))
        # printer.print_sci("xtol * max(1.0, numpy.linalg.norm(x, numpy.inf)) = ", xtol * max(1.0, numpy.linalg.norm(x, numpy.inf)))
        if (xtol is not None) and (numpy.linalg.norm(x - x_old, numpy.inf)/max(1.0, numpy.linalg.norm(x_old, numpy.inf)) <= xtol):
            printer.print_str("xtol reached")
            success = True; break
            
        # printer.print_sci("numpy.linalg.norm(x - x_old, numpy.inf) = ", numpy.linalg.norm(x - x_old, numpy.inf))
        if (xatol is not None) and (numpy.linalg.norm(x - x_old, numpy.inf) <= xatol):
            printer.print_str("xatol reached")
            success = True; break

        # printer.print_sci("abs(f - f_old) = ", abs(f - f_old))
        # printer.print_sci("ftol * max(1.0, abs(f)) = ", ftol * max(1.0, abs(f)))
        if (ftol is not None) and (f_old is not None) and (abs(f - f_old)/max(1.0, abs(f_old)) <= ftol):
            printer.print_str("ftol reached")
            success = True; break
            
        # if (f_old is not None): printer.print_sci("abs(f - f_old) = ", abs(f - f_old))
        if (fatol is not None) and (f_old is not None) and (abs(f - f_old) <= fatol):
            printer.print_str("fatol reached")
            success = True; break

        if (callback is not None):
            callback(x)

        if (nit >= maxiter):
            printer.print_str("maximum number of iterations reached")
            success = False; break

    return scipy.optimize.OptimizeResult(x=x, fun=f, jac=g, nit=nit, nfev=nfev, njev=njev, success=success)

def ncg(
        fun                   ,
        x0                    ,
        jac                   ,
        hessp                 ,
        args           = ()   ,
        cg_tol         = 1e-3 ,
        cg_maxiter     = 20   ,
        learning_rate  = 1e-3 ,
        lr_contraction = 2.0  ,
        lr_min         = 1e-9 ,
        lr_expansion   = 1.1  ,
        lr_max         = None ,
        max_disp_inc   = None ,
        maxiter        = 1000 ,
        callback       = None ,
        gtol           = None ,
        ftol           = None ,
        fatol          = None ,
        fabstol        = None ,
        xtol           = None ,
        xatol          = None ,
        **kwargs              ):

    printer = kwargs.get("printer", None)

    nit  = 0
    nfev = 0
    njev = 0
    nhev = 0

    x     = x0
    x_old = numpy.zeros_like(x)
    f     = fun(x); nfev += 1
    f_old = None
    g     = jac(x); njev += 1

    printer.print_sci("f", f)
    
    while (True):
        nit += 1

        if (fabstol is not None) and (f <= fabstol):
            printer.print_str("fabstol reached")
            success = True; break

        if (gtol is not None) and (numpy.linalg.norm(g, numpy.inf) <= gtol):
            printer.print_str("gtol reached")
            success = True; break

        # Inner CG solve: H * step_dir = -g
        step_dir = numpy.zeros_like(g)
        r = -g.copy()
        p = r.copy()
        rsold = numpy.dot(r, r)
        
        for cg_iter in range(cg_maxiter):
            Ap = hessp(x, p); nhev += 1
            pAp = numpy.dot(p, Ap)
            
            # Check for negative curvature
            if (pAp <= 1e-12):
                if (cg_iter == 0):
                    step_dir = -g.copy() # fallback to steepest descent
                break
                
            alpha = rsold / pAp
            step_dir += alpha * p
            r -= alpha * Ap
            rsnew = numpy.dot(r, r)
            
            if (numpy.sqrt(rsnew) < cg_tol * numpy.linalg.norm(g)):
                break
                
            p = r + (rsnew / rsold) * p
            rsold = rsnew
        
        learning_rate_old = learning_rate
        while (True):
            printer.print_sci("learning_rate", learning_rate)
            x_new = x + learning_rate * step_dir
            f_new = fun(x_new); nfev += 1
            printer.print_sci("f_new", f_new)
            
            if numpy.isnan(f_new) or (f_new >= f) or ((max_disp_inc is not None) and (numpy.linalg.norm(x_new - x, numpy.inf) > max_disp_inc)):
                if numpy.isnan(f_new):
                    printer.print_str("NaNs detected in energy. Halving learning rate.")
                if (f_new >= f):
                    printer.print_str("Energy did not decrease. Halving learning rate.")
                if ((max_disp_inc is not None) and (numpy.linalg.norm(x_new - x, numpy.inf) > max_disp_inc)):
                    printer.print_str("Displacement too large. Halving learning rate.")
                learning_rate /= lr_contraction
                if (learning_rate < lr_min):
                    learning_rate = 0.0
                    break
            else:
                break
                
        if (learning_rate == 0.0):
            printer.print_str("minimum learning rate reached")
            success = False; break
                
        if (learning_rate == learning_rate_old):
            learning_rate *= lr_expansion
            if (lr_max is not None) and (learning_rate > lr_max):
                printer.print_str("maximum learning rate reached")
                learning_rate = lr_max
            
        x_old[:] = x; x = x_new
        f_old = f; f = f_new
        g[:] = jac(x); njev += 1
        
        if (xtol is not None) and (numpy.linalg.norm(x - x_old, numpy.inf)/max(1.0, numpy.linalg.norm(x_old, numpy.inf)) <= xtol):
            printer.print_str("xtol reached")
            success = True; break
            
        if (xatol is not None) and (numpy.linalg.norm(x - x_old, numpy.inf) <= xatol):
            printer.print_str("xatol reached")
            success = True; break

        if (ftol is not None) and (f_old is not None) and (abs(f - f_old)/max(1.0, abs(f_old)) <= ftol):
            printer.print_str("ftol reached")
            success = True; break
            
        if (fatol is not None) and (f_old is not None) and (abs(f - f_old) <= fatol):
            printer.print_str("fatol reached")
            success = True; break

        if (callback is not None):
            callback(x)

        if (nit >= maxiter):
            printer.print_str("maximum number of iterations reached")
            success = False; break

    return scipy.optimize.OptimizeResult(x=x, fun=f, jac=g, nit=nit, nfev=nfev, njev=njev, nhev=nhev, success=success)

def trust_ncg(
        fun                          ,
        x0                           ,
        jac                          ,
        hessp                        ,
        args                 = ()    ,
        trust_radius         = 1e-2  ,
        tr_contraction       = 2.0   ,
        tr_min               = 1e-4  ,
        tr_expansion         = 1.1   ,
        tr_max               = 1e-1  ,
        eta                  = 0.15  ,
        cg_tol               = 1e-3  ,
        cg_maxiter           = 20    ,
        max_disp_inc         = 1e-2  ,
        maxiter              = 100   ,
        callback             = None  ,
        gtol                 = None  ,
        ftol                 = None  ,
        fatol                = None  ,
        fabstol              = None  ,
        xtol                 = None  ,
        xatol                = None  ,
        **kwargs                     ):

    printer = kwargs.get("printer", None)

    nit  = 0
    nfev = 0
    njev = 0
    nhev = 0

    x     = x0
    x_old = numpy.zeros_like(x)
    f     = fun(x); nfev += 1
    f_old = None
    g     = jac(x); njev += 1

    trust_radius_old = trust_radius

    printer.print_sci("f", f)
    
    while (True):
        nit += 1

        if (fabstol is not None) and (f <= fabstol):
            printer.print_str("fabstol reached")
            success = True; break

        if (gtol is not None) and (numpy.linalg.norm(g, numpy.inf) <= gtol):
            printer.print_str("gtol reached")
            success = True; break

        # Inner Steihaug-Toint CG solve: H * step_dir = -g
        step_dir = numpy.zeros_like(g)
        r = g.copy()
        d = -r.copy()
        
        for cg_iter in range(cg_maxiter):
            Bd = hessp(x, d); nhev += 1
            dBd = numpy.dot(d, Bd)
            
            # 1. Negative curvature detection
            if (dBd <= 0):
                a_coef = numpy.dot(d, d)
                b_coef = 2 * numpy.dot(step_dir, d)
                c_coef = numpy.dot(step_dir, step_dir) - trust_radius**2
                tau = (-b_coef + numpy.sqrt(max(b_coef**2 - 4*a_coef*c_coef, 0))) / (2*a_coef)
                step_dir += tau * d
                break
                
            alpha = numpy.dot(r, r) / dBd
            step_next = step_dir + alpha * d
            
            # 2. Trust-region boundary intersection
            if (numpy.linalg.norm(step_next) >= trust_radius):
                a_coef = numpy.dot(d, d)
                b_coef = 2 * numpy.dot(step_dir, d)
                c_coef = numpy.dot(step_dir, step_dir) - trust_radius**2
                tau = (-b_coef + numpy.sqrt(max(b_coef**2 - 4*a_coef*c_coef, 0))) / (2*a_coef)
                step_dir += tau * d
                break
                
            step_dir = step_next
            r_next = r + alpha * Bd
            
            # 3. CG convergence
            if (numpy.linalg.norm(r_next) < cg_tol * numpy.linalg.norm(g)):
                break
                
            beta = numpy.dot(r_next, r_next) / numpy.dot(r, r)
            d = -r_next + beta * d
            r = r_next

        # Evaluate the proposed step
        x_new = x + step_dir
        f_new = fun(x_new); nfev += 1

        printer.print_sci("f_new", f_new)

        # 4. Guardrails: NaNs or user constraints
        if numpy.isnan(f_new) or ((max_disp_inc is not None) and (numpy.linalg.norm(step_dir, numpy.inf) > max_disp_inc)):
            if numpy.isnan(f_new):
                printer.print_str("NaNs detected in energy. Shrinking trust region.")
            if ((max_disp_inc is not None) and (numpy.linalg.norm(step_dir, numpy.inf) > max_disp_inc)):
                printer.print_str("Displacement too large. Shrinking trust region.")
            trust_radius /= tr_contraction
            if (trust_radius < tr_min):
                printer.print_str("Trust radius reached minimum limit.")
                success = False; break
            continue

        # Evaluate predicted reduction
        Hp = hessp(x, step_dir); nhev += 1
        pred_reduction = -(numpy.dot(g, step_dir) + 0.5 * numpy.dot(step_dir, Hp))
        actual_reduction = f - f_new

        # 5. Bad quadratic model safeguard
        if (pred_reduction <= 0):
            printer.print_str("Negative predicted reduction. Shrinking trust region.")
            trust_radius /= tr_contraction
            if (trust_radius < tr_min):
                printer.print_str("Trust radius reached minimum limit.")
                success = False; break
            continue

        ratio = actual_reduction / pred_reduction

        # 6. Trust-region updates
        if   (ratio < 0.25):
            trust_radius /= tr_contraction
        elif (ratio > 0.75) and (numpy.linalg.norm(step_dir) >= 0.99 * trust_radius):
            trust_radius *= tr_expansion
            if (tr_max is not None) and (trust_radius > tr_max):
                printer.print_str("maximum trust radius reached")
                trust_radius = tr_max

        # 7. Step acceptance
        if (ratio > eta):
            x_old[:] = x; x = x_new
            f_old = f; f = f_new
            g[:] = jac(x); njev += 1

            if (xtol is not None) and (numpy.linalg.norm(x - x_old, numpy.inf)/max(1.0, numpy.linalg.norm(x_old, numpy.inf)) <= xtol):
                printer.print_str("xtol reached")
                success = True; break
                
            if (xatol is not None) and (numpy.linalg.norm(x - x_old, numpy.inf) <= xatol):
                printer.print_str("xatol reached")
                success = True; break

            if (ftol is not None) and (f_old is not None) and (abs(f - f_old)/max(1.0, abs(f_old)) <= ftol):
                printer.print_str("ftol reached")
                success = True; break
                
            if (fatol is not None) and (f_old is not None) and (abs(f - f_old) <= fatol):
                printer.print_str("fatol reached")
                success = True; break

            if (callback is not None):
                callback(x)
        else:
            printer.print_str("Step rejected. Trust region was shrunk.")
            if (trust_radius < tr_min):
                success = False; break

        if (nit >= maxiter):
            printer.print_str("maximum number of iterations reached")
            success = False; break

    return scipy.optimize.OptimizeResult(x=x, fun=f, jac=g, nit=nit, nfev=nfev, njev=njev, nhev=nhev, success=success)

################################################################################

class ScipyNonlinearSolver(NonlinearSolver):



    def __init__(self,
            problem,
            parameters={}):

        self.problem = problem
        self.printer = self.problem.printer
        
        self.working_folder   = parameters["working_folder"]
        self.working_basename = parameters["working_basename"]

        options = parameters.get("options", {}).copy()
        method = options.pop("method", "Nelder-Mead")

        if (method.startswith("custom-")):
            options["printer"] = self.printer

        if   (method == "custom-SGD"):
            custom_method = sgd
        elif (method == "custom-ADAM"):
            custom_method = adam
        elif (method == "custom-NCG"):
            custom_method = ncg
        elif (method == "custom-trust-NCG"):
            custom_method = trust_ncg
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
        
        zero_order_methods   = ["Nelder-Mead"]
        first_order_methods  = ["CG", "BFGS", "L-BFGS-B", "custom-SGD", "custom-ADAM"]
        second_order_methods = ["Newton-CG", "trust-NCG", "custom-NCG", "custom-trust-NCG"]

        assert (method in zero_order_methods + first_order_methods + second_order_methods),\
            "method ("+str(method)+ ") not implemented. Aborting."

        if (use_finite_difference):
            assert (method in first_order_methods + second_order_methods),\
                "Finite difference gradient can only be used with first or second-order methods. Aborting."
            finite_difference_schemes = ["2-point", "3-point"]
            assert (finite_difference_scheme in finite_difference_schemes),\
                "finite_difference_scheme ("+str(finite_difference_scheme)+ ") not implemented. Aborting."
            assert (not use_combined_jac),\
                "use_combined_jac incompatible with use_finite_difference. Aborting."

        if (use_combined_jac):
            assert (method in first_order_methods + second_order_methods),\
                "use_combined_jac can only be used with first or second-order methods. Aborting."

        if (method in zero_order_methods):
            self.scipy_fun = self._fun
        else:
            if (use_combined_jac):
                self.scipy_fun = self._fun_and_jac
            else:
                self.scipy_fun = self._fun

        if (method in first_order_methods + second_order_methods):
            if (use_finite_difference):
                finite_difference_step = options.get("finite_difference_step", None)
                self.finite_difference_dynamic_step = options.get("finite_difference_dynamic_step", False)
                self.scipy_kwargs["jac"] = lambda x: self._jac_numdiff(x, abs_step=finite_difference_step)
                self.finite_difference_scheme = finite_difference_scheme
            elif (use_combined_jac):
                self.scipy_kwargs["jac"] = True
            else:
                self.scipy_kwargs["jac"] = self._jac

            if   (type(self.problem) is dwarp.FullKinematicsWarpingProblem):
                self.res_vec = self.problem.U.vector().copy()
            elif (type(self.problem) is dwarp.ReducedKinematicsWarpingProblem):
                self.res_vec = self.problem.reduced_displacement.vector().copy()

        if (method in second_order_methods):
            if (use_finite_difference):
                self.scipy_kwargs["hessp"] = lambda x, p: self._hessp_numdiff(x, p, abs_step=finite_difference_step)
            else:
                self.scipy_kwargs["hessp"] = self._hessp
            self.p_func = dolfin.Function(self.problem.U.function_space())
            self.hvp_vec = self.p_func.vector().copy()

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

        if numpy.any(numpy.isnan(x)):
            return False

        if (x is self._cached_x):
            return False
            
        if (self._cached_x is not None) and numpy.array_equal(x, self._cached_x):
            self._cached_x = x
            return False

        if   (type(self.problem) is dwarp.FullKinematicsWarpingProblem):
            self.problem.U.vector().set_local(x); self.problem.U.vector().apply("insert")
        elif (type(self.problem) is dwarp.ReducedKinematicsWarpingProblem):
            self.problem.reduced_displacement.vector().set_local(x); self.problem.reduced_displacement.vector().apply("insert")
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
            
            if hasattr(self, "finite_difference_dynamic_step") and (self.finite_difference_dynamic_step):
                if hasattr(self, "_jac_old_x") and (self._jac_old_x is not None):
                    if ((type(self.finite_difference_dynamic_step) is not str) and (self.finite_difference_dynamic_step            )) or \
                       ((type(self.finite_difference_dynamic_step) is     str) and (self.finite_difference_dynamic_step == "global"))    :
                        dx = numpy.linalg.norm(x - self._jac_old_x)
                        dynamic_abs_step = max(dx, 1e-6)
                    elif (type(self.finite_difference_dynamic_step) is str) and (self.finite_difference_dynamic_step == "local"):
                        dx = x - self._jac_old_x
                        dynamic_abs_step = numpy.maximum(numpy.abs(dx), 1e-6)
                    else:
                        raise ValueError("Invalid value for finite_difference_dynamic_step. Must be 'global' or 'local'.")
                else:
                    dynamic_abs_step = abs_step if (abs_step is not None) else 1e-6
                self.printer.print_var("dynamic_abs_step", dynamic_abs_step)
            else:
                dynamic_abs_step = abs_step if (abs_step is not None) else 1e-6
                
            while (True):
                self._cached_jac = scipy.optimize._numdiff.approx_derivative(
                    fun=self._fun,
                    x0=x,
                    method=self.finite_difference_scheme,
                    rel_step=rel_step,
                    abs_step=dynamic_abs_step).flatten()

                if numpy.any(numpy.isnan(self._cached_jac)):
                    self.printer.print_str("NaNs detected in FD gradient. Halving step size.")
                    dynamic_abs_step /= 2.0
                    if numpy.all(dynamic_abs_step < 1e-9):
                        raise RuntimeError("Finite difference step size reached minimum limit. Aborting.")
                else:
                    break

            self._jac_old_x = x.copy()

        return self._cached_jac



    def _fun_and_jac(self, x): # MG20260410: Not sure this is anywhere faster

        self._update_state(x)

        if (self._cached_fun is None) or (self._cached_jac is None):
            self._cached_fun = self.problem.assemble_ener()
            self.problem.assemble_res(res_vec=self.res_vec)
            self._cached_jac = self.res_vec.get_local()

        return self._cached_fun, self._cached_jac



    def _hessp(self, x, p):

        # 1. Update the mesh to the current displacement 'x'
        x_was_updated = self._update_state(x)
        
        # 2. Convert SciPy's numpy vector 'p' into a FEniCS Function
        self.p_func.vector().set_local(p); self.p_func.vector().apply("insert")
        p_vec = self.p_func.vector()

        # 3. Sum contributions
        self.hvp_vec.zero()
        for energy in self.problem.energies:
            # 3.1. Continuous part (UFL action)
            if hasattr(energy, "jac_form") and (energy.jac_form is not None):
                hvp_form = dolfin.action(energy.jac_form, self.p_func)
                dolfin.assemble(hvp_form, tensor=self.hvp_vec, add_values=True)
            
            # 3.2. Discrete part (Custom hessp method)
            elif hasattr(energy, "hessp"):
                # Ensure internal state (like matrices) is updated if x changed
                if (x_was_updated) and hasattr(energy, "update_jac"):
                    energy.update_jac()
                
                energy.hessp(p_vec, self.hvp_vec, add_values=True)

        return self.hvp_vec.get_local()



    def _hessp_numdiff(self, x, p, rel_step=None, abs_step=None):

        norm_p = numpy.linalg.norm(p)
        if (norm_p < 1e-12):
            return numpy.zeros_like(x)

        self._update_state(x)

        if hasattr(self, "finite_difference_dynamic_step") and (self.finite_difference_dynamic_step):
            if hasattr(self, "_hessp_old_x") and (self._hessp_old_x is not None):
                if ((type(self.finite_difference_dynamic_step) is not str) and (self.finite_difference_dynamic_step            )) or \
                   ((type(self.finite_difference_dynamic_step) is     str) and (self.finite_difference_dynamic_step == "global"))    :
                    dx = numpy.linalg.norm(x - self._hessp_old_x)
                    dynamic_abs_step = max(dx, 1e-6)
                elif (type(self.finite_difference_dynamic_step) is str) and (self.finite_difference_dynamic_step == "local"):
                    dx = x - self._hessp_old_x
                    dynamic_abs_step = numpy.maximum(numpy.abs(dx), 1e-6)
                else:
                    raise ValueError("Invalid value for finite_difference_dynamic_step. Must be 'global' or 'local'.")
            else:
                dynamic_abs_step = abs_step if (abs_step is not None) else 1e-6
            self.printer.print_var("dynamic_abs_step", dynamic_abs_step)
        else:
            dynamic_abs_step = abs_step if (abs_step is not None) else 1e-6
            
        jac_func = self.scipy_kwargs.get("jac")
        
        if callable(jac_func) and (jac_func != True):
            g0 = jac_func(x).copy()
        else:
            g0 = self._jac_numdiff(x=x, abs_step=dynamic_abs_step)
                
        while (True):
            step = numpy.mean(dynamic_abs_step) / norm_p
            
            if callable(jac_func) and (jac_func != True):
                g1 = jac_func(x + step * p)
            else:
                g1 = self._jac_numdiff(x=x + step * p, abs_step=dynamic_abs_step)

            res = (g1 - g0) / step
            
            if numpy.any(numpy.isnan(res)):
                self.printer.print_str("NaNs detected in FD hessp. Halving step size.")
                dynamic_abs_step /= 2.0
                if numpy.all(dynamic_abs_step < 1e-9):
                    raise RuntimeError("Finite difference step size reached minimum limit. Aborting.")
            else:
                break

        self._update_state(x) # restore state
        self._hessp_old_x = x.copy()
        return res



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
        self.printer.print_var("k_iter",self.k_iter,-1)

        return self.k_iter



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
        self._jac_old_x = None # Reset dynamic FD tracker

        # Run optimizer with pre-baked kwargs
        self.k_iter = 1
        self.printer.print_var("k_iter",self.k_iter)
        self.printer.inc()
        res = scipy.optimize.minimize(
            fun = self.scipy_fun,
            x0  = x0            ,
            **self.scipy_kwargs )
        self.printer.dec()
        if hasattr(res, "success"): self.printer.print_var("success", res.success)
        if hasattr(res, "message"): self.printer.print_var("message", res.message)
        if hasattr(res, "nit"    ): self.printer.print_var("nit"    , res.nit    )
        if hasattr(res, "nfev"   ): self.printer.print_var("nfev"   , res.nfev   )
        if hasattr(res, "njev"   ): self.printer.print_var("njev"   , res.njev   )
        if hasattr(res, "nhev"   ): self.printer.print_var("nhev"   , res.nhev   )
        # if hasattr(res, "x"      ): self.printer.print_var("x"      , res.x      )
        if hasattr(res, "fun"    ): self.printer.print_var("fun"    , res.fun    )

        if (res.success):
            self.printer.print_str("Nonlinear solver converged…")
        else:
            self.printer.print_str("Warning! Nonlinear solver failed to converge… (k_frame = "+str(self.k_frame)+")")

        # Update FEniCS state
        if   (type(self.problem) is dwarp.FullKinematicsWarpingProblem):
            self.problem.U.vector().set_local(res.x); self.problem.U.vector().apply("insert")
        elif (type(self.problem) is dwarp.ReducedKinematicsWarpingProblem):
            self.problem.reduced_displacement.vector().set_local(res.x); self.problem.reduced_displacement.vector().apply("insert")
            self.problem.update_disp()
        
        return res.success, res.nit
