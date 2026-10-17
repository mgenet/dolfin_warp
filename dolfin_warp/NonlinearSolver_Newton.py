#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2016-2026                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

import dolfin
import glob
import numpy
import os
import time

import myPythonLibrary as mypy

import dolfin_mech as dmech
import dolfin_warp as dwarp

from .NonlinearSolver                 import NonlinearSolver
from .NonlinearSolverMixin_BFGS       import BFGSNonlinearSolverMixin
from .NonlinearSolverMixin_Relaxation import RelaxationNonlinearSolverMixin

################################################################################

class NewtonNonlinearSolver(NonlinearSolver, RelaxationNonlinearSolverMixin, BFGSNonlinearSolverMixin):



    def __init__(self,
            problem,
            parameters={}):

        self.problem = problem
        self.printer = self.problem.printer

        # linear solver
        self.linear_solver_name = parameters["linear_solver_name"] if ("linear_solver_name" in parameters) and (parameters["linear_solver_name"] is not None) else "mumps"

        # self.res_vec = dolfin.PETScVector()
        # self.jac_mat = dolfin.PETScMatrix()
        # self.res_vec = dolfin.Vector()
        if (type(self.problem) is dwarp.FullKinematicsWarpingProblem):
            self.res_vec = self.problem.U.vector().copy()
        elif (type(self.problem) is dwarp.ReducedKinematicsWarpingProblem):
            self.res_vec = self.problem.reduced_displacement.vector().copy()
        self.res_vec.zero()
        self.res_norm = 0.
        self.jac_mat = dolfin.Matrix()

        self.linear_solver = dolfin.LUSolver(
            self.jac_mat,
            self.linear_solver_name)
        self.linear_solver.parameters['report']               = bool(0)
        # self.linear_solver.parameters['reuse_factorization']  = bool(0)
        # self.linear_solver.parameters['same_nonzero_pattern'] = bool(1)
        self.linear_solver.parameters['symmetric']            = bool(1)
        self.linear_solver.parameters['verbose']              = bool(0)

        options = parameters.get("options")
        if options is None: options = {}

        # bfgs
        self.init_bfgs(parameters=options)

        # relaxation
        self.init_relax(parameters=options)

        # iterations control
        self.tol_dU           = options.get("tol_dU"          , None)
        self.tol_dU_rel_U     = options.get("tol_dU_rel_U"    , None)
        self.tol_dU_rel_DU    = options.get("tol_dU_rel_DU"   , None)
        self.tol_res          = options.get("tol_res"         , None)
        self.tol_dres_rel_res = options.get("tol_dres_rel_res", None)
        self.n_iter_max       = options.get("n_iter_max"      , 32  )
        self.fail_on_line_search_failure = options.get("fail_on_line_search_failure", False)

        # damping (Levenberg-Marquardt-like): the Newton matrix is augmented with
        # mu*D, where D is a fixed metric (elastic stiffness or mass matrix of the
        # reference mesh). Since it only acts on the increment, it does not change
        # the solution, only the path to it. It is useful when the Newton matrix
        # has a large (near) null space, e.g., with discrete equilibrium gap
        # regularizations, for which undamped Newton steps can be huge.
        self.damping_type = options.get("damping_type", None) # None, elastic, mass
        if (self.damping_type is not None):
            assert (type(self.problem) is dwarp.FullKinematicsWarpingProblem),\
                "Damping is only implemented for full kinematics. Aborting."
            self.damping_init     = options.get("damping_init"    , 1e-2) # relative to the average diagonal of the Newton matrix
            self.damping_min      = options.get("damping_min"     , 1e-6)
            self.damping_max      = options.get("damping_max"     , 1e+2)
            self.damping_increase = options.get("damping_increase", 10. ) # when the line search fails
            self.damping_decrease = options.get("damping_decrease", 0.5 ) # when the full step is accepted
            self.damping_tol      = options.get("damping_tol"     , 1e-2) # convergence is only accepted when the damping is below this value
            damping_poisson       = options.get("damping_poisson" , 0.3 )
            if (self.damping_type == "elastic"):
                eps_trial = dolfin.sym(dolfin.grad(self.problem.dU_trial))
                eps_test  = dolfin.sym(dolfin.grad(self.problem.dU_test ))
                lmbda = damping_poisson/(1+damping_poisson)/(1-2*damping_poisson)
                mu    = 1./2/(1+damping_poisson)
                damping_form = (lmbda * dolfin.tr(eps_trial) * dolfin.tr(eps_test) + 2*mu * dolfin.inner(eps_trial, eps_test)) * dolfin.dx(domain=self.problem.mesh)
            elif (self.damping_type == "mass"):
                damping_form = dolfin.inner(self.problem.dU_trial, self.problem.dU_test) * dolfin.dx(domain=self.problem.mesh)
            else:
                assert (0), "damping_type (="+str(self.damping_type)+") must be None, \"elastic\" or \"mass\". Aborting."
            self.damping_mat = dolfin.assemble(damping_form)
            self.damping_diag_vec = self.problem.U.vector().copy()
            self.damping_mat.get_diagonal(self.damping_diag_vec)
            self.damping_diag_avg = self.damping_diag_vec.norm("l1")/self.damping_diag_vec.size()
            self.jac_diag_vec = self.problem.U.vector().copy()

        # write iterations
        self.write_iterations = parameters["write_iterations"] if ("write_iterations" in parameters) and (parameters["write_iterations"] is not None) else False

        if (self.write_iterations):
            self.working_folder   = parameters["working_folder"]
            self.working_basename = parameters["working_basename"]

            for filename in glob.glob(self.working_folder+"/"+self.working_basename+"-frame=[0-9]*.*"):
                os.remove(filename)



    def solve(self,
            k_frame=None):

        self.k_frame = k_frame

        if (self.write_iterations):
            self.frame_filebasename = self.working_folder+"/"+self.working_basename+"-frame="+str(self.k_frame).zfill(len(str(self.problem.images_n_frames)))

            self.frame_printer = mypy.DataPrinter(
                names=["k_iter", "res_norm", "err_dres_rel_res", "relax", "dU_norm", "U_norm", "err_dU_rel_U"],
                filename=self.frame_filebasename+".dat")

            dmech.write_VTU_file(
                filebasename=self.frame_filebasename,
                function=self.problem.U,
                time=0)
        else:
            self.frame_filebasename = None

        self.k_iter = 0
        self.problem.DU.vector().zero()
        self.success = False
        if (self.damping_type is not None):
            self.damping = self.damping_init
        self.printer.inc()
        while (True):
            self.k_iter += 1
            self.printer.print_var("k_iter",self.k_iter,-1)

            # linear problem
            self.linear_success = self.linear_solve()
            if not (self.linear_success):
                break

            # relaxation
            self.compute_relax()

            # damping update
            if (self.damping_type is not None):
                if (self.relax == 0.) and (self.damping < self.damping_max):
                    self.damping = min(self.damping*self.damping_increase, self.damping_max)
                    self.printer.print_sci("Line search failed, increasing damping",self.damping)
                    continue
                elif (self.relax == getattr(self, "relax_init", 1.)):
                    self.damping = max(self.damping*self.damping_decrease, self.damping_min)
                self.printer.print_sci("damping",self.damping)

            # line search failure: the energy could not be decreased along the
            # Newton direction (relax = 0), so the update is zero, which used to
            # be silently reported as convergence. Near the minimum this is
            # expected (the image energy gradient is only approximate), but it
            # also happens far from it, e.g., when every step creates inverted
            # elements. The failure is now reported, and can be treated as a
            # solver failure with the "fail_on_line_search_failure" option.
            if (self.relax == 0.):
                if (self.fail_on_line_search_failure):
                    self.printer.print_str("Warning! Line search failed: nonlinear solver failed to converge… (k_frame = "+str(self.k_frame)+")")
                    self.success = False
                    break
                else:
                    self.printer.print_str("Warning! Line search failed: zero update, nonlinear solver will be considered converged… (k_frame = "+str(self.k_frame)+")")

            # solution update
            self.problem.update_displacement(relax=self.relax)
            self.printer.print_sci("U_norm",self.problem.U_norm)

            self.problem.DU.vector().zero(); self.problem.DU.vector().axpy(+1.0, self.problem.U.vector()); self.problem.DU.vector().axpy(-1.0, self.problem.Uold.vector())
            self.problem.DU_norm = self.problem.DU.vector().norm("l2")
            self.printer.print_sci("DU_norm",self.problem.DU_norm)

            if (self.write_iterations):
                dmech.write_VTU_file(
                    filebasename=self.frame_filebasename,
                    function=self.problem.U,
                    time=self.k_iter)

            # displacement error
            self.problem.err_dU = abs(self.relax)*self.problem.dU_norm
            self.printer.print_sci("err_dU",self.problem.err_dU)
   
            if (self.problem.U_norm == 0.):
                if (self.problem.Uold_norm == 0.):
                    self.problem.err_dU_rel_U = 0.
                else:
                    self.problem.err_dU_rel_U = abs(self.relax)*self.problem.dU_norm/self.problem.Uold_norm
            else:
                self.problem.err_dU_rel_U = abs(self.relax)*self.problem.dU_norm/self.problem.U_norm
            self.printer.print_sci("err_dU_rel_U",self.problem.err_dU_rel_U)

            if (self.problem.DU_norm == 0.):
                self.problem.err_dU_rel_DU = 1.
            else:
                self.problem.err_dU_rel_DU = abs(self.relax)*self.problem.dU_norm/self.problem.DU_norm
            self.printer.print_sci("err_dU_rel_DU",self.problem.err_dU_rel_DU)

            # write iteration data
            if (self.write_iterations):
                self.frame_printer.write_line([self.k_iter, self.res_norm, self.err_dres_rel_res, self.relax, self.problem.dU_norm, self.problem.U_norm, self.problem.err_dU_rel_U])

            # store s vector for BFGS
            if (self.use_bfgs):
                if (type(self.problem) is dwarp.FullKinematicsWarpingProblem):
                    if not hasattr(self, "s_vec_cur") or self.s_vec_cur is None:
                        self.s_vec_cur = self.problem.dU.vector().copy()
                    else:
                        self.s_vec_cur.zero(); self.s_vec_cur.axpy(1.0, self.problem.dU.vector())
                    self.s_vec_cur *= self.relax
                elif (type(self.problem) is dwarp.ReducedKinematicsWarpingProblem):
                    if not hasattr(self, "s_vec_cur") or self.s_vec_cur is None:
                        self.s_vec_cur = self.problem.dreduced_displacement.vector().copy()
                    else:
                        self.s_vec_cur.zero(); self.s_vec_cur.axpy(1.0, self.problem.dreduced_displacement.vector())
                    self.s_vec_cur *= self.relax

            # exit test
            self.success = True
            if (self.tol_res          is not None) and (self.res_norm              > self.tol_res         ):
                self.success = False
            if (self.tol_dres_rel_res is not None) and (self.err_dres_rel_res      > self.tol_dres_rel_res):
                self.success = False
            if (self.tol_dU           is not None) and (self.problem.err_dU        > self.tol_dU          ):
                self.success = False
            if (self.tol_dU_rel_U     is not None) and (self.problem.err_dU_rel_U  > self.tol_dU_rel_U    ):
                self.success = False
            if (self.tol_dU_rel_DU    is not None) and (self.problem.err_dU_rel_DU > self.tol_dU_rel_DU   ):
                self.success = False
            if (self.damping_type is not None) and (self.damping > self.damping_tol) and (self.relax > 0.): # if the line search failed even with maximal damping, same treatment as without damping (cf. above)
                self.success = False

            # exit
            if (self.success):
                self.printer.print_str("Nonlinear solver converged…")
                break

            if (self.k_iter == self.n_iter_max):
                self.printer.print_str("Warning! Nonlinear solver failed to converge… (k_frame = "+str(self.k_frame)+")")
                break

        self.printer.dec()

        if (self.write_iterations):
            self.frame_printer.close()
            commandline  = "gnuplot -e \"set terminal pdf noenhanced;"
            commandline += " set output '"+self.frame_filebasename+".pdf';"
            commandline += " set key box textcolor variable;"
            commandline += " set grid;"
            commandline += " set logscale y;"
            commandline += " set yrange [1e-3:1e0];"
            commandline += " plot '"+self.frame_filebasename+".dat' u 1:7 pt 1 lw 3 title 'err_dU_rel_U', "+str(self.tol_dU_rel_U)+" lt -1 notitle;"
            commandline += " unset logscale y;"
            commandline += " set yrange [*:*];"
            commandline += " plot '' u 1:4 pt 1 lw 3 title 'relax'\""
            os.system(commandline)

        return self.success, self.k_iter



    def linear_solve(self):

        # res_old
        if (hasattr(self, "res_old_vec")):
            self.res_old_vec.zero(); self.res_old_vec.axpy(1.0, self.res_vec)
        else:
            self.res_old_vec = self.res_vec.copy()
        self.res_old_norm = self.res_norm

        self.problem.call_before_assembly(
            write_iterations=self.write_iterations,
            basename=self.frame_filebasename,
            k_iter=self.k_iter)

        # linear system: residual assembly
        self.printer.print_str("Residual assembly…",newline=False)
        timer = time.time()
        self.problem.assemble_res(
            res_vec=self.res_vec)
        timer = time.time() - timer
        self.printer.print_str(" "+str(timer)+" s",tab=False)
        # self.printer.print_var("res_vec",self.res_vec.get_local())

        self.printer.inc()

        # res_norm
        self.res_norm = self.res_vec.norm("l2")
        self.printer.print_sci("res_norm",self.res_norm)
        if not (numpy.isfinite(self.res_norm)):
            self.printer.print_str("Warning! Residual is NaN!",tab=False)
            return False

        # dres
        if (hasattr(self, "dres_vec")):
            self.dres_vec.zero(); self.dres_vec.axpy(1.0, self.res_vec); self.dres_vec.axpy(-1.0, self.res_old_vec)
        else:
            self.dres_vec = self.res_vec - self.res_old_vec
        self.dres_norm = self.dres_vec.norm("l2")
        self.printer.print_sci("dres_norm",self.dres_norm)

        if self.use_bfgs and self.s_vec_cur is not None:
            self.update_bfgs_history(self.s_vec_cur, self.dres_vec)

        # err_dres_rel_res
        if (self.res_norm == 0.) and (self.res_old_norm == 0.):
            self.err_dres_rel_res = 0.
        elif (self.res_norm == 0.):
            self.err_dres_rel_res = self.dres_norm / self.res_old_norm
        else:
            self.err_dres_rel_res = self.dres_norm / self.res_norm
        self.printer.print_sci("err_dres_rel_res",self.err_dres_rel_res)

        self.printer.dec()

        # linear system: matrix assembly
        if self.use_bfgs and self.k_iter > 1 and ((self.k_iter - 1) % self.bfgs_restart_iter != 0):
            assemble_new_jacobian = False
        else:
            assemble_new_jacobian = True

        if (assemble_new_jacobian):
            self.printer.print_str("Jacobian assembly…",newline=False)
            timer = time.time()
            self.problem.assemble_jac(
                jac_mat=self.jac_mat)
            timer = time.time() - timer
            self.printer.print_str(" "+str(timer)+" s",tab=False)
            if (self.damping_type is not None):
                self.jac_mat.get_diagonal(self.jac_diag_vec)
                jac_diag_avg = self.jac_diag_vec.norm("l1")/self.jac_diag_vec.size()
                self.jac_mat.axpy(self.damping*jac_diag_avg/self.damping_diag_avg, self.damping_mat, False)
            # self.printer.print_var("jac_mat",self.jac_mat.array())
            if self.use_bfgs:
                self.reset_bfgs_history()

            # linear system: solve
            try:
                self.printer.print_str("Solve…",newline=False)
                timer = time.time()
                if (type(self.problem) is dwarp.FullKinematicsWarpingProblem):
                    self.linear_solver.solve(
                        self.problem.dU.vector(),
                        -self.res_vec)
                    # self.printer.print_var("dU",dU.vector().get_local())
                elif (type(self.problem) is dwarp.ReducedKinematicsWarpingProblem):
                    self.linear_solver.solve(
                        self.problem.dreduced_displacement.vector(),
                        -self.res_vec)
                # self.problem.dreduced_displacement.vector()[:] = numpy.linalg.solve(
                #     self.jac_mat.array(),
                #     -self.res_vec.get_local())
                # self.printer.print_var("dreduced_displacement",self.problem.dreduced_displacement.vector().get_local())
                timer = time.time() - timer
                self.printer.print_str(" "+str(timer)+" s",tab=False)
            except:
                self.printer.print_str("Warning! Linear solver failed!",tab=False)
                return False
        else:
            try:
                self.printer.print_str("BFGS Solve…",newline=False)
                timer = time.time()
                if (type(self.problem) is dwarp.FullKinematicsWarpingProblem):
                    self.compute_bfgs_search_direction(self.res_vec, self.problem.dU.vector(), self.linear_solver)
                elif (type(self.problem) is dwarp.ReducedKinematicsWarpingProblem):
                    self.compute_bfgs_search_direction(self.res_vec, self.problem.dreduced_displacement.vector(), self.linear_solver)
                timer = time.time() - timer
                self.printer.print_str(" "+str(timer)+" s",tab=False)
            except:
                self.printer.print_str("Warning! BFGS solver failed!",tab=False)
                return False

        self.printer.inc()

        if (type(self.problem) is dwarp.FullKinematicsWarpingProblem):
            self.problem.dU_norm = self.problem.dU.vector().norm("l2")
            self.printer.print_sci("dU_norm",self.problem.dU_norm)
            if not (numpy.isfinite(self.problem.dU_norm)):
                self.printer.print_str("Warning! Solution increment is NaN! Setting it to 0.",tab=False)
                self.problem.dU.vector().zero()
                return False
        elif (type(self.problem) is dwarp.ReducedKinematicsWarpingProblem):
            self.problem.dreduced_displacement_norm = self.problem.dreduced_displacement.vector().norm("l2")
            self.printer.print_sci("dreduced_displacement_norm",self.problem.dreduced_displacement_norm)
            if not (numpy.isfinite(self.problem.dreduced_displacement_norm)):
                self.printer.print_str("Warning! Solution increment is NaN! Setting it to 0.",tab=False)
                self.problem.dreduced_displacement.vector().zero()
                return False
            self.problem.U_vec_cp.zero(); self.problem.U_vec_cp.axpy(1.0, self.problem.U.vector())
            self.problem.update_displacement(relax=+1.)
            self.problem.dU.vector().zero(); self.problem.dU.vector().axpy(+1.0, self.problem.U.vector()); self.problem.dU.vector().axpy(-1.0, self.problem.U_vec_cp)
            self.problem.dU_norm = self.problem.dU.vector().norm("l2")
            self.printer.print_sci("dU_norm",self.problem.dU_norm)
            self.problem.update_displacement(relax=-1.)

        self.printer.dec()

        return True
