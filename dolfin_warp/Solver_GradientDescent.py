#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2016-2024                                       ###
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

from .NonlinearSolver            import           NonlinearSolver
from .NonlinearSolver_Relaxation import RelaxationNonlinearSolver

################################################################################

class GradientDescentSolver(RelaxationNonlinearSolver):



    def __init__(self,
            problem,
            parameters={}):

        self.problem = problem
        self.printer = self.problem.printer

        self.res_vec = dolfin.Vector()          # Pre allocatin of res_vec


        # relaxation
        RelaxationNonlinearSolver.__init__(self, parameters=parameters)

        # iterations control
        self.min_gradient_step      = parameters.get("min_gradient_step"    , 1e-6)
        self.step                   = parameters.get("step"                 , 1)
        self.tol_dU_rel             = parameters.get("tol_dU_rel"           , None)
        self.tol_res_rel            = parameters.get("tol_res_rel"          , None)
        self.n_iter_max             = parameters.get("n_iter_max"           , 32  )

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

            # DEBUG
            # self.frame_printer = mypy.DataPrinter(
            #     names=["k_iter", "res_norm", "res_err_rel", "relax", "dU_norm", "U_norm", "dU_err"],
            #     filename=self.frame_filebasename+".dat")

            dmech.write_VTU_file(
                filebasename=self.frame_filebasename,
                function=self.problem.U,
                time=0)
        else:
            self.frame_filebasename = None

        self.k_iter = 0
        self.problem.DU.vector().zero()
        self.success = False
        self.printer.inc()
        while (True):
            self.k_iter += 1
            self.printer.print_var("k_iter",self.k_iter,-1)

            # Gradient descent direction computation
            self.problem.assemble_res(
                res_vec     =self.res_vec, 
                add_values  = False)                                        # DEBUG: compute gradient from scratch

            self.problem.DU.vector()[:] = self.res_vec[:]
            # relaxation
            self.compute_relax()

            # solution update
            # self.problem.update_displacement(relax=self.relax)            # Why needed already done in compute relax right #DEBUG?
            # self.printer.print_sci("U_norm",self.problem.U_norm)

            self.problem.DU_norm = self.problem.DU.vector().norm("l2")
            self.printer.print_sci("DU_norm",self.problem.DU_norm)

            if (self.write_iterations):
                dmech.write_VTU_file(
                    filebasename=self.frame_filebasename,
                    function=self.problem.U,
                    time=self.k_iter)

            # displacement error
            if (self.problem.U_norm == 0.):
                if (self.problem.Uold_norm == 0.):
                    self.problem.dU_err = 0.
                else:
                    self.problem.dU_err = self.problem.dU_norm/self.problem.Uold_norm
            else:
                self.problem.dU_err = self.problem.dU_norm/self.problem.U_norm
            self.printer.print_sci("dU_err",self.problem.dU_err)

            if (self.problem.DU_norm == 0.):
                self.problem.dU_err_rel = 1.
            else:
                self.problem.dU_err_rel = self.problem.dU_norm/self.problem.DU_norm
            self.printer.print_sci("dU_err_rel",self.problem.dU_err_rel)

            if (self.write_iterations):
                self.frame_printer.write_line([self.k_iter, self.res_norm, self.res_err_rel, self.relax, self.problem.dU_norm, self.problem.U_norm, self.problem.dU_err])

            # exit test
            self.success = True
            if (self.tol_res_rel is not None) and (self.res_err_rel        > self.tol_res_rel):
                self.success = False
            if (self.tol_dU      is not None) and (self.problem.dU_err     > self.tol_dU     ):
                self.success = False
            if (self.tol_dU_rel  is not None) and (self.problem.dU_err_rel > self.tol_dU_rel ):
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
            commandline += " plot '"+self.frame_filebasename+".dat' u 1:7 pt 1 lw 3 title 'dU_err', "+str(self.tol_dU)+" lt -1 notitle;"
            commandline += " unset logscale y;"
            commandline += " set yrange [*:*];"
            commandline += " plot '' u 1:4 pt 1 lw 3 title 'relax'\""
            os.system(commandline)

        return self.success, self.k_iter


