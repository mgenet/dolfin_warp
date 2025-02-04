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

        if self.problem.kinematics_type == "reduced":
            self.res_vec_funct = dolfin.Function(self.problem.reduced_displacement_fs)
        else:
            self.res_vec_funct = dolfin.Function(self.problem.U_fs)
        # relaxation
        RelaxationNonlinearSolver.__init__(self, parameters=parameters)

        # iterations control
        self.min_gradient_step          = parameters.get("min_gradient_step"        , 1e-6)
        self.step                       = parameters.get("step"                     , 1)
        self.tol_dU_rel                 = parameters.get("tol_dU_rel"               , None)
        self.tol_dU                     = parameters.get("tol_dU"                   , None)
        self.tol_res_rel                = parameters.get("tol_res_rel"              , None)
        self.n_iter_max                 = parameters.get("n_iter_max"               , 32  )
        self.relax_n_iter_max           = parameters.get("relax_n_iter_max"         , None)
        self.relax_type                 = parameters.get("relax_type"               , None)
        self.gradient_type              = parameters.get("gradient_type"            , "L2")
        self.inner_product_H1_weight    = parameters.get("inner_product_H1_weight"  , 1e-2)
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
                names=["k_iter", "res_norm", "relax", "dU_norm", "U_norm", "dU_err"],
                filename=self.frame_filebasename+".dat")

            dmech.write_VTU_file(
                filebasename=self.frame_filebasename,
                function=self.problem.U,
                time=0)
        else:
            self.frame_filebasename = None

        self.k_iter = 0
        self.problem.dU.vector().zero()
        self.problem.DU.vector().zero()
        self.success = False
        self.printer.inc()
        while (True):
            self.k_iter += 1
            self.printer.print_var("k_iter",self.k_iter,-1)

            self.problem.call_before_assembly(
                write_iterations    =self.write_iterations  ,
                basename            =self.frame_filebasename,
                k_iter              =self.k_iter            )

            # Gradient descent direction computation

            if self.gradient_type == "Sobolev": 
                found_energy = False
                for energy in self.problem.energies:
                    if isinstance(energy, dwarp.Energy_Shape_Registration.SignedImageEnergy): #DEBUG should make it more general and allow sobolev for all possible energy ?
                        energy_shape = energy
                        found_energy = True
                        break
                assert found_energy, "No SignedImageEnergy. Aborting."
                alpha           = self.inner_product_H1_weight
                # # inner_product pulling back inner product on reference body:

                # grad_u_trial_ref = dolfin.inv(energy_shape.problem.F).T * dolfin.grad(energy_shape.problem.dU_trial) * dolfin.inv(energy_shape.problem.F)
                # grad_u_test_ref = dolfin.inv(energy_shape.problem.F).T * dolfin.grad(energy_shape.problem.dU_test) * dolfin.inv(energy_shape.problem.F)

                # symmetric_grad_trial_ref =  grad_u_trial_ref + grad_u_trial_ref.T
                # symmetric_grad_test_ref = grad_u_test_ref + grad_u_test_ref.T

                # #DEBUG u cdot v = FU cdot FV : BRENO OK

                # inner_product   = dolfin.inner(symmetric_grad_trial_ref, symmetric_grad_test_ref) * self.problem.J * energy_shape.dV \
                #                 + alpha * dolfin.inner(energy_shape.problem.F*energy_shape.problem.dU_trial, energy_shape.problem.F*energy_shape.problem.dU_test) * self.problem.J * energy_shape.dV
                



                ##DEBUG only chain rule

                grad_x_trial = dolfin.grad(energy_shape.problem.dU_trial)*dolfin.inv(energy_shape.problem.F)
                grad_x_test = dolfin.grad(energy_shape.problem.dU_test)*dolfin.inv(energy_shape.problem.F)
                inner_product = dolfin.inner(grad_x_trial+grad_x_trial.T,grad_x_test+grad_x_test.T) * self.problem.J * energy_shape.dV + alpha * dolfin.inner(energy_shape.problem.dU_trial, energy_shape.problem.dU_test) * self.problem.J * energy_shape.dV
                

                #DEBUG res_form: 
                # res_form        = self.problem.J*dolfin.inner(energy_shape.DIdef, energy_shape.problem.dU_test) * energy_shape.dV 
                # res_form        += self.problem.J*energy_shape.Idef*dolfin.inner(dolfin.inv(energy_shape.problem.F).T, dolfin.grad(energy_shape.problem.dU_test))* energy_shape.dV 
                # dolfin.solve(inner_product == res_form, self.res_vec_funct); print("* DEBUG")
                dolfin.solve(inner_product == energy_shape.res_form, self.res_vec_funct)


                self.res_vec    = self.res_vec_funct.vector()
                print(f"self.res_vec : {self.res_vec[:]}")#DEBUG
            else:
                self.problem.assemble_res(
                    res_vec = self.res_vec) 



            if (type(self.problem) is dwarp.FullKinematicsWarpingProblem):
                self.problem.dU.vector()[:] = -self.res_vec[:]
            elif (type(self.problem) is dwarp.ReducedKinematicsWarpingProblem):
                self.problem.dreduced_displacement.vector()[:] = -self.res_vec[:]

            self.res_norm               = self.res_vec.norm("l2")

            # relaxation
            if self.k_iter == 1:
                self.relax = 1

            # self.compute_relax(
            #                     2*self.relax
            #                     ) 
      
            # solution update
            # self.problem.update_displacement(relax=self.relax)                                  # Somehow need although it's already done in compute_relax() #DEBUG?
            self.problem.update_displacement(relax=1)                                  #DEBUG relax = 1 for comparison Somehow need although it's already done in compute_relax() #DEBUG?
            self.printer.print_sci("U_norm",self.problem.U_norm)

            self.problem.DU.vector()[:] = self.problem.U.vector() - self.problem.Uold.vector()
            self.problem.DU_norm        = self.problem.DU.vector().norm("l2")
            self.problem.dU_norm        = self.problem.dU.vector().norm("l2")
            self.problem.dU_norm_relax  = self.relax*self.problem.dU.vector().norm('l2')
            self.printer.print_sci("DU_norm",self.problem.DU_norm)

            if (self.write_iterations):
                dmech.write_VTU_file(
                    filebasename    =self.frame_filebasename,
                    function        =self.problem.U         ,
                    time            =self.k_iter            )

            # displacement error
            if (self.problem.U_norm == 0.):
                if (self.problem.Uold_norm == 0.):
                    self.problem.dU_err     = 0.
                else:
                    self.problem.dU_err     = self.problem.dU_norm_relax/self.problem.Uold_norm
            else:
                self.problem.dU_err = self.problem.dU_norm_relax/self.problem.U_norm
            self.printer.print_sci("dU_err",self.problem.dU_err)


            if (self.problem.DU_norm == 0.):
                self.problem.dU_err_rel = 1.
            else:
                self.problem.dU_err_rel = self.problem.dU_norm_relax/self.problem.DU_norm
            self.printer.print_sci("dU_err_rel",self.problem.dU_err_rel)

            if (self.write_iterations):
                self.frame_printer.write_line([self.k_iter, self.res_norm, self.relax, self.problem.dU_norm, self.problem.U_norm, self.problem.dU_err])

            # exit test

            self.success = True

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


