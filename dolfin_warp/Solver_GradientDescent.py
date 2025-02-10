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
                alpha           = dolfin.Constant(self.inner_product_H1_weight)



                inner_compiler_parameters = {
                    "quadrature_degree":2,
                    "quadrature_scheme":"default"}
                dV_inner = dolfin.Measure(
                    "dx",
                    domain=self.problem.mesh,
                    metadata=inner_compiler_parameters)


                #DEBUG PSEUDO INV
                F_inv = dolfin.inv((self.problem.F.T*self.problem.F)) * self.problem.F.T
                # F_inv = dolfin.inv(self.problem.F)


                # grad_x_trial    = dolfin.dot(dolfin.grad(self.problem.dU_trial),dolfin.inv(self.problem.F))
                # grad_x_test     = dolfin.dot(dolfin.grad(self.problem.dU_test),dolfin.inv(self.problem.F))
                grad_x_trial    = dolfin.dot(dolfin.grad(self.problem.dU_trial),F_inv)
                grad_x_test     = dolfin.dot(dolfin.grad(self.problem.dU_test),F_inv)


                inner_product_1   = dolfin.Constant(4)*dolfin.inner(dolfin.sym(grad_x_trial),dolfin.sym(grad_x_test)) * self.problem.J * dV_inner
                inner_product_2   = alpha * dolfin.inner(self.problem.dU_trial, self.problem.dU_test) * self.problem.J * dV_inner
                inner_product   = inner_product_1 + inner_product_2

                # F_inner = dolfin.inner(dolfin.inv(self.problem.F),dolfin.inv(self.problem.F)) * dV_inner
                # F_inner_assemble = dolfin.assemble(dolfin.inner(dolfin.inv(self.problem.F),dolfin.inv(self.problem.F)) * dV_inner)
                F_inner = dolfin.inner(F_inv,F_inv) 
                F_inner_assemble = dolfin.assemble(F_inner* dV_inner)



                #DEBUG
                # u = self.problem.dU_test
                # v = self.problem.dU_trial
                # inner_product       = dolfin.inner(dolfin.grad(u) + dolfin.grad(u).T, dolfin.grad(v) + dolfin.grad(v).T) * energy_shape.dV + alpha * dolfin.inner(u, v) * energy_shape.dV


                
                # dolfin.solve(inner_product == energy_shape.res_form, self.res_vec_funct)
                       
                #DEBUG res_form: 
                res_form       = self.problem.J*energy_shape.Idef*dolfin.inner(dolfin.inv(self.problem.F).T, dolfin.grad(self.problem.dU_test))* energy_shape.dV 

                inner_assembled = dolfin.assemble(inner_product)
                rhs_assembled = dolfin.assemble(res_form)

                dolfin.solve(inner_assembled,self.res_vec_funct.vector(),rhs_assembled)


                # res_form      += self.problem.J*dolfin.inner(energy_shape.DIdef, self.problem.dU_test) * energy_shape.dV 
                # dolfin.solve(inner_product == res_form, self.res_vec_funct); print("* DEBUG")

                self.res_vec    = self.res_vec_funct.vector()

                ##DEBUG save intermediate numpy arrays
                # res_numpy = self.res_vec[:]
                # print(res_numpy.shape)
                # import os
                # file_res = "res_lagrange_noF.dat"
                # if os.path.exists(file_res):
                #     res_data = numpy.loadtxt(file_res)
                #     if res_data.ndim == 1:  # If file has only one row, reshape to column
                #         res_data = res_data[:, numpy.newaxis]
                #     updated_res_data = numpy.column_stack((res_data, res_numpy))
                # else:
                #     updated_res_data = res_numpy[:, numpy.newaxis]  
                # numpy.savetxt(file_res, updated_res_data, fmt="%.6f")

                # F_inner_numpy = numpy.array([F_inner_assemble])

                # print(F_inner_numpy.shape)
                # import os
                # file_F_inner = "F_inner_lagrange"
                # if os.path.exists(file_F_inner+".npy"):
                #     F_inner_data = numpy.load(file_F_inner+".npy")
                #     print(f"Loaded F_inner_data {F_inner_data.shape}, F_inner_numpy {F_inner_numpy.shape}")
                #     updated_F_inner_data = numpy.concatenate((F_inner_data, F_inner_numpy))
                #     numpy.save(file_F_inner, updated_F_inner_data)
                # else:
                #     updated_F_inner_data = F_inner_numpy
                #     print(f"Init F_inner_data {updated_F_inner_data.shape}, F_inner_numpy {F_inner_numpy.shape}")
                #     numpy.save(file_F_inner, F_inner_numpy)


                # form_bi_numpy = dolfin.assemble(inner_product).array()[:,:]
                # print(form_bi_numpy.shape)

                # file_form_bi = "form_bi_lagrange_noF"
                # if os.path.exists(file_form_bi+".npy"):
                #     form_bi_data = numpy.load(file_form_bi+".npy")
                #     if form_bi_data.ndim == 2:  # If file has only matrix, reshape to tensor
                #         form_bi_data = form_bi_data[:, :,numpy.newaxis]
                #         print(f"form_bi_dataafter expand {form_bi_data.shape}")
                #     print(f"form_bi_data {form_bi_data.shape}, form_bi_numpy {form_bi_numpy.shape}")
                #     updated_form_bi_data = numpy.concatenate((form_bi_data, form_bi_numpy[:, :,numpy.newaxis]), axis = 2)
                #     print(f"updated_form_bi_data {updated_form_bi_data.shape}")
                # else:
                #     updated_form_bi_data = form_bi_numpy[:, :, numpy.newaxis]  
                #     print(f"updated_form_bi_data initial {updated_form_bi_data.shape}")
                # numpy.save(file_form_bi, updated_form_bi_data)

                # form_bi_numpy_1 = dolfin.assemble(inner_product_1).array()[:,:]
                # print(form_bi_numpy_1.shape)

                # file_form_bi_1 = "form_bi_lagrange_1_noF"
                # if os.path.exists(file_form_bi_1+".npy"):
                #     form_bi_data_1 = numpy.load(file_form_bi_1+".npy")
                #     if form_bi_data.ndim == 2:  # If file has only matrix, reshape to tensor
                #         form_bi_data_1 = form_bi_data_1[:, :,numpy.newaxis]
                #         print(f"form_bi_dataafter expand {form_bi_data_1.shape}")
                #     print(f"form_bi_data {form_bi_data_1.shape}, form_bi_numpy {form_bi_numpy_1.shape}")
                #     updated_form_bi_data_1 = numpy.concatenate((form_bi_data_1, form_bi_numpy_1[:, :,numpy.newaxis]), axis = 2)
                #     print(f"updated_form_bi_data {updated_form_bi_data_1.shape}")
                # else:
                #     updated_form_bi_data_1 = form_bi_numpy_1[:, :, numpy.newaxis]  
                #     print(f"updated_form_bi_data initial {updated_form_bi_data_1.shape}")
                # numpy.save(file_form_bi_1, updated_form_bi_data_1)


                # form_bi_numpy_2 = dolfin.assemble(inner_product_2).array()[:,:]
                # file_form_bi_2 = "form_bi_lagrange_2_noF"
                # if os.path.exists(file_form_bi_2+".npy"):
                #     form_bi_data_2 = numpy.load(file_form_bi_2+".npy")
                #     if form_bi_data.ndim == 2:  # If file has only matrix, reshape to tensor
                #         form_bi_data_2 = form_bi_data_2[:, :,numpy.newaxis]
                #         print(f"form_bi_dataafter expand {form_bi_data_2.shape}")
                #     print(f"form_bi_data {form_bi_data_2.shape}, form_bi_numpy {form_bi_numpy_2.shape}")
                #     updated_form_bi_data_2 = numpy.concatenate((form_bi_data_2, form_bi_numpy_2[:, :,numpy.newaxis]), axis = 2)
                #     print(f"updated_form_bi_data {updated_form_bi_data_2.shape}")
                # else:
                #     updated_form_bi_data_2 = form_bi_numpy_2[:, :, numpy.newaxis]  
                #     print(f"updated_form_bi_data initial {updated_form_bi_data_2.shape}")
                # numpy.save(file_form_bi_2, updated_form_bi_data_2)

                # form_numpy = dolfin.assemble(res_form)[:]
                # print(form_numpy.shape)
                # import os
                # file_form = "form_lagrange_noF"
                # if os.path.exists(file_form+".npy"):
                #     form_data = numpy.load(file_form+".npy")
                #     if form_data.ndim == 1:  #If file has only one row, reshape to column
                #         form_data = form_data[:, numpy.newaxis]
                #     updated_form_data = numpy.column_stack((form_data, form_numpy))
                # else:
                #     updated_form_data = form_numpy[:, numpy.newaxis]  
                # numpy.save(file_form, updated_form_data)

                # #### END DEBUGGING

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
                self.relax = 1e-2

            self.compute_relax(
                                1.5*self.relax
                                    ) 
      
            # solution update
            self.problem.update_displacement(relax=self.relax)                       
            # self.problem.update_displacement(relax=1)                                  #DEBUG relax = 1 for comparison Somehow need although it's already done in compute_relax() #DEBUG?
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


