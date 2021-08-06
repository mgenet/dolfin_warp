#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2016-2021                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

from builtins import range

import dolfin
import glob
import math
import numpy
import os
import time

import myPythonLibrary    as mypy
import myVTKPythonLibrary as myvtk

import dolfin_warp as dwarp
from .NonlinearSolver import NonlinearSolver

################################################################################

class RelaxationNonlinearSolver(NonlinearSolver):



    def compute_relax_aitken(self):

        if (self.k_iter == 1):
            self.relax = 1.
        else:
            self.relax *= (-1.) * self.res_old_vec.inner(self.dres_vec) / self.dres_norm**2
        self.printer.print_sci("relax",self.relax)



    def compute_relax_constant(self):

        self.relax = self.relax_val
        self.printer.print_sci("relax",self.relax)



    def compute_relax_gss(self):

        phi = (1 + math.sqrt(5)) / 2
        relax_a = (1-phi)/(2-phi)
        relax_b = 1./(2-phi)
        need_update_c = True
        need_update_d = True
        relax_cur = 0.
        relax_list = []
        ener_list = []
        self.printer.inc()
        k_relax = 1
        while (True):
            self.printer.print_var("k_relax",k_relax,-1)
            # self.printer.print_sci("relax_a",relax_a)
            # self.printer.print_sci("relax_b",relax_b)
            self.problem.call_before_assembly()
            if (need_update_c):
                relax_c = relax_b - (relax_b - relax_a) / phi
                relax_list.append(relax_c)
                self.printer.print_sci("relax_c",relax_c)
                self.problem.U.vector().axpy(relax_c-relax_cur, self.problem.dU.vector())
                relax_cur = relax_c
                relax_fc  = self.problem.assemble_ener()
                #self.printer.print_sci("relax_fc",relax_fc)
                if (numpy.isnan(relax_fc)):
                    relax_fc = float('+inf')
                    #self.printer.print_sci("relax_fc",relax_fc)
                self.printer.print_sci("relax_fc",relax_fc)
                ener_list.append(relax_fc)
            if (need_update_d):
                relax_d = relax_a + (relax_b - relax_a) / phi
                relax_list.append(relax_d)
                self.printer.print_sci("relax_d",relax_d)
                self.problem.U.vector().axpy(relax_d-relax_cur, self.problem.dU.vector())
                relax_cur = relax_d
                relax_fd  = self.problem.assemble_ener()
                if (numpy.isnan(relax_fd)):
                    relax_fd = float('+inf')
                    #self.printer.print_sci("relax_fd",relax_fd)
                self.printer.print_sci("relax_fd",relax_fd)
                ener_list.append(relax_fd)
            # self.printer.print_var("relax_list",relax_list)
            # self.printer.print_var("ener_list",ener_list)
            if (k_relax > 1):
                ener_min_old = ener_min
            ener_min = min(ener_list)
            # self.printer.print_sci("ener_min",ener_min)
            relax_min = relax_list[numpy.argmin(ener_list)]
            # self.printer.print_sci("relax_min",relax_min)
            if (ener_list[0] > 0) and (k_relax > 1):
                dener_min = ener_min-ener_min_old
                self.printer.print_sci("dener_min",dener_min)
                relax_err = dener_min/ener_list[0]
                self.printer.print_sci("relax_err",relax_err)
                if (relax_min != 0.) and (abs(relax_err) < self.relax_tol):
                    break
            if (k_relax == self.relax_n_iter_max):
                break
            if (relax_fc < relax_fd):
                relax_b = relax_d
                relax_d = relax_c
                relax_fd = relax_fc
                need_update_c = True
                need_update_d = False
            elif (relax_fc >= relax_fd):
                relax_a = relax_c
                relax_c = relax_d
                relax_fc = relax_fd
                need_update_c = False
                need_update_d = True
            else: assert(0)
            k_relax += 1
        self.printer.dec()
        self.problem.U.vector().axpy(-relax_cur, self.problem.dU.vector())
        #self.printer.print_var("ener_list",ener_list)

        if (self.write_iterations):
            self.iter_filebasename = self.frame_filebasename+"-iter="+str(self.k_iter).zfill(3)
            file_dat_iter = open(self.iter_filebasename+".dat","w")
            file_dat_iter.write("\n".join([" ".join([str(val) for val in [relax_list[k_relax], ener_list[k_relax]]]) for k_relax in range(len(relax_list))]))
            file_dat_iter.close()
            commandline  = "gnuplot -e \"set terminal pdf noenhanced;"
            commandline += " set output '"+self.iter_filebasename+".pdf';"
            commandline += " plot '"+self.iter_filebasename+".dat' using 1:2 with points title 'psi_int';"
            commandline += " plot '"+self.iter_filebasename+".dat' using 1:2 with points title 'psi_int', '"+self.iter_filebasename+".dat' using (\$2=='inf'?\$1:1/0):(GPVAL_Y_MIN+(0.8)*(GPVAL_Y_MAX-GPVAL_Y_MIN)):(0):((0.2)*(GPVAL_Y_MAX-GPVAL_Y_MIN)) with vectors notitle\""
            os.system(commandline)

        self.relax = relax_list[numpy.argmin(ener_list)]
        self.printer.print_sci("relax",self.relax)
        if (self.relax == 0.):
            self.printer.print_str("Warning! Optimal relaxation is null…")
