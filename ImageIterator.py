#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2016-2018                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

import dolfin
import glob
import numpy
import os
import time
import vtk

import myPythonLibrary as mypy
import myVTKPythonLibrary as myvtk

import dolfin_dic as ddic

################################################################################

class ImageIterator():



    def __init__(self,
            problem,
            solver,
            parameters={}):

        self.problem = problem
        self.printer = self.problem.printer
        self.solver  = solver

        self.working_folder           = parameters["working_folder"]           if ("working_folder"           in parameters) else "."
        self.working_basename         = parameters["working_basename"]         if ("working_basename"         in parameters) else "sol"
        self.initialize_U_from_file   = parameters["initialize_U_from_file"]   if ("initialize_U_from_file"   in parameters) else False
        self.initialize_U_folder      = parameters["initialize_U_folder"]      if ("initialize_U_folder"      in parameters) else "."
        self.initialize_U_basename    = parameters["initialize_U_basename"]    if ("initialize_U_basename"    in parameters) else None
        self.initialize_U_ext         = parameters["initialize_U_ext"]         if ("initialize_U_ext"         in parameters) else "vtu"
        self.initialize_U_array_name  = parameters["initialize_U_array_name"]  if ("initialize_U_array_name"  in parameters) else "displacement"
        self.initialize_DU_with_DUold = parameters["initialize_DU_with_DUold"] if ("initialize_DU_with_DUold" in parameters) else False



    def iterate(self):

        self.printer.print_str("Writing initial solution…")
        self.printer.inc()

        if not os.path.exists(self.working_folder):
            os.mkdir(self.working_folder)
        pvd_basename = self.working_folder+"/"+self.working_basename
        for vtu_filename in glob.glob(pvd_basename+"_[0-9]*.vtu"):
            os.remove(vtu_filename)
        ddic.write_VTU_file(
            filebasename=pvd_basename,
            function=self.problem.U,
            time=self.problem.images_ref_frame)

        self.printer.dec()
        self.printer.print_str("Initializing QOI file…")
        self.printer.inc()

        qoi_names = ["k_frame"]+self.problem.get_qoi_names()
        qoi_filebasename = self.working_folder+"/"+self.working_basename+"-qoi"
        qoi_printer = mypy.DataPrinter(
            names=qoi_names,
            filename=qoi_filebasename+".dat")
        qoi_values = [self.problem.images_ref_frame]+self.problem.get_qoi_values()
        qoi_printer.write_line(
            values=qoi_values)

        self.printer.dec()
        self.printer.print_str("Looping over frames…")

        if (self.initialize_U_from_file):
            mesh_series = ddic.MeshSeries(
                problem=self.problem,
                folder=self.initialize_U_folder,
                basename=self.initialize_U_basename,
                ext=self.initialize_U_ext)

        n_iter_tot = 0
        global_success = True
        for forward_or_backward in ["forward","backward"]:
            self.printer.print_var("forward_or_backward",forward_or_backward)

            if   (forward_or_backward == "forward"):
                k_frames = range(self.problem.images_ref_frame+1, self.problem.images_n_frames, +1)
            elif (forward_or_backward == "backward"):
                k_frames = range(self.problem.images_ref_frame-1,                           -1, -1)
            #self.printer.print_var("k_frames",k_frames)

            if (forward_or_backward == "backward"):
                self.problem.reinit()

            self.printer.inc()
            success = True
            for k_frame in k_frames:
                self.printer.print_var("k_frame",k_frame,-1)

                if   (forward_or_backward == "forward"):
                    k_frame_old = k_frame-1
                elif (forward_or_backward == "backward"):
                    k_frame_old = k_frame+1
                #self.printer.print_var("k_frame_old",k_frame_old,-1)

                if (self.initialize_U_from_file):
                    mesh = mesh_series.get_mesh(k_frame)
                    array_U = mesh.GetPointData().GetArray(self.initialize_U_array_name)
                    array_U = vtk.util.numpy_support.vtk_to_numpy(array_U)
                    array_U = array_U.astype(float)
                    array_U = numpy.reshape(array_U, array_U.size)
                    self.problem.U.vector()[:] = array_U[dolfin.dof_to_vertex_map(self.problem.U_fs)]

                elif (self.initialize_DU_with_DUold):
                    self.problem.U.vector().axpy(1., self.problem.DUold.vector())

                self.problem.call_before_solve(
                    k_frame=k_frame,
                    k_frame_old=k_frame_old)

                self.printer.print_str("Running registration…")

                success, n_iter = self.solver.solve(
                    k_frame=k_frame)
                n_iter_tot += n_iter

                if not (success):
                    global_success = False
                    break

                self.problem.call_after_solve()

                self.printer.print_str("Writing solution…")
                self.printer.inc()

                ddic.write_VTU_file(
                    filebasename=pvd_basename,
                    function=self.problem.U,
                    time=k_frame)

                self.printer.dec()
                self.printer.print_str("Writing QOI file…")
                self.printer.inc()

                qoi_printer.write_line(
                    [k_frame]+self.problem.get_qoi_values())

                self.printer.dec()

            self.printer.dec()

            if not (global_success):
                break

        self.printer.print_str("Image iterator finished…")
        self.printer.inc()

        self.printer.print_var("n_iter_tot",n_iter_tot)

        self.printer.dec()
        self.printer.print_str("Plotting QOI…")

        qoi_printer.close()
        commandline  = "gnuplot -e \"set terminal pdf;"
        commandline += " set output '"+qoi_filebasename+".pdf';"
        commandline += " set grid;"
        for k_qoi in xrange(1,len(qoi_names)):
            commandline += " plot '"+qoi_filebasename+".dat' u 1:"+str(1+k_qoi)+" lw 3 title '"+qoi_names[k_qoi]+"';"
        commandline += "\""
        os.system(commandline)

        return global_success
