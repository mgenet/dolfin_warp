#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2016-2024                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

import glob
import dolfin
import meshio
import myPythonLibrary    as mypy
import myVTKPythonLibrary as myvtk

from .FilesSeries import FilesSeries
from .Problem         import Problem

################################################################################

class MappingsSeries(FilesSeries):



    def __init__(self,
            folder          : str,
            basename        : str,
            ext             : str       = "vti",
            verbose         : bool      = True,
            printer                     = None,
            warping_type                = "barycenter", 
            problem                     = Problem, 
            U_family        : str       = "Lagrange",
            U_degree        : int       = 1 ):

        self.folder        = folder
        self.basename      = basename
        self.ext           = ext

        self.verbose = verbose
        if (printer is None):
            self.printer = mypy.Printer()
        else:
            self.printer = printer

        if (verbose): self.printer.print_str("Reading mappings series…")
        if (verbose): self.printer.inc()

        self.filenames = glob.glob(self.folder+"/"+self.basename+"_[0-9]*"+"."+self.ext)



        self.n_mappings = len(self.filenames)

        if (verbose): self.printer.print_var("n_mappings",self.n_mappings)

        self.zfill = len(self.filenames[0].rsplit("_",1)[-1].split(".",1)[0])
        if (verbose): self.printer.print_var("zfill",self.zfill)



        self.U_family = U_family
        self.U_degree = U_degree
        self.U_fe = dolfin.VectorElement(
            family=self.U_family,
            cell=problem.mesh.ufl_cell(),
            degree=self.U_degree)
        self.U_fs = dolfin.FunctionSpace(
            problem.mesh,
            self.U_fe)



        self.printer.print_str("Defining measure…")

        # dV

        self.dV = dolfin.Measure(
            "dx",
            domain=problem.mesh)


    def get_image_filename(self,
            k_frame=None,
            suffix=None,
            ext=None):

        return self.folder+"/"+self.basename+("-"+suffix if bool(suffix) else "")+("_"+str(k_frame).zfill(self.zfill) if (k_frame is not None) else "")+"."+(ext if bool(ext) else self.ext)



    def get_mapping(self,
            k_mapping):

        loaded_mapping  = meshio.read(self.filenames[k_mapping-1])
        u_mapping       = loaded_mapping.point_data["displacement"]   

        u = dolfin.Function(
            self.U_fs,
            name="mapping")
        u.vector()[:] = u_mapping.flatten()
        return u



    def get_image_grad_filename(self,
            k_frame=None,
            suffix=None,
            ext=None):

        if (self.grad_basename is None):
            return self.get_image_filename(k_frame, suffix)
        else:
            return self.grad_folder+"/"+self.grad_basename+("-"+suffix if bool(suffix) else "")+("_"+str(k_frame).zfill(self.zfill) if (k_frame is not None) else "")+"."+(ext if bool(ext) else self.ext)



    def get_image_grad(self,
            k_frame):

        return myvtk.readImage(
            filename=self.get_image_grad_filename(k_frame))
