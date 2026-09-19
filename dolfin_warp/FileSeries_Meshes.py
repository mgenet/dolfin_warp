#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2016-2026                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

import glob
import vtk.numpy_interface.dataset_adapter as dsa

import myPythonLibrary    as mypy
import myVTKPythonLibrary as myvtk

from .FileSeries import FileSeries

################################################################################

class MeshSeries(FileSeries):



    def __init__(self,
            folder   : str          ,
            basename : str          ,
            n_frames : int  = None  ,
            zfill    : int  = None  ,
            ext      : str  = "vtu" ,
            verbose  : bool = True  ,
            printer         = None  ):

        self.folder   = folder
        self.basename = basename
        self.n_frames = n_frames
        self.zfill    = zfill
        self.ext      = ext

        self.verbose = verbose
        if (printer is None):
            self.printer = mypy.Printer()
        else:
            self.printer = printer

        if (self.n_frames is not None) and (self.zfill is not None): # the series is declared, not read, so that the meshes do not need to exist yet

            if (verbose): self.printer.print_str("Declaring mesh series…")
            if (verbose): self.printer.inc()

            assert (self.n_frames >= 1),\
                "n_frames = "+str(self.n_frames)+" < 1. Aborting."
            if (verbose): self.printer.print_var("n_frames",self.n_frames)
            if (verbose): self.printer.print_var("zfill",self.zfill)

            self.filenames = [self.get_mesh_filename(k_frame=k_frame) for k_frame in range(self.n_frames)]

            if (verbose): self.printer.dec()

        else:

            if (verbose): self.printer.print_str("Reading mesh series…")
            if (verbose): self.printer.inc()

            self.filenames = glob.glob(self.folder+"/"+self.basename+"_[0-9]*"+"."+self.ext)
            assert (len(self.filenames) >= 1),\
                "Not enough meshes ("+self.folder+"/"+self.basename+"_[0-9]*"+"."+self.ext+"). Aborting."

            if (self.n_frames is None):
                self.n_frames = len(self.filenames)
            else:
                assert (self.n_frames <= len(self.filenames))
            assert (self.n_frames >= 1),\
                "n_frames = "+str(self.n_frames)+" < 2. Aborting."
            if (verbose): self.printer.print_var("n_frames",self.n_frames)

            self.zfill = len(self.filenames[0].rsplit("_",1)[-1].split(".",1)[0])
            if (verbose): self.printer.print_var("zfill",self.zfill)

            if (verbose): self.printer.dec()



    def get_mesh_filebasename(self,
            k_frame = None ,
            suffix  = None ,
            sep     = "-"  ):

        return self.folder+"/"+self.basename+(sep+suffix if bool(suffix) else "")+("_"+str(k_frame).zfill(self.zfill) if (k_frame is not None) else "")



    def get_mesh_filename(self,
            k_frame = None ,
            suffix  = None ,
            sep     = "-"  ,
            ext     = None ):

        return self.get_mesh_filebasename(
            k_frame=k_frame,
            suffix=suffix,
            sep=sep)+"."+(ext if bool(ext) else self.ext)



    def get_mesh(self,
            k_frame):

        return myvtk.readDataSet(
            filename=self.get_mesh_filename(
                k_frame=k_frame))



    def get_np_mesh(self,
            k_frame):

        return dsa.WrapDataObject(
            myvtk.readDataSet(
                filename=self.get_mesh_filename(
                    k_frame=k_frame)))
