#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2016-2022                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

import glob

import myPythonLibrary    as mypy
import myVTKPythonLibrary as myvtk

from .FilesSeries import FilesSeries

################################################################################

class MeshesSeries(FilesSeries):



    def __init__(self,
            folder  : str         ,
            basename: str         ,
            n_frames       = None ,
            ext     : str  = "vtu",
            verbose : bool = True ,
            printer        = None ):

        self.folder   = folder
        self.basename = basename
        self.n_frames = n_frames
        self.ext      = ext

        self.verbose = verbose
        if (printer is None):
            self.printer = mypy.Printer()
        else:
            self.printer = printer

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
            k_frame = None,
            suffix = None):

        return self.folder+"/"+self.basename+("-"+suffix if bool(suffix) else "")+("_"+str(k_frame).zfill(self.zfill) if (k_frame is not None) else "")



    def get_mesh_filename(self,
            k_frame = None,
            suffix = None,
            ext = None):

        return self.get_mesh_filebasename(
            k_frame=k_frame,
            suffix=suffix)+"."+(ext if bool(ext) else self.ext)



    def get_mesh(self,
            k_frame):

        return myvtk.readDataSet(
            filename=self.get_mesh_filename(
                k_frame=k_frame))
