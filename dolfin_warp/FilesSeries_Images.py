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

class ImagesSeries(FilesSeries):



    def __init__(self,
            folder: str,
            basename: str,
            grad_folder = None,
            grad_basename = None,
            n_frames = None,
            ext: str = "vti",
            verbose: bool = True,
            printer = None):

        self.folder        = folder
        self.basename      = basename
        self.grad_folder   = grad_folder
        self.grad_basename = grad_basename
        self.n_frames      = n_frames
        self.ext           = ext

        self.verbose = verbose
        if (printer is None):
            self.printer = mypy.Printer()
        else:
            self.printer = printer

        if (verbose): self.printer.print_str("Reading image series…")
        if (verbose): self.printer.inc()

        self.filenames = glob.glob(self.folder+"/"+self.basename+"_[0-9]*"+"."+self.ext)
        assert (len(self.filenames) >= 2),\
            "Not enough images ("+self.folder+"/"+self.basename+"_[0-9]*"+"."+self.ext+"). Aborting."

        if (self.n_frames is None):
            self.n_frames = len(self.filenames)
        else:
            assert (self.n_frames <= len(self.filenames))
        assert (self.n_frames >= 1),\
            "n_frames = "+str(self.n_frames)+" < 2. Aborting."
        if (verbose): self.printer.print_var("n_frames",self.n_frames)

        self.zfill = len(self.filenames[0].rsplit("_",1)[-1].split(".",1)[0])
        if (verbose): self.printer.print_var("zfill",self.zfill)

        if (self.grad_basename is not None):
            if (self.grad_folder is None):
                self.grad_folder = self.folder
            self.grad_filenames = glob.glob(self.grad_folder+"/"+self.grad_basename+"_[0-9]*"+"."+self.ext)
            assert (len(self.grad_filenames) >= self.n_frames)

        image = myvtk.readImage(
            filename=self.get_image_filename(
                k_frame=0),
            verbose=0)
        self.dimension = myvtk.getImageDimensionality(
            image=image,
            verbose=0)
        if (verbose): self.printer.print_var("dimension",self.dimension)

        if (verbose): self.printer.dec()



    def get_image_filename(self,
            k_frame=None,
            suffix=None,
            ext=None):

        return self.folder+"/"+self.basename+("-"+suffix if bool(suffix) else "")+("_"+str(k_frame).zfill(self.zfill) if (k_frame is not None) else "")+"."+(ext if bool(ext) else self.ext)



    def get_image(self,
            k_frame):

        return myvtk.readImage(
            filename=self.get_image_filename(k_frame))



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
