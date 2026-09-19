#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2016-2026                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

import glob

import myPythonLibrary    as mypy
import myVTKPythonLibrary as myvtk

from .FileSeries import FileSeries

################################################################################

class ImageSeries(FileSeries):



    def __init__(self,
            folder        : str          ,
            basename      : str          ,
            grad_folder   : str  = None  ,
            grad_basename : str  = None  ,
            n_frames      : int  = None  ,
            zfill         : int  = None  ,
            ext           : str  = "vti" ,
            verbose       : bool = True  ,
            printer              = None  ):

        self.folder        = folder
        self.basename      = basename
        self.grad_folder   = grad_folder
        self.grad_basename = grad_basename
        self.n_frames      = n_frames
        self.zfill         = zfill
        self.ext           = ext

        self.verbose = verbose
        if (printer is None):
            self.printer = mypy.Printer()
        else:
            self.printer = printer

        if (self.n_frames is not None) and (self.zfill is not None): # the series is declared, not read, so that the images do not need to exist yet

            if (verbose): self.printer.print_str("Declaring image series…")
            if (verbose): self.printer.inc()

            assert (self.n_frames >= 1),\
                "n_frames = "+str(self.n_frames)+" < 1. Aborting."
            if (verbose): self.printer.print_var("n_frames",self.n_frames)
            if (verbose): self.printer.print_var("zfill",self.zfill)

            self.filenames = [self.get_image_filename(k_frame=k_frame) for k_frame in range(self.n_frames)]

            if (self.grad_basename is not None):
                if (self.grad_folder is None):
                    self.grad_folder = self.folder
                self.grad_filenames = [self.get_image_grad_filename(k_frame=k_frame) for k_frame in range(self.n_frames)]

            self.dimension = None # the images do not exist yet, so this cannot be determined

            if (verbose): self.printer.dec()

        else:

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
            k_frame = None ,
            suffix  = None ,
            sep     = "-"  ,
            ext     = None ):

        return self.folder+"/"+self.basename+(sep+suffix if bool(suffix) else "")+("_"+str(k_frame).zfill(self.zfill) if (k_frame is not None) else "")+"."+(ext if bool(ext) else self.ext)



    def get_image(self,
            k_frame):

        return myvtk.readImage(
            filename=self.get_image_filename(k_frame))



    def get_image_grad_filename(self,
            k_frame = None ,
            suffix  = None ,
            sep     = "-"  ,
            ext     = None ):

        if (self.grad_basename is None):
            return self.get_image_filename(k_frame, suffix, sep)
        else:
            return self.grad_folder+"/"+self.grad_basename+(sep+suffix if bool(suffix) else "")+("_"+str(k_frame).zfill(self.zfill) if (k_frame is not None) else "")+"."+(ext if bool(ext) else self.ext)



    def get_image_grad(self,
            k_frame):

        return myvtk.readImage(
            filename=self.get_image_grad_filename(k_frame))
