#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2016-2021                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

import dolfin
import glob

import myPythonLibrary    as mypy
import myVTKPythonLibrary as myvtk

import dolfin_warp as dwarp

################################################################################



class ImageSeries():



    def __init__(self,
            problem,
            folder,
            basename,
            grad_folder=None,
            grad_basename=None,
            n_frames=None,
            ext="vti"):

        self.problem       = problem
        self.printer       = self.problem.printer
        self.folder        = folder
        self.basename      = basename
        self.grad_folder   = grad_folder
        self.grad_basename = grad_basename
        self.n_frames      = n_frames
        self.ext           = ext

        self.printer.print_str("Reading image series…")
        self.printer.inc()

        self.filenames = glob.glob(self.folder+"/"+self.basename+"_[0-9]*"+"."+self.ext)
        assert (len(self.filenames) >= 2),\
            "Not enough images ("+self.folder+"/"+self.basename+"_[0-9]*"+"."+self.ext+"). Aborting."

        if (self.n_frames is None):
            self.n_frames = len(self.filenames)
        else:
            assert (self.n_frames <= len(self.filenames))
        assert (self.n_frames >= 1),\
            "n_frames = "+str(self.n_frames)+" < 2. Aborting."
        self.printer.print_var("n_frames",self.n_frames)

        self.zfill = len(self.filenames[0].rsplit("_",1)[-1].split(".",1)[0])

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
        self.printer.print_var("dimension",self.dimension)

        self.printer.dec()



    def get_image_filename(self,
            k_frame):

        return self.folder+"/"+self.basename+"_"+str(k_frame).zfill(self.zfill)+"."+self.ext



    def get_image_grad_filename(self,
            k_frame):

        if (self.grad_basename is None):
            return self.get_image_filename(k_frame)
        else:
            return self.grad_folder+"/"+self.grad_basename+"_"+str(k_frame).zfill(self.zfill)+"."+self.ext
