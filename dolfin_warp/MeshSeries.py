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



class MeshSeries():



    def __init__(self,
            problem,
            folder,
            basename,
            n_frames=None,
            ext="vtu"):

        self.problem       = problem
        self.printer       = self.problem.printer
        self.folder        = folder
        self.basename      = basename
        self.n_frames      = n_frames
        self.ext           = ext

        self.printer.print_str("Reading mesh series…")
        self.printer.inc()

        self.filenames = glob.glob(self.folder+"/"+self.basename+"_[0-9]*"+"."+self.ext)
        assert (len(self.filenames) >= 1),\
            "Not enough meshes ("+self.folder+"/"+self.basename+"_[0-9]*"+"."+self.ext+"). Aborting."

        if (self.n_frames is None):
            self.n_frames = len(self.filenames)
        else:
            assert (self.n_frames <= len(self.filenames))
        assert (self.n_frames >= 1),\
            "n_frames = "+str(self.n_frames)+" < 2. Aborting."
        self.printer.print_var("n_frames",self.n_frames)

        self.zfill = len(self.filenames[0].rsplit("_",1)[-1].split(".",1)[0])

        self.printer.dec()



    def get_mesh_filename(self,
            k_frame):

        return self.folder+"/"+self.basename+"_"+str(k_frame).zfill(self.zfill)+"."+self.ext



    def get_mesh(self,
            k_frame):

        return myvtk.readDataSet(
            filename=self.get_mesh_filename(
                k_frame=k_frame))
