#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2016-2018                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

import glob
import math
import numpy
import os
import random
import vtk

import myPythonLibrary as mypy
import myVTKPythonLibrary as myvtk

import dolfin_dic as ddic

################################################################################

#class ImagesInfo():
    #def __init__(self, n_dim, L, n_voxels, n_integration, T, n_frames, data_type, images_folder, images_basename):
        #assert (n_dim in (1,2,3))
        #self.n_dim = n_dim

        #if (type(L) == float):
            #assert (L>0)
            #self.L = numpy.array([L]*self.n_dim)
        #elif (type(L) == int):
            #assert (L>0)
            #self.L = numpy.array([float(L)]*self.n_dim)
        #else:
            #assert (len(L) == self.n_dim)
            #self.L = numpy.array(L)
            #assert ((self.L>0).all())

        #if (type(n_voxels) == int):
            #assert (n_voxels>0)
            #self.n_voxels = numpy.array([n_voxels]*self.n_dim)
        #else:
            #assert (len(n_voxels) == self.n_dim)
            #self.n_voxels = numpy.array(n_voxels)
            #assert ((self.n_voxels>0).all())

        #if (type(n_integration) == int):
            #assert (n_integration>0)
            #self.n_integration = numpy.array([n_integration]*self.n_dim)
        #else:
            #assert (len(n_integration) == self.n_dim)
            #self.n_integration = numpy.array(n_integration)
            #assert ((self.n_integration>0).all())

        #assert (T>0.)
        #self.T = T

        #assert (n_frames>0)
        #self.n_frames = n_frames

        #assert (data_type in ("int", "float", "unsigned char", "unsigned short", "unsigned int", "unsigned long", "unsigned float" "uint8", "uint16", "uint32", "uint64", "ufloat"))
        #self.data_type = data_type

        #self.images_folder = images_folder
        #self.images_basename = images_basename

#class StructureInfo():
    #def __init__(self, images, type, **kwargs):
        #assert (type in ("no", "heart"))
        #self["type"] = type
        #if (self["type"] == "heart"):
            #self.Ri = kwargs["Ri"]
            #self.Re = kwargs["Re"]
            #if (images.n_dim == 3):
                #self.Zmin = kwargs["Zmin"] if ("Zmin" in kwargs.keys()) else 0.
                #self.Zmax = kwargs["Zmax"] if ("Zmax" in kwargs.keys()) else images.L[2]

#class TextureInfo():
    #def __init__(self, type, **kwargs):
        #assert (type in ("no", "taggX", "taggY", "taggZ"))
        #self["type"] = type

#class NoiseInfo():
    #def __init__(self, type, **kwargs):
        #self["type"] = type

#class DeformationInfo():
    #def __init__(self, type, **kwargs):
        #self["type"] = type

#class EvolutionInfo():
    #def __init__(self, type, **kwargs):
        #self["type"] = type

################################################################################

class Image():
    def __init__(self, images, structure, texture, noise):
        self.L = images["L"]

        # structure
        if (structure["type"] == "no"):
            self.I0_structure = self.I0_structure_no
        elif (structure["type"] == "heart"):
            if (images["n_dim"] == 2):
                self.I0_structure = self.I0_structure_heart_2
                self.R = float()
                self.Ri = structure["Ri"]
                self.Re = structure["Re"]
            elif (images["n_dim"] == 3):
                self.I0_structure = self.I0_structure_heart_3
                self.R = float()
                self.Ri = structure["Ri"]
                self.Re = structure["Re"]
                self.Zmin = structure.Zmin if ("Zmin" in structure.keys()) else 0.
                self.Zmax = structure.Zmax if ("Zmax" in structure.keys()) else images["L"][2]
            else:
                assert (0), "n_dim must be \"2\" or \"3 for \"heart\" type structure. Aborting."
        else:
            assert (0), "structure type must be \"no\" or \"heart\". Aborting."

        # texture
        if (texture["type"] == "no"):
            self.I0_texture = self.I0_texture_no
        elif (texture["type"].startswith("tagging")):
            if   (images["n_dim"] == 1):
                if ("-signed" in texture["type"]):
                    if ("-normalized" in texture["type"]):
                        self.I0_texture = self.I0_texture_tagging_signed_normalized_X
                    else:
                        self.I0_texture = self.I0_texture_tagging_signed_X
                else:
                    self.I0_texture = self.I0_texture_tagging_X
            elif (images["n_dim"] == 2):
                if ("-signed" in texture["type"]):
                    if ("-normalized" in texture["type"]):
                        self.I0_texture = self.I0_texture_tagging_signed_normalized_XY
                    else:
                        self.I0_texture = self.I0_texture_tagging_signed_XY
                else:
                    self.I0_texture = self.I0_texture_tagging_XY
            elif (images["n_dim"] == 3):
                if ("-signed" in texture["type"]):
                    if ("-normalized" in texture["type"]):
                        self.I0_texture = self.I0_texture_tagging_signed_normalized_XYZ
                    else:
                        self.I0_texture = self.I0_texture_tagging_signed_XYZ
                else:
                    self.I0_texture = self.I0_texture_tagging_XYZ
            else:
                assert (0), "n_dim must be \"1\", \"2\" or \"3\". Aborting."
            self.s = texture["s"]
        elif (texture["type"].startswith("taggX")):
            if ("-signed" in texture["type"]):
                if ("-normalized" in texture["type"]):
                    self.I0_texture = self.I0_texture_tagging_signed_normalized_X
                else:
                    self.I0_texture = self.I0_texture_tagging_signed_X
            else:
                self.I0_texture = self.I0_texture_tagging_X
            self.s = texture["s"]
        elif (texture["type"].startswith("taggY")):
            if ("-signed" in texture["type"]):
                if ("-normalized" in texture["type"]):
                    self.I0_texture = self.I0_texture_tagging_signed_normalized_Y
                else:
                    self.I0_texture = self.I0_texture_tagging_signed_Y
            else:
                self.I0_texture = self.I0_texture_tagging_Y
            self.s = texture["s"]
        elif (texture["type"].startswith("taggZ")):
            if ("-signed" in texture["type"]):
                if ("-normalized" in texture["type"]):
                    self.I0_texture = self.I0_texture_tagging_signed_normalized_Z
                else:
                    self.I0_texture = self.I0_texture_tagging_signed_Z
            else:
                self.I0_texture = self.I0_texture_tagging_Z
            self.s = texture["s"]
        else:
            assert (0), "texture type must be \"no\", \"tagging\", \"taggX\", \"taggY\" or \"taggZ\". Aborting."

        # noise
        if (noise["type"] == "no"):
            self.I0_noise = self.I0_noise_no
        elif (noise["type"] == "normal"):
            self.I0_noise = self.I0_noise_normal
            self.avg = noise["avg"] if ("avg" in noise.keys()) else 0.
            self.std = noise["stdev"]
        else:
            assert (0), "noise type must be \"no\" or \"normal\". Aborting."

    def I0(self, X, i, g=None):
        self.I0_structure(X, i, g)
        self.I0_texture(X, i, g)
        self.I0_noise(i, g)

    def I0_structure_no(self, X, i, g=None):
        i[0] = 1.
        if (g is not None): g[:] = 1. # MG 20180806: gradient is given by texture; here it is just indicator function

    def I0_structure_heart_2(self, X, i, g=None):
        self.R = ((X[0]-self.L[0]/2)**2 + (X[1]-self.L[1]/2)**2)**(1./2)
        if (self.R >= self.Ri) and (self.R <= self.Re):
            i[0] = 1.
            if (g is not None): g[:] = 1. # MG 20180806: gradient is given by texture; here it is just indicator function
        else:
            i[0] = 0.
            if (g is not None): g[:] = 0.

    def I0_structure_heart_3(self, X, i, g=None):
        self.R = ((X[0]-self.L[0]/2)**2 + (X[1]-self.L[1]/2)**2)**(1./2)
        if (self.R >= self.Ri) and (self.R <= self.Re) and (X[2] >= self.Zmin) and (X[2] <= self.Zmax):
            i[0] = 1.
            if (g is not None): g[:] = 1. # MG 20180806: gradient is given by texture; here it is just indicator function
        else:
            i[0] = 0.
            if (g is not None): g[:] = 0.

    def I0_texture_no(self, X, i, g=None):
        i[0] *= 1.
        if (g is not None): g[:] *= 0.

    def I0_texture_tagging_X(self, X, i, g=None):
        i[0] *= abs(math.sin(math.pi*X[0]/self.s))
        if (g is not None):
            g[0] *= math.copysign(1, math.sin(math.pi*X[0]/self.s)) * (math.pi/self.s)*math.cos(math.pi*X[0]/self.s)
            g[1] *= 0
            g[2] *= 0

    def I0_texture_tagging_Y(self, X, i, g=None):
        i[0] *= abs(math.sin(math.pi*X[1]/self.s))
        if (g is not None):
            g[0] *= 0
            g[1] *= math.copysign(1, math.sin(math.pi*X[1]/self.s)) * (math.pi/self.s)*math.cos(math.pi*X[1]/self.s)
            g[2] *= 0

    def I0_texture_tagging_Z(self, X, i, g=None):
        i[0] *= abs(math.sin(math.pi*X[2]/self.s))
        if (g is not None):
            g[0] *= 0
            g[1] *= 0
            g[2] *= math.copysign(1, math.sin(math.pi*X[2]/self.s)) * (math.pi/self.s)*math.cos(math.pi*X[2]/self.s)

    def I0_texture_tagging_XY(self, X, i, g=None):
        i[0] *= (1 + 3 * abs(math.sin(math.pi*X[0]/self.s))
                       * abs(math.sin(math.pi*X[1]/self.s)))**(1./2) - 1
        if (g is not None):
            g[0] *= 3 * math.copysign(1, math.sin(math.pi*X[0]/self.s)) * (math.pi/self.s)*math.cos(math.pi*X[0]/self.s) * abs(math.sin(math.pi*X[1]/self.s)) / 2 / (i[0] + 1)
            g[1] *= 3 * abs(math.sin(math.pi*X[0]/self.s)) * math.copysign(1, math.sin(math.pi*X[1]/self.s)) * (math.pi/self.s)*math.cos(math.pi*X[1]/self.s) / 2 / (i[0] + 1)
            g[2] *= 0

    #def I0_texture_tagging_XY(self, X, i, g=None):
        #i[0] *= (abs(math.sin(math.pi*X[0]/self.s))
              #+  abs(math.sin(math.pi*X[1]/self.s)))/2
        #if (g is not None):
            #g[0] *= math.copysign(1, math.sin(math.pi*X[0]/self.s)) * (math.pi/self.s)*math.cos(math.pi*X[0]/self.s) / 2
            #g[1] *= math.copysign(1, math.sin(math.pi*X[1]/self.s)) * (math.pi/self.s)*math.cos(math.pi*X[1]/self.s) / 2
            #g[2] *= 0

    def I0_texture_tagging_XYZ(self, X, i, g=None):
        i[0] *= (1 + 7 * abs(math.sin(math.pi*X[0]/self.s))
                       * abs(math.sin(math.pi*X[1]/self.s))
                       * abs(math.sin(math.pi*X[2]/self.s)))**(1./3) - 1
        if (g is not None):
            g[0] *= 7 * math.copysign(1, math.sin(math.pi*X[0]/self.s)) * (math.pi/self.s)*math.cos(math.pi*X[0]/self.s) * abs(math.sin(math.pi*X[1]/self.s)) * abs(math.sin(math.pi*X[2]/self.s)) / 3 / (i[0] + 1)
            g[1] *= 7 * abs(math.sin(math.pi*X[0]/self.s)) * math.copysign(1, math.sin(math.pi*X[1]/self.s)) * (math.pi/self.s)*math.cos(math.pi*X[1]/self.s) * abs(math.sin(math.pi*X[2]/self.s)) / 3 / (i[0] + 1)
            g[2] *= 7 * abs(math.sin(math.pi*X[0]/self.s)) * abs(math.sin(math.pi*X[1]/self.s)) * math.copysign(1, math.sin(math.pi*X[2]/self.s)) * (math.pi/self.s)*math.cos(math.pi*X[2]/self.s) / 3 / (i[0] + 1)

    #def I0_texture_tagging_XYZ(self, X, i, g=None):
        #i[0] *= (abs(math.sin(math.pi*X[0]/self.s))
              #+  abs(math.sin(math.pi*X[1]/self.s))
              #+  abs(math.sin(math.pi*X[2]/self.s)))/3
        #if (g is not None):
            #g[0] *= math.copysign(1, math.sin(math.pi*X[0]/self.s)) * (math.pi/self.s)*math.cos(math.pi*X[0]/self.s) / 3
            #g[1] *= math.copysign(1, math.sin(math.pi*X[1]/self.s)) * (math.pi/self.s)*math.cos(math.pi*X[1]/self.s) / 3
            #g[2] *= math.copysign(1, math.sin(math.pi*X[2]/self.s)) * (math.pi/self.s)*math.cos(math.pi*X[2]/self.s) / 3

    def I0_texture_tagging_signed_X(self, X, i, g=None):
        i[0] *= math.sin(math.pi*X[0]/self.s)
        if (g is not None):
            g[0] *= (math.pi/self.s)*math.cos(math.pi*X[0]/self.s)
            g[1] *= 0
            g[2] *= 0

    def I0_texture_tagging_signed_Y(self, X, i, g=None):
        i[0] *= math.sin(math.pi*X[1]/self.s)
        if (g is not None):
            g[0] *= 0
            g[1] *= (math.pi/self.s)*math.cos(math.pi*X[1]/self.s)
            g[2] *= 0

    def I0_texture_tagging_signed_Z(self, X, i, g=None):
        i[0] *= math.sin(math.pi*X[2]/self.s)
        if (g is not None):
            g[0] *= 0
            g[1] *= 0
            g[2] *= (math.pi/self.s)*math.cos(math.pi*X[2]/self.s)

    def I0_texture_tagging_signed_XY(self, X, i, g=None):
        i[0] *= (math.sin(math.pi*X[0]/self.s)
              +  math.sin(math.pi*X[1]/self.s))/2
        if (g is not None):
            g[0] *= (math.pi/self.s)*math.cos(math.pi*X[0]/self.s) / 2
            g[1] *= (math.pi/self.s)*math.cos(math.pi*X[1]/self.s) / 2
            g[2] *= 0

    def I0_texture_tagging_signed_XYZ(self, X, i, g=None):
        i[0] *= (math.sin(math.pi*X[0]/self.s)
              +  math.sin(math.pi*X[1]/self.s)
              +  math.sin(math.pi*X[2]/self.s))/3
        if (g is not None):
            g[0] *= (math.pi/self.s)*math.cos(math.pi*X[0]/self.s) / 3
            g[1] *= (math.pi/self.s)*math.cos(math.pi*X[1]/self.s) / 3
            g[2] *= (math.pi/self.s)*math.cos(math.pi*X[2]/self.s) / 3

    def I0_texture_tagging_signed_normalized_X(self, X, i, g=None):
        i[0] *= (1+math.sin(math.pi*X[0]/self.s))/2
        if (g is not None):
            g[0] *= (math.pi/self.s)*math.cos(math.pi*X[0]/self.s)/2
            g[1] *= 0
            g[2] *= 0

    def I0_texture_tagging_signed_normalized_Y(self, X, i, g=None):
        i[0] *= (1+math.sin(math.pi*X[1]/self.s))/2
        if (g is not None):
            g[0] *= 0
            g[1] *= (math.pi/self.s)*math.cos(math.pi*X[1]/self.s)/2
            g[2] *= 0

    def I0_texture_tagging_signed_normalized_Z(self, X, i, g=None):
        i[0] *= (1+math.sin(math.pi*X[2]/self.s))/2
        if (g is not None):
            g[0] *= 0
            g[1] *= 0
            g[2] *= (math.pi/self.s)*math.cos(math.pi*X[2]/self.s)/2

    def I0_texture_tagging_signed_normalized_XY(self, X, i, g=None):
        i[0] *= (1 + 3 * (1+math.sin(math.pi*X[0]/self.s))/2
                       * (1+math.sin(math.pi*X[1]/self.s))/2)**(1./2) - 1
        if (g is not None):
            g[0] *= 3 * (math.pi/self.s)*math.cos(math.pi*X[0]/self.s)/2 * (1+math.sin(math.pi*X[1]/self.s))/2 / 2 / (i[0] + 1)
            g[1] *= 3 * (1+math.sin(math.pi*X[0]/self.s))/2 * (math.pi/self.s)*math.cos(math.pi*X[1]/self.s)/2 / 2 / (i[0] + 1)
            g[2] *= 0

    #def I0_texture_tagging_signed_normalized_XY(self, X, i, g=None):
        #i[0] *= ((1+math.sin(math.pi*X[0]/self.s))/2
              #+  (1+math.sin(math.pi*X[1]/self.s))/2)/2
        #if (g is not None):
            #g[0] *= (math.pi/self.s)*math.cos(math.pi*X[0]/self.s)/2 / 2
            #g[1] *= (math.pi/self.s)*math.cos(math.pi*X[1]/self.s)/2 / 2
            #g[2] *= 0.

    def I0_texture_tagging_signed_normalized_XYZ(self, X, i, g=None):
        i[0] *= (1 + 7 * (1+math.sin(math.pi*X[0]/self.s))/2
                       * (1+math.sin(math.pi*X[1]/self.s))/2
                       * (1+math.sin(math.pi*X[2]/self.s))/2)**(1./3) - 1
        if (g is not None):
            g[0] *= 7 * (math.pi/self.s)*math.cos(math.pi*X[0]/self.s)/2 * (1+math.sin(math.pi*X[1]/self.s))/2 * (1+math.sin(math.pi*X[2]/self.s))/2 / 3 / (i[0] + 1)
            g[1] *= 7 * (1+math.sin(math.pi*X[0]/self.s))/2 * (math.pi/self.s)*math.cos(math.pi*X[1]/self.s)/2 * (1+math.sin(math.pi*X[2]/self.s))/2 / 3 / (i[0] + 1)
            g[2] *= 7 * (1+math.sin(math.pi*X[0]/self.s))/2 * (1+math.sin(math.pi*X[1]/self.s))/2 * (math.pi/self.s)*math.cos(math.pi*X[2]/self.s)/2 / 3 / (i[0] + 1)

    #def I0_texture_tagging_signed_normalized_XYZ(self, X, i, g=None):
        #i[0] *= ((1+math.sin(math.pi*X[0]/self.s))/2
              #+  (1+math.sin(math.pi*X[1]/self.s))/2
              #+  (1+math.sin(math.pi*X[2]/self.s))/2)/3
        #if (g is not None):
            #g[0] *= (math.pi/self.s)*math.cos(math.pi*X[0]/self.s)/2 / 3
            #g[1] *= (math.pi/self.s)*math.cos(math.pi*X[1]/self.s)/2 / 3
            #g[2] *= (math.pi/self.s)*math.cos(math.pi*X[2]/self.s)/2 / 3

    def I0_noise_no(self, i, g=None):
        pass

    def I0_noise_normal(self, i, g=None):
        i[0] += random.normalvariate(self.avg, self.std)
        if (g is not None): g[k] += [2*random.normalvariate(self.avg, self.std) for k in xrange(len(g))]

################################################################################

class Mapping:
    def __init__(self, images, structure, deformation, evolution):
        self.deformation = deformation
        if (self.deformation["type"] == "no"):
            self.init_t = self.init_t_no
            self.X = self.X_no
            self.x = self.x_no
        elif (self.deformation["type"] == "translation"):
            self.init_t = self.init_t_translation
            self.X = self.X_translation
            self.x = self.x_translation
            self.D = numpy.empty(3)
        elif (self.deformation["type"] == "rotation"):
            self.init_t = self.init_t_rotation
            self.X = self.X_rotation
            self.x = self.x_rotation
            self.C = numpy.empty(3)
            self.R = numpy.empty((3,3))
            self.Rinv = numpy.empty((3,3))
        elif (self.deformation["type"] == "homogeneous"):
            self.init_t = self.init_t_homogeneous
            self.X = self.X_homogeneous
            self.x = self.x_homogeneous
        elif (self.deformation["type"] == "heart"):
            assert (structure["type"] == "heart"), "structure type must be \"heart\" for \"heart\" type deformation. Aborting."
            self.init_t = self.init_t_heart
            self.X = self.X_heart
            self.x = self.x_heart
            self.x_inplane = numpy.empty(2)
            self.X_inplane = numpy.empty(2)
            self.rt = numpy.empty(2)
            self.RT = numpy.empty(2)
            self.L = images["L"]
            self.Ri = structure["Ri"]
            self.Re = structure["Re"]
            self.R = numpy.empty((3,3))
        else:
            assert (0), "deformation type must be \"no\", \"translation\", \"rotation\", \"homogeneous\" or \"heart\". Aborting."

        if (evolution["type"] == "linear"):
            self.phi = self.phi_linear
        elif (evolution["type"] == "sine"):
            self.phi = self.phi_sine
            self.T = evolution["T"]
        else:
            assert (0), "evolution ("+evolution["type"]+") type must be \"linear\" or \"sine\". Aborting."

    def phi_linear(self, t):
        return t

    def phi_sine(self, t):
        return math.sin(math.pi*t/self.T)**2

    def init_t_no(self, t):
        pass

    def init_t_translation(self, t):
        self.D[0] = self.deformation["Dx"]*self.phi(t) if ("Dx" in self.deformation.keys()) else 0.
        self.D[1] = self.deformation["Dy"]*self.phi(t) if ("Dy" in self.deformation.keys()) else 0.
        self.D[2] = self.deformation["Dz"]*self.phi(t) if ("Dz" in self.deformation.keys()) else 0.

    def init_t_rotation(self, t):
        self.C[0] = self.deformation["Cx"] if ("Cx" in self.deformation.keys()) else 0.
        self.C[1] = self.deformation["Cy"] if ("Cy" in self.deformation.keys()) else 0.
        self.C[2] = self.deformation["Cz"] if ("Cz" in self.deformation.keys()) else 0.
        Rx = self.deformation["Rx"]*math.pi/180*self.phi(t) if ("Rx" in self.deformation.keys()) else 0.
        Ry = self.deformation["Ry"]*math.pi/180*self.phi(t) if ("Ry" in self.deformation.keys()) else 0.
        Rz = self.deformation["Rz"]*math.pi/180*self.phi(t) if ("Rz" in self.deformation.keys()) else 0.
        RRx = numpy.array([[          1. ,           0. ,           0. ],
                           [          0. , +math.cos(Rx), -math.sin(Rx)],
                           [          0. , +math.sin(Rx), +math.cos(Rx)]])
        RRy = numpy.array([[+math.cos(Ry),           0. , +math.sin(Ry)],
                           [          0. ,           1. ,           0. ],
                           [-math.sin(Ry),           0. , +math.cos(Ry)]])
        RRz = numpy.array([[+math.cos(Rz), -math.sin(Rz),           0. ],
                           [+math.sin(Rz), +math.cos(Rz),           0. ],
                           [          0. ,           0. ,           1. ]])
        self.R[:,:] = numpy.dot(numpy.dot(RRx, RRy), RRz)
        self.Rinv[:,:] = numpy.linalg.inv(self.R)

    def init_t_homogeneous(self, t):
        Exx = self.deformation["Exx"]*self.phi(t) if ("Exx" in self.deformation.keys()) else 0.
        Eyy = self.deformation["Eyy"]*self.phi(t) if ("Eyy" in self.deformation.keys()) else 0.
        Ezz = self.deformation["Ezz"]*self.phi(t) if ("Ezz" in self.deformation.keys()) else 0.
        Exy = self.deformation["Exy"]*self.phi(t) if ("Exy" in self.deformation.keys()) else 0.
        Eyx = self.deformation["Eyx"]*self.phi(t) if ("Eyx" in self.deformation.keys()) else 0.
        Exz = self.deformation["Exz"]*self.phi(t) if ("Exz" in self.deformation.keys()) else 0.
        Ezx = self.deformation["Ezx"]*self.phi(t) if ("Ezx" in self.deformation.keys()) else 0.
        Eyz = self.deformation["Eyz"]*self.phi(t) if ("Eyz" in self.deformation.keys()) else 0.
        Ezy = self.deformation["Ezy"]*self.phi(t) if ("Ezy" in self.deformation.keys()) else 0.
        self.F = numpy.array([[Exx, Exy, Exz],
                              [Eyx, Eyy, Eyz],
                              [Ezx, Ezy, Ezz]])
        self.F *= 2
        self.F += numpy.eye(3)
        w, v = numpy.linalg.eig(self.F)
        #assert (numpy.diag(numpy.dot(numpy.dot(numpy.transpose(v), self.F), v)) == w).all(), str(numpy.dot(numpy.dot(numpy.transpose(v), self.F), v))+" ≠ "+str(numpy.diag(w))+". Aborting."
        self.F = numpy.dot(numpy.dot(v, numpy.diag(numpy.sqrt(w))), numpy.transpose(v))
        self.Finv = numpy.linalg.inv(self.F)

    def init_t_heart(self, t):
        self.dRi = self.deformation["dRi"]*self.phi(t) if ("dRi" in self.deformation.keys()) else 0.
        self.dRe = self.deformation["dRi"]*self.phi(t) if ("dRi" in self.deformation.keys()) else 0.
        self.dTi = self.deformation["dTi"]*self.phi(t) if ("dTi" in self.deformation.keys()) else 0.
        self.dTe = self.deformation["dTe"]*self.phi(t) if ("dTe" in self.deformation.keys()) else 0.
        self.A = numpy.array([[1.-(self.dRi-self.dRe)/(self.Re-self.Ri), 0.],
                              [  -(self.dTi-self.dTe)/(self.Re-self.Ri), 1.]])
        self.Ainv = numpy.linalg.inv(self.A)
        self.B = numpy.array([(1.+self.Ri/(self.Re-self.Ri))*self.dRi-self.Ri/(self.Re-self.Ri)*self.dRe,
                              (1.+self.Ri/(self.Re-self.Ri))*self.dTi-self.Ri/(self.Re-self.Ri)*self.dTe])

    def X_no(self, x, X, Finv=None):
        X[:] = x
        if (Finv is not None): Finv[:,:] = numpy.identity(numpy.sqrt(numpy.size(Finv)))

    def X_translation(self, x, X, Finv=None):
        X[:] = x - self.D
        if (Finv is not None): Finv[:,:] = numpy.identity(numpy.sqrt(numpy.size(Finv)))

    def X_rotation(self, x, X, Finv=None):
        X[:] = numpy.dot(self.Rinv, x - self.C) + self.C
        if (Finv is not None): Finv[:,:] = self.Rinv

    def X_homogeneous(self, x, X, Finv=None):
        X[:] = numpy.dot(self.Finv, x)
        if (Finv is not None): Finv[:,:] = self.Finv

    def X_heart(self, x, X, Finv=None):
        #print "x = "+str(x)
        self.x_inplane[0] = x[0] - self.L[0]/2
        self.x_inplane[1] = x[1] - self.L[1]/2
        #print "x_inplane = "+str(self.x_inplane)
        self.rt[0] = numpy.linalg.norm(self.x_inplane)
        self.rt[1] = math.atan2(self.x_inplane[1], self.x_inplane[0])
        #print "rt = "+str(self.rt)
        self.RT[:] = numpy.dot(self.Ainv, self.rt-self.B)
        #print "RT = "+str(self.RT)
        X[0] = self.RT[0] * math.cos(self.RT[1]) + self.L[0]/2
        X[1] = self.RT[0] * math.sin(self.RT[1]) + self.L[1]/2
        X[2] = x[2]
        #print "X = "+str(X)
        if (Finv is not None):
            Finv[0,0] = 1.+(self.dRe-self.dRi)/(self.Re-self.Ri)
            Finv[0,1] = 0.
            Finv[0,2] = 0.
            Finv[1,0] = (self.dTe-self.dTi)/(self.Re-self.Ri)*self.rt[0]
            Finv[1,1] = self.rt[0]/self.RT[0]
            Finv[1,2] = 0.
            Finv[2,0] = 0.
            Finv[2,1] = 0.
            Finv[2,2] = 1.
            #print "F = "+str(Finv)
            Finv[:,:] = numpy.linalg.inv(Finv)
            #print "Finv = "+str(Finv)
            self.R[0,0] = +math.cos(self.RT[1])
            self.R[0,1] = +math.sin(self.RT[1])
            self.R[0,2] = 0.
            self.R[1,0] = -math.sin(self.RT[1])
            self.R[1,1] = +math.cos(self.RT[1])
            self.R[1,2] = 0.
            self.R[2,0] = 0.
            self.R[2,1] = 0.
            self.R[2,2] = 1.
            #print "R = "+str(self.R)
            Finv[:] = numpy.dot(numpy.transpose(self.R), numpy.dot(Finv, self.R))
            #print "Finv = "+str(Finv)

    def x_no(self, X, x, F=None):
        x[:] = X
        if (F is not None): F[:,:] = numpy.identity(numpy.sqrt(numpy.size(F)))

    def x_translation(self, X, x, F=None):
        x[:] = X + self.D
        if (F is not None): F[:,:] = numpy.identity(numpy.sqrt(numpy.size(F)))

    def x_rotation(self, X, x, F=None):
        x[:] = numpy.dot(self.R, X - self.C) + self.C
        if (F is not None): F[:,:] = self.R

    def x_homogeneous(self, X, x, F=None):
        x[:] = numpy.dot(self.F, X)
        if (F is not None): F[:,:] = self.F

    def x_heart(self, X, x, F=None):
        #print "X = "+str(X)
        self.X_inplane[0] = X[0] - self.L[0]/2
        self.X_inplane[1] = X[1] - self.L[1]/2
        #print "X_inplane = "+str(self.X_inplane)
        self.RT[0] = numpy.linalg.norm(self.X_inplane)
        self.RT[1] = math.atan2(self.X_inplane[1], self.X_inplane[0])
        #print "RT = "+str(self.RT)
        self.rt[:] = numpy.dot(self.A, self.RT) + self.B
        #print "rt = "+str(self.rt)
        x[0] = self.rt[0] * math.cos(self.rt[1]) + self.L[0]/2
        x[1] = self.rt[0] * math.sin(self.rt[1]) + self.L[1]/2
        x[2] = X[2]
        #print "x = "+str(x)
        if (F is not None):
            F[0,0] = 1.+(self.dRe-self.dRi)/(self.Re-self.Ri)
            F[0,1] = 0.
            F[0,2] = 0.
            F[1,0] = (self.dTe-self.dTi)/(self.Re-self.Ri)*self.rt[0]
            F[1,1] = self.rt[0]/self.RT[0]
            F[1,2] = 0.
            F[2,0] = 0.
            F[2,1] = 0.
            F[2,2] = 1.
            #print "F = "+str(F)
            self.R[0,0] = +math.cos(self.RT[1])
            self.R[0,1] = +math.sin(self.RT[1])
            self.R[0,2] = 0.
            self.R[1,0] = -math.sin(self.RT[1])
            self.R[1,1] = +math.cos(self.RT[1])
            self.R[1,2] = 0.
            self.R[2,0] = 0.
            self.R[2,1] = 0.
            self.R[2,2] = 1.
            F[:] = numpy.dot(numpy.transpose(self.R), numpy.dot(F, self.R))
            #print "F = "+str(F)

################################################################################

def generateImages(
        images,
        structure,
        texture,
        noise,
        deformation,
        evolution,
        generate_image_gradient=False,
        verbose=0):

    mypy.my_print(verbose, "*** generateImages ***")

    vtk_image = vtk.vtkImageData()

    if   (images["n_dim"] == 1):
        vtk_image.SetExtent([0, images["n_voxels"][0]-1, 0,                       0, 0,                       0])
    elif (images["n_dim"] == 2):
        vtk_image.SetExtent([0, images["n_voxels"][0]-1, 0, images["n_voxels"][1]-1, 0,                       0])
    elif (images["n_dim"] == 3):
        vtk_image.SetExtent([0, images["n_voxels"][0]-1, 0, images["n_voxels"][1]-1, 0, images["n_voxels"][2]-1])
    else:
        assert (0), "n_dim must be \"1\", \"2\" or \"3\". Aborting."

    #spacing = numpy.array(images["L"])/numpy.array(images["n_voxels"])
    if   (images["n_dim"] == 1):
        spacing = numpy.array([images["L"][0]/images["n_voxels"][0], 1., 1.])
    elif (images["n_dim"] == 2):
        spacing = numpy.array([images["L"][0]/images["n_voxels"][0], images["L"][1]/images["n_voxels"][1], 1.])
    elif (images["n_dim"] == 3):
        spacing = numpy.array([images["L"][0]/images["n_voxels"][0], images["L"][1]/images["n_voxels"][1], images["L"][2]/images["n_voxels"][2]])
    vtk_image.SetSpacing(spacing)

    origin = numpy.array(vtk_image.GetSpacing())/2
    if   (images["n_dim"] == 1):
        origin[1] = 0.
        origin[2] = 0.
    elif (images["n_dim"] == 2):
        origin[2] = 0.
    vtk_image.SetOrigin(origin)

    n_points = vtk_image.GetNumberOfPoints()
    vtk_image_scalars = myvtk.createFloatArray(
        name="ImageScalars",
        n_components=1,
        n_tuples=n_points,
        verbose=verbose-1)
    vtk_image.GetPointData().SetScalars(vtk_image_scalars)
    if (generate_image_gradient):
        vtk_image_gradient = myvtk.createFloatArray(
            name="ImageScalarsGradient",
            n_components=3,
            n_tuples=n_points,
            verbose=verbose-1)
        vtk_image.GetPointData().SetVectors(vtk_image_gradient)

    if not os.path.exists(images["folder"]):
        os.mkdir(images["folder"])

    x0 = numpy.empty(3)
    x  = numpy.empty(3)
    X  = numpy.empty(3)
    if (generate_image_gradient):
        F    = numpy.empty((3,3))
        Finv = numpy.empty((3,3))
    else:
        F    = None
        Finv = None
    dx = spacing[0:images["n_dim"]]/images["n_integration"][0:images["n_dim"]]
    global_min = float("+Inf")
    global_max = float("-Inf")
    I = numpy.empty(1)
    i = numpy.empty(1)
    if (generate_image_gradient):
        G = numpy.empty(3)
        g = numpy.empty(3)
    else:
        G = None
        g = None
    image = Image(images, structure, texture, noise)
    mapping = Mapping(images, structure, deformation, evolution)
    if ("zfill" not in images.keys()):
        images["zfill"] = len(str(images["n_frames"]))
    if ("ext" not in images.keys()):
        images["ext"] = "vti"
    for k_frame in xrange(images["n_frames"]):
        t = images["T"]*float(k_frame)/(images["n_frames"]-1) if (images["n_frames"]>1) else 0.
        print "t = "+str(t)
        mapping.init_t(t)
        for k_point in xrange(n_points):
            vtk_image.GetPoint(k_point, x0)
            #print "x0 = "+str(x0)
            x[:] = x0[:]
            #print "x = "+str(x)
            I[0] = 0.
            #print "I = "+str(I)
            if (generate_image_gradient):
                G[:] = 0.
                #print "G = "+str(G)
            if   (images["n_dim"] == 1):
                for k_x in xrange(images["n_integration"][0]):
                    x[0] = x0[0] - dx[0]/2 + (k_x+1./2)*dx[0]/images["n_integration"][0]
                    mapping.X(x, X, Finv)
                    image.I0(X, i, g)
                    I += i
                    if (generate_image_gradient): G += numpy.dot(g, Finv)
                I /= images["n_integration"][0]
                if (generate_image_gradient): G /= images["n_integration"][0]
            elif (images["n_dim"] == 2):
                for k_y in xrange(images["n_integration"][1]):
                    x[1] = x0[1] - dx[1]/2 + (k_y+1./2)*dx[1]/images["n_integration"][1]
                    for k_x in xrange(images["n_integration"][0]):
                        x[0] = x0[0] - dx[0]/2 + (k_x+1./2)*dx[0]/images["n_integration"][0]
                        #print "x = "+str(x)
                        mapping.X(x, X, Finv)
                        #print "X = "+str(X)
                        #print "Finv = "+str(Finv)
                        image.I0(X, i, g)
                        #print "i = "+str(i)
                        #print "g = "+str(g)
                        I += i
                        if (generate_image_gradient): G += numpy.dot(g, Finv)
                I /= images["n_integration"][1]*images["n_integration"][0]
                if (generate_image_gradient): G /= images["n_integration"][1]*images["n_integration"][0]
            elif (images["n_dim"] == 3):
                for k_z in xrange(images["n_integration"][2]):
                    x[2] = x0[2] - dx[2]/2 + (k_z+1./2)*dx[2]/images["n_integration"][2]
                    for k_y in xrange(images["n_integration"][1]):
                        x[1] = x0[1] - dx[1]/2 + (k_y+1./2)*dx[1]/images["n_integration"][1]
                        for k_x in xrange(images["n_integration"][0]):
                            x[0] = x0[0] - dx[0]/2 + (k_x+1./2)*dx[0]/images["n_integration"][0]
                            mapping.X(x, X, Finv)
                            image.I0(X, i, g)
                            I += i
                            if (generate_image_gradient): G += numpy.dot(g, Finv)
                I /= images["n_integration"][2]*images["n_integration"][1]*images["n_integration"][0]
                if (generate_image_gradient): G /= images["n_integration"][2]*images["n_integration"][1]*images["n_integration"][0]
            else:
                assert (0), "n_dim must be \"1\", \"2\" or \"3\". Aborting."
            vtk_image_scalars.SetTuple(k_point, I)
            if (generate_image_gradient): vtk_image_gradient.SetTuple(k_point, G)
            if (I[0] < global_min): global_min = I[0]
            if (I[0] > global_max): global_max = I[0]
        myvtk.writeImage(
            image=vtk_image,
            filename=images["folder"]+"/"+images["basename"]+"_"+str(k_frame).zfill(images["zfill"])+"."+images["ext"],
            verbose=verbose-1)

    if (images["data_type"] in ("float")):
        pass
    elif (images["data_type"] in ("unsigned char", "unsigned short", "unsigned int", "unsigned long", "unsigned float", "uint8", "uint16", "uint32", "uint64", "ufloat")):
        #print "global_min = "+str(global_min)
        #print "global_max = "+str(global_max)
        shifter = vtk.vtkImageShiftScale()
        shifter.SetShift(-global_min)
        if   (images["data_type"] in ("unsigned char", "uint8")):
            shifter.SetScale(float(2**8-1)/(global_max-global_min))
            shifter.SetOutputScalarTypeToUnsignedChar()
        elif (images["data_type"] in ("unsigned short", "uint16")):
            shifter.SetScale(float(2**16-1)/(global_max-global_min))
            shifter.SetOutputScalarTypeToUnsignedShort()
        elif (images["data_type"] in ("unsigned int", "uint32")):
            shifter.SetScale(float(2**32-1)/(global_max-global_min))
            shifter.SetOutputScalarTypeToUnsignedInt()
        elif (images["data_type"] in ("unsigned long", "uint64")):
            shifter.SetScale(float(2**64-1)/(global_max-global_min))
            shifter.SetOutputScalarTypeToUnsignedLong()
        elif (images["data_type"] in ("unsigned float", "ufloat")):
            shifter.SetScale(1./(global_max-global_min))
            shifter.SetOutputScalarTypeToFloat()
        for k_frame in xrange(images["n_frames"]):
            vtk_image = myvtk.readImage(
                filename=images["folder"]+"/"+images["basename"]+"_"+str(k_frame).zfill(images["zfill"])+"."+images["ext"],
                verbose=verbose-1)
            if (vtk.vtkVersion.GetVTKMajorVersion() >= 6):
                shifter.SetInputData(vtk_image)
            else:
                shifter.SetInput(vtk_image)
            shifter.Update()
            vtk_image = shifter.GetOutput()
            myvtk.writeImage(
                image=vtk_image,
                filename=images["folder"]+"/"+images["basename"]+"_"+str(k_frame).zfill(images["zfill"])+"."+images["ext"],
                verbose=verbose-1)
    else:
        assert (0), "Wrong data type. Aborting."
