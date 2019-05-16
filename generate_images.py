#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2016-2019                                       ###
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

    def __init__(self, images, structure, texture, noise, generate_image_gradient=False):

        self.L = images["L"]

        # structure
        if (structure["type"] == "no"):
            self.I0_structure = self.I0_structure_no_wGrad if (generate_image_gradient) else self.I0_structure_no
        elif (structure["type"] == "box"):
            self.I0_structure = self.I0_structure_box_wGrad if (generate_image_gradient) else self.I0_structure_box
            self.Xmin = structure["Xmin"]
            self.Xmax = structure["Xmax"]
        elif (structure["type"] == "heart"):
            if (images["n_dim"] == 2):
                self.I0_structure = self.I0_structure_heart_2_wGrad if (generate_image_gradient) else self.I0_structure_heart_2
                self.R = float()
                self.Ri = structure["Ri"]
                self.Re = structure["Re"]
            elif (images["n_dim"] == 3):
                self.I0_structure = self.I0_structure_heart_3_wGrad if (generate_image_gradient) else self.I0_structure_heart_3
                self.R = float()
                self.Ri = structure["Ri"]
                self.Re = structure["Re"]
                self.Zmin = structure.Zmin if ("Zmin" in structure) else 0.
                self.Zmax = structure.Zmax if ("Zmax" in structure) else images["L"][2]
            else:
                assert (0), "n_dim must be \"2\" or \"3 for \"heart\" type structure. Aborting."
        else:
            assert (0), "structure type must be \"no\", \"box\" or \"heart\". Aborting."

        # texture
        if (texture["type"] == "no"):
            self.I0_texture = self.I0_texture_no_wGrad if (generate_image_gradient) else self.I0_texture_no
        elif (texture["type"].startswith("tagging")):
            if   (images["n_dim"] == 1):
                if ("-signed" in texture["type"]):
                    self.I0_texture = self.I0_texture_tagging_signed_X_wGrad if (generate_image_gradient) else self.I0_texture_tagging_signed_X
                else:
                    self.I0_texture = self.I0_texture_tagging_X_wGrad if (generate_image_gradient) else self.I0_texture_tagging_X
            elif (images["n_dim"] == 2):
                if ("-signed" in texture["type"]):
                    if   ("-addComb" in texture["type"]):
                        self.I0_texture = self.I0_texture_tagging_signed_XY_wAdditiveCombination_wGrad       if (generate_image_gradient) else self.I0_texture_tagging_signed_XY_wAdditiveCombination
                    elif ("-diffComb" in texture["type"]):
                        self.I0_texture = self.I0_texture_tagging_signed_XY_wDifferentiableCombination_wGrad if (generate_image_gradient) else self.I0_texture_tagging_signed_XY_wDifferentiableCombination
                    else:
                        self.I0_texture = self.I0_texture_tagging_signed_XY_wMultiplicativeCombination_wGrad if (generate_image_gradient) else self.I0_texture_tagging_signed_XY_wMultiplicativeCombination
                else:
                    if   ("-addComb" in texture["type"]):
                        self.I0_texture = self.I0_texture_tagging_XY_wAdditiveCombination_wGrad       if (generate_image_gradient) else self.I0_texture_tagging_XY_wAdditiveCombination
                    elif ("-diffComb" in texture["type"]):
                        self.I0_texture = self.I0_texture_tagging_XY_wDifferentiableCombination_wGrad if (generate_image_gradient) else self.I0_texture_tagging_XY_wDifferentiableCombination
                    else:
                        self.I0_texture = self.I0_texture_tagging_XY_wMultiplicativeCombination_wGrad if (generate_image_gradient) else self.I0_texture_tagging_XY_wMultiplicativeCombination
            elif (images["n_dim"] == 3):
                if ("-signed" in texture["type"]):
                    if   ("-addComb" in texture["type"]):
                        self.I0_texture = self.I0_texture_tagging_signed_XYZ_wAdditiveCombination_wGrad       if (generate_image_gradient) else self.I0_texture_tagging_signed_XYZ_wAdditiveCombination
                    elif ("-diffComb" in texture["type"]):
                        self.I0_texture = self.I0_texture_tagging_signed_XYZ_wDifferentiableCombination_wGrad if (generate_image_gradient) else self.I0_texture_tagging_signed_XYZ_wDifferentiableCombination
                    else:
                        self.I0_texture = self.I0_texture_tagging_signed_XYZ_wMultiplicativeCombination_wGrad if (generate_image_gradient) else self.I0_texture_tagging_signed_XYZ_wMultiplicativeCombination
                else:
                    if   ("-addComb" in texture["type"]):
                        self.I0_texture = self.I0_texture_tagging_XYZ_wAdditiveCombination_wGrad       if (generate_image_gradient) else self.I0_texture_tagging_XYZ_wAdditiveCombination
                    elif ("-diffComb" in texture["type"]):
                        self.I0_texture = self.I0_texture_tagging_XYZ_wDifferentiableCombination_wGrad if (generate_image_gradient) else self.I0_texture_tagging_XYZ_wDifferentiableCombination
                    else:
                        self.I0_texture = self.I0_texture_tagging_XYZ_wMultiplicativeCombination_wGrad if (generate_image_gradient) else self.I0_texture_tagging_XYZ_wMultiplicativeCombination
            else:
                assert (0), "n_dim must be \"1\", \"2\" or \"3\". Aborting."
            self.s = texture["s"]
        elif (texture["type"].startswith("taggX")):
            if ("-signed" in texture["type"]):
                self.I0_texture = self.I0_texture_tagging_signed_X_wGrad if (generate_image_gradient) else self.I0_texture_tagging_signed_X
            else:
                self.I0_texture = self.I0_texture_tagging_X_wGrad if (generate_image_gradient) else self.I0_texture_tagging_X
            self.s = texture["s"]
        elif (texture["type"].startswith("taggY")):
            if ("-signed" in texture["type"]):
                self.I0_texture = self.I0_texture_tagging_signed_Y_wGrad if (generate_image_gradient) else self.I0_texture_tagging_signed_Y
            else:
                self.I0_texture = self.I0_texture_tagging_Y_wGrad if (generate_image_gradient) else self.I0_texture_tagging_Y
            self.s = texture["s"]
        elif (texture["type"].startswith("taggZ")):
            if ("-signed" in texture["type"]):
                self.I0_texture = self.I0_texture_tagging_signed_Z_wGrad if (generate_image_gradient) else self.I0_texture_tagging_signed_Z
            else:
                self.I0_texture = self.I0_texture_tagging_Z_wGrad if (generate_image_gradient) else self.I0_texture_tagging_Z
            self.s = texture["s"]
        else:
            assert (0), "texture type must be \"no\", \"tagging\", \"taggX\", \"taggY\" or \"taggZ\". Aborting."

        # noise
        if (noise["type"] == "no"):
            self.I0_noise = self.I0_noise_no_wGrad if (generate_image_gradient) else self.I0_noise_no
        elif (noise["type"] == "normal"):
            self.I0_noise = self.I0_noise_normal_wGrad if (generate_image_gradient) else self.I0_noise_normal
            self.avg = noise["avg"] if ("avg" in noise) else 0.
            self.std = noise["stdev"]
        else:
            assert (0), "noise type must be \"no\" or \"normal\". Aborting."

    def I0(self, X, I):
        self.I0_structure(X, I)
        self.I0_texture(X, I)
        self.I0_noise(I)

    def I0_wGrad(self, X, I, G):
        self.I0_structure_wGrad(X, I, G)
        self.I0_texture_wGrad(X, I, G)
        self.I0_noise_wGrad(I, G)

    def I0_structure_no(self, X, I):
        I[0] = 1.

    def I0_structure_no_wGrad(self, X, I, G):
        self.I0_structure_no(X, I)
        G[:] = 1. # MG 20180806: gradient is given by texture; here it is just indicator function

    def I0_structure_box(self, X, I):
        if all(numpy.greater_equal(X, self.Xmin)) and all(numpy.less_equal(X, self.Xmax)):
            I[0] = 1.
        else:
            I[0] = 0.

    def I0_structure_box_wGrad(self, X, I, G):
        if all(numpy.greater_equal(X, self.Xmin)) and all(numpy.less_equal(X, self.Xmax)):
            I[0] = 1.
            G[:] = 1. # MG 20180806: gradient is given by texture; here it is just indicator function
        else:
            I[0] = 0.
            G[:] = 0. # MG 20180806: gradient is given by texture; here it is just indicator function

    def I0_structure_heart_2(self, X, I):
        self.R = ((X[0]-self.L[0]/2)**2 + (X[1]-self.L[1]/2)**2)**(1./2)
        if (self.R >= self.Ri) and (self.R <= self.Re):
            I[0] = 1.
        else:
            I[0] = 0.

    def I0_structure_heart_2_wGrad(self, X, I, G):
        self.R = ((X[0]-self.L[0]/2)**2 + (X[1]-self.L[1]/2)**2)**(1./2)
        if (self.R >= self.Ri) and (self.R <= self.Re):
            I[0] = 1.
            G[:] = 1. # MG 20180806: gradient is given by texture; here it is just indicator function
        else:
            I[0] = 0.
            G[:] = 0. # MG 20180806: gradient is given by texture; here it is just indicator function

    def I0_structure_heart_3(self, X, I):
        self.R = ((X[0]-self.L[0]/2)**2 + (X[1]-self.L[1]/2)**2)**(1./2)
        if (self.R >= self.Ri) and (self.R <= self.Re) and (X[2] >= self.Zmin) and (X[2] <= self.Zmax):
            I[0] = 1.
        else:
            I[0] = 0.

    def I0_structure_heart_3_wGrad(self, X, I, G):
        self.R = ((X[0]-self.L[0]/2)**2 + (X[1]-self.L[1]/2)**2)**(1./2)
        if (self.R >= self.Ri) and (self.R <= self.Re) and (X[2] >= self.Zmin) and (X[2] <= self.Zmax):
            I[0] = 1.
            G[:] = 1. # MG 20180806: gradient is given by texture; here it is just indicator function
        else:
            I[0] = 0.
            G[:] = 0. # MG 20180806: gradient is given by texture; here it is just indicator function

    def I0_texture_no(self, X, I):
        I[0] *= 1.

    def I0_texture_no_wGrad(self, X, I, G):
        self.I0_texture_no(X, I)
        G[:] *= 0.

    def I0_texture_tagging_X(self, X, I):
        I[0] *= abs(math.sin(math.pi*X[0]/self.s))

    def I0_texture_tagging_X_wGrad(self, X, I, G):
        self.I0_texture_tagging_X(X, I)
        G[0] *= math.copysign(1, math.sin(math.pi*X[0]/self.s)) * (math.pi/self.s) * math.cos(math.pi*X[0]/self.s)
        G[1] *= 0.
        G[2] *= 0.

    def I0_texture_tagging_Y(self, X, I):
        I[0] *= abs(math.sin(math.pi*X[1]/self.s))

    def I0_texture_tagging_Y_wGrad(self, X, I, G):
        self.I0_texture_tagging_Y(X, I)
        G[0] *= 0.
        G[1] *= math.copysign(1, math.sin(math.pi*X[1]/self.s)) * (math.pi/self.s) * math.cos(math.pi*X[1]/self.s)
        G[2] *= 0.

    def I0_texture_tagging_Z(self, X, I):
        I[0] *= abs(math.sin(math.pi*X[2]/self.s))

    def I0_texture_tagging_Z_wGrad(self, X, I, G):
        self.I0_texture_tagging_Z(X, I)
        G[0] *= 0.
        G[1] *= 0.
        G[2] *= math.copysign(1, math.sin(math.pi*X[2]/self.s)) * (math.pi/self.s) * math.cos(math.pi*X[2]/self.s)

    def I0_texture_tagging_XY_wDifferentiableCombination(self, X, I):
        I[0] *= (1 + 3 * abs(math.sin(math.pi*X[0]/self.s))
                       * abs(math.sin(math.pi*X[1]/self.s)))**(1./2) - 1

    def I0_texture_tagging_XY_wDifferentiableCombination_wGrad(self, X, I, G):
        self.I0_texture_tagging_XY_wDifferentiableCombination(X, I)
        G[0] *= 3 * math.copysign(1, math.sin(math.pi*X[0]/self.s)) * (math.pi/self.s) * math.cos(math.pi*X[0]/self.s) * abs(math.sin(math.pi*X[1]/self.s)) / 2 / (I[0] + 1)
        G[1] *= 3 * math.copysign(1, math.sin(math.pi*X[1]/self.s)) * (math.pi/self.s) * math.cos(math.pi*X[1]/self.s) * abs(math.sin(math.pi*X[0]/self.s)) / 2 / (I[0] + 1)
        G[2] *= 0.

    def I0_texture_tagging_XY_wMultiplicativeCombination(self, X, I):
        I[0] *= (abs(math.sin(math.pi*X[0]/self.s))
             *   abs(math.sin(math.pi*X[1]/self.s)))**(1./2)

    def I0_texture_tagging_XY_wMultiplicativeCombination_wGrad(self, X, I, G):
        self.I0_texture_tagging_XY_wMultiplicativeCombination(X, I)
        G[0] *= math.copysign(1, math.sin(math.pi*X[0]/self.s)) * (math.pi/self.s) * math.cos(math.pi*X[0]/self.s) * abs(math.sin(math.pi*X[1]/self.s)) / 2 / I[0]
        G[1] *= math.copysign(1, math.sin(math.pi*X[1]/self.s)) * (math.pi/self.s) * math.cos(math.pi*X[1]/self.s) * abs(math.sin(math.pi*X[0]/self.s)) / 2 / I[0]
        G[2] *= 0.

    def I0_texture_tagging_XYZ_wDifferentiableCombination(self, X, I):
        I[0] *= (1 + 7 * abs(math.sin(math.pi*X[0]/self.s))
                       * abs(math.sin(math.pi*X[1]/self.s))
                       * abs(math.sin(math.pi*X[2]/self.s)))**(1./3) - 1

    def I0_texture_tagging_XYZ_wDifferentiableCombination_wGrad(self, X, I, G):
        self.I0_texture_tagging_XYZ_wDifferentiableCombination(X, I)
        G[0] *= 7 * math.copysign(1, math.sin(math.pi*X[0]/self.s)) * (math.pi/self.s) * math.cos(math.pi*X[0]/self.s) * abs(math.sin(math.pi*X[1]/self.s)) * abs(math.sin(math.pi*X[2]/self.s)) / 3 / (I[0] + 1)
        G[1] *= 7 * math.copysign(1, math.sin(math.pi*X[1]/self.s)) * (math.pi/self.s) * math.cos(math.pi*X[1]/self.s) * abs(math.sin(math.pi*X[0]/self.s)) * abs(math.sin(math.pi*X[2]/self.s)) / 3 / (I[0] + 1)
        G[2] *= 7 * math.copysign(1, math.sin(math.pi*X[2]/self.s)) * (math.pi/self.s) * math.cos(math.pi*X[2]/self.s) * abs(math.sin(math.pi*X[0]/self.s)) * abs(math.sin(math.pi*X[1]/self.s)) / 3 / (I[0] + 1)

    def I0_texture_tagging_XYZ_wMultiplicativeCombination(self, X, I):
        I[0] *= (abs(math.sin(math.pi*X[0]/self.s))
             *   abs(math.sin(math.pi*X[1]/self.s))
             *   abs(math.sin(math.pi*X[2]/self.s)))**(1./3)

    def I0_texture_tagging_XYZ_wMultiplicativeCombination_wGrad(self, X, I, G):
        self.I0_texture_tagging_XYZ_wMultiplicativeCombination(X, I)
        G[0] *= math.copysign(1, math.sin(math.pi*X[0]/self.s)) * (math.pi/self.s) * math.cos(math.pi*X[0]/self.s) * abs(math.sin(math.pi*X[1]/self.s)) * abs(math.sin(math.pi*X[2]/self.s)) / 3 / I[0]**2
        G[1] *= math.copysign(1, math.sin(math.pi*X[1]/self.s)) * (math.pi/self.s) * math.cos(math.pi*X[1]/self.s) * abs(math.sin(math.pi*X[0]/self.s)) * abs(math.sin(math.pi*X[2]/self.s)) / 3 / I[0]**2
        G[2] *= math.copysign(1, math.sin(math.pi*X[2]/self.s)) * (math.pi/self.s) * math.cos(math.pi*X[2]/self.s) * abs(math.sin(math.pi*X[0]/self.s)) * abs(math.sin(math.pi*X[1]/self.s)) / 3 / I[0]**2

    def I0_texture_tagging_signed_X(self, X, I):
        I[0] *= (1+math.sin(math.pi*X[0]/self.s-math.pi/2))/2

    def I0_texture_tagging_signed_X_wGrad(self, X, I, G):
        self.I0_texture_tagging_signed_X(X, I)
        G[0] *= (math.pi/self.s) * math.cos(math.pi*X[0]/self.s-math.pi/2) / 2
        G[1] *= 0.
        G[2] *= 0.

    def I0_texture_tagging_signed_Y(self, X, I):
        I[0] *= (1+math.sin(math.pi*X[1]/self.s-math.pi/2))/2

    def I0_texture_tagging_signed_Y_wGrad(self, X, I, G):
        self.I0_texture_tagging_signed_Y(X, I)
        G[0] *= 0.
        G[1] *= (math.pi/self.s) * math.cos(math.pi*X[1]/self.s-math.pi/2) / 2
        G[2] *= 0.

    def I0_texture_tagging_signed_Z(self, X, I):
        I[0] *= (1+math.sin(math.pi*X[1]/self.s-math.pi/2))/2

    def I0_texture_tagging_signed_Z_wGrad(self, X, I, G):
        self.I0_texture_tagging_signed_Z(X, I)
        G[0] *= 0.
        G[1] *= 0.
        G[2] *= (math.pi/self.s) * math.cos(math.pi*X[2]/self.s-math.pi/2) / 2

    def I0_texture_tagging_signed_XY_wDifferentiableCombination(self, X, I):
        I[0] *= (1 + 3 * (1+math.sin(math.pi*X[0]/self.s-math.pi/2))/2
                       * (1+math.sin(math.pi*X[1]/self.s-math.pi/2))/2)**(1./2) - 1

    def I0_texture_tagging_signed_XY_wDifferentiableCombination_wGrad(self, X, I, G):
        self.I0_texture_tagging_signed_XY_wDifferentiableCombination(X, I)
        G[0] *= 3 * (math.pi/self.s) * math.cos(math.pi*X[0]/self.s-math.pi/2)/2 * (1+math.sin(math.pi*X[1]/self.s-math.pi/2))/2 / 2 / (I[0] + 1)
        G[1] *= 3 * (math.pi/self.s) * math.cos(math.pi*X[1]/self.s-math.pi/2)/2 * (1+math.sin(math.pi*X[0]/self.s-math.pi/2))/2 / 2 / (I[0] + 1)
        G[2] *= 0.

    def I0_texture_tagging_signed_XY_wAdditiveCombination(self, X, I):
        I[0] *= ((1+math.sin(math.pi*X[0]/self.s-math.pi/2))/2
              +  (1+math.sin(math.pi*X[1]/self.s-math.pi/2))/2) / 2

    def I0_texture_tagging_signed_XY_wAdditiveCombination_wGrad(self, X, I, G):
        self.I0_texture_tagging_signed_XY_wAdditiveCombination(X, I)
        G[0] *= (math.pi/self.s) * math.cos(math.pi*X[0]/self.s-math.pi/2)/2 / 2
        G[1] *= (math.pi/self.s) * math.cos(math.pi*X[1]/self.s-math.pi/2)/2 / 2
        G[2] *= 0.

    def I0_texture_tagging_signed_XYZ_wDifferentiableCombination(self, X, I):
        I[0] *= (1 + 7 * (1+math.sin(math.pi*X[0]/self.s-math.pi/2))/2
                       * (1+math.sin(math.pi*X[1]/self.s-math.pi/2))/2
                       * (1+math.sin(math.pi*X[2]/self.s-math.pi/2))/2)**(1./3) - 1

    def I0_texture_tagging_signed_XYZ_wDifferentiableCombination_wGrad(self, X, I, G):
        self.I0_texture_tagging_signed_XYZ_wDifferentiableCombination(X, I)
        G[0] *= 7 * (math.pi/self.s) * math.cos(math.pi*X[0]/self.s-math.pi/2)/2 * (1+math.sin(math.pi*X[1]/self.s-math.pi/2))/2 * (1+math.sin(math.pi*X[2]/self.s-math.pi/2))/2 / 3 / (I[0] + 1)
        G[1] *= 7 * (math.pi/self.s) * math.cos(math.pi*X[1]/self.s-math.pi/2)/2 * (1+math.sin(math.pi*X[0]/self.s-math.pi/2))/2 * (1+math.sin(math.pi*X[2]/self.s-math.pi/2))/2 / 3 / (I[0] + 1)
        G[2] *= 7 * (math.pi/self.s) * math.cos(math.pi*X[2]/self.s-math.pi/2)/2 * (1+math.sin(math.pi*X[0]/self.s-math.pi/2))/2 * (1+math.sin(math.pi*X[1]/self.s-math.pi/2))/2 / 3 / (I[0] + 1)

    def I0_texture_tagging_signed_XYZ_wAdditiveCombination(self, X, I):
        I[0] *= ((1+math.sin(math.pi*X[0]/self.s-math.pi/2))/2
              +  (1+math.sin(math.pi*X[1]/self.s-math.pi/2))/2
              +  (1+math.sin(math.pi*X[2]/self.s-math.pi/2))/2) / 3

    def I0_texture_tagging_signed_XYZ_wAdditiveCombination_wGrad(self, X, I, G):
        self.I0_texture_tagging_signed_XYZ_wAdditiveCombination(X, I)
        G[0] *= (math.pi/self.s) * math.cos(math.pi*X[0]/self.s-math.pi/2)/2 / 3
        G[1] *= (math.pi/self.s) * math.cos(math.pi*X[1]/self.s-math.pi/2)/2 / 3
        G[2] *= (math.pi/self.s) * math.cos(math.pi*X[2]/self.s-math.pi/2)/2 / 3

    def I0_noise_no(self, I):
        pass

    def I0_noise_no_wGrad(self, I, G):
        pass

    def I0_noise_normal(self, I):
        I[0] += random.normalvariate(self.avg, self.std)

    def I0_noise_normal_wGrad(self, I, G):
        self.I0_noise_normal(I)
        G[:] += [2*random.normalvariate(self.avg, self.std) for k in xrange(len(G))]

################################################################################

class Mapping:

    def __init__(self, images, structure, deformation, evolution, generate_image_gradient):

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
        self.D[0] = self.deformation["Dx"] if ("Dx" in self.deformation) else 0.
        self.D[1] = self.deformation["Dy"] if ("Dy" in self.deformation) else 0.
        self.D[2] = self.deformation["Dz"] if ("Dz" in self.deformation) else 0.
        self.D *= self.phi(t)

    def init_t_rotation(self, t):
        self.C[0] = self.deformation["Cx"] if ("Cx" in self.deformation) else 0.
        self.C[1] = self.deformation["Cy"] if ("Cy" in self.deformation) else 0.
        self.C[2] = self.deformation["Cz"] if ("Cz" in self.deformation) else 0.
        Rx = self.deformation["Rx"]*math.pi/180*self.phi(t) if ("Rx" in self.deformation) else 0.
        Ry = self.deformation["Ry"]*math.pi/180*self.phi(t) if ("Ry" in self.deformation) else 0.
        Rz = self.deformation["Rz"]*math.pi/180*self.phi(t) if ("Rz" in self.deformation) else 0.
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
        if (any(E in self.deformation for E in ("Exx", "Eyy", "Ezz"))): # build F from E
            Exx = self.deformation["Exx"] if ("Exx" in self.deformation) else 0.
            Eyy = self.deformation["Eyy"] if ("Eyy" in self.deformation) else 0.
            Ezz = self.deformation["Ezz"] if ("Ezz" in self.deformation) else 0.
            self.F = numpy.array([[Exx,  0.,  0.],
                                  [ 0., Eyy,  0.],
                                  [ 0.,  0., Ezz]])*self.phi(t)
            self.F *= 2
            self.F += numpy.eye(3)
            w, v = numpy.linalg.eig(self.F)
            # assert (numpy.diag(numpy.dot(numpy.dot(numpy.transpose(v), self.F), v)) == w).all(), str(numpy.dot(numpy.dot(numpy.transpose(v), self.F), v))+" ≠ "+str(numpy.diag(w))+". Aborting."
            self.F = numpy.dot(numpy.dot(v, numpy.diag(numpy.sqrt(w))), numpy.transpose(v))
        else:
            Fxx = self.deformation["Fxx"] if ("Fxx" in self.deformation) else 0.
            Fyy = self.deformation["Fyy"] if ("Fyy" in self.deformation) else 0.
            Fzz = self.deformation["Fzz"] if ("Fzz" in self.deformation) else 0.
            Fxy = self.deformation["Fxy"] if ("Fxy" in self.deformation) else 0.
            Fyx = self.deformation["Fyx"] if ("Fyx" in self.deformation) else 0.
            Fyz = self.deformation["Fyz"] if ("Fyz" in self.deformation) else 0.
            Fzy = self.deformation["Fzy"] if ("Fzy" in self.deformation) else 0.
            Fzx = self.deformation["Fzx"] if ("Fzx" in self.deformation) else 0.
            Fxz = self.deformation["Fxz"] if ("Fxz" in self.deformation) else 0.
            self.F = numpy.eye(3) + (numpy.array([[Fxx, Fxy, Fxz],
                                                  [Fyx, Fyy, Fyz],
                                                  [Fzx, Fzy, Fzz]])-numpy.eye(3))*self.phi(t)
        self.Finv = numpy.linalg.inv(self.F)

    def init_t_heart(self, t):
        self.dRi = self.deformation["dRi"]*self.phi(t) if ("dRi" in self.deformation) else 0.
        self.dRe = self.deformation["dRi"]*self.phi(t) if ("dRi" in self.deformation) else 0.
        self.dTi = self.deformation["dTi"]*self.phi(t) if ("dTi" in self.deformation) else 0.
        self.dTe = self.deformation["dTe"]*self.phi(t) if ("dTe" in self.deformation) else 0.
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

def set_I_woGrad(
        image,
        X,
        I,
        vtk_image_scalars,
        k_point,
        G=None,
        Finv=None,
        vtk_gradient_vectors=None):

    image.I0(X, I)
    vtk_image_scalars.SetTuple(k_point, I)

def set_I_wGrad(
        image,
        X,
        I,
        vtk_image_scalars,
        k_point,
        G,
        Finv,
        vtk_gradient_vectors):

    image.I0_wGrad(X, I, G)
    vtk_image_scalars.SetTuple(k_point, I)
    G = numpy.dot(G, Finv)
    vtk_gradient_vectors.SetTuple(k_point, G)

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

    assert ("n_integration" not in images),\
        "\"n_integration\" has been deprecated. Use \"resampling\" instead. Aborting."

    if ("resampling" not in images):
        images["resampling"] = [1]*images["n_dim"]
    if ("zfill" not in images):
        images["zfill"] = len(str(images["n_frames"]))
    if ("ext" not in images):
        images["ext"] = "vti"

    if not os.path.exists(images["folder"]):
        os.mkdir(images["folder"])

    image = Image(
        images,
        structure,
        texture,
        noise,
        generate_image_gradient)
    mapping = Mapping(
        images,
        structure,
        deformation,
        evolution,
        generate_image_gradient)

    vtk_image = vtk.vtkImageData()

    if   (images["n_dim"] == 1):
        vtk_image.SetExtent([0, images["n_voxels"][0]*images["resampling"][0]-1, 0,                                               0, 0,                                               0])
    elif (images["n_dim"] == 2):
        vtk_image.SetExtent([0, images["n_voxels"][0]*images["resampling"][0]-1, 0, images["n_voxels"][1]*images["resampling"][1]-1, 0,                                               0])
    elif (images["n_dim"] == 3):
        vtk_image.SetExtent([0, images["n_voxels"][0]*images["resampling"][0]-1, 0, images["n_voxels"][1]*images["resampling"][1]-1, 0, images["n_voxels"][2]*images["resampling"][2]-1])
    else:
        assert (0), "n_dim must be \"1\", \"2\" or \"3\". Aborting."

    if   (images["n_dim"] == 1):
        spacing = numpy.array([images["L"][0]/images["n_voxels"][0]/images["resampling"][0],                                                           1.,                                                           1.])
    elif (images["n_dim"] == 2):
        spacing = numpy.array([images["L"][0]/images["n_voxels"][0]/images["resampling"][0], images["L"][1]/images["n_voxels"][1]/images["resampling"][1],                                                           1.])
    elif (images["n_dim"] == 3):
        spacing = numpy.array([images["L"][0]/images["n_voxels"][0]/images["resampling"][0], images["L"][1]/images["n_voxels"][1]/images["resampling"][1], images["L"][2]/images["n_voxels"][2]/images["resampling"][2]])
    vtk_image.SetSpacing(spacing)

    if   (images["n_dim"] == 1):
        origin = numpy.array([spacing[0]/2,           0.,           0.])
    elif (images["n_dim"] == 2):
        origin = numpy.array([spacing[0]/2, spacing[1]/2,           0.])
    elif (images["n_dim"] == 3):
        origin = numpy.array([spacing[0]/2, spacing[1]/2, spacing[2]/2])
    vtk_image.SetOrigin(origin)

    n_points = vtk_image.GetNumberOfPoints()
    vtk_image_scalars = myvtk.createFloatArray(
        name="ImageScalars",
        n_components=1,
        n_tuples=n_points,
        verbose=verbose-1)
    vtk_image.GetPointData().SetScalars(vtk_image_scalars)

    if (generate_image_gradient):
        vtk_gradient = vtk.vtkImageData()
        vtk_gradient.DeepCopy(vtk_image)

        vtk_gradient_vectors = myvtk.createFloatArray(
            name="ImageScalarsGradient",
            n_components=3,
            n_tuples=n_points,
            verbose=verbose-1)
        vtk_gradient.GetPointData().SetScalars(vtk_gradient_vectors)
    else:
        vtk_gradient         = None
        vtk_gradient_vectors = None

    x = numpy.empty(3)
    X = numpy.empty(3)
    I = numpy.empty(1)
    global_min = float("+Inf")
    global_max = float("-Inf")
    if (generate_image_gradient):
        G     = numpy.empty(3)
        Finv  = numpy.empty((3,3))
        set_I = set_I_wGrad
    else:
        G     = None
        Finv  = None
        set_I = set_I_woGrad

    for k_frame in xrange(images["n_frames"]):
        t = images["T"]*float(k_frame)/(images["n_frames"]-1) if (images["n_frames"]>1) else 0.
        print "t = "+str(t)
        mapping.init_t(t)
        for k_point in xrange(n_points):
            vtk_image.GetPoint(k_point, x)
            #print "x0 = "+str(x)
            mapping.X(x, X, Finv)
            #print "X = "+str(X)
            set_I(image, X, I, vtk_image_scalars, k_point, G, Finv, vtk_gradient_vectors)
            global_min = min(global_min, I[0])
            global_max = max(global_max, I[0])
        #print vtk_image
        myvtk.writeImage(
            image=vtk_image,
            filename=images["folder"]+"/"+images["basename"]+"_"+str(k_frame).zfill(images["zfill"])+"."+images["ext"],
            verbose=verbose-1)
        if (generate_image_gradient):
            #print vtk_gradient
            myvtk.writeImage(
                image=vtk_gradient,
                filename=images["folder"]+"/"+images["basename"]+"-grad"+"_"+str(k_frame).zfill(images["zfill"])+"."+images["ext"],
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
