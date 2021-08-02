#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2016-2021                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

from builtins import range

import dolfin
import glob
import math
import numpy
import os
import random
import vtk

import myPythonLibrary    as mypy
import myVTKPythonLibrary as myvtk

import dolfin_warp as dwarp

################################################################################

class Image():

    def __init__(
            self,
            images,
            structure,
            texture,
            noise,
            generate_image_gradient=0):

        self.L = images["L"]

        # structure
        if (structure["type"] == "no"):
            self.I0_structure = self.I0_structure_no_wGrad if (generate_image_gradient) else self.I0_structure_no
        elif (structure["type"] == "box"):
            self.I0_structure = self.I0_structure_box_wGrad if (generate_image_gradient) else self.I0_structure_box
            self.Xmin = structure["Xmin"]+[float("-Inf")]*(3-images["n_dim"])
            self.Xmax = structure["Xmax"]+[float("+Inf")]*(3-images["n_dim"])
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

    def I0_texture_tagging_XY_wAdditiveCombination(self, X, I):
        I[0] *= (abs(math.sin(math.pi*X[0]/self.s))
               + abs(math.sin(math.pi*X[1]/self.s)))/2

    def I0_texture_tagging_XY_wAdditiveCombination_wGrad(self, X, I, G):
        self.I0_texture_tagging_XY_wAdditiveCombination(X, I)
        G[0] *= math.copysign(1, math.sin(math.pi*X[0]/self.s)) * (math.pi/self.s) * math.cos(math.pi*X[0]/self.s) / 2
        G[1] *= math.copysign(1, math.sin(math.pi*X[1]/self.s)) * (math.pi/self.s) * math.cos(math.pi*X[1]/self.s) / 2
        G[2] *= 0.

    def I0_texture_tagging_XY_wMultiplicativeCombination(self, X, I):
        I[0] *= (abs(math.sin(math.pi*X[0]/self.s))
             *   abs(math.sin(math.pi*X[1]/self.s)))**(1./2)

    def I0_texture_tagging_XY_wMultiplicativeCombination_wGrad(self, X, I, G):
        self.I0_texture_tagging_XY_wMultiplicativeCombination(X, I)
        G[0] *= math.copysign(1, math.sin(math.pi*X[0]/self.s)) * (math.pi/self.s) * math.cos(math.pi*X[0]/self.s) * abs(math.sin(math.pi*X[1]/self.s)) / 2 / I[0]
        G[1] *= math.copysign(1, math.sin(math.pi*X[1]/self.s)) * (math.pi/self.s) * math.cos(math.pi*X[1]/self.s) * abs(math.sin(math.pi*X[0]/self.s)) / 2 / I[0]
        G[2] *= 0.

    def I0_texture_tagging_XY_wDifferentiableCombination(self, X, I):
        I[0] *= (1 + 3 * abs(math.sin(math.pi*X[0]/self.s))
                       * abs(math.sin(math.pi*X[1]/self.s)))**(1./2) - 1

    def I0_texture_tagging_XY_wDifferentiableCombination_wGrad(self, X, I, G):
        self.I0_texture_tagging_XY_wDifferentiableCombination(X, I)
        G[0] *= 3 * math.copysign(1, math.sin(math.pi*X[0]/self.s)) * (math.pi/self.s) * math.cos(math.pi*X[0]/self.s) * abs(math.sin(math.pi*X[1]/self.s)) / 2 / (I[0] + 1)
        G[1] *= 3 * math.copysign(1, math.sin(math.pi*X[1]/self.s)) * (math.pi/self.s) * math.cos(math.pi*X[1]/self.s) * abs(math.sin(math.pi*X[0]/self.s)) / 2 / (I[0] + 1)
        G[2] *= 0.

    def I0_texture_tagging_XYZ_wAdditiveCombination(self, X, I):
        I[0] *= (abs(math.sin(math.pi*X[0]/self.s))
               + abs(math.sin(math.pi*X[1]/self.s))
               + abs(math.sin(math.pi*X[2]/self.s)))/3

    def I0_texture_tagging_XYZ_wAdditiveCombination_wGrad(self, X, I, G):
        self.I0_texture_tagging_XYZ_wAdditiveCombination(X, I)
        G[0] *= math.copysign(1, math.sin(math.pi*X[0]/self.s)) * (math.pi/self.s) * math.cos(math.pi*X[0]/self.s) / 3
        G[1] *= math.copysign(1, math.sin(math.pi*X[1]/self.s)) * (math.pi/self.s) * math.cos(math.pi*X[1]/self.s) / 3
        G[2] *= math.copysign(1, math.sin(math.pi*X[2]/self.s)) * (math.pi/self.s) * math.cos(math.pi*X[2]/self.s) / 3

    def I0_texture_tagging_XYZ_wMultiplicativeCombination(self, X, I):
        I[0] *= (abs(math.sin(math.pi*X[0]/self.s))
             *   abs(math.sin(math.pi*X[1]/self.s))
             *   abs(math.sin(math.pi*X[2]/self.s)))**(1./3)

    def I0_texture_tagging_XYZ_wMultiplicativeCombination_wGrad(self, X, I, G):
        self.I0_texture_tagging_XYZ_wMultiplicativeCombination(X, I)
        G[0] *= math.copysign(1, math.sin(math.pi*X[0]/self.s)) * (math.pi/self.s) * math.cos(math.pi*X[0]/self.s) * abs(math.sin(math.pi*X[1]/self.s)) * abs(math.sin(math.pi*X[2]/self.s)) / 3 / I[0]**2
        G[1] *= math.copysign(1, math.sin(math.pi*X[1]/self.s)) * (math.pi/self.s) * math.cos(math.pi*X[1]/self.s) * abs(math.sin(math.pi*X[0]/self.s)) * abs(math.sin(math.pi*X[2]/self.s)) / 3 / I[0]**2
        G[2] *= math.copysign(1, math.sin(math.pi*X[2]/self.s)) * (math.pi/self.s) * math.cos(math.pi*X[2]/self.s) * abs(math.sin(math.pi*X[0]/self.s)) * abs(math.sin(math.pi*X[1]/self.s)) / 3 / I[0]**2

    def I0_texture_tagging_XYZ_wDifferentiableCombination(self, X, I):
        I[0] *= (1 + 7 * abs(math.sin(math.pi*X[0]/self.s))
                       * abs(math.sin(math.pi*X[1]/self.s))
                       * abs(math.sin(math.pi*X[2]/self.s)))**(1./3) - 1

    def I0_texture_tagging_XYZ_wDifferentiableCombination_wGrad(self, X, I, G):
        self.I0_texture_tagging_XYZ_wDifferentiableCombination(X, I)
        G[0] *= 7 * math.copysign(1, math.sin(math.pi*X[0]/self.s)) * (math.pi/self.s) * math.cos(math.pi*X[0]/self.s) * abs(math.sin(math.pi*X[1]/self.s)) * abs(math.sin(math.pi*X[2]/self.s)) / 3 / (I[0] + 1)
        G[1] *= 7 * math.copysign(1, math.sin(math.pi*X[1]/self.s)) * (math.pi/self.s) * math.cos(math.pi*X[1]/self.s) * abs(math.sin(math.pi*X[0]/self.s)) * abs(math.sin(math.pi*X[2]/self.s)) / 3 / (I[0] + 1)
        G[2] *= 7 * math.copysign(1, math.sin(math.pi*X[2]/self.s)) * (math.pi/self.s) * math.cos(math.pi*X[2]/self.s) * abs(math.sin(math.pi*X[0]/self.s)) * abs(math.sin(math.pi*X[1]/self.s)) / 3 / (I[0] + 1)

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

    def I0_texture_tagging_signed_XY_wAdditiveCombination(self, X, I):
        I[0] *= ((1+math.sin(math.pi*X[0]/self.s-math.pi/2))/2
              +  (1+math.sin(math.pi*X[1]/self.s-math.pi/2))/2) / 2

    def I0_texture_tagging_signed_XY_wAdditiveCombination_wGrad(self, X, I, G):
        self.I0_texture_tagging_signed_XY_wAdditiveCombination(X, I)
        G[0] *= (math.pi/self.s) * math.cos(math.pi*X[0]/self.s-math.pi/2)/2 / 2
        G[1] *= (math.pi/self.s) * math.cos(math.pi*X[1]/self.s-math.pi/2)/2 / 2
        G[2] *= 0.

    def I0_texture_tagging_signed_XY_wMultiplicativeCombination(self, X, I):
        I[0] *= ((1+math.sin(math.pi*X[0]/self.s-math.pi/2))/2
             *   (1+math.sin(math.pi*X[1]/self.s-math.pi/2))/2)**(1./2)

    def I0_texture_tagging_signed_XY_wMultiplicativeCombination_wGrad(self, X, I, G):
        self.I0_texture_tagging_XY_wMultiplicativeCombination(X, I)
        G[0] *= (math.pi/self.s) * math.cos(math.pi*X[0]/self.s-math.pi/2)/2 / 2 / I[0]
        G[1] *= (math.pi/self.s) * math.cos(math.pi*X[1]/self.s-math.pi/2)/2 / 2 / I[0]
        G[2] *= 0.

    def I0_texture_tagging_signed_XY_wDifferentiableCombination(self, X, I):
        I[0] *= (1 + 3 * (1+math.sin(math.pi*X[0]/self.s-math.pi/2))/2
                       * (1+math.sin(math.pi*X[1]/self.s-math.pi/2))/2)**(1./2) - 1

    def I0_texture_tagging_signed_XY_wDifferentiableCombination_wGrad(self, X, I, G):
        self.I0_texture_tagging_signed_XY_wDifferentiableCombination(X, I)
        G[0] *= 3 * (math.pi/self.s) * math.cos(math.pi*X[0]/self.s-math.pi/2)/2 * (1+math.sin(math.pi*X[1]/self.s-math.pi/2))/2 / 2 / (I[0] + 1)
        G[1] *= 3 * (math.pi/self.s) * math.cos(math.pi*X[1]/self.s-math.pi/2)/2 * (1+math.sin(math.pi*X[0]/self.s-math.pi/2))/2 / 2 / (I[0] + 1)
        G[2] *= 0.

    def I0_texture_tagging_signed_XYZ_wAdditiveCombination(self, X, I):
        I[0] *= ((1+math.sin(math.pi*X[0]/self.s-math.pi/2))/2
              +  (1+math.sin(math.pi*X[1]/self.s-math.pi/2))/2
              +  (1+math.sin(math.pi*X[2]/self.s-math.pi/2))/2) / 3

    def I0_texture_tagging_signed_XYZ_wAdditiveCombination_wGrad(self, X, I, G):
        self.I0_texture_tagging_signed_XYZ_wAdditiveCombination(X, I)
        G[0] *= (math.pi/self.s) * math.cos(math.pi*X[0]/self.s-math.pi/2)/2 / 3
        G[1] *= (math.pi/self.s) * math.cos(math.pi*X[1]/self.s-math.pi/2)/2 / 3
        G[2] *= (math.pi/self.s) * math.cos(math.pi*X[2]/self.s-math.pi/2)/2 / 3

    def I0_texture_tagging_signed_XYZ_wMultiplicativeCombination(self, X, I):
        I[0] *= ((1+math.sin(math.pi*X[0]/self.s-math.pi/2))/2
             *   (1+math.sin(math.pi*X[1]/self.s-math.pi/2))/2
             *   (1+math.sin(math.pi*X[2]/self.s-math.pi/2))/2)**(1./3)

    def I0_texture_tagging_signed_XYZ_wMultiplicativeCombination_wGrad(self, X, I, G):
        self.I0_texture_tagging_XYZ_wMultiplicativeCombination(X, I)
        G[0] *= (math.pi/self.s) * math.cos(math.pi*X[0]/self.s-math.pi/2)/2 / 3 / I[0]
        G[1] *= (math.pi/self.s) * math.cos(math.pi*X[1]/self.s-math.pi/2)/2 / 3 / I[0]
        G[2] *= (math.pi/self.s) * math.cos(math.pi*X[2]/self.s-math.pi/2)/2 / 3 / I[0]

    def I0_texture_tagging_signed_XYZ_wDifferentiableCombination(self, X, I):
        I[0] *= (1 + 7 * (1+math.sin(math.pi*X[0]/self.s-math.pi/2))/2
                       * (1+math.sin(math.pi*X[1]/self.s-math.pi/2))/2
                       * (1+math.sin(math.pi*X[2]/self.s-math.pi/2))/2)**(1./3) - 1

    def I0_texture_tagging_signed_XYZ_wDifferentiableCombination_wGrad(self, X, I, G):
        self.I0_texture_tagging_signed_XYZ_wDifferentiableCombination(X, I)
        G[0] *= 7 * (math.pi/self.s) * math.cos(math.pi*X[0]/self.s-math.pi/2)/2 * (1+math.sin(math.pi*X[1]/self.s-math.pi/2))/2 * (1+math.sin(math.pi*X[2]/self.s-math.pi/2))/2 / 3 / (I[0] + 1)
        G[1] *= 7 * (math.pi/self.s) * math.cos(math.pi*X[1]/self.s-math.pi/2)/2 * (1+math.sin(math.pi*X[0]/self.s-math.pi/2))/2 * (1+math.sin(math.pi*X[2]/self.s-math.pi/2))/2 / 3 / (I[0] + 1)
        G[2] *= 7 * (math.pi/self.s) * math.cos(math.pi*X[2]/self.s-math.pi/2)/2 * (1+math.sin(math.pi*X[0]/self.s-math.pi/2))/2 * (1+math.sin(math.pi*X[1]/self.s-math.pi/2))/2 / 3 / (I[0] + 1)

    def I0_noise_no(self, I):
        pass

    def I0_noise_no_wGrad(self, I, G):
        pass

    def I0_noise_normal(self, I):
        I[0] += random.normalvariate(self.avg, self.std)

    def I0_noise_normal_wGrad(self, I, G):
        self.I0_noise_normal(I)
        G[:] += [2*random.normalvariate(self.avg, self.std) for k in range(len(G))]
