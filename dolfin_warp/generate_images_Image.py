#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2016-2026                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

import math
import numpy
import random

################################################################################

class Image():

    def __init__(
            self,
            images,
            structure,
            texture,
            noise):

        self.L = images["L"]

        # structure
        if (structure["type"] == "no"):
            self.I0_structure = self.I0_structure_no
        elif (structure["type"] == "box"):
            self.I0_structure = self.I0_structure_box
            self.Xmin = structure["Xmin"]+[float("-Inf")]*(3-images["n_dim"])
            self.Xmax = structure["Xmax"]+[float("+Inf")]*(3-images["n_dim"])
        elif (structure["type"] in ("ring", "heart")):
            self.C  = structure["C" ] if ("C" in structure) else [images["L"][0]/2, images["L"][1]/2]
            self.Ri = structure["Ri"]
            self.Re = structure["Re"]
            self.R = float()
            if (images["n_dim"] == 2):
                self.I0_structure = self.I0_structure_ring_2
            elif (images["n_dim"] == 3):
                self.I0_structure = self.I0_structure_ring_3
                self.Zmin = structure.Zmin if ("Zmin" in structure) else 0.
                self.Zmax = structure.Zmax if ("Zmax" in structure) else images["L"][2]
            else:
                assert (0), "n_dim must be \"2\" or \"3 for \"ring\"/\"heart\" type structure. Aborting."
        else:
            assert (0), "structure type must be \"no\", \"box\", \"ring\" or \"heart\". Aborting."

        # texture
        if (texture["type"] == "no"):
            self.I0_texture = self.I0_texture_no
        elif (texture["type"].startswith("tagging")):
            if   (images["n_dim"] == 1):
                if ("-signed" in texture["type"]):
                    self.I0_texture = self.I0_texture_tagging_signed_X
                else:
                    self.I0_texture = self.I0_texture_tagging_X
            elif (images["n_dim"] == 2):
                if ("-signed" in texture["type"]):
                    if   ("-addComb" in texture["type"]):
                        self.I0_texture = self.I0_texture_tagging_signed_XY_wAdditiveCombination
                    elif ("-diffComb" in texture["type"]):
                        self.I0_texture = self.I0_texture_tagging_signed_XY_wDifferentiableCombination
                    else:
                        self.I0_texture = self.I0_texture_tagging_signed_XY_wMultiplicativeCombination
                else:
                    if   ("-addComb" in texture["type"]):
                        self.I0_texture = self.I0_texture_tagging_XY_wAdditiveCombination
                    elif ("-diffComb" in texture["type"]):
                        self.I0_texture = self.I0_texture_tagging_XY_wDifferentiableCombination
                    else:
                        self.I0_texture = self.I0_texture_tagging_XY_wMultiplicativeCombination
            elif (images["n_dim"] == 3):
                if ("-signed" in texture["type"]):
                    if   ("-addComb" in texture["type"]):
                        self.I0_texture = self.I0_texture_tagging_signed_XYZ_wAdditiveCombination
                    elif ("-diffComb" in texture["type"]):
                        self.I0_texture = self.I0_texture_tagging_signed_XYZ_wDifferentiableCombination
                    else:
                        self.I0_texture = self.I0_texture_tagging_signed_XYZ_wMultiplicativeCombination
                else:
                    if   ("-addComb" in texture["type"]):
                        self.I0_texture = self.I0_texture_tagging_XYZ_wAdditiveCombination
                    elif ("-diffComb" in texture["type"]):
                        self.I0_texture = self.I0_texture_tagging_XYZ_wDifferentiableCombination
                    else:
                        self.I0_texture = self.I0_texture_tagging_XYZ_wMultiplicativeCombination
            else:
                assert (0), "n_dim must be \"1\", \"2\" or \"3\". Aborting."
        elif (texture["type"].startswith("taggX")):
            if ("-signed" in texture["type"]):
                self.I0_texture = self.I0_texture_tagging_signed_X
            else:
                self.I0_texture = self.I0_texture_tagging_X
        elif (texture["type"].startswith("taggY")):
            if ("-signed" in texture["type"]):
                self.I0_texture = self.I0_texture_tagging_signed_Y
            else:
                self.I0_texture = self.I0_texture_tagging_Y
        elif (texture["type"].startswith("taggZ")):
            if ("-signed" in texture["type"]):
                self.I0_texture = self.I0_texture_tagging_signed_Z
            else:
                self.I0_texture = self.I0_texture_tagging_Z
        else:
            assert (0), "texture type must be \"no\", \"tagging\", \"taggX\", \"taggY\" or \"taggZ\". Aborting."

        if (texture["type"] != "no"): # the tagging parameters are common to all tagging textures
            self.s = texture["s"]
            self.X0 = numpy.empty(3)
            self.X0[0] = texture["X0"] if ("X0" in texture) else 0.
            if (images["n_dim"] >= 2): self.X0[1] = texture["Y0"] if ("Y0" in texture) else 0.
            if (images["n_dim"] >= 3): self.X0[2] = texture["Z0"] if ("Z0" in texture) else 0.

        # noise (MG20220818: Should use dwarp.Noise)
        if (noise["type"] == "no"):
            self.I0_noise = self.I0_noise_no
        elif (noise["type"] == "normal"):
            self.I0_noise = self.I0_noise_normal
            self.avg = noise["avg"] if ("avg" in noise) else 0.
            self.std = noise["stdev"]
        else:
            assert (0), "noise type must be \"no\" or \"normal\". Aborting."

    def I0(self, X, I):
        self.I0_structure(X, I)
        self.I0_texture(X, I)
        self.I0_noise(I)
    def I0_structure_no(self, X, I):
        I[0] = 1.
    def I0_structure_box(self, X, I):
        if all(numpy.greater_equal(X, self.Xmin)) and all(numpy.less_equal(X, self.Xmax)):
            I[0] = 1.
        else:
            I[0] = 0.
    def I0_structure_ring_2(self, X, I):
        self.R = ((X[0]-self.C[0])**2 + (X[1]-self.C[1])**2)**(1./2)
        if (self.R >= self.Ri) and (self.R <= self.Re):
            I[0] = 1.
        else:
            I[0] = 0.
    def I0_structure_ring_3(self, X, I):
        self.R = ((X[0]-self.C[0])**2 + (X[1]-self.C[1])**2)**(1./2)
        if (self.R >= self.Ri) and (self.R <= self.Re) and (X[2] >= self.Zmin) and (X[2] <= self.Zmax):
            I[0] = 1.
        else:
            I[0] = 0.
    def I0_texture_no(self, X, I):
        I[0] *= 1.
    def I0_texture_tagging_X(self, X, I):
        I[0] *= abs(math.sin(math.pi*(X[0]-self.X0[0])/self.s))
    def I0_texture_tagging_Y(self, X, I):
        I[0] *= abs(math.sin(math.pi*(X[1]-self.X0[1])/self.s))
    def I0_texture_tagging_Z(self, X, I):
        I[0] *= abs(math.sin(math.pi*(X[2]-self.X0[2])/self.s))
    def I0_texture_tagging_XY_wAdditiveCombination(self, X, I):
        I[0] *= (abs(math.sin(math.pi*(X[0]-self.X0[0])/self.s))
               + abs(math.sin(math.pi*(X[1]-self.X0[1])/self.s)))/2
    def I0_texture_tagging_XY_wMultiplicativeCombination(self, X, I):
        I[0] *= (abs(math.sin(math.pi*(X[0]-self.X0[0])/self.s))
             *   abs(math.sin(math.pi*(X[1]-self.X0[1])/self.s)))**(1./2)
    def I0_texture_tagging_XY_wDifferentiableCombination(self, X, I):
        I[0] *= (1 + 3 * abs(math.sin(math.pi*(X[0]-self.X0[0])/self.s))
                       * abs(math.sin(math.pi*(X[1]-self.X0[1])/self.s)))**(1./2) - 1
    def I0_texture_tagging_XYZ_wAdditiveCombination(self, X, I):
        I[0] *= (abs(math.sin(math.pi*(X[0]-self.X0[0])/self.s))
               + abs(math.sin(math.pi*(X[1]-self.X0[1])/self.s))
               + abs(math.sin(math.pi*(X[2]-self.X0[2])/self.s)))/3
    def I0_texture_tagging_XYZ_wMultiplicativeCombination(self, X, I):
        I[0] *= (abs(math.sin(math.pi*(X[0]-self.X0[0])/self.s))
             *   abs(math.sin(math.pi*(X[1]-self.X0[1])/self.s))
             *   abs(math.sin(math.pi*(X[2]-self.X0[2])/self.s)))**(1./3)
    def I0_texture_tagging_XYZ_wDifferentiableCombination(self, X, I):
        I[0] *= (1 + 7 * abs(math.sin(math.pi*(X[0]-self.X0[0])/self.s))
                       * abs(math.sin(math.pi*(X[1]-self.X0[1])/self.s))
                       * abs(math.sin(math.pi*(X[2]-self.X0[2])/self.s)))**(1./3) - 1
    def I0_texture_tagging_signed_X(self, X, I):
        I[0] *= (1+math.sin(math.pi*(X[0]-self.X0[0])/self.s-math.pi/2))/2
    def I0_texture_tagging_signed_Y(self, X, I):
        I[0] *= (1+math.sin(math.pi*(X[1]-self.X0[1])/self.s-math.pi/2))/2
    def I0_texture_tagging_signed_Z(self, X, I):
        I[0] *= (1+math.sin(math.pi*(X[1]-self.X0[1])/self.s-math.pi/2))/2
    def I0_texture_tagging_signed_XY_wAdditiveCombination(self, X, I):
        I[0] *= ((1+math.sin(math.pi*(X[0]-self.X0[0])/self.s-math.pi/2))/2
              +  (1+math.sin(math.pi*(X[1]-self.X0[1])/self.s-math.pi/2))/2) / 2
    def I0_texture_tagging_signed_XY_wMultiplicativeCombination(self, X, I):
        I[0] *= ((1+math.sin(math.pi*(X[0]-self.X0[0])/self.s-math.pi/2))/2
             *   (1+math.sin(math.pi*(X[1]-self.X0[1])/self.s-math.pi/2))/2)**(1./2)
    def I0_texture_tagging_signed_XY_wDifferentiableCombination(self, X, I):
        I[0] *= (1 + 3 * (1+math.sin(math.pi*(X[0]-self.X0[0])/self.s-math.pi/2))/2
                       * (1+math.sin(math.pi*(X[1]-self.X0[1])/self.s-math.pi/2))/2)**(1./2) - 1
    def I0_texture_tagging_signed_XYZ_wAdditiveCombination(self, X, I):
        I[0] *= ((1+math.sin(math.pi*(X[0]-self.X0[0])/self.s-math.pi/2))/2
              +  (1+math.sin(math.pi*(X[1]-self.X0[1])/self.s-math.pi/2))/2
              +  (1+math.sin(math.pi*(X[2]-self.X0[2])/self.s-math.pi/2))/2) / 3
    def I0_texture_tagging_signed_XYZ_wMultiplicativeCombination(self, X, I):
        I[0] *= ((1+math.sin(math.pi*(X[0]-self.X0[0])/self.s-math.pi/2))/2
             *   (1+math.sin(math.pi*(X[1]-self.X0[1])/self.s-math.pi/2))/2
             *   (1+math.sin(math.pi*(X[2]-self.X0[2])/self.s-math.pi/2))/2)**(1./3)
    def I0_texture_tagging_signed_XYZ_wDifferentiableCombination(self, X, I):
        I[0] *= (1 + 7 * (1+math.sin(math.pi*(X[0]-self.X0[0])/self.s-math.pi/2))/2
                       * (1+math.sin(math.pi*(X[1]-self.X0[1])/self.s-math.pi/2))/2
                       * (1+math.sin(math.pi*(X[2]-self.X0[2])/self.s-math.pi/2))/2)**(1./3) - 1
    def I0_noise_no(self, I):
        pass
    def I0_noise_normal(self, I):
        I[0] += random.normalvariate(self.avg, self.std)