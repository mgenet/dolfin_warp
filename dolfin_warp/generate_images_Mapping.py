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
import math
import numpy
import os
import random
import vtk

import myPythonLibrary    as mypy
import myVTKPythonLibrary as myvtk

import dolfin_warp as dwarp

################################################################################

class Mapping():

    def __init__(
            self,
            images,
            structure,
            deformation,
            evolution,
            generate_image_gradient=0):

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
            Fxx = self.deformation["Fxx"] if ("Fxx" in self.deformation) else 1.
            Fyy = self.deformation["Fyy"] if ("Fyy" in self.deformation) else 1.
            Fzz = self.deformation["Fzz"] if ("Fzz" in self.deformation) else 1.
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
        #print("x = "+str(x))
        self.x_inplane[0] = x[0] - self.L[0]/2
        self.x_inplane[1] = x[1] - self.L[1]/2
        #print("x_inplane = "+str(self.x_inplane))
        self.rt[0] = numpy.linalg.norm(self.x_inplane)
        self.rt[1] = math.atan2(self.x_inplane[1], self.x_inplane[0])
        #print("rt = "+str(self.rt))
        self.RT[:] = numpy.dot(self.Ainv, self.rt-self.B)
        #print("RT = "+str(self.RT))
        X[0] = self.RT[0] * math.cos(self.RT[1]) + self.L[0]/2
        X[1] = self.RT[0] * math.sin(self.RT[1]) + self.L[1]/2
        X[2] = x[2]
        #print("X = "+str(X))
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
            #print("F = "+str(Finv))
            Finv[:,:] = numpy.linalg.inv(Finv)
            #print("Finv = "+str(Finv))
            self.R[0,0] = +math.cos(self.RT[1])
            self.R[0,1] = +math.sin(self.RT[1])
            self.R[0,2] = 0.
            self.R[1,0] = -math.sin(self.RT[1])
            self.R[1,1] = +math.cos(self.RT[1])
            self.R[1,2] = 0.
            self.R[2,0] = 0.
            self.R[2,1] = 0.
            self.R[2,2] = 1.
            #print("R = "+str(self.R))
            Finv[:] = numpy.dot(numpy.transpose(self.R), numpy.dot(Finv, self.R))
            #print("Finv = "+str(Finv))

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
        #print("X = "+str(X))
        self.X_inplane[0] = X[0] - self.L[0]/2
        self.X_inplane[1] = X[1] - self.L[1]/2
        #print("X_inplane = "+str(self.X_inplane))
        self.RT[0] = numpy.linalg.norm(self.X_inplane)
        self.RT[1] = math.atan2(self.X_inplane[1], self.X_inplane[0])
        #print("RT = "+str(self.RT))
        self.rt[:] = numpy.dot(self.A, self.RT) + self.B
        #print("rt = "+str(self.rt))
        x[0] = self.rt[0] * math.cos(self.rt[1]) + self.L[0]/2
        x[1] = self.rt[0] * math.sin(self.rt[1]) + self.L[1]/2
        x[2] = X[2]
        #print("x = "+str(x))
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
            #print("F = "+str(F))
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
            #print("F = "+str(F))
