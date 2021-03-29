#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2016-2021                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

import dolfin
import numpy

import myPythonLibrary    as mypy
import myVTKPythonLibrary as myvtk

import dolfin_warp as dwarp

################################################################################

def getScalingFactor(scalar_type_as_string):
    if   (scalar_type_as_string == 'unsigned char' ): return float(2**8 -1)
    elif (scalar_type_as_string == 'unsigned short'): return float(2**16-1)
    elif (scalar_type_as_string == 'unsigned int'  ): return float(2**32-1)
    elif (scalar_type_as_string == 'unsigned long' ): return float(2**64-1)
    elif (scalar_type_as_string == 'float'         ): return 1.
    elif (scalar_type_as_string == 'double'        ): return 1.
    else: assert (0), "Wrong image scalar type. Aborting."

class ExprIm2(dolfin.Expression):
    def __init__(self, filename=None, Z=0., **kwargs):
        if filename is not None:
            self.init_image(filename=filename)
        self.X = numpy.array([float()]*2+[Z])

    def init_image(self, filename):
        self.image = myvtk.readImage(
            filename=filename,
            verbose=0)
        self.s = getScalingFactor(
            scalar_type_as_string=self.image.GetScalarTypeAsString())
        self.interpolator = myvtk.getImageInterpolator(
            image=self.image,
            mode="linear",
            out_value=0.,
            verbose=0)

    def eval(self, Expr, X):
        #print("    X = " + str(X))
        self.X[0:2] = X[0:2]
        #print("    X = " + str(self.X))
        self.interpolator.Interpolate(self.X, Expr)
        #print("    Expr = " + str(Expr))
        Expr /= self.s
        #print("    Expr = " + str(Expr))

class ExprIm3(dolfin.Expression):
    def __init__(self, filename=None, **kwargs):
        if filename is not None:
            self.init_image(filename=filename)

    def init_image(self, filename):
        self.image = myvtk.readImage(
            filename=filename,
            verbose=0)
        self.s = getScalingFactor(
            scalar_type_as_string=self.image.GetScalarTypeAsString())
        self.interpolator = myvtk.getImageInterpolator(
            image=self.image,
            mode="linear",
            out_value=0.,
            verbose=0)

    def eval(self, Expr, X):
        #print("    X = " + str(X))
        self.interpolator.Interpolate(X, Expr)
        #print("    Expr = " + str(Expr))
        Expr /= self.s
        #print("    Expr = " + str(Expr))

class ExprGradIm2(dolfin.Expression):
    def __init__(self, filename=None, Z=0., **kwargs):
        if filename is not None:
            self.init_image(filename=filename)
        self.X = numpy.array([float()]*2+[Z])

    def init_image(self, filename):
        self.image = myvtk.readImage(
            filename=filename,
            verbose=0)
        self.s = getScalingFactor(
            scalar_type_as_string=self.image.GetScalarTypeAsString())
        self.image = myvtk.addImageGradient(
            image=self.image,
            verbose=0)
        self.interpolator = myvtk.getImageInterpolator(
            image=self.image,
            mode="nearest",
            out_value=self.s,
            verbose=0)

    def value_shape(self):
        return (2,)

    def eval(self, Expr, X):
        #print("    X = " + str(X))
        self.X[0:2] = X[0:2]
        #print("    X = " + str(self.X))
        self.interpolator.Interpolate(self.X, Expr)
        #print("    Expr = " + str(Expr))
        Expr /= self.s

class ExprGradIm3(dolfin.Expression):
    def __init__(self, filename=None, **kwargs):
        if filename is not None:
            self.init_image(filename=filename)

    def init_image(self, filename):
        self.image = myvtk.readImage(
            filename=filename,
            verbose=0)
        self.s = getScalingFactor(
            scalar_type_as_string=self.image.GetScalarTypeAsString())
        self.image = myvtk.addImageGradient(
            image=self.image,
            verbose=0)
        self.interpolator = myvtk.getImageInterpolator(
            image=self.image,
            mode="nearest",
            out_value=self.s,
            verbose=0)

    def value_shape(self):
        return (3,)

    def eval(self, Expr, X):
        #print("    X = " + str(X))
        self.interpolator.Interpolate(X, Expr)
        #print("    Expr = " + str(Expr))
        Expr /= self.s

class ExprHessIm2(dolfin.Expression):
    def __init__(self, filename=None, Z=0., **kwargs):
        if filename is not None:
            self.init_image(filename=filename)
        self.X = numpy.array([float()]*2+[Z])

    def init_image(self, filename):
        self.image = myvtk.readImage(
            filename=filename,
            verbose=0)
        self.s = getScalingFactor(
            scalar_type_as_string=self.image.GetScalarTypeAsString())
        self.image = myvtk.addImageHessian(
            image=self.image,
            verbose=0)
        self.interpolator = myvtk.getImageInterpolator(
            image=self.image,
            mode="nearest",
            out_value=self.s,
            verbose=0)

    def value_shape(self):
        return (2,2)

    def eval(self, Expr, X):
        self.X[0:2] = X[0:2]
        self.interpolator.Interpolate(self.X, Expr)
        Expr /= self.s

class ExprHessIm3(dolfin.Expression):
    def __init__(self, filename=None, **kwargs):
        if filename is not None:
            self.init_image(filename=filename)

    def init_image(self, filename):
        self.image = myvtk.readImage(
            filename=filename,
            verbose=0)
        self.s = getScalingFactor(
            scalar_type_as_string=self.image.GetScalarTypeAsString())
        self.image = myvtk.addImageHessian(
            image=self.image,
            verbose=0)
        self.interpolator = myvtk.getImageInterpolator(
            image=self.image,
            mode="nearest",
            out_value=self.s,
            verbose=0)

    def value_shape(self):
        return (3,3)

    def eval(self, Expr, X):
        self.interpolator.Interpolate(X, Expr)
        Expr /= self.s

class ExprDefIm2(dolfin.Expression):
    def __init__(self, U, filename=None, scaling=[1.,0.], Z=0., **kwargs):
        if filename is not None:
            self.init_image(filename=filename)
        self.U = U
        self.UX = numpy.empty(2)
        self.x = numpy.array([float()]*2+[Z])
        self.scaling = scaling

    def init_image(self, filename):
        self.image = myvtk.readImage(
            filename=filename,
            verbose=0)
        self.s = getScalingFactor(
            scalar_type_as_string=self.image.GetScalarTypeAsString())
        self.interpolator = myvtk.getImageInterpolator(
            image=self.image,
            mode="linear",
            out_value=0.,
            verbose=0)

    def eval(self, Expr, X):
        #print("    X = " + str(X))
        self.U.eval(self.UX, X)
        #print("    UX = " + str(self.UX))
        self.x[0:2] = X[0:2] + self.UX[0:2]
        #print("    x = " + str(self.x))
        self.interpolator.Interpolate(self.x, Expr)
        #print("    Expr = " + str(Expr))
        Expr /= self.s
        #print("    Expr = " + str(Expr))
        Expr *= self.scaling[0]
        Expr += self.scaling[1]
        #print("    Expr = " + str(Expr))

class ExprDefIm3(dolfin.Expression):
    def __init__(self, U, filename=None, scaling=[1.,0.], **kwargs):
        if filename is not None:
            self.init_image(filename=filename)
        self.U = U
        self.UX = numpy.empty(3)
        self.x = numpy.empty(3)
        self.scaling = scaling

    def init_image(self, filename):
        self.image = myvtk.readImage(
            filename=filename,
            verbose=0)
        self.s = getScalingFactor(
            scalar_type_as_string=self.image.GetScalarTypeAsString())
        self.interpolator = myvtk.getImageInterpolator(
            image=self.image,
            mode="linear",
            out_value=0.,
            verbose=0)

    def eval(self, Expr, X):
        #print("    X = " + str(X))
        self.U.eval(self.UX, X)
        #print("    UX = " + str(self.UX))
        self.x[:] = X + self.UX
        #print("    x = " + str(self.x))
        self.interpolator.Interpolate(self.x, Expr)
        #print("    Expr = " + str(Expr))
        Expr /= self.s
        #print("    Expr = " + str(Expr))
        Expr *= self.scaling[0]
        Expr += self.scaling[1]
        #print("    Expr = " + str(Expr))

class ExprGradDefIm2(dolfin.Expression):
    def __init__(self, U, filename=None, scaling=[1.,0.], Z=0., **kwargs):
        if filename is not None:
            self.init_image(filename=filename)
        self.U = U
        self.UX = numpy.empty(2)
        self.x = numpy.array([float()]*2+[Z])
        self.scaling = scaling

    def init_image(self, filename):
        self.image = myvtk.readImage(
            filename=filename,
            verbose=0)
        self.s = getScalingFactor(
            scalar_type_as_string=self.image.GetScalarTypeAsString())
        self.image = myvtk.addImageGradient(
            image=self.image,
            verbose=0)
        self.interpolator = myvtk.getImageInterpolator(
            image=self.image,
            mode="nearest",
            out_value=self.s,
            verbose=0)

    def value_shape(self):
        return (2,)

    def eval(self, Expr, X):
        #print("    X = " + str(X))
        self.U.eval(self.UX, X)
        #print("    UX = " + str(self.UX))
        self.x[0:2] = X[0:2] + self.UX[0:2]
        #print("    x = " + str(self.x))
        self.interpolator.Interpolate(self.x, Expr)
        #print("    Expr = " + str(Expr))
        Expr /= self.s
        #print("    Expr = " + str(Expr))
        Expr *= self.scaling[0]
        #print("    Expr = " + str(Expr))

class ExprGradDefIm3(dolfin.Expression):
    def __init__(self, U, filename=None, scaling=[1.,0.], **kwargs):
        if filename is not None:
            self.init_image(filename=filename)
        self.U = U
        self.UX = numpy.empty(3)
        self.x = numpy.empty(3)
        self.scaling = scaling

    def init_image(self, filename):
        self.image = myvtk.readImage(
            filename=filename,
            verbose=0)
        self.s = getScalingFactor(
            scalar_type_as_string=self.image.GetScalarTypeAsString())
        self.image = myvtk.addImageGradient(
            image=self.image,
            verbose=0)
        self.interpolator = myvtk.getImageInterpolator(
            image=self.image,
            mode="nearest",
            out_value=self.s,
            verbose=0)

    def value_shape(self):
        return (3,)

    def eval(self, Expr, X):
        #print("    X = " + str(X))
        self.U.eval(self.UX, X)
        #print("    UX = " + str(self.UX))
        self.x[:] = X + self.UX
        #print("    x = " + str(self.x))
        self.interpolator.Interpolate(self.x, Expr)
        #print("    Expr = " + str(Expr))
        Expr /= self.s
        #print("    Expr = " + str(Expr))
        Expr *= self.scaling[0]
        #print("    Expr = " + str(Expr))

class ExprHessDefIm2(dolfin.Expression):
    def __init__(self, U, filename=None, scaling=[1.,0.], Z=0., **kwargs):
        if filename is not None:
            self.init_image(filename=filename)
        self.U = U
        self.UX = numpy.empty(2)
        self.x = numpy.array([float()]*2+[Z])
        self.scaling = scaling

    def init_image(self, filename):
        self.image = myvtk.readImage(
            filename=filename,
            verbose=0)
        self.s = getScalingFactor(
            scalar_type_as_string=self.image.GetScalarTypeAsString())
        self.image = myvtk.addImageHessian(
            image=self.image,
            verbose=0)
        self.interpolator = myvtk.getImageInterpolator(
            image=self.image,
            mode="nearest",
            out_value=self.s,
            verbose=0)

    def value_shape(self):
        return (2,2)

    def eval(self, Expr, X):
        self.U.eval(self.UX, X)
        self.x[0:2] = X[0:2] + self.UX[0:2]
        self.interpolator.Interpolate(self.x, Expr)
        Expr /= self.s
        Expr *= self.scaling[0]

class ExprHessDefIm3(dolfin.Expression):
    def __init__(self, U, filename=None, scaling=[1.,0.], **kwargs):
        if filename is not None:
            self.init_image(filename=filename)
        self.U = U
        self.UX = numpy.empty(3)
        self.x = numpy.empty(3)
        self.scaling = scaling

    def init_image(self, filename):
        self.image = myvtk.readImage(
            filename=filename,
            verbose=0)
        self.s = getScalingFactor(
            scalar_type_as_string=self.image.GetScalarTypeAsString())
        self.image = myvtk.addImageHessian(
            image=self.image,
            verbose=0)
        self.interpolator = myvtk.getImageInterpolator(
            image=self.image,
            mode="nearest",
            out_value=self.s,
            verbose=0)

    def value_shape(self):
        return (3,3)

    def eval(self, Expr, X):
        self.U.eval(self.UX, X)
        self.x[:] = X + self.UX
        self.interpolator.Interpolate(self.x, Expr)
        Expr /= self.s
        Expr *= self.scaling[0]
