#coding=utf8

########################################################################
###                                                                  ###
### Created by Martin Genet, 2016                                    ###
###                                                                  ###
### École Polytechnique, Palaiseau, France                           ###
###                                                                  ###
########################################################################

import dolfin
import numpy

import myVTKPythonLibrary as myVTK

########################################################################

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
        #self.evalcount = 0

    def init_image(self, filename):
        self.image = myVTK.readImage(
            filename=filename,
            verbose=0)
        self.s = getScalingFactor(self.image.GetScalarTypeAsString())
        self.interpolator = myVTK.createImageInterpolator(
            image=self.image,
            verbose=0)

    def eval(self, Im, X):
        #print "Im"
        self.X[0] = X[0]
        self.X[1] = X[1]
        #print "    X = " + str(X)
        self.interpolator.Interpolate(self.X, Im)
        #print "    Im = " + str(Im)
        Im /= self.s
        #print "    Im = " + str(Im)
        #self.evalcount += 1

class ExprIm3(dolfin.Expression):
    def __init__(self, filename, **kwargs):
        if filename is not None:
            self.init_image(filename=filename)
        #self.evalcount = 0

    def init_image(self, filename):
        self.image = myVTK.readImage(
            filename=filename,
            verbose=0)
        self.s = getScalingFactor(self.image.GetScalarTypeAsString())
        self.interpolator = myVTK.createImageInterpolator(
            image=self.image,
            verbose=0)

    def eval(self, Im, X):
        #print "Im"
        #print "    X = " + str(X)
        self.interpolator.Interpolate(X, Im)
        #print "    Im = " + str(Im)
        Im /= self.s
        #print "    Im = " + str(Im)
        #self.evalcount += 1

class ExprGradIm2(dolfin.Expression): # for some reason this does not work with quadrature finite elements, hence the following scalar gradients definition
    def __init__(self, filename=None, Z=0., **kwargs):
        if filename is not None:
            self.init_image(filename=filename)
        self.X = numpy.array([float()]*2+[Z])

    def init_image(self, filename):
        self.image = myVTK.readImage(
            filename=filename,
            verbose=0)
        self.s = getScalingFactor(self.image.GetScalarTypeAsString())
        self.image = myVTK.computeImageGradient(
            image=self.image,
            verbose=0)
        self.interpolator = myVTK.createImageInterpolator(
            image=self.image,
            verbose=0)

    def value_shape(self):
        return (2,)

    def eval(self, GradIm, X):
        self.X[0] = X[0]
        self.X[1] = X[1]
        self.interpolator.Interpolate(self.X, GradIm)
        GradIm /= self.s

class ExprGradXIm2(dolfin.Expression):
    def __init__(self, filename=None, Z=0., **kwargs):
        if filename is not None:
            self.init_image(filename=filename)
        self.X = numpy.array([float()]*2+[Z])

    def init_image(self, filename):
        self.image = myVTK.readImage(
            filename=filename,
            verbose=0)
        self.s = getScalingFactor(self.image.GetScalarTypeAsString())
        self.image = myVTK.computeImageGradient(
            image=self.image,
            verbose=0)
        self.interpolator = myVTK.createImageInterpolator(
            image=self.image,
            verbose=0)

    def eval(self, GradXIm, X):
        self.X[0] = X[0]
        self.X[1] = X[1]
        GradXIm[0] = self.interpolator.Interpolate(self.X[0], self.X[1], self.X[2], 0)
        GradXIm /= self.s

class ExprGradYIm2(dolfin.Expression):
    def __init__(self, filename=None, Z=0., **kwargs):
        if filename is not None:
            self.init_image(filename=filename)
        self.X = numpy.array([float()]*2+[Z])

    def init_image(self, filename):
        self.image = myVTK.readImage(
            filename=filename,
            verbose=0)
        self.s = getScalingFactor(self.image.GetScalarTypeAsString())
        self.image = myVTK.computeImageGradient(
            image=self.image,
            verbose=0)
        self.interpolator = myVTK.createImageInterpolator(
            image=self.image,
            verbose=0)

    def eval(self, GradYIm, X):
        self.X[0] = X[0]
        self.X[1] = X[1]
        GradYIm[0] = self.interpolator.Interpolate(self.X[0], self.X[1], self.X[2], 1)
        GradYIm /= self.s

class ExprGradIm3(dolfin.Expression): # for some reason this does not work with quadrature finite elements, hence the following scalar gradients definition
    def __init__(self, filename=None, **kwargs):
        if filename is not None:
            self.init_image(filename=filename)

    def init_image(self, filename):
        self.image = myVTK.readImage(
            filename=filename,
            verbose=0)
        self.s = getScalingFactor(self.image.GetScalarTypeAsString())
        self.image = myVTK.computeImageGradient(
            image=self.image,
            verbose=0)
        self.interpolator = myVTK.createImageInterpolator(
            image=self.image,
            verbose=0)

    def value_shape(self):
        return (3,)

    def eval(self, GradIm, X):
        self.interpolator.Interpolate(X, GradIm)
        GradIm /= self.s

class ExprGradXIm3(dolfin.Expression):
    def __init__(self, filename=None, **kwargs):
        if filename is not None:
            self.init_image(filename=filename)

    def init_image(self, filename):
        self.image = myVTK.readImage(
            filename=filename,
            verbose=0)
        self.s = getScalingFactor(self.image.GetScalarTypeAsString())
        self.image = myVTK.computeImageGradient(
            image=self.image,
            verbose=0)
        self.interpolator = myVTK.createImageInterpolator(
            image=self.image,
            verbose=0)

    def eval(self, GradXIm, X):
        GradXIm[0] = self.interpolator.Interpolate(X[0], X[1], X[2], 0)
        GradXIm /= self.s

class ExprGradYIm3(dolfin.Expression):
    def __init__(self, filename=None, **kwargs):
        if filename is not None:
            self.init_image(filename=filename)

    def init_image(self, filename):
        self.image = myVTK.readImage(
            filename=filename,
            verbose=0)
        self.s = getScalingFactor(self.image.GetScalarTypeAsString())
        self.image = myVTK.computeImageGradient(
            image=self.image,
            verbose=0)
        self.interpolator = myVTK.createImageInterpolator(
            image=self.image,
            verbose=0)

    def eval(self, GradYIm, X):
        GradYIm[0] = self.interpolator.Interpolate(X[0], X[1], X[2], 1)
        GradYIm /= self.s

class ExprGradZIm3(dolfin.Expression):
    def __init__(self, filename=None, **kwargs):
        if filename is not None:
            self.init_image(filename=filename)

    def init_image(self, filename):
        self.image = myVTK.readImage(
            filename=filename,
            verbose=0)
        self.s = getScalingFactor(self.image.GetScalarTypeAsString())
        self.image = myVTK.computeImageGradient(
            image=self.image,
            verbose=0)
        self.interpolator = myVTK.createImageInterpolator(
            image=self.image,
            verbose=0)

    def eval(self, GradZIm, X):
        GradZIm[0] = self.interpolator.Interpolate(X[0], X[1], X[2], 2)
        GradZIm /= self.s

class ExprDefIm2(dolfin.Expression):
    def __init__(self, U, filename=None, Z=0., **kwargs):
        if filename is not None:
            self.init_image(filename=filename)
        self.U = U
        self.UX = numpy.array([float()]*2)
        self.x = numpy.array([float()]*2+[Z])

    def init_image(self, filename):
        self.image = myVTK.readImage(
            filename=filename,
            verbose=0)
        self.s = getScalingFactor(self.image.GetScalarTypeAsString())
        self.interpolator = myVTK.createImageInterpolator(
            image=self.image,
            verbose=0)

    def eval(self, DefIm, X):
        #print "DefIm"
        #print "    U = " + str(U.vector().array())
        #print "    X = " + str(X)
        #print "    UX = " + str(self.UX)
        self.U.eval(self.UX, X)
        #print "    UX = " + str(self.UX)
        self.x[0] = X[0] + self.UX[0]
        self.x[1] = X[1] + self.UX[1]
        #print "    x = " + str(self.x)
        #print "    DefIm = " + str(DefIm)
        self.interpolator.Interpolate(self.x, DefIm)
        #print "    DefIm = " + str(DefIm)
        DefIm /= self.s
        #print "    DefIm = " + str(DefIm)

class ExprDefIm3(dolfin.Expression):
    def __init__(self, U, filename=None, **kwargs):
        if filename is not None:
            self.init_image(filename=filename)
        self.U = U
        self.UX = numpy.array([float()]*3)
        self.x = numpy.array([float()]*3)

    def init_image(self, filename):
        self.image = myVTK.readImage(
            filename=filename,
            verbose=0)
        self.s = getScalingFactor(self.image.GetScalarTypeAsString())
        self.interpolator = myVTK.createImageInterpolator(
            image=self.image,
            verbose=0)

    def eval(self, DefIm, X):
        #print "DefIm"
        #print "    U = " + str(U.vector().array())
        #print "    X = " + str(X)
        #print "    UX = " + str(self.UX)
        self.U.eval(self.UX, X)
        #print "    UX = " + str(self.UX)
        self.x[:] = X + self.UX
        #print "    x = " + str(self.x)
        #print "    DefIm = " + str(DefIm)
        self.interpolator.Interpolate(self.x, DefIm)
        #print "    DefIm = " + str(DefIm)
        DefIm /= self.s
        #print "    DefIm = " + str(DefIm)

class ExprGradDefIm2(dolfin.Expression): # for some reason this does not work with quadrature finite elements, hence the following scalar gradients definition
    def __init__(self, U, filename=None, Z=0., **kwargs):
        if filename is not None:
            self.init_image(filename=filename)
        self.U = U
        self.UX = numpy.array([float()]*2)
        self.x = numpy.array([float()]*2+[Z])

    def init_image(self, filename):
        self.image = myVTK.readImage(
            filename=filename,
            verbose=0)
        self.s = getScalingFactor(self.image.GetScalarTypeAsString())
        self.image = myVTK.computeImageGradient(
            image=self.image,
            verbose=0)
        self.interpolator = myVTK.createImageInterpolator(
            image=self.image,
            verbose=0)

    def value_shape(self):
        return (2,)

    def eval(self, GradDefIm, X):
        #print "GradDefIm"
        #print "    U = " + str(U.vector().array())
        #print "    X = " + str(X)
        #print "    UX = " + str(self.UX)
        self.U.eval(self.UX, X)
        #print "    UX = " + str(self.UX)
        self.x[0] = X[0] + self.UX[0]
        self.x[1] = X[1] + self.UX[1]
        #print "    x = " + str(self.x)
        #print "    GradDefIm = " + str(GradDefIm)
        self.interpolator.Interpolate(self.x, GradDefIm)
        #print "    GradDefIm = " + str(GradDefIm)
        GradDefIm /= self.s
        #print "    GradDefIm = " + str(GradDefIm)

class ExprGradXDefIm2(dolfin.Expression):
    def __init__(self, U, filename=None, Z=0., **kwargs):
        if filename is not None:
            self.init_image(filename=filename)
        self.U = U
        self.UX = numpy.array([float()]*2)
        self.x = numpy.array([float()]*2+[Z])

    def init_image(self, filename):
        self.image = myVTK.readImage(
            filename=filename,
            verbose=0)
        self.s = getScalingFactor(self.image.GetScalarTypeAsString())
        self.image = myVTK.computeImageGradient(
            image=self.image,
            verbose=0)
        self.interpolator = myVTK.createImageInterpolator(
            image=self.image,
            verbose=0)

    def eval(self, GradXDefIm, X):
        #print "GradXDefIm"
        #print "    U = " + str(U.vector().array())
        #print "    X = " + str(X)
        #print "    UX = " + str(self.UX)
        self.U.eval(self.UX, X)
        #print "    UX = " + str(self.UX)
        self.x[0] = X[0] + self.UX[0]
        self.x[1] = X[1] + self.UX[1]
        #print "    x = " + str(self.x)
        #print "    GradXDefIm = " + str(GradXDefIm)
        GradXDefIm[0] = self.interpolator.Interpolate(self.x[0], self.x[1], self.x[2], 0)
        #print "    GradXDefIm = " + str(GradXDefIm)
        GradXDefIm /= self.s
        #print "    GradXDefIm = " + str(GradXDefIm)

class ExprGradYDefIm2(dolfin.Expression):
    def __init__(self, U, filename=None, Z=0., **kwargs):
        if filename is not None:
            self.init_image(filename=filename)
        self.U = U
        self.UX = numpy.array([float()]*2)
        self.x = numpy.array([float()]*2+[Z])

    def init_image(self, filename):
        self.image = myVTK.readImage(
            filename=filename,
            verbose=0)
        self.s = getScalingFactor(self.image.GetScalarTypeAsString())
        self.image = myVTK.computeImageGradient(
            image=self.image,
            verbose=0)
        self.interpolator = myVTK.createImageInterpolator(
            image=self.image,
            verbose=0)

    def eval(self, GradYDefIm, X):
        #print "GradYDefIm"
        #print "    U = " + str(U.vector().array())
        #print "    X = " + str(X)
        #print "    UX = " + str(self.UX)
        self.U.eval(self.UX, X)
        #print "    UX = " + str(self.UX)
        self.x[0] = X[0] + self.UX[0]
        self.x[1] = X[1] + self.UX[1]
        #print "    x = " + str(self.x)
        #print "    GradYDefIm = " + str(GradYDefIm)
        GradYDefIm[0] = self.interpolator.Interpolate(self.x[0], self.x[1], self.x[2], 1)
        #print "    GradYDefIm = " + str(GradYDefIm)
        GradYDefIm /= self.s
        #print "    GradYDefIm = " + str(GradYDefIm)

class ExprGradDefIm3(dolfin.Expression): # for some reason this does not work with quadrature finite elements, hence the following scalar gradients definition
    def __init__(self, U, filename=None, **kwargs):
        if filename is not None:
            self.init_image(filename=filename)
        self.U = U
        self.UX = numpy.array([float()]*3)
        self.x = numpy.array([float()]*3)

    def init_image(self, filename):
        self.image = myVTK.readImage(
            filename=filename,
            verbose=0)
        self.s = getScalingFactor(self.image.GetScalarTypeAsString())
        self.image = myVTK.computeImageGradient(
            image=self.image,
            verbose=0)
        self.interpolator = myVTK.createImageInterpolator(
            image=self.image,
            verbose=0)

    def value_shape(self):
        return (3,)

    def eval(self, GradDefIm, X):
        #print "GradDefIm"
        #print "    U = " + str(U.vector().array())
        #print "    X = " + str(X)
        #print "    UX = " + str(self.UX)
        self.U.eval(self.UX, X)
        #print "    UX = " + str(self.UX)
        self.x[:] = X + self.UX
        #print "    x = " + str(self.x)
        #print "    GradDefIm = " + str(GradDefIm)
        self.interpolator.Interpolate(self.x, GradDefIm)
        #print "    GradDefIm = " + str(GradDefIm)
        GradDefIm /= self.s
        #print "    GradDefIm = " + str(GradDefIm)

class ExprGradXDefIm3(dolfin.Expression):
    def __init__(self, U, filename=None, **kwargs):
        if filename is not None:
            self.init_image(filename=filename)
        self.U = U
        self.UX = numpy.array([float()]*3)
        self.x = numpy.array([float()]*3)

    def init_image(self, filename):
        self.image = myVTK.readImage(
            filename=filename,
            verbose=0)
        self.s = getScalingFactor(self.image.GetScalarTypeAsString())
        self.image = myVTK.computeImageGradient(
            image=self.image,
            verbose=0)
        self.interpolator = myVTK.createImageInterpolator(
            image=self.image,
            verbose=0)

    def eval(self, GradXDefIm, X):
        #print "GradXDefIm"
        #print "    U = " + str(U.vector().array())
        #print "    X = " + str(X)
        #print "    UX = " + str(self.UX)
        self.U.eval(self.UX, X)
        #print "    UX = " + str(self.UX)
        self.x[:] = X + self.UX
        #print "    x = " + str(self.x)
        #print "    GradXDefIm = " + str(GradXDefIm)
        GradXDefIm[0] = self.interpolator.Interpolate(self.x[0], self.x[1], self.x[2], 0)
        #print "    GradXDefIm = " + str(GradXDefIm)
        GradXDefIm /= self.s
        #print "    GradXDefIm = " + str(GradXDefIm)

class ExprGradYDefIm3(dolfin.Expression):
    def __init__(self, U, filename=None, **kwargs):
        if filename is not None:
            self.init_image(filename=filename)
        self.U = U
        self.UX = numpy.array([float()]*3)
        self.x = numpy.array([float()]*3)

    def init_image(self, filename):
        self.image = myVTK.readImage(
            filename=filename,
            verbose=0)
        self.s = getScalingFactor(self.image.GetScalarTypeAsString())
        self.image = myVTK.computeImageGradient(
            image=self.image,
            verbose=0)
        self.interpolator = myVTK.createImageInterpolator(
            image=self.image,
            verbose=0)

    def eval(self, GradYDefIm, X):
        #print "GradYDefIm"
        #print "    U = " + str(U.vector().array())
        #print "    X = " + str(X)
        #print "    UX = " + str(self.UX)
        self.U.eval(self.UX, X)
        #print "    UX = " + str(self.UX)
        self.x[:] = X + self.UX
        #print "    x = " + str(self.x)
        #print "    GradYDefIm = " + str(GradYDefIm)
        GradYDefIm[0] = self.interpolator.Interpolate(self.x[0], self.x[1], self.x[2], 1)
        #print "    GradYDefIm = " + str(GradYDefIm)
        GradYDefIm /= self.s
        #print "    GradYDefIm = " + str(GradYDefIm)

class ExprGradZDefIm3(dolfin.Expression):
    def __init__(self, U, filename=None, **kwargs):
        if filename is not None:
            self.init_image(filename=filename)
        self.U = U
        self.UX = numpy.array([float()]*3)
        self.x = numpy.array([float()]*3)

    def init_image(self, filename):
        self.image = myVTK.readImage(
            filename=filename,
            verbose=0)
        self.s = getScalingFactor(self.image.GetScalarTypeAsString())
        self.image = myVTK.computeImageGradient(
            image=self.image,
            verbose=0)
        self.interpolator = myVTK.createImageInterpolator(
            image=self.image,
            verbose=0)

    def eval(self, GradZDefIm, X):
        #print "GradZDefIm"
        #print "    U = " + str(U.vector().array())
        #print "    X = " + str(X)
        #print "    UX = " + str(self.UX)
        self.U.eval(self.UX, X)
        #print "    UX = " + str(self.UX)
        self.x[:] = X + self.UX
        #print "    x = " + str(self.x)
        #print "    GradZDefIm = " + str(GradZDefIm)
        GradZDefIm[0] = self.interpolator.Interpolate(self.x[0], self.x[1], self.x[2], 2)
        #print "    GradZDefIm = " + str(GradZDefIm)
        GradZDefIm /= self.s
        #print "    GradZDefIm = " + str(GradZDefIm)

cppcode_ExprIm='''
#include <vtkSmartPointer.h>
#include <vtkXMLImageDataReader.h>
#include <vtkImageData.h>
#include <vtkImageInterpolator.h>

namespace dolfin
{

class MyFun : public Expression
{
    public:
        MyFun(): Expression()
        {
            vtkSmartPointer<vtkXMLImageDataReader> reader = vtkSmartPointer<vtkXMLImageDataReader>::New();
            reader->SetFileName("<<filename>>");
            reader->Update();

            interpolator->Initialize(reader->GetOutput());
            interpolator->Update();
        };

        void eval(Array<double>& values, const Array<double>& x) const
        {
            interpolator->Interpolate(x.data(), values.data());
        }

    private:
        vtkSmartPointer<vtkImageInterpolator> interpolator = vtkSmartPointer<vtkImageInterpolator>::New();
};

}'''
