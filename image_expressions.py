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

    def init_image(self, filename):
        self.image = myVTK.readImage(
            filename=filename,
            verbose=0)
        self.s = getScalingFactor(self.image.GetScalarTypeAsString())
        #print self.s
        self.interpolator = myVTK.createImageInterpolator(
            image=self.image,
            verbose=0)

    def eval(self, Im, X):
        #print "Im"
        self.X[0:2] = X[0:2]
        #print "    X = " + str(X)
        self.interpolator.Interpolate(self.X, Im)
        #print "    Im = " + str(Im)
        Im /= self.s
        #print "    Im = " + str(Im)

class ExprIm3(dolfin.Expression):
    def __init__(self, filename, **kwargs):
        if filename is not None:
            self.init_image(filename=filename)

    def init_image(self, filename):
        self.image = myVTK.readImage(
            filename=filename,
            verbose=0)
        self.s = getScalingFactor(self.image.GetScalarTypeAsString())
        #print self.s
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

class ExprGradIm2(dolfin.Expression):
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
        self.X[0:2] = X[0:2]
        self.interpolator.Interpolate(self.X, GradIm)
        GradIm /= self.s

class ExprGradIm3(dolfin.Expression):
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

class ExprDefIm2(dolfin.Expression):
    def __init__(self, U, filename=None, Z=0., **kwargs):
        if filename is not None:
            self.init_image(filename=filename)
        self.U = U
        self.UX = numpy.empty(2)
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
        self.x[0:2] = X[0:2] + self.UX[0:2]
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
        self.UX = numpy.empty(3)
        self.x = numpy.empty(3)

    def init_image(self, filename):
        self.image = myVTK.readImage(
            filename=filename,
            verbose=0)
        self.s = getScalingFactor(self.image.GetScalarTypeAsString())
        #print self.s
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

class ExprGradDefIm2(dolfin.Expression):
    def __init__(self, U, filename=None, Z=0., **kwargs):
        if filename is not None:
            self.init_image(filename=filename)
        self.U = U
        self.UX = numpy.empty(2)
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
        self.x[0:2] = X[0:2] + self.UX[0:2]
        #print "    x = " + str(self.x)
        #print "    GradDefIm = " + str(GradDefIm)
        self.interpolator.Interpolate(self.x, GradDefIm)
        #print "    GradDefIm = " + str(GradDefIm)
        GradDefIm /= self.s
        #print "    GradDefIm = " + str(GradDefIm)

class ExprGradDefIm3(dolfin.Expression):
    def __init__(self, U, filename=None, **kwargs):
        if filename is not None:
            self.init_image(filename=filename)
        self.U = U
        self.UX = numpy.empty(3)
        self.x = numpy.empty(3)

    def init_image(self, filename):
        self.image = myVTK.readImage(
            filename=filename,
            verbose=0)
        self.s = getScalingFactor(self.image.GetScalarTypeAsString())
        #print self.s
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
