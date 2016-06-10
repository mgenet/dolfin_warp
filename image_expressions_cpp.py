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
from myVTKPythonLibrary.mat_vec_tools import *

########################################################################

def get_ExprIm_cpp(
        im_dim,
        im_type="im",
        im_is_def=False,
        verbose=False):

    assert (im_dim in (2,3))
    assert (im_type in ("im","grad"))

    ExprIm_cpp = '''\
#include <string.h>

#include <vtkSmartPointer.h>
#include <vtkXMLImageDataReader.h>
#include <vtkImageData.h>'''+('''
#include <vtkImageGradient.h>''')*(im_type=="grad")+'''
#include <vtkImageInterpolator.h>

namespace dolfin
{

class MyExpr : public Expression
{
    vtkSmartPointer<vtkImageInterpolator> interpolator;'''+('''
    const Function* U;
    mutable Array<double> UX;
    mutable Array<double> x;''')*(im_is_def)+('''
    mutable Array<double> X;''')*(not im_is_def)*(im_dim==2)+'''

public:

    MyExpr():
        Expression('''+('''2''')*(im_type=="grad")+''')'''+(''',
        UX('''+str(im_dim)+'''),
        x(3)''')*(im_is_def)+(''',
        X(3)''')*(not im_is_def)*(im_dim==2)+'''
    {
    }

    void init('''+('''
        const Function &UU,''')*(im_is_def)+('''
        const double &Z=0.''')*(im_dim==2)+''')
    {'''+('''
        U = &UU;''')*(im_is_def)+('''

        x[2] = Z;''')*(im_is_def)*(im_dim==2)+('''

        X[2] = Z;''')*(not im_is_def)*(im_dim==2)+'''
    }

    void init_image(
        const char* filename,
        const char* interpol_mode="'''+('''linear''')*(im_type=="im")+('''nearest''')*(im_type=="grad")+'''",
        const double &interpol_out_value='''+('''0.''')*(im_type=="im")+('''1.''')*(im_type=="grad")+''')
    {
        vtkSmartPointer<vtkXMLImageDataReader> reader = vtkSmartPointer<vtkXMLImageDataReader>::New();
        reader->SetFileName(filename);
        reader->Update();'''+('''

        vtkSmartPointer<vtkImageGradient> gradient = vtkSmartPointer<vtkImageGradient>::New();
#if VTK_MAJOR_VERSION >= 6
        gradient->SetInputData(reader->GetOutput());
#else
        gradient->SetInput(reader->GetOutput());
#endif
        gradient->SetDimensionality('''+str(im_dim)+''');
        gradient->Update();''')*(im_type=="grad")+'''

        interpolator = vtkSmartPointer<vtkImageInterpolator>::New();
        if (strcmp(interpol_mode, "nearest") == 0)
        {
            interpolator->SetInterpolationModeToNearest();
        }
        else if (strcmp(interpol_mode, "linear") == 0)
        {
            interpolator->SetInterpolationModeToLinear();
        }
        else if (strcmp(interpol_mode, "cubic") == 0)
        {
            interpolator->SetInterpolationModeToCubic();
        }
        else
        {
            std::cout << "Interpolator interpol_mode (" << interpol_mode << ") must be \\"nearest\\", \\"linear\\" or \\"cubic\\". Aborting." << std::endl;
            assert(0);
        }
        interpolator->SetOutValue(interpol_out_value);
        interpolator->Initialize('''+('''reader->GetOutput()''')*(im_type=="im")+('''gradient->GetOutput()''')*(im_type=="grad")+''');
        interpolator->Update();
    }

    void eval(Array<double>& values, const Array<double>& position) const
    {'''+('''
        std::cout << "position = " << position.str(1) << std::endl;''')*(verbose)+('''

        U->eval(UX, position);'''+('''
        std::cout << "UX = " << UX.str(1) << std::endl;''')*(verbose)+'''
        x[0] = position[0] + UX[0];
        x[1] = position[1] + UX[1];'''+('''
        std::cout << "x = " << x.str(1) << std::endl;''')*(verbose)+'''
        interpolator->Interpolate(x.data(), values.data());''')*(im_is_def)*(im_dim==2)+('''

        U->eval(UX, position);'''+('''
        std::cout << "UX = " << UX.str(1) << std::endl;''')*(verbose)+'''
        x = position + UX;'''+('''
        std::cout << "x = " << x.str(1) << std::endl;''')*(verbose)+'''
        interpolator->Interpolate(x.data(), values.data());''')*(im_is_def)*(im_dim==3)+('''

        X[0] = position[0];
        X[1] = position[1];'''+('''
        std::cout << "X = " << X.str(1) << std::endl;''')*(verbose)+'''
        interpolator->Interpolate(X.data(), values.data());''')*(not im_is_def)*(im_dim==2)+('''

        interpolator->Interpolate(position.data(), values.data());''')*(not im_is_def)*(im_dim==3)+('''

        std::cout << "values = " << values.str(1) << std::endl;''')*(verbose)+'''
    }
};

}'''
    return ExprIm_cpp
