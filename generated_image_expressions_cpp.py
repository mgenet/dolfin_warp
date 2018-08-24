#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2016-2018                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

################################################################################

def get_ExprGenIm_cpp(
        im_dim,
        verbose=0):

    assert (im_dim in (2,3))

    ExprGenIm_cpp = '''\
#include <string.h>

#include <vtkSmartPointer.h>
#include <vtkStructuredPointsReader.h>
#include <vtkXMLImageDataReader.h>
#include <vtkImageData.h>
#include <vtkImageInterpolator.h>

double getStaticScalingFactor(const char* scalar_type_as_string)
{
    if (strcmp(scalar_type_as_string, "unsigned char" ) == 0) return pow(2,  8)-1;
    if (strcmp(scalar_type_as_string, "unsigned short") == 0) return pow(2, 16)-1;
    if (strcmp(scalar_type_as_string, "unsigned int"  ) == 0) return pow(2, 32)-1;
    if (strcmp(scalar_type_as_string, "unsigned long" ) == 0) return pow(2, 64)-1;
    if (strcmp(scalar_type_as_string, "float"         ) == 0) return 1.;
    if (strcmp(scalar_type_as_string, "double"        ) == 0) return 1.;
    assert (0);
}

namespace dolfin
{

class MyExpr : public Expression
{
    vtkSmartPointer<vtkImageInterpolator> interpolator;
    double static_scaling;

public:

    MyExpr():
        Expression()
    {
    }

    void init_image(
        const char* filename)
    {
        vtkSmartPointer<vtkXMLImageDataReader> reader = vtkSmartPointer<vtkXMLImageDataReader>::New();
        reader->SetFileName(filename);
        reader->Update();

        static_scaling = getStaticScalingFactor(reader->GetOutput()->GetScalarTypeAsString());

        interpolator = vtkSmartPointer<vtkImageInterpolator>::New();
        interpolator->SetInterpolationModeToLinear();
        interpolator->SetOutValue(0.);
        interpolator->Initialize(reader->GetOutput());
        interpolator->Update();
    }

    void eval(Array<double>& expr, const Array<double>& X) const
    {'''+('''
        std::cout << "X = " << X.str(1) << std::endl;''')*(verbose)+('''

        X3D[0] = X[0];
        X3D[1] = X[1];'''+('''
        std::cout << "X3D = " << X3D.str(1) << std::endl;''')*(verbose)+'''
        interpolator->Interpolate(X3D.data(), expr.data());''')*(im_dim==2)+('''

        interpolator->Interpolate(X.data(), expr.data());''')*(im_dim==3)+('''

        std::cout << "expr = " << expr.str(1) << std::endl;''')*(verbose)+'''

        expr[0] /= static_scaling;'''+('''

        std::cout << "expr = " << expr.str(1) << std::endl;''')*(verbose)+'''
    }
};

}'''
    #print ExprGenIm_cpp
    return ExprGenIm_cpp
