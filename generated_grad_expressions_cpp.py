#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2016-2019                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

from builtins import *

################################################################################

def get_ExprGenGrad_cpp(
        im_dim,
        im_type="grad",
        im_is_def=0,
        disp_type="fenics", # "vtk"
        verbose=0):

    assert (im_dim in (2,3))
    assert (im_type in ("grad"))

    ExprGenGrad_cpp = '''\
#include <string.h>

#include <vtkSmartPointer.h>
#include <vtkStructuredPointsReader.h>
#include <vtkXMLImageDataReader.h>
#include <vtkImageData.h>
#include <vtkImageGradient.h>
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
    vtkSmartPointer<vtkImageData> image;
    vtkSmartPointer<vtkImageGradient> gradient;
    vtkSmartPointer<vtkImageInterpolator> interpolator;
    double static_scaling;
    std::shared_ptr<dolfin::Mesh> mesh;
    std::shared_ptr<dolfin::Function> U;
    mutable Array<double> UX;
    mutable Array<double> x;

public:

    MyExpr():
        Expression('''+str(im_dim)*(im_type in ("grad"))+'''),
        UX('''+str(im_dim)+'''),
        x(3),
        interpolator(vtkSmartPointer<vtkImageInterpolator>::New())

    {
    }

    void init_mesh(
        const std::shared_ptr<dolfin::Mesh> mesh_)
    {
        mesh = mesh_;

        std::cout << mesh->num_vertices() << std::endl;
        std::cout << mesh->num_cells() << std::endl;
    }

    void init_disp(
        const std::shared_ptr<dolfin::Function> U_)
    {
        U = U_;
    }

    void init_image()
    {
        image = vtkSmartPointer<vtkImageData>::New(); // MG20180913: another way to instantiate object
        gradient = vtkSmartPointer<vtkImageGradient>::New();
    }



    void eval(Array<double>& expr, const Array<double>& X) const
    {'''+('''

        std::cout << "X = " << X.str(1) << std::endl;''')*(verbose)+'''

        U->eval(UX, X);'''+('''
        std::cout << "UX = " << UX.str(1) << std::endl;''')*(verbose)+('''
        x[0] = X[0] + UX[0];
        x[1] = X[1] + UX[1];''')*(im_dim==2)+('''
        x[0] = X[0] + UX[0];
        x[1] = X[1] + UX[1];
        x[2] = X[2] + UX[2];''')*(im_dim==3)+('''
        std::cout << "x = " << x.str(1) << std::endl;''')*(verbose)+'''

        interpolator->Interpolate(x.data(), expr.data());'''+('''
        std::cout << "expr = " << expr.str(1) << std::endl;''')*(verbose)+('''

        expr[0] /= static_scaling;''')*(im_type=="im")+('''

        expr[0] /= static_scaling;
        expr[1] /= static_scaling;''')*(im_type=="grad")*(im_dim==2)+('''

        expr[0] /= static_scaling;
        expr[1] /= static_scaling;
        expr[2] /= static_scaling;''')*(im_type=="grad")*(im_dim==3)+('''
        std::cout << "expr = " << expr.str(1) << std::endl;''')*(verbose)+'''
    }


    void prepareImage( int n1, int n2, int n3, float origin1, float origin2, float origin3, float spac1, float spac2, float spac3)
    {
        image->SetDimensions(n1,n2,n3);
        image->AllocateScalars(VTK_FLOAT,1);

        image->SetSpacing(spac1, spac2, spac3);
        image->SetOrigin(origin1, origin2, origin3);

        static_scaling = getStaticScalingFactor("double");
    }


    void assignValues( int i, int j, int k, float value)
    {
        image->SetScalarComponentFromFloat(i,j,k,0,value);
    }

    void setGradient_Interpol()
    {
        interpolator->SetInterpolationModeToLinear();
        interpolator->SetOutValue(.0);

        gradient->SetInputData(image);
        gradient->SetDimensionality('''+str(im_dim)+''');
        gradient->Update();
        interpolator->Initialize(gradient->GetOutput());

        interpolator->Update();
    }


    void prepareImage_2D( int n1, int n2, float origin1, float origin2, float spac1, float spac2)
    {
        const double &Z=0.;
        image->SetExtent(0, n1-1, 0, n2-1, 0, 0);

        image->AllocateScalars(VTK_FLOAT,1);

        image->SetSpacing(spac1, spac2, 1);
        image->SetOrigin(origin1, origin2, 0);

        static_scaling = getStaticScalingFactor("double");
        x[2] = Z;

    }


    void assignValues_2D( int i, int j, float value)
    {
        image->SetScalarComponentFromFloat(i,j,0,0,value);
    }
};

}'''
    #print(ExprIm_cpp)
    return ExprGenGrad_cpp
