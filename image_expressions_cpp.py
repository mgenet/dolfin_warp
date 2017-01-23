#coding=utf8

########################################################################
###                                                                  ###
### Created by Martin Genet, 2016-2017                               ###
###                                                                  ###
### École Polytechnique, Palaiseau, France                           ###
###                                                                  ###
########################################################################

########################################################################

def get_ExprIm_cpp(
        im_dim,
        im_type="im",
        im_is_def=0,
        verbose=0):

    assert (im_dim in (2,3))
    assert (im_type in ("im","grad"))

    ExprIm_cpp = '''\
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
    vtkSmartPointer<vtkImageInterpolator> interpolator;
    double static_scaling;'''+('''
    Array<double>* dynamic_scaling; // does not work
    double dynamic_scaling_a;       // should not be needed
    double dynamic_scaling_b;       // should not be needed
    Function* U;
    mutable Array<double> UX;
    mutable Array<double> x;''')*(im_is_def)+('''
    mutable Array<double> X3D;''')*(not im_is_def)*(im_dim==2)+'''

public:

    MyExpr():
        Expression('''+str(im_dim)*(im_type=="grad")+''')'''+(''',
        dynamic_scaling_a(1.),
        dynamic_scaling_b(0.),
        UX('''+str(im_dim)+'''),
        x(3)''')*(im_is_def)+(''',
        X3D(3)''')*(not im_is_def)*(im_dim==2)+'''
    {
    }'''+('''

    void init_dynamic_scaling(
        const Array<double> &scaling)
    {
        //dynamic_scaling = scaling;                                                  // does not work
        // std::cout << "dynamic_scaling = " << dynamic_scaling->str(1) << std::endl; // does not work
        dynamic_scaling_a = scaling[0];                                               // should not be needed
        dynamic_scaling_b = scaling[1];                                               // should not be needed
    }

    void update_dynamic_scaling(        // should not be needed
        const Array<double> &scaling)   // should not be needed
    {                                   // should not be needed
        dynamic_scaling_a = scaling[0]; // should not be needed
        dynamic_scaling_b = scaling[1]; // should not be needed
    }                                   // should not be needed

    void init_disp(
        Function* UU)
    {
        U = UU;
    }''')*(im_is_def)+'''

    void init_image(
        const char* filename,
        const char* interpol_mode="'''+('''linear''')*(im_type=="im")+('''linear''')*(im_type=="grad")+'''",
        const double &interpol_out_value='''+('''0.''')*(im_type=="im")+('''1.''')*(im_type=="grad")+(''',
        const double &Z=0.''')*(im_dim==2)+''')
    {
        vtkSmartPointer<vtkXMLImageDataReader> reader = vtkSmartPointer<vtkXMLImageDataReader>::New();
        reader->SetFileName(filename);
        reader->Update();

        static_scaling = getStaticScalingFactor(reader->GetOutput()->GetScalarTypeAsString());'''+('''

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
        interpolator->Update();'''+('''

        x[2] = Z;''')*(im_is_def)*(im_dim==2)+('''

        X3D[2] = Z;''')*(not im_is_def)*(im_dim==2)+'''
    }

    void eval(Array<double>& expr, const Array<double>& X) const
    {'''+('''
        std::cout << "X = " << X.str(1) << std::endl;''')*(verbose)+('''

        U->eval(UX, X);'''+('''
        std::cout << "UX = " << UX.str(1) << std::endl;''')*(verbose)+('''
        x[0] = X[0] + UX[0];
        x[1] = X[1] + UX[1];''')*(im_dim==2)+('''
        x[0] = X[0] + UX[0];
        x[1] = X[1] + UX[1];
        x[2] = X[2] + UX[2];''')*(im_dim==3)+('''
        std::cout << "x = " << x.str(1) << std::endl;''')*(verbose)+'''
        interpolator->Interpolate(x.data(), expr.data());''')*(im_is_def)+('''

        X3D[0] = X[0];
        X3D[1] = X[1];'''+('''
        std::cout << "X3D = " << X3D.str(1) << std::endl;''')*(verbose)+'''
        interpolator->Interpolate(X3D.data(), expr.data());''')*(not im_is_def)*(im_dim==2)+('''

        interpolator->Interpolate(X.data(), expr.data());''')*(not im_is_def)*(im_dim==3)+('''

        std::cout << "expr = " << expr.str(1) << std::endl;''')*(verbose)+('''

        expr[0] /= static_scaling;''')*(im_type=="im")+('''

        expr[0] /= static_scaling;
        expr[1] /= static_scaling;''')*(im_type=="grad")*(im_dim==2)+('''

        expr[0] /= static_scaling;
        expr[1] /= static_scaling;
        expr[2] /= static_scaling;''')*(im_type=="grad")*(im_dim==3)+('''

        std::cout << "expr = " << expr.str(1) << std::endl;''')*(verbose)+(('''

        // std::cout << "in (im)" << std::endl;
        // std::cout << "expr = " << expr.str(1) << std::endl;
        // expr[0] *= (*dynamic_scaling)[0]; // does not work
        // expr[0] += (*dynamic_scaling)[1]; // does not work
        expr[0] *= dynamic_scaling_a;        // should not be needed
        expr[0] += dynamic_scaling_b;        // should not be needed
        // std::cout << "expr = " << expr.str(1) << std::endl;
        // std::cout << "out (im)" << std::endl;''')*(im_type=="im")+('''

        // std::cout << "in (grad)" << std::endl;
        // std::cout << "expr = " << expr.str(1) << std::endl;
        // expr[0] *= (*dynamic_scaling)[0]; // does not work
        // expr[1] *= (*dynamic_scaling)[0]; // does not work
        expr[0] *= dynamic_scaling_a;        // should not be needed
        expr[1] *= dynamic_scaling_a;        // should not be needed
        // std::cout << "expr = " << expr.str(1) << std::endl;
        // std::cout << "out (grad)" << std::endl;''')*(im_type=="grad")*(im_dim==2)+('''

        // std::cout << "in (grad)" << std::endl;
        // std::cout << "expr = " << expr.str(1) << std::endl;
        // expr[0] *= (*dynamic_scaling)[0]; // does not work
        // expr[1] *= (*dynamic_scaling)[0]; // does not work
        // expr[2] *= (*dynamic_scaling)[0]; // does not work
        expr[0] *= dynamic_scaling_a;        // should not be needed
        expr[1] *= dynamic_scaling_a;        // should not be needed
        expr[2] *= dynamic_scaling_a;        // should not be needed
        // std::cout << "expr = " << expr.str(1) << std::endl;
        // std::cout << "out (grad)" << std::endl;''')*(im_type=="grad")*(im_dim==3)+('''

        std::cout << "expr = " << expr.str(1) << std::endl;''')*(verbose))*(im_is_def)+'''
    }
};

}'''
    #print ExprIm_cpp
    return ExprIm_cpp
