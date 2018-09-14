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
public:

    vtkSmartPointer<vtkImageData> image; // MG20180913: pointer would be instantiated at class object construction, but would point toward nothing; object would not be instantiated
    vtkSmartPointer<vtkImageInterpolator> interpolator; // MG20180913: pointer would be instantiated at class object construction, but would point toward nothing; object would not be instantiated
    double static_scaling;
    std::shared_ptr<dolfin::Mesh> mesh;
    std::shared_ptr<dolfin::Function> U;

    MyExpr():
        Expression(),
        interpolator(vtkSmartPointer<vtkImageInterpolator>::New()) // MG20180913: object is instantiated together with pointer
    {
    }

    void init_image(
        const char* filename)
    {
        vtkSmartPointer<vtkXMLImageDataReader> reader = vtkSmartPointer<vtkXMLImageDataReader>::New();
        reader->SetFileName(filename);
        reader->Update();

        image = reader->GetOutput();
        // image = vtkSmartPointer<vtkImageData>::New(); // MG20180913: another way to instantiate object

        static_scaling = getStaticScalingFactor(image->GetScalarTypeAsString());

        // interpolator = vtkSmartPointer<vtkImageInterpolator>::New(); // MG20180913: another way to instantiate object
        interpolator->SetInterpolationModeToLinear();
        interpolator->SetOutValue(0.);
        interpolator->Initialize(image);
        interpolator->Update();
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

    void eval(Array<double>& expr, const Array<double>& X) const
    {'''+('''
        std::cout << "X = " << X.str(1) << std::endl;''')*(verbose)+'''

        interpolator->Interpolate(X.data(), expr.data());'''+('''

        std::cout << "expr = " << expr.str(1) << std::endl;''')*(verbose)+'''

        expr[0] /= static_scaling;'''+('''

        std::cout << "expr = " << expr.str(1) << std::endl;''')*(verbose)+'''
    }
};

}'''
    #print ExprGenIm_cpp
    return ExprGenIm_cpp
