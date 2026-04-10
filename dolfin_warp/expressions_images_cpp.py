#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2016-2026                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

import os

import jinja2

import dolfin_warp as dwarp

################################################################################

def get_ExprIm_cpp(
        im_dim, # 2, 3
        im_type="im", # im, grad, grad_direct, im+grad
        im_is_def=0,
        u_type="dolfin", # dolfin, vtk
        static_scaling_factor=0,
        dynamic_scaling=0,
        use_jinja2=1,
        verbose=0):

    assert (im_dim  in (2,3))
    assert (im_type in ("im","grad","grad_direct","im+grad"))
    if (im_is_def):
        assert (u_type in ("dolfin","vtk"))
    if (not im_is_def):
        assert (not dynamic_scaling)

    name  = "Expr"
    name += str(im_dim)
    if   (im_type == "im"):
        name += "Im"
    elif (im_type in ("grad","grad_direct")):
        name += "Grad"
    elif (im_type == "im+grad"):
        name += "ImGrad"
    if   (im_is_def == 0):
        name += "Ref"
    elif (im_is_def == 1):
        name += "Def"
    # print(name)

    if (use_jinja2):
        env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(os.path.dirname(__file__)),
            trim_blocks=True,
            lstrip_blocks=True)
        template = env.get_template("expressions_images_cpp.j2")
        
        cpp = template.render(
            name=name,
            im_dim=im_dim,
            im_type=im_type,
            im_is_def=im_is_def,
            u_type=u_type,
            static_scaling_factor=static_scaling_factor,
            static_scaling_cpp=dwarp.get_StaticScaling_cpp(),
            dynamic_scaling=dynamic_scaling,
            verbose=verbose)
    else:
        cpp = '''\
#include <string.h>

#include <dolfin/function/Expression.h>
#include <dolfin/function/Function.h>

#include <vtkImageData.h>
#include <vtkImageGradient.h>
#include <vtkImageInterpolator.h>
#include <vtkPolyData.h>
#include <vtkPointData.h>
#include <vtkProbeFilter.h>
#include <vtkSmartPointer.h>
#include <vtkUnstructuredGrid.h>
#include <vtkXMLImageDataReader.h>
#include <vtkXMLUnstructuredGridReader.h>

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

'''+dwarp.get_StaticScaling_cpp()+'''\

class '''+name+''' : public dolfin::Expression
{
public:

    static constexpr unsigned int n_dim = '''+str(im_dim)+''';

    vtkSmartPointer<vtkXMLImageDataReader>        reader                = vtkSmartPointer<vtkXMLImageDataReader>::New()       ;
    vtkSmartPointer<vtkImageData>                 image                 = nullptr                                             ;
    double                                        static_scaling                                                              ;'''+(('''
    std::unique_ptr<Eigen::Ref<Eigen::Vector2d>>  dynamic_scaling                                                             ;''')*(dynamic_scaling)+('''
    std::shared_ptr<dolfin::Function>             U                     = nullptr                                             ;''')*(u_type=="dolfin")+('''
    vtkSmartPointer<vtkXMLUnstructuredGridReader> ugrid_reader          = vtkSmartPointer<vtkXMLUnstructuredGridReader>::New();
    vtkSmartPointer<vtkUnstructuredGrid>          ugrid                 = nullptr                                             ;
    vtkSmartPointer<vtkPoints>                    probe_points          = vtkSmartPointer<vtkPoints>::New()                   ;
    vtkSmartPointer<vtkPolyData>                  probe_polydata        = vtkSmartPointer<vtkPolyData>::New()                 ;
    vtkSmartPointer<vtkProbeFilter>               probe_filter          = vtkSmartPointer<vtkProbeFilter>::New()              ;''')*(u_type=="vtk"))*(im_is_def)+('''
    mutable Eigen::Vector3d                       X_3D                                                                        ;''')*(not im_is_def)*(im_dim==2)+('''
    mutable Eigen::Vector3d                       x_3D                                                                        ;''')*(im_is_def)+('''
    vtkSmartPointer<vtkImageGradient>             gradient_filter       = vtkSmartPointer<vtkImageGradient>::New()            ;
    vtkSmartPointer<vtkImageData>                 gradient_image        = nullptr                                             ;''')*(im_type in ("grad","im+grad"))+'''
    vtkSmartPointer<vtkImageInterpolator>         interpolator          = vtkSmartPointer<vtkImageInterpolator>::New()        ;'''+('''
    vtkSmartPointer<vtkImageInterpolator>         gradient_interpolator = vtkSmartPointer<vtkImageInterpolator>::New()        ;''')*(im_type=="im+grad")+('''
    mutable Eigen::Matrix<double, n_dim, 1>       UX                                                                          ;''')*(im_is_def)+'''

    '''+name+'''(
        const char* image_interpol_mode="'''+('''linear''')*(im_type in ("im","im+grad"))+('''linear''')*(im_type in ("grad","grad_direct"))+'''",
        const double &image_interpol_out_value=0.'''+(''',
        const char* gradient_interpol_mode="linear",
        const double &gradient_interpol_out_value=0.''')*(im_type=="im+grad")+(''',
        const double &Z=0.''')*(im_dim==2)+'''
    ) :
        dolfin::Expression('''+str(im_dim)*(im_type in ("grad","grad_direct"))+str(1+im_dim)*(im_type=="im+grad")+''')
    {'''+('''
        std::cout << "constructor" << std::endl;''')*(verbose)+'''

        reader->UpdateDataObject();
        image = reader->GetOutput();'''+(('''

        X_3D[2] = Z;''')*(not im_is_def)+('''
        x_3D[2] = Z;''')*(im_is_def))*(im_dim==2)+('''

        gradient_filter->SetDimensionality(n_dim);
        gradient_filter->SetInputDataObject(image);
        gradient_filter->UpdateDataObject();
        gradient_image = gradient_filter->GetOutput();''')*(im_type in ("grad","im+grad"))+('''

        if (strcmp(image_interpol_mode, "nearest") == 0)
        {
            interpolator->SetInterpolationModeToNearest();
        }
        else if (strcmp(image_interpol_mode, "linear") == 0)
        {
            interpolator->SetInterpolationModeToLinear();
        }
        else
        {
            std::cout << "Interpolator image_interpol_mode (" << image_interpol_mode << ") must be \\"nearest\\" or \\"linear\\". Aborting." << std::endl;
            std::exit(0);
        }
        interpolator->SetOutValue(image_interpol_out_value);
        // interpolator->Initialize('''+('''image''')*(im_type=="im")+('''gradient_image''')*(im_type=="grad")+'''); // MG20240524: Possible here? Nope! Apparently, after modifying the image content, the interpolator must be initialized again…''')*(im_type in ("im", "grad", "grad_direct", "im+grad"))+('''

        if (strcmp(gradient_interpol_mode, "nearest") == 0)
        {
            gradient_interpolator->SetInterpolationModeToNearest();
        }
        else if (strcmp(gradient_interpol_mode, "linear") == 0)
        {
            gradient_interpolator->SetInterpolationModeToLinear();
        }
        else
        {
            std::cout << "Interpolator gradient_interpol_mode (" << gradient_interpol_mode << ") must be \\"nearest\\" or \\"linear\\". Aborting." << std::endl;
            std::exit(0);
        }
        gradient_interpolator->SetOutValue(gradient_interpol_out_value);''')*(im_type=="im+grad")+'''
    }

    void init_image
    (
        const char* filename
    )
    {'''+('''
        std::cout << "init_image" << std::endl;''')*(verbose)+'''

        reader->SetFileName(filename);
        reader->Update();'''+('''

        static_scaling = getStaticScalingFactor(image->GetScalarTypeAsString());''')*(not static_scaling_factor)+('''
        static_scaling = '''+str(static_scaling_factor)+''';''')*(static_scaling_factor)+('''

        gradient_filter->Update();''')*(im_type in ("grad","im+grad"))+'''

        interpolator->Initialize('''+('''image''')*(im_type in ("im","grad_direct","im+grad"))+('''gradient_image''')*(im_type=="grad")+''');'''+('''
        gradient_interpolator->Initialize(gradient_image);''')*(im_type=="im+grad")+'''
    }

    void update_image
    (
        const char* filename
    )
    {'''+('''
        std::cout << "update_image" << std::endl;''')*(verbose)+'''

        reader->SetFileName(filename);
        reader->Update();'''+('''

        gradient_filter->Update();''')*(im_type in ("grad","im+grad"))+'''

        interpolator->Initialize('''+('''image''')*(im_type in ("im","grad_direct","im+grad"))+('''gradient_image''')*(im_type=="grad")+''');'''+('''
        gradient_interpolator->Initialize(gradient_image);''')*(im_type=="im+grad")+'''
    }'''+(('''

    void init_dynamic_scaling
    (
        Eigen::Ref<Eigen::Vector2d> dynamic_scaling_
    )
    {'''+('''
        std::cout << "init_dynamic_scaling" << std::endl;''')*(verbose)+'''

        dynamic_scaling.reset(new Eigen::Ref<Eigen::Vector2d>(dynamic_scaling_));
    }''')*(dynamic_scaling)+('''

    void init_disp
    (
        std::shared_ptr<dolfin::Function> U_
    )
    {'''+('''
        std::cout << "init_disp" << std::endl;''')*(verbose)+'''

        U = U_;
    }''')*(u_type=="dolfin")+('''

    void init_disp
    (
        const char* ugrid_filename
    )
    {'''+('''
        std::cout << "init_disp" << std::endl;''')*(verbose)+'''

        ugrid_reader->SetFileName(ugrid_filename);
        ugrid_reader->Update();
        ugrid = ugrid_reader->GetOutput();

        probe_points->SetNumberOfPoints(1);
        probe_polydata->SetPoints(probe_points);
        probe_filter->SetInputData(probe_polydata);
        probe_filter->SetSourceData(ugrid);
    }''')*(u_type=="vtk"))*(im_is_def)+'''

    void eval
    (
        Eigen::Ref<      Eigen::VectorXd> expr,
        Eigen::Ref<const Eigen::VectorXd> X
    ) const
    {'''+('''
        // std::cout << "X = " << X << std::endl;''')*(verbose)+(('''

        X_3D.head<n_dim>() = X;'''+('''
        // std::cout << "X_3D = " << X_3D << std::endl;''')*(verbose)+'''

        interpolator->Interpolate(X_3D.data(), expr.data());'''+('''
        gradient_interpolator->Interpolate(X_3D.data(), expr.data()+1);''')*(im_type=="im+grad"))*(im_dim==2)+('''

        interpolator->Interpolate(X.data(), expr.data());'''+('''
        gradient_interpolator->Interpolate(X.data(), expr.data()+1);''')*(im_type=="im+grad"))*(im_dim==3))*(not im_is_def)+(('''

        U->eval(UX, X);''')*(u_type=="dolfin")+('''

        probe_points->SetPoint(0,X.data());
        probe_filter->Update();
        probe_filter->GetOutput()->GetPointData()->GetArray("U")->GetTuple(0, UX.data());''')*(u_type=="vtk")+('''

        // std::cout << "UX = " << UX << std::endl;''')*(verbose)+('''

        x_3D.head<n_dim>() = X + UX;''')*(im_dim==2)+('''
        x_3D               = X + UX;''')*(im_dim==3)+('''
        // std::cout << "x_3D = " << x_3D << std::endl;''')*(verbose)+'''

        interpolator->Interpolate(x_3D.data(), expr.data());'''+('''
        gradient_interpolator->Interpolate(x_3D.data(), expr.data()+1);''')*(im_type=="im+grad"))*(im_is_def)+('''
        // std::cout << "expr = " << expr << std::endl;''')*(verbose)+'''

        expr /= static_scaling;'''+('''
        // std::cout << "expr = " << expr << std::endl;''')*(verbose)+(('''

        expr *= (*dynamic_scaling)[0];
        expr += (*dynamic_scaling)[1];''')*(im_type=="im")+('''

        expr *= (*dynamic_scaling)[0];''')*(im_type in ("grad", "grad_direct"))+('''

        expr    *= (*dynamic_scaling)[0];
        expr[0] += (*dynamic_scaling)[1];''')*(im_type=="im+grad")+('''

        // std::cout << "expr = " << expr << std::endl;''')*(verbose))*(dynamic_scaling)*(im_is_def)+'''
    }
};

PYBIND11_MODULE(SIGNATURE, m)
{
    pybind11::class_<'''+name+''', std::shared_ptr<'''+name+'''>, dolfin::Expression>(m, "'''+name+'''")
    .def(pybind11::init<const char*, const double&'''+(''', const char*, const double&''')*(im_type=="im+grad")+(''', const double&''')*(im_dim==2)+'''>(), pybind11::arg("image_interpol_mode") = "'''+('''linear''')*(im_type in ("im","im+grad"))+('''linear''')*(im_type in ("grad","grad_direct"))+'''", pybind11::arg("image_interpol_out_value") = 0.'''+(''', pybind11::arg("gradient_interpol_mode") = "linear", pybind11::arg("gradient_interpol_out_value") = 0.''')*(im_type=="im+grad")+(''', pybind11::arg("Z") = 0.''')*(im_dim==2)+''')
    .def("init_image", &'''+name+'''::init_image, pybind11::arg("filename"))
    .def("update_image", &'''+name+'''::update_image, pybind11::arg("filename"))'''+(('''
    .def("init_dynamic_scaling", &'''+name+'''::init_dynamic_scaling, pybind11::arg("dynamic_scaling_"))''')*(dynamic_scaling)+('''
    .def("init_disp", &'''+name+'''::init_disp, pybind11::arg("U_"))''')*(u_type=="dolfin")+('''
    .def("init_disp", &'''+name+'''::init_disp, pybind11::arg("ugrid_filename"))''')*(u_type=="vtk"))*(im_is_def)+''';
}
'''
    # print(cpp)

    return name, cpp

