#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2016-2021                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

import dolfin_warp as dwarp

################################################################################

def get_ExprIm_cpp_pybind(
        im_dim, # 2, 3
        im_type="im", # im, grad, grad_no_deriv
        im_is_def=0,
        u_type="dolfin", # dolfin, vtk
        static_scaling_factor=0,
        dynamic_scaling=0,
        verbose=0):

    assert (im_dim  in (2,3))
    assert (im_type in ("im","grad","grad_no_deriv"))
    if (im_is_def):
        assert ( u_type in ("dolfin","vtk"))
    if (not im_is_def):
        assert (not dynamic_scaling)

    name  = "Expr"
    name += str(im_dim)
    if   (im_type == "im"):
        name += "Im"
    elif (im_type in ("grad", "grad_no_deriv")):
        name += "Grad"
    if   (im_is_def == 0):
        name += "Ref"
    elif (im_is_def == 1):
        name += "Def"
    # print(name)

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

    vtkSmartPointer<vtkImageInterpolator> interpolator;

    double static_scaling;'''+(('''

    std::unique_ptr<Eigen::Ref<Eigen::Vector2d>> dynamic_scaling;''')*(dynamic_scaling)+('''

    std::shared_ptr<dolfin::Function> U;''')*(u_type=="dolfin")+('''

    vtkSmartPointer<vtkProbeFilter> probe_filter;
    vtkSmartPointer<vtkPoints>      probe_points;
    vtkSmartPointer<vtkPolyData>    probe_polydata;''')*(u_type=="vtk")+'''

    mutable Eigen::Vector'''+str(im_dim)+'''d UX;
    mutable Eigen::Vector3d x;''')*(im_is_def)+('''

    mutable Eigen::Vector3d X3D;''')*(not im_is_def)*(im_dim==2)+'''

    '''+name+'''() :
        dolfin::Expression('''+str(im_dim)*(im_type in ("grad", "grad_no_deriv"))+''')'''+'''
    {'''+('''

        probe_filter = vtkSmartPointer<vtkProbeFilter>::New();
        probe_points = vtkSmartPointer<vtkPoints>::New();
        probe_polydata = vtkSmartPointer<vtkPolyData>::New();''')*(u_type=="vtk")*(im_is_def)+'''
    }

    void init_image
    (
        const char* filename,
        const char* interpol_mode="linear",
        const double &interpol_out_value=0.'''+(''',
        const double &Z=0.''')*(im_dim==2)+'''
    )
    {
        vtkSmartPointer<vtkXMLImageDataReader> reader = vtkSmartPointer<vtkXMLImageDataReader>::New();
        reader->SetFileName(filename);
        reader->Update();'''+('''

        static_scaling = getStaticScalingFactor(reader->GetOutput()->GetScalarTypeAsString());''')*(not static_scaling_factor)+('''
        static_scaling = '''+str(static_scaling_factor)+''';''')*(static_scaling_factor)+('''

        vtkSmartPointer<vtkImageGradient> gradient = vtkSmartPointer<vtkImageGradient>::New();
        gradient->SetInputData(reader->GetOutput());
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
            std::exit(0);
        }
        interpolator->SetOutValue(interpol_out_value);
        interpolator->Initialize('''+('''reader->GetOutput()''')*(im_type in ("im", "grad_no_deriv"))+('''gradient->GetOutput()''')*(im_type=="grad")+''');'''+(('''

        x[2] = Z;''')*(im_is_def)+('''

        X3D[2] = Z;''')*(not im_is_def))*(im_dim==2)+'''
    }'''+(('''

    void init_dynamic_scaling
    (
        Eigen::Ref<Eigen::Vector2d> dynamic_scaling_
    )
    {
        dynamic_scaling.reset(new Eigen::Ref<Eigen::Vector2d>(dynamic_scaling_));
    }''')*(dynamic_scaling)+('''

    void init_disp
    (
        std::shared_ptr<dolfin::Function> U_
    )
    {
        U = U_;
    }''')*(u_type=="dolfin")+('''

    void init_disp
    (
        const char* mesh_filename
    )
    {
        vtkSmartPointer<vtkXMLUnstructuredGridReader> reader = vtkSmartPointer<vtkXMLUnstructuredGridReader>::New();
        reader->SetFileName(mesh_filename);
        reader->Update();

        vtkSmartPointer<vtkUnstructuredGrid> mesh = reader->GetOutput();

        probe_filter->SetSourceData(mesh);
    }''')*(u_type=="vtk"))*(im_is_def)+'''

    void eval
    (
        Eigen::Ref<      Eigen::VectorXd> expr,
        Eigen::Ref<const Eigen::VectorXd> X
    ) const
    {'''+('''
        std::cout << "X = " << X << std::endl;''')*(verbose)+(('''

        U->eval(UX, X);''')*(u_type=="dolfin")+('''

        probe_points->SetNumberOfPoints(1);
        probe_points->SetPoint(0,X.data());
        probe_polydata->SetPoints(probe_points);
        probe_filter->SetInputData(probe_polydata);
        probe_filter->Update();
        probe_filter->GetOutput()->GetPointData()->GetArray("U")->GetTuple(0, UX.data());

        ''')*(u_type=="vtk")+('''
        std::cout << "UX = " << UX << std::endl;''')*(verbose)+('''

        x[0] = X[0] + UX[0];
        x[1] = X[1] + UX[1];''')*(im_dim==2)+('''
        x[0] = X[0] + UX[0];
        x[1] = X[1] + UX[1];
        x[2] = X[2] + UX[2];''')*(im_dim==3)+('''
        std::cout << "x = " << x << std::endl;''')*(verbose)+'''
        interpolator->Interpolate(x.data(), expr.data());''')*(im_is_def)+(('''

        X3D[0] = X[0];
        X3D[1] = X[1];'''+('''
        std::cout << "X3D = " << X3D << std::endl;''')*(verbose)+'''
        interpolator->Interpolate(X3D.data(), expr.data());''')*(im_dim==2)+('''

        interpolator->Interpolate(X.data(), expr.data());''')*(im_dim==3))*(not im_is_def)+('''

        std::cout << "expr = " << expr << std::endl;''')*(verbose)+('''

        expr[0] /= static_scaling;''')*(im_type=="im")+(('''

        expr[0] /= static_scaling;
        expr[1] /= static_scaling;''')*(im_dim==2)+('''

        expr[0] /= static_scaling;
        expr[1] /= static_scaling;
        expr[2] /= static_scaling;''')*(im_dim==3))*(im_type=="grad")+('''

        std::cout << "expr = " << expr << std::endl;''')*(verbose)+(('''

        expr[0] *= (*dynamic_scaling)[0];
        expr[0] += (*dynamic_scaling)[1];''')*(im_type=="im")+(('''

        expr[0] *= (*dynamic_scaling)[0];
        expr[1] *= (*dynamic_scaling)[0];''')*(im_dim==2)+('''

        expr[0] *= (*dynamic_scaling)[0];
        expr[1] *= (*dynamic_scaling)[0];
        expr[2] *= (*dynamic_scaling)[0];''')*(im_dim==3))*(im_type=="grad")+('''

        std::cout << "expr = " << expr << std::endl;''')*(verbose))*(dynamic_scaling)*(im_is_def)+'''
    }
};

PYBIND11_MODULE(SIGNATURE, m)
{
    pybind11::class_<'''+name+''', std::shared_ptr<'''+name+'''>, dolfin::Expression>
    (m, "'''+name+'''")
    .def(pybind11::init<>())
    .def("init_image", &'''+name+'''::init_image, pybind11::arg("filename"), pybind11::arg("interpol_mode") = "linear", pybind11::arg("interpol_out_value") = 0.'''+(''', pybind11::arg("Z") = 0.''')*(im_dim==2)+''')'''+(('''
    .def("init_dynamic_scaling", &'''+name+'''::init_dynamic_scaling, pybind11::arg("dynamic_scaling_"))''')*(dynamic_scaling)+('''
    .def("init_disp", &'''+name+'''::init_disp, pybind11::arg("U_"))''')*(u_type=="dolfin")+('''
    .def("init_disp", &'''+name+'''::init_disp, pybind11::arg("mesh_filename"))''')*(u_type=="vtk"))*(im_is_def)+''';
}
'''
    # print(cpp)

    return name, cpp

################################################################################

def get_ExprIm_cpp_swig(
        im_dim, # 2, 3
        im_type="im", # im, grad, grad_no_deriv
        im_is_def=0,
        u_type="dolfin", # dolfin, vtk
        static_scaling_factor=0,
        verbose=0):

    assert (im_dim  in (2,3))
    assert (im_type in ("im","grad","grad_no_deriv"))
    assert (u_type in ("dolfin","vtk"))

    ExprIm_cpp = '''\
#include <string.h>

#include <vtkSmartPointer.h>
#include <vtkXMLImageDataReader.h>
#include <vtkImageData.h>'''+('''
#include <vtkImageGradient.h>''')*(im_type=="grad")+'''
#include <vtkImageInterpolator.h>'''+('''
#include <vtkXMLUnstructuredGridReader.h>
#include <vtkUnstructuredGrid.h>
#include <vtkProbeFilter.h>
#include <vtkPointData.h>
#include <vtkPolyData.h>''')*(im_is_def)*(u_type=="vtk")+'''

'''+dwarp.get_StaticScaling_cpp()+'''\

namespace dolfin
{

class MyExpr : public Expression
{
    vtkSmartPointer<vtkImageInterpolator> interpolator;

    double static_scaling;'''+('''

    double dynamic_scaling_a;
    double dynamic_scaling_b;'''+('''

    std::shared_ptr<Function> U;''')*(u_type=="dolfin")+('''

    vtkSmartPointer<vtkProbeFilter> probe_filter;
    vtkSmartPointer<vtkPoints>      probe_points;
    vtkSmartPointer<vtkPolyData>    probe_polydata;''')*(u_type=="vtk")+'''
    mutable Array<double> UX;
    mutable Array<double> x;''')*(im_is_def)+('''
    mutable Array<double> X3D;''')*(not im_is_def)*(im_dim==2)+'''

public:

    MyExpr
    (
    ) :
        Expression('''+str(im_dim)*(im_type in ("grad", "grad_no_deriv"))+''')'''+(''',
        dynamic_scaling_a(1.),
        dynamic_scaling_b(0.),
        UX('''+str(im_dim)+'''),
        x(3)''')*(im_is_def)+(''',
        X3D(3)''')*(not im_is_def)*(im_dim==2)+'''
    {'''+('''

        probe_filter = vtkSmartPointer<vtkProbeFilter>::New();
        probe_points = vtkSmartPointer<vtkPoints>::New();
        probe_polydata = vtkSmartPointer<vtkPolyData>::New();''')*(im_is_def)*(u_type=="vtk")+'''
    }

    void init_image
    (
        const char* filename,
        const char* interpol_mode="linear",
        const double &interpol_out_value=0.'''+(''',
        const double &Z=0.''')*(im_dim==2)+'''
    )
    {
        vtkSmartPointer<vtkXMLImageDataReader> reader = vtkSmartPointer<vtkXMLImageDataReader>::New();
        reader->SetFileName(filename);
        reader->Update();'''+('''

        static_scaling = getStaticScalingFactor(reader->GetOutput()->GetScalarTypeAsString());''')*(not static_scaling_factor)+('''
        static_scaling = '''+str(static_scaling_factor)+''';''')*(static_scaling_factor)+('''

        vtkSmartPointer<vtkImageGradient> gradient = vtkSmartPointer<vtkImageGradient>::New();
        gradient->SetInputData(reader->GetOutput());
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
        interpolator->Initialize('''+('''reader->GetOutput()''')*(im_type in ("im", "grad_no_deriv"))+('''gradient->GetOutput()''')*(im_type=="grad")+''');
        //interpolator->Update();'''+(('''

        x[2] = Z;''')*(im_is_def)+('''

        X3D[2] = Z;''')*(not im_is_def))*(im_dim==2)+'''
    }'''+(('''

    void init_disp
    (
        std::shared_ptr<Function> U_
    )
    {
        U = U_;
    }''')*(u_type=="dolfin")+('''

    void init_disp
    (
        const char* mesh_filename
    )
    {
        vtkSmartPointer<vtkXMLUnstructuredGridReader> reader = vtkSmartPointer<vtkXMLUnstructuredGridReader>::New();
        reader->SetFileName(mesh_filename);
        reader->Update();

        vtkSmartPointer<vtkUnstructuredGrid> mesh = reader->GetOutput();

        probe_filter->SetSourceData(mesh);
    }''')*(u_type=="vtk")+'''

    void init_dynamic_scaling
    (
        const Array<double> &dynamic_scaling
    )
    {
        dynamic_scaling_a = dynamic_scaling[0];
        dynamic_scaling_b = dynamic_scaling[1];
    }

    void update_dynamic_scaling
    (
        const Array<double> &dynamic_scaling
    )
    {
        dynamic_scaling_a = dynamic_scaling[0];
        dynamic_scaling_b = dynamic_scaling[1];
    }''')*(im_is_def)+'''

    void eval
    (
              Array<double>& expr,
        const Array<double>& X
    ) const
    {'''+('''
        std::cout << "X = " << X.str(1) << std::endl;''')*(verbose)+(('''

        U->eval(UX, X);''')*(u_type=="dolfin")+('''

        probe_points->SetNumberOfPoints(1);
        probe_points->SetPoint(0,X.data());
        probe_polydata->SetPoints(probe_points);
        probe_filter->SetInputData(probe_polydata);
        probe_filter->Update();
        probe_filter->GetOutput()->GetPointData()->GetArray("U")->GetTuple(0, UX.data());

        ''')*(u_type=="vtk")+('''
        std::cout << "UX = " << UX.str(1) << std::endl;''')*(verbose)+('''
        x[0] = X[0] + UX[0];
        x[1] = X[1] + UX[1];''')*(im_dim==2)+('''
        x[0] = X[0] + UX[0];
        x[1] = X[1] + UX[1];
        x[2] = X[2] + UX[2];''')*(im_dim==3)+('''
        std::cout << "x = " << x.str(1) << std::endl;''')*(verbose)+'''
        interpolator->Interpolate(x.data(), expr.data());''')*(im_is_def)+(('''

        X3D[0] = X[0];
        X3D[1] = X[1];'''+('''
        std::cout << "X3D = " << X3D.str(1) << std::endl;''')*(verbose)+'''
        interpolator->Interpolate(X3D.data(), expr.data());''')*(im_dim==2)+('''

        interpolator->Interpolate(X.data(), expr.data());''')*(im_dim==3))*(not im_is_def)+('''

        std::cout << "expr = " << expr.str(1) << std::endl;''')*(verbose)+('''

        expr[0] /= static_scaling;''')*(im_type=="im")+(('''

        expr[0] /= static_scaling;
        expr[1] /= static_scaling;''')*(im_dim==2)+('''

        expr[0] /= static_scaling;
        expr[1] /= static_scaling;
        expr[2] /= static_scaling;''')*(im_dim==3))*(im_type=="grad")+('''

        std::cout << "expr = " << expr.str(1) << std::endl;''')*(verbose)+(('''

        expr[0] *= dynamic_scaling_a;
        expr[0] += dynamic_scaling_b;''')*(im_type=="im")+(('''

        expr[0] *= dynamic_scaling_a;
        expr[1] *= dynamic_scaling_a;''')*(im_dim==2)+('''

        expr[0] *= dynamic_scaling_a;
        expr[1] *= dynamic_scaling_a;
        expr[2] *= dynamic_scaling_a;''')*(im_dim==3))*(im_type=="grad")+('''

        std::cout << "expr = " << expr.str(1) << std::endl;''')*(verbose))*(im_is_def)+'''
    }
};

}'''
    # print(ExprIm_cpp)

    return ExprIm_cpp
