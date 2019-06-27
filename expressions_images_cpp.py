#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2016-2019                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

# from future.utils import native_str

import dolfin

import dolfin_dic as ddic

################################################################################

def get_ExprIm_cpp(
        *args,
        **kwargs):

    if (int(dolfin.dolfin_version().split('.')[0]) >= 2018):
        return get_ExprIm_cpp_pybind(
            *args,
            **kwargs)
    else:
        return get_ExprIm_cpp_swig(
            *args,
            **kwargs)


################################################################################

def get_ExprIm_cpp_swig(
        im_dim, # 2, 3
        im_type="im", # im, grad, grad_no_deriv
        im_is_def=0,
        u_is_vtk=0,
        im_default_interpol_mode="linear", # linear, nearest
        im_default_interpol_out_value="0.",
        grad_default_interpol_mode="linear", # linear, nearest
        grad_default_interpol_out_value="0.",
        static_scaling_factor=0,
        verbose=0):

    assert (im_dim in (2,3))
    assert (im_type in ("im","grad","grad_no_deriv"))

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
#include <vtkPolyData.h>''')*(im_is_def)*(u_is_vtk)+'''

'''+ddic.get_StaticScaling_cpp()+'''\

namespace dolfin
{

class MyExpr : public Expression
{
    vtkSmartPointer<vtkImageInterpolator> interpolator;

    double static_scaling;'''+('''

    // std::shared_ptr<Array<double>> dynamic_scaling; // MG20160719: does not work
    double dynamic_scaling_a;                          // MG20160719: should not be needed
    double dynamic_scaling_b;                          // MG20160719: should not be needed'''+('''

    vtkSmartPointer<vtkProbeFilter> probe_filter;
    vtkSmartPointer<vtkPoints> probe_points;
    vtkSmartPointer<vtkPolyData> probe_polydata;''')*(u_is_vtk)+('''

    std::shared_ptr<Function> U;''')*(not u_is_vtk)+'''

    mutable Array<double> UX;
    mutable Array<double> x;''')*(im_is_def)+('''
    mutable Array<double> X3D;''')*(not im_is_def)*(im_dim==2)+'''

public:

    MyExpr():
        Expression('''+str(im_dim)*(im_type in ("grad", "grad_no_deriv"))+''')'''+(''',
        dynamic_scaling_a(1.), // MG20160719: should not be needed
        dynamic_scaling_b(0.), // MG20160719: should not be needed
        UX('''+str(im_dim)+'''),
        x(3)''')*(im_is_def)+(''',
        X3D(3)''')*(not im_is_def)*(im_dim==2)+'''
    {'''+('''
        probe_filter = vtkSmartPointer<vtkProbeFilter>::New();
        probe_points = vtkSmartPointer<vtkPoints>::New();
        probe_polydata = vtkSmartPointer<vtkPolyData>::New();''')*(u_is_vtk)+'''
    }'''+('''

    void init_dynamic_scaling(
        const Array<double> &scaling)
    {
        // dynamic_scaling = scaling;                                                  // MG20160719: does not work
        // std::cout << "dynamic_scaling = " << dynamic_scaling->str(1) << std::endl; // MG20160719: does not work
        dynamic_scaling_a = scaling[0];                                               // MG20160719: should not be needed
        dynamic_scaling_b = scaling[1];                                               // MG20160719: should not be needed
    }

    void update_dynamic_scaling(        // MG20160719: should not be needed
        const Array<double> &scaling)   // MG20160719: should not be needed
    {                                   // MG20160719: should not be needed
        dynamic_scaling_a = scaling[0]; // MG20160719: should not be needed
        dynamic_scaling_b = scaling[1]; // MG20160719: should not be needed
    }                                   // MG20160719: should not be needed

    '''+('''
    void init_disp(
        std::shared_ptr<Function> U_)
    {
        U = U_;
    }''')*(not u_is_vtk)+('''
    void init_disp(
        const char* mesh_filename)
    {
        vtkSmartPointer<vtkXMLUnstructuredGridReader> reader = vtkSmartPointer<vtkXMLUnstructuredGridReader>::New();
        reader->SetFileName(mesh_filename);
        reader->Update();

        vtkSmartPointer<vtkUnstructuredGrid> mesh = reader->GetOutput();

        probe_filter->SetSourceData(mesh);
    }''')*(u_is_vtk))*(im_is_def)+'''

    void init_image(
        const char* filename,
        const char* interpol_mode="'''+(im_default_interpol_mode)*(im_type=="im")+(grad_default_interpol_mode)*(im_type in ("grad", "grad_no_deriv"))+'''",
        const double &interpol_out_value='''+(im_default_interpol_out_value)*(im_type=="im")+(grad_default_interpol_out_value)*(im_type in ("grad", "grad_no_deriv"))+(''',
        const double &Z=0.''')*(im_dim==2)+''')
    {
        vtkSmartPointer<vtkXMLImageDataReader> reader = vtkSmartPointer<vtkXMLImageDataReader>::New();
        reader->SetFileName(filename);
        reader->Update();'''+('''

        static_scaling = getStaticScalingFactor(reader->GetOutput()->GetScalarTypeAsString());''')*(not static_scaling_factor)+('''
        static_scaling = '''+str(static_scaling_factor)+''';''')*(static_scaling_factor)+('''

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
        interpolator->Initialize('''+('''reader->GetOutput()''')*(im_type in ("im", "grad_no_deriv"))+('''gradient->GetOutput()''')*(im_type=="grad")+''');'''+('''

        x[2] = Z;''')*(im_is_def)*(im_dim==2)+('''

        X3D[2] = Z;''')*(not im_is_def)*(im_dim==2)+'''
    }

    void eval(
              Array<double>& expr,
        const Array<double>& X_  ) const
    {'''+('''
        std::cout << "X_ = " << X_.str(1) << std::endl;''')*(verbose)+(('''

        U->eval(UX, X_);''')*(not u_is_vtk)+('''

        probe_points->SetNumberOfPoints(1);
        probe_points->SetPoint(0,X_.data());
        probe_polydata->SetPoints(probe_points);
        probe_filter->SetInputData(probe_polydata);
        probe_filter->GetOutput()->GetPointData()->GetArray("U")->GetTuple(0, UX.data());

        ''')*(u_is_vtk)+('''
        std::cout << "UX = " << UX.str(1) << std::endl;''')*(verbose)+('''
        x[0] = X_[0] + UX[0];
        x[1] = X_[1] + UX[1];''')*(im_dim==2)+('''
        x[0] = X_[0] + UX[0];
        x[1] = X_[1] + UX[1];
        x[2] = X_[2] + UX[2];''')*(im_dim==3)+('''
        std::cout << "x = " << x.str(1) << std::endl;''')*(verbose)+'''
        interpolator->Interpolate(x.data(), expr.data());''')*(im_is_def)+('''

        X3D[0] = X_[0];
        X3D[1] = X_[1];'''+('''
        std::cout << "X3D = " << X3D.str(1) << std::endl;''')*(verbose)+'''
        interpolator->Interpolate(X3D.data(), expr.data());''')*(not im_is_def)*(im_dim==2)+('''

        interpolator->Interpolate(X_.data(), expr.data());''')*(not im_is_def)*(im_dim==3)+('''

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
        // expr[0] *= (*dynamic_scaling)[0]; // MG20160719: does not work
        // expr[0] += (*dynamic_scaling)[1]; // MG20160719: does not work
        expr[0] *= dynamic_scaling_a;        // MG20160719: should not be needed
        expr[0] += dynamic_scaling_b;        // MG20160719: should not be needed
        // std::cout << "expr = " << expr.str(1) << std::endl;
        // std::cout << "out (im)" << std::endl;''')*(im_type=="im")+('''

        // std::cout << "in (grad)" << std::endl;
        // std::cout << "expr = " << expr.str(1) << std::endl;
        // expr[0] *= (*dynamic_scaling)[0]; // MG20160719: does not work
        // expr[1] *= (*dynamic_scaling)[0]; // MG20160719: does not work
        expr[0] *= dynamic_scaling_a;        // MG20160719: should not be needed
        expr[1] *= dynamic_scaling_a;        // MG20160719: should not be needed
        // std::cout << "expr = " << expr.str(1) << std::endl;
        // std::cout << "out (grad)" << std::endl;''')*(im_type=="grad")*(im_dim==2)+('''

        // std::cout << "in (grad)" << std::endl;
        // std::cout << "expr = " << expr.str(1) << std::endl;
        // expr[0] *= (*dynamic_scaling)[0]; // MG20160719: does not work
        // expr[1] *= (*dynamic_scaling)[0]; // MG20160719: does not work
        // expr[2] *= (*dynamic_scaling)[0]; // MG20160719: does not work
        expr[0] *= dynamic_scaling_a;        // MG20160719: should not be needed
        expr[1] *= dynamic_scaling_a;        // MG20160719: should not be needed
        expr[2] *= dynamic_scaling_a;        // MG20160719: should not be needed
        // std::cout << "expr = " << expr.str(1) << std::endl;
        // std::cout << "out (grad)" << std::endl;''')*(im_type=="grad")*(im_dim==3)+('''

        std::cout << "expr = " << expr.str(1) << std::endl;''')*(verbose))*(im_is_def)+'''
    }
};

}'''
    # print(ExprIm_cpp)
    # print(type(ExprIm_cpp))
    # print(type(native_str(ExprIm_cpp)))
    # return native_str(ExprIm_cpp)
    return ExprIm_cpp

################################################################################

def get_ExprCharFuncIm_cpp(
        im_dim,
        im_is_def=0,
        im_is_cone=0,
        verbose=0):

    assert (im_dim in (2,3))
    if (im_is_cone):
        assert (im_dim == 3)

    ExprCharFuncIm_cpp = '''\
#include <math.h>
#include <string.h>

#include <vtkSmartPointer.h>
#include <vtkXMLImageDataReader.h>
#include <vtkImageData.h>

namespace dolfin
{

class MyExpr : public Expression
{
    double xmin, xmax, ymin, ymax, zmin, zmax;
    mutable Array<double> UX;
    mutable Array<double> x;'''+('''
    Function* U;''')*(im_is_def)+('''
    Array<double> O;
    Array<double> n1;
    Array<double> n2;
    Array<double> n3;
    Array<double> n4;
    mutable double d1, d2, d3, d4;''')*(im_is_cone)+'''

public:

    MyExpr():
        Expression(),
        UX('''+str(im_dim)+'''),
        x('''+str(im_dim)+''')'''+(''',
        O(3),
        n1(3),
        n2(3),
        n3(3),
        n4(3)''')*(im_is_cone)+'''
    {
    }'''+('''

    void init_disp(
        Function* U_)
    {
        U = U_;
    }''')*(im_is_def)+'''

    void init_image(
        const char* filename)
    {
        vtkSmartPointer<vtkXMLImageDataReader> reader = vtkSmartPointer<vtkXMLImageDataReader>::New();
        reader->SetFileName(filename);
        reader->Update();

        vtkSmartPointer<vtkImageData> image = reader->GetOutput();
        double* bounds = image->GetBounds();
        xmin = bounds[0];
        xmax = bounds[1];
        ymin = bounds[2];
        ymax = bounds[3];
        zmin = bounds[4];
        zmax = bounds[5];'''+('''
        std::cout << "bounds = " << bounds[0] << " " << bounds[1] << " " << bounds[2] << " " << bounds[3] << " " << bounds[4] << " " << bounds[5] << std::endl;
        std::cout << "xmin = " << xmin << std::endl;
        std::cout << "xmax = " << xmax << std::endl;
        std::cout << "ymin = " << ymin << std::endl;
        std::cout << "ymax = " << ymax << std::endl;
        std::cout << "zmin = " << zmin << std::endl;
        std::cout << "zmax = " << zmax << std::endl;''')*(verbose)+('''

        O[0] = (xmin+xmax)/2;
        O[1] = (ymin+ymax)/2;
        O[2] = zmax;

        n1[0] = +cos(35. * M_PI/180.);
        n1[1] = 0.;
        n1[2] = -sin(35. * M_PI/180.);

        n2[0] = -cos(35. * M_PI/180.);
        n2[1] = 0.;
        n2[2] = -sin(35. * M_PI/180.);

        n3[0] = 0.;
        n3[1] = +cos(40. * M_PI/180.);
        n3[2] = -sin(40. * M_PI/180.);

        n4[0] = 0.;
        n4[1] = -cos(40. * M_PI/180.);
        n4[2] = -sin(40. * M_PI/180.);''')*(im_is_cone)+'''
    }

    void eval(
              Array<double>& expr,
        const Array<double>& X_  ) const
    {'''+('''
        std::cout << "X_ = " << X_.str(1) << std::endl;''')*(verbose)+('''

        U->eval(UX, X_);''')*(im_is_def)+('''
        std::cout << "UX = " << UX.str(1) << std::endl;''')*(verbose)+('''

        x[0] = X_[0] + UX[0];
        x[1] = X_[1] + UX[1];''')*(im_dim==2)+('''
        x[0] = X_[0] + UX[0];
        x[1] = X_[1] + UX[1];
        x[2] = X_[2] + UX[2];''')*(im_dim==3)+('''
        std::cout << "x = " << x.str(1) << std::endl;''')*(verbose)+(('''

        if ((x[0] >= xmin)
         && (x[0] <= xmax)
         && (x[1] >= ymin)
         && (x[1] <= ymax))''')*(im_dim==2)+('''
        if ((x[0] >= xmin)
         && (x[0] <= xmax)
         && (x[1] >= ymin)
         && (x[1] <= ymax)
         && (x[2] >= zmin)
         && (x[2] <= zmax))''')*(im_dim==3))*(not im_is_cone)+('''

        d1 = (x[0]-O[0])*n1[0]
           + (x[1]-O[1])*n1[1]
           + (x[2]-O[2])*n1[2];
        d2 = (x[0]-O[0])*n2[0]
           + (x[1]-O[1])*n2[1]
           + (x[2]-O[2])*n2[2];
        d3 = (x[0]-O[0])*n3[0]
           + (x[1]-O[1])*n3[1]
           + (x[2]-O[2])*n3[2];
        d4 = (x[0]-O[0])*n4[0]
           + (x[1]-O[1])*n4[1]
           + (x[2]-O[2])*n4[2];

        if ((d1 >= 0.)
         && (d2 >= 0.)
         && (d3 >= 0.)
         && (d4 >= 0.))''')*(im_is_cone)+'''
        {
            expr[0] = 1.;
        }
        else
        {
            expr[0] = 0.;
        }'''+('''
        std::cout << "expr = " << expr.str(1) << std::endl;''')*(verbose)+'''
    }
};

}'''
    # print(ExprCharFuncIm_cpp)
    # return native_str(ExprCharFuncIm_cpp)
    return ExprCharFuncIm_cpp
