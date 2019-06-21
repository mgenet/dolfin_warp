#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2016-2019                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

from builtins import *

import dolfin_dic as ddic

################################################################################

def get_ExprGenIm_cpp(
        im_dim, # 2, 3
        im_type="im", # im
        im_default_interpol_mode="linear", # nearest, linear, cubic
        im_default_interpol_out_value="0.",
        grad_default_interpol_mode="linear", # nearest, linear, cubic
        grad_default_interpol_out_value="0.",
        verbose=0):

    assert (im_dim in (2,3))
    assert (im_type in ("im"))

    ExprGenIm_cpp = '''\
#include <string.h>

#include <vtkSmartPointer.h>
#include <vtkXMLImageDataReader.h>
#include <vtkImageData.h>
#include <vtkImageInterpolator.h>
#include <vtkPointData.h>
#include <vtkDataArray.h>
#include <vtkUnstructuredGrid.h>
#include <vtkWarpVector.h>
#include <vtkProbeFilter.h>

'''+ddic.get_StaticScaling_cpp()+'''\

namespace dolfin
{

class MyExpr : public Expression
{
    unsigned int k_point;
    unsigned int image_n_points;

    mutable Array<double>  X;
    mutable Array<double> UX; // MG20190521: these guys cannot be public for some reason
    mutable Array<double>  x; // MG20190521: these guys cannot be public for some reason
    mutable Array<double> ux;

    double static_scaling;
    double m[1];
    double I[1];

    vtkSmartPointer<vtkXMLImageDataReader> reader;
    vtkSmartPointer<vtkImageData> image;
    vtkSmartPointer<vtkDataArray> array_image_scalars;
    // vtkSmartPointer<vtkImageData> image_probed; // MG20190607: does not work for some reason
    vtkSmartPointer<vtkDataArray> array_probed_mask;
    vtkSmartPointer<vtkDataArray> array_probed_disp;
    vtkSmartPointer<vtkImageInterpolator> interpolator;
    vtkSmartPointer<vtkUnstructuredGrid> ugrid;
    vtkSmartPointer<vtkWarpVector> warp;
    vtkSmartPointer<vtkProbeFilter> probe;

    std::shared_ptr<Mesh> mesh;
    std::shared_ptr<Function> U;

public:

    MyExpr():
        Expression(),
        X(3),
        UX('''+str(im_dim)+'''),
        x(3),
        ux('''+str(im_dim)+'''),
        reader(vtkSmartPointer<vtkXMLImageDataReader>::New()),
        interpolator(vtkSmartPointer<vtkImageInterpolator>::New()),
        warp(vtkSmartPointer<vtkWarpVector>::New()),
        probe(vtkSmartPointer<vtkProbeFilter>::New())
    {
    }

    void init_image(
        const char* filename,
        const char* interpol_mode="'''+(  im_default_interpol_mode)*(im_type=="im")\
                                      +(grad_default_interpol_mode)*(im_type in ("grad", "grad_no_deriv"))+'''",
        const double &interpol_out_value='''+(  im_default_interpol_out_value)*(im_type=="im")\
                                            +(grad_default_interpol_out_value)*(im_type in ("grad", "grad_no_deriv"))+(''',
        const double &Z=0.''')*(im_dim==2)+''')
    {
        // reader = vtkSmartPointer<vtkXMLImageDataReader>::New();
        reader->SetFileName(filename);
        reader->Update();

        image.TakeReference(reader->GetOutput());
        image_n_points = image->GetNumberOfPoints();
        array_image_scalars.TakeReference(image->GetPointData()->GetScalars());
        static_scaling = getStaticScalingFactor(image->GetScalarTypeAsString());

        // interpolator = vtkSmartPointer<vtkImageInterpolator>::New();
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
        interpolator->Initialize(image);'''+('''

        X[2] = Z;
        x[2] = Z;''')*(im_dim==2)+'''
    }

    void init_mesh(
        const std::shared_ptr<Mesh> mesh_)
    {
        mesh = mesh_;

        std::cout << mesh->num_vertices() << std::endl;
        std::cout << mesh->num_cells()    << std::endl;
    }

    void init_disp(
        const std::shared_ptr<Function> U_)
    {
        U = U_;
    }

    void init_ugrid(
        const vtkSmartPointer<vtkUnstructuredGrid> ugrid_)
    {
        ugrid = ugrid_;

        std::cout << "n_points = " << ugrid->GetNumberOfPoints() << std::endl;
        std::cout << "n_cells  = " << ugrid->GetNumberOfCells()  << std::endl;
    }

    void generate_image()
    {
        // std::cout << "n_points = " << ugrid->GetNumberOfPoints() << std::endl;
        // std::cout << "n_cells  = " << ugrid->GetNumberOfCells()  << std::endl;

        warp->SetInputData(ugrid);
        warp->Update();

        probe->SetInputData(image);
        probe->SetSourceData(ugrid);
        probe->Update();
        // image_probed.TakeReference(probe->GetOutput()); // MG20190607: does not work for some reason
        array_probed_mask.TakeReference(probe->GetOutput()->GetPointData()->GetArray("vtkValidPointMask"));
        array_probed_disp.TakeReference(probe->GetOutput()->GetPointData()->GetArray("U"));
        for (k_point=0;
             k_point<image_n_points;
           ++k_point)
        {
            array_probed_mask->GetTuple(k_point, m);
            if (m[0] == 0)
            {
                I[0] = 0.;
            }
            else
            {
                image->GetPoint(k_point, x.data());
                array_probed_disp->GetTuple(k_point, ux.data());'''+('''
                X[0] = x[0] - ux[0];
                X[1] = x[1] - ux[1];''')*(im_dim==2)+('''
                X[0] = x[0] - ux[0];
                X[1] = x[1] - ux[1];
                X[2] = x[2] - ux[2];''')*(im_dim==3)+'''
                I[0] = 1.;
                // image_model->I(X, I);
            }
            array_image_scalars->SetTuple(k_point, I);
        }
    }

    void eval(
              Array<double>& expr,
        const Array<double>& X_  ) const
    {'''+('''
        std::cout << "X_ = " << X_.str(1) << std::endl;
        ''')*(verbose)+'''
        U->eval(UX, X_);'''+('''
        std::cout << "UX = " << UX.str(1) << std::endl;''')*(verbose)+('''

        x[0] = X_[0] + UX[0];
        x[1] = X_[1] + UX[1];''')*(im_dim==2)+('''
        x[0] = X_[0] + UX[0];
        x[1] = X_[1] + UX[1];
        x[2] = X_[2] + UX[2];''')*(im_dim==3)+('''
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
};

}

'''+ddic.get_ImagingModel_cpp()+''' // MG20190617: Cannot be defined before expression for some reason

'''
    # print(ExprGenIm_cpp)
    return ExprGenIm_cpp
