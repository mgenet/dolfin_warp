#coding=utf8

################################################################################
###                                                                          ###
### Created by Ezgi Berberoğlu, 2018-2021                                    ###
###                                                                          ###
### Swiss Federal Institute of Technology (ETH), Zurich, Switzerland         ###
###                                                                          ###
### And Martin Genet, 2016-2019                                              ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

import dolfin
import numpy

import myPythonLibrary    as mypy
import myVTKPythonLibrary as myvtk

import dolfin_warp as dwarp

################################################################################

def get_ExprProbedGrid_swig(
    image_dim=3,
    image_field_name="displacement"):

    assert (ProbedGridExpr <= 3)

    cpp = '''
#include <vtkPolyData.h>
#include <vtkProbeFilter.h>
#include <vtkSmartPointer.h>
#include <vtkStructuredGrid.h>
#include <vtkStructuredGridReader.h>

namespace dolfin
{

class MyExpr : public Expression
{
    vtkSmartPointer<vtkStructuredGrid> sgrid;
    vtkSmartPointer<vtkPoints> probe_points;
    vtkSmartPointer<vtkPolyData> probe_polydata;
    vtkSmartPointer<vtkProbeFilter> probe_filter;

public:

    MyExpr():
        Expression('''+str(image_dim)+''')
    {
        sgrid = vtkSmartPointer<vtkStructuredGrid>::New();
        probe_points = vtkSmartPointer<vtkPoints>::New();
        probe_polydata = vtkSmartPointer<vtkPolyData>::New();
        probe_filter = vtkSmartPointer<vtkProbeFilter>::New();
    }

    void init_image(
        const char* image_filename)
    {
        vtkSmartPointer<vtkStructuredGridReader> reader = vtkSmartPointer<vtkStructuredGridReader>::New();
        reader->SetFileName(image_filename);
        reader->Update();
        sgrid = reader->GetOutput();
        probe_filter->SetSourceData(sgrid);
    }

    void eval(
        dolfin::Array<double>& expr,
        const dolfin::Array<double>& X) const
    {
        probe_points->SetNumberOfPoints(1);
        probe_points->SetPoint(0, X.data());
        probe_polydata->SetPoints(probe_points);
        probe_filter->SetInputData(probe_polydata);
        probe_filter->Update();
        probe_filter->GetOutput()->GetPointData()->GetArray("'''+image_field_name+'''")->GetTuple(0, expr.data());
    }
};

}
'''

    return cpp

def get_ExprProbedGrid_pybind(
        image_dim=3,
        image_field_name="displacement"):

    assert (image_dim <= 3)

    cpp = '''
#include <dolfin/function/Expression.h>

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

#include <vtkPolyData.h>
#include <vtkPointData.h>
#include <vtkProbeFilter.h>
#include <vtkSmartPointer.h>
#include <vtkStructuredGrid.h>
#include <vtkStructuredGridReader.h>

class ProbedGridExpr : public dolfin::Expression
{
    vtkSmartPointer<vtkStructuredGrid> sgrid;
    vtkSmartPointer<vtkPoints> probe_points;
    vtkSmartPointer<vtkPolyData> probe_polydata;
    vtkSmartPointer<vtkProbeFilter> probe_filter;

public:

    ProbedGridExpr():
        Expression('''+str(image_dim)+''')
    {
        sgrid = vtkSmartPointer<vtkStructuredGrid>::New();
        probe_points = vtkSmartPointer<vtkPoints>::New();
        probe_polydata = vtkSmartPointer<vtkPolyData>::New();
        probe_filter = vtkSmartPointer<vtkProbeFilter>::New();
    }

    void init_image(
        const char* image_filename)
    {
        vtkSmartPointer<vtkStructuredGridReader> reader = vtkSmartPointer<vtkStructuredGridReader>::New();
        reader->SetFileName(image_filename);
        reader->Update();
        sgrid = reader->GetOutput();
        probe_filter->SetSourceData(sgrid);
    }

    void eval(
        dolfin::Array<double>& expr,
        const dolfin::Array<double>& X) const
    {
        probe_points->SetNumberOfPoints(1);
        probe_points->SetPoint(0, X.data());
        probe_polydata->SetPoints(probe_points);
        probe_filter->SetInputData(probe_polydata);
        probe_filter->Update();
        probe_filter->GetOutput()->GetPointData()->GetArray("'''+image_field_name+'''")->GetTuple(0, expr.data());
    }
};

PYBIND11_MODULE(SIGNATURE, m)
{
    pybind11::class_<ProbedGridExpr, std::shared_ptr<ProbedGridExpr>, dolfin::Expression>
    (m, "ProbedGridExpr")
    .def(pybind11::init<>())
    .def("init_image", &ProbedGridExpr::init_image, pybind11::arg("filename"));
}
'''
    # print(cpp)

    return cpp

def get_ExprImageData_pybind(
        image_field_name="porosity"):

    cpp = '''
#include <dolfin/function/Expression.h>

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

#include <vtkPolyData.h>
#include <vtkPointData.h>
#include <vtkImageData.h>
#include <vtkProbeFilter.h>
#include <vtkSmartPointer.h>
#include <vtkStructuredGrid.h>
#include <vtkXMLImageDataReader.h>

class ImageDataExpr : public dolfin::Expression
{
    vtkSmartPointer<vtkImageData> image_data;
    vtkSmartPointer<vtkPoints> probe_points;
    vtkSmartPointer<vtkPolyData> probe_polydata;
    vtkSmartPointer<vtkProbeFilter> probe_filter;

public:

    ImageDataExpr():
        Expression()
    {
        image_data = vtkSmartPointer<vtkImageData>::New();
        probe_points = vtkSmartPointer<vtkPoints>::New();
        probe_polydata = vtkSmartPointer<vtkPolyData>::New();
        probe_filter = vtkSmartPointer<vtkProbeFilter>::New();
    }

    void init_image(
        const char* image_filename)
    {
        vtkSmartPointer<vtkXMLImageDataReader> reader = vtkSmartPointer<vtkXMLImageDataReader>::New();
        reader->SetFileName(image_filename);
        reader->Update();
        image_data = reader->GetOutput();
        probe_filter->SetSourceData(image_data);
    }

    void eval(
        dolfin::Array<double>& expr,
        const dolfin::Array<double>& X) const
    {
        probe_points->SetNumberOfPoints(1);
        probe_points->SetPoint(0, X.data());
        probe_polydata->SetPoints(probe_points);
        probe_filter->SetInputData(probe_polydata);
        probe_filter->Update();
        probe_filter->GetOutput()->GetPointData()->GetArray("'''+image_field_name+'''")->GetTuple(0, expr.data());
    }
};

PYBIND11_MODULE(SIGNATURE, m)
{
    pybind11::class_<ImageDataExpr, std::shared_ptr<ImageDataExpr>, dolfin::Expression>
    (m, "ImageDataExpr")
    .def(pybind11::init<>())
    .def("init_image", &ImageDataExpr::init_image, pybind11::arg("filename"));
}
'''
    # print(cpp)

    return cpp

def compute_projected_image(
        mesh,
        image_filename,
        image_field_name="displacement",
        image_field_dim=1,
        image_field_family="Lagrange",
        image_field_degree=1,
        image_quadrature=1):

    dV = dolfin.Measure("dx", domain=mesh)
    form_compiler_parameters_for_images = {}
    form_compiler_parameters_for_images["quadrature_degree"] = image_quadrature

    if (image_field_dim == 1):
        fe = dolfin.FiniteElement(
            family="Quadrature",
            cell=mesh.ufl_cell(),
            degree=image_quadrature,
            quad_scheme="default")

        fs = dolfin.FunctionSpace(
            mesh,
            image_field_family,
            image_field_degree)
    else:
        fe = dolfin.VectorElement(
            family="Quadrature",
            cell=mesh.ufl_cell(),
            degree=image_quadrature,
            quad_scheme="default")

        fs = dolfin.VectorFunctionSpace(
            mesh=mesh,
            family=image_field_family,
            degree=image_field_degree)

    projected_func = dolfin.Function(
        fs,
        name=image_field_name)
    U = dolfin.TrialFunction(
        fs)
    V = dolfin.TestFunction(
        fs)

    if (int(dolfin.__version__.split('.')[0]) >= 2018):
        if (image_field_dim == 1): # CL 03/2021: not a good conditional statement here
            cpp = get_ExprImageData_pybind(
                image_field_name=image_field_name)
            module = dolfin.compile_cpp_code(cpp)
            expr = getattr(module, "ImageDataExpr")
        else:
            cpp = get_ExprProbedGrid_pybind(
                image_dim=3,
                image_field_name=image_field_name)
            module = dolfin.compile_cpp_code(cpp)
            expr = getattr(module, "ProbedGridExpr")
        source_expr = dolfin.CompiledExpression(
            expr(),
            element=fe)
    else:
        source_expr = dolfin.Expression(
            cppcode=get_ExprProbedGrid_swig(
                image_dim=3,
                image_field_name=image_field_name),
            element=fe)
    source_expr.init_image(
        image_filename)

    M = dolfin.assemble(
        dolfin.inner(U,V)*dV)

    N = dolfin.assemble(
        dolfin.inner(source_expr, V) * dV,
        form_compiler_parameters=form_compiler_parameters_for_images)

    dolfin.solve(M, projected_func.vector(), N)

    return projected_func
