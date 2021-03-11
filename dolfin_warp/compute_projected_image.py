#coding=utf8

################################################################################
###                                                                          ###
### Created by Ezgi Berberoğlu, 2018-2020                                    ###
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

def compute_projected_image(
        mesh,
        image_filename,
        image_field_name="displacement",
        image_quadrature=1):

    dV = dolfin.Measure("dx", domain=mesh)
    form_compiler_parameters_for_images = {}
    form_compiler_parameters_for_images["quadrature_degree"] = image_quadrature

    vfe = dolfin.VectorElement(
        family="Quadrature",
        cell=mesh.ufl_cell(),
        degree=image_quadrature,
        quad_scheme="default")

    vfs = dolfin.VectorFunctionSpace(
        mesh=mesh,
        family="Lagrange",
        degree=1)

    projected_func = dolfin.Function(
        vfs,
        name=image_field_name)
    U = dolfin.TrialFunction(
        vfs)
    V = dolfin.TestFunction(
        vfs)

    source_expr = dolfin.Expression(
        cppcode='''\
#include <vtkSmartPointer.h>
#include <vtkStructuredGridReader.h>
#include <vtkProbeFilter.h>
#include <vtkStructuredGrid.h>
#include <vtkPolyData.h>
#include <vtkPointData.h>

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
        Expression(3)
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
        Array<double>& expr,
        const Array<double>& X) const
    {
        probe_points->SetNumberOfPoints(1);
        probe_points->SetPoint(0, X.data());
        probe_polydata->SetPoints(probe_points);
        probe_filter->SetInputData(probe_polydata);
        probe_filter->Update();
        probe_filter->GetOutput()->GetPointData()->GetArray("'''+image_field_name+'''")->GetTuple(0, expr.data());
    }
};

}''',
        element=vfe)
    source_expr.init_image(
        image_filename)

    M = dolfin.assemble(
        dolfin.inner(U,V)*dV)

    N = dolfin.assemble(
        dolfin.inner(source_expr,V)*dV,
        form_compiler_parameters=form_compiler_parameters_for_images)

    dolfin.solve(M, projected_func.vector(), N)

    return projected_func
