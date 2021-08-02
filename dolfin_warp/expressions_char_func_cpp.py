#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2016-2021                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

import dolfin

import dolfin_warp as dwarp

################################################################################

def get_ExprCharFuncIm_cpp_pybind(
        im_dim,
        im_is_def=0,
        im_is_cone=0,
        verbose=0):

    assert (im_dim in (2,3))
    if (im_is_cone):
        assert (im_dim == 3)

    name  = "Expr"
    name += str(im_dim)
    name += "CharFuncIm"
    if   (im_is_def == 0):
        name += "Ref"
    elif (im_is_def == 1):
        name += "Def"
    # print(name)

    if   (im_is_def == 0):
        X_or_x = "X"
    elif (im_is_def == 1):
        X_or_x = "x"

    cpp = '''\
#include <math.h>
#include <string.h>

#include <dolfin/function/Expression.h>
#include <dolfin/function/Function.h>

#include <vtkSmartPointer.h>
#include <vtkXMLImageDataReader.h>
#include <vtkImageData.h>

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

class '''+name+''' : public dolfin::Expression
{
public:

    double xmin, xmax, ymin, ymax, zmin, zmax;'''+('''

    mutable Eigen::Vector'''+str(im_dim)+'''d UX, x;
    std::shared_ptr<dolfin::Function> U;''')*(im_is_def)+('''

    Eigen::Vector3d O, n1, n2, n3, n4;
    mutable double d1, d2, d3, d4;''')*(im_is_cone)+'''

    '''+name+'''() : dolfin::Expression() {}

    void init_image(
        const char* filename) {

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
        n4[2] = -sin(40. * M_PI/180.);''')*(im_is_cone)+'''}'''+('''

    void init_disp(
        std::shared_ptr<dolfin::Function> U_) {

        U = U_;}''')*(im_is_def)+'''

    void eval(
        Eigen::Ref<      Eigen::VectorXd> expr,
        Eigen::Ref<const Eigen::VectorXd> X   ) const {'''+('''

        std::cout << "X = " << X << std::endl;''')*(verbose)+('''

        U->eval(UX, X);'''+('''
        std::cout << "UX = " << UX << std::endl;''')*(verbose)+('''

        x[0] = X[0] + UX[0];
        x[1] = X[1] + UX[1];''')*(im_dim==2)+('''
        x[0] = X[0] + UX[0];
        x[1] = X[1] + UX[1];
        x[2] = X[2] + UX[2];''')*(im_dim==3)+('''
        std::cout << "x = " << x << std::endl;''')*(verbose))*(im_is_def)+('''

        d1 = ('''+X_or_x+'''[0]-O[0])*n1[0]
           + ('''+X_or_x+'''[1]-O[1])*n1[1]
           + ('''+X_or_x+'''[2]-O[2])*n1[2];
        d2 = ('''+X_or_x+'''[0]-O[0])*n2[0]
           + ('''+X_or_x+'''[1]-O[1])*n2[1]
           + ('''+X_or_x+'''[2]-O[2])*n2[2];
        d3 = ('''+X_or_x+'''[0]-O[0])*n3[0]
           + ('''+X_or_x+'''[1]-O[1])*n3[1]
           + ('''+X_or_x+'''[2]-O[2])*n3[2];
        d4 = ('''+X_or_x+'''[0]-O[0])*n4[0]
           + ('''+X_or_x+'''[1]-O[1])*n4[1]
           + ('''+X_or_x+'''[2]-O[2])*n4[2];''')*(im_is_cone)+('''

        if (('''+X_or_x+'''[0] >= xmin)
         && ('''+X_or_x+'''[0] <= xmax)
         && ('''+X_or_x+'''[1] >= ymin)
         && ('''+X_or_x+'''[1] <= ymax))''')*(im_dim==2)+('''
        if (('''+X_or_x+'''[0] >= xmin)
         && ('''+X_or_x+'''[0] <= xmax)
         && ('''+X_or_x+'''[1] >= ymin)
         && ('''+X_or_x+'''[1] <= ymax)
         && ('''+X_or_x+'''[2] >= zmin)
         && ('''+X_or_x+'''[2] <= zmax)'''+('''
         && (d1 >= 0.)
         && (d2 >= 0.)
         && (d3 >= 0.)
         && (d4 >= 0.)''')*(im_is_cone)+''')''')*(im_dim==3)+''' {
            expr[0] = 1.;}
        else {
            expr[0] = 0.;}'''+('''
        std::cout << "expr = " << expr << std::endl;''')*(verbose)+'''}
};

PYBIND11_MODULE(SIGNATURE, m)
{
    pybind11::class_<'''+name+''', std::shared_ptr<'''+name+'''>, dolfin::Expression>
    (m, "'''+name+'''")
    .def(pybind11::init<>())
    .def("init_image", &'''+name+'''::init_image, pybind11::arg("filename"))'''+('''
    .def("init_disp", &'''+name+'''::init_disp, pybind11::arg("U_"))''')*(im_is_def)+''';
}
'''
    # print(cpp)

    return name, cpp

################################################################################

def get_ExprCharFuncIm_cpp_swig(
        im_dim,
        im_is_def=0,
        im_is_cone=0,
        verbose=0):

    assert (im_dim in (2,3))
    if (im_is_cone):
        assert (im_dim == 3)

    if   (im_is_def == 0):
        X_or_x = "X"
    elif (im_is_def == 1):
        X_or_x = "x"

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

    double xmin, xmax, ymin, ymax, zmin, zmax;'''+('''

    mutable Array<double> UX, x;
    std::shared_ptr<Function> U;''')*(im_is_def)+('''

    Array<double> O, n1, n2, n3, n4;
    mutable double d1, d2, d3, d4;''')*(im_is_cone)+'''

public:

    MyExpr() :
        Expression()'''+(''',
        UX('''+str(im_dim)+'''),
        x('''+str(im_dim)+''')''')*(im_is_def)+(''',
        O(3),
        n1(3),
        n2(3),
        n3(3),
        n4(3)''')*(im_is_cone)+''' {}

    void init_image(
        const char* filename) {

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
        n4[2] = -sin(40. * M_PI/180.);''')*(im_is_cone)+'''}'''+('''

    void init_disp(
        std::shared_ptr<Function> U_) {

        U = U_;}''')*(im_is_def)+'''

    void eval(
              Array<double>& expr,
        const Array<double>& X   ) const {'''+('''

        std::cout << "X = " << X.str(1) << std::endl;''')*(verbose)+('''

        U->eval(UX, X);'''+('''
        std::cout << "UX = " << UX.str(1) << std::endl;''')*(verbose)+('''

        x[0] = X[0] + UX[0];
        x[1] = X[1] + UX[1];''')*(im_dim==2)+('''
        x[0] = X[0] + UX[0];
        x[1] = X[1] + UX[1];
        x[2] = X[2] + UX[2];''')*(im_dim==3)+('''
        std::cout << "x = " << x.str(1) << std::endl;''')*(verbose))*(im_is_def)+('''

       d1 = ('''+X_or_x+'''[0]-O[0])*n1[0]
          + ('''+X_or_x+'''[1]-O[1])*n1[1]
          + ('''+X_or_x+'''[2]-O[2])*n1[2];
       d2 = ('''+X_or_x+'''[0]-O[0])*n2[0]
          + ('''+X_or_x+'''[1]-O[1])*n2[1]
          + ('''+X_or_x+'''[2]-O[2])*n2[2];
       d3 = ('''+X_or_x+'''[0]-O[0])*n3[0]
          + ('''+X_or_x+'''[1]-O[1])*n3[1]
          + ('''+X_or_x+'''[2]-O[2])*n3[2];
       d4 = ('''+X_or_x+'''[0]-O[0])*n4[0]
          + ('''+X_or_x+'''[1]-O[1])*n4[1]
          + ('''+X_or_x+'''[2]-O[2])*n4[2];''')*(im_is_cone)+('''

        if (('''+X_or_x+'''[0] >= xmin)
         && ('''+X_or_x+'''[0] <= xmax)
         && ('''+X_or_x+'''[1] >= ymin)
         && ('''+X_or_x+'''[1] <= ymax))''')*(im_dim==2)+('''
        if (('''+X_or_x+'''[0] >= xmin)
         && ('''+X_or_x+'''[0] <= xmax)
         && ('''+X_or_x+'''[1] >= ymin)
         && ('''+X_or_x+'''[1] <= ymax)
         && ('''+X_or_x+'''[2] >= zmin)
         && ('''+X_or_x+'''[2] <= zmax)'''+('''
         && (d1 >= 0.)
         && (d2 >= 0.)
         && (d3 >= 0.)
         && (d4 >= 0.)''')*(im_is_cone)+''')''')*(im_dim==3)+''' {
            expr[0] = 1.;}
        else {
            expr[0] = 0.;}'''+('''
        std::cout << "expr = " << expr.str(1) << std::endl;''')*(verbose)+'''}
};

}
'''
    # print(ExprCharFuncIm_cpp)

    return ExprCharFuncIm_cpp
