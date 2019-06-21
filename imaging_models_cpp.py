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

def get_ImagingModel_cpp():

    return '''\
class ImageModel
{
public:

    virtual void I(
        const dolfin::Array<double>& X,
              dolfin::Array<double>& I) const = 0;
};

class TaggingImageModel : public ImageModel
{
    std::shared_ptr<dolfin::Array<double>> K0;
    std::shared_ptr<dolfin::Array<double>> X0;

public:

    TaggingImageModel(
        std::shared_ptr<dolfin::Array<double>> K0_,
        std::shared_ptr<dolfin::Array<double>> X0_):
            K0(K0_),
            X0(X0_)
    {
    }

    void I(
        const dolfin::Array<double>& X,
              dolfin::Array<double>& I)
    {
        I[0] = abs(sin((*K0)[0]*(X[0]-(*X0)[0])
                      +(*K0)[1]*(X[1]-(*X0)[1])
                      +(*K0)[2]*(X[2]-(*X0)[2])));
        // I[0] = abs(sin(dolfin::dot(K0, X-X0)));
    }
};
'''
