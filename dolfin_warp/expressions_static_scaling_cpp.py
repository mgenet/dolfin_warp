#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2016-2021                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

################################################################################

def get_StaticScaling_cpp():

    return '''\
double getStaticScalingFactor
(
    const char* scalar_type_as_string
)
{
    if (strcmp(scalar_type_as_string, "unsigned char" ) == 0)
    {
        return pow(2,  8)-1;
    }
    if (strcmp(scalar_type_as_string, "unsigned short") == 0)
    {
        return pow(2, 16)-1;
    }
    if (strcmp(scalar_type_as_string, "unsigned int"  ) == 0)
    {
        return pow(2, 32)-1;
    }
    if (strcmp(scalar_type_as_string, "unsigned long" ) == 0)
    {
        return pow(2, 64)-1;
    }
    if (strcmp(scalar_type_as_string, "float"         ) == 0)
    {
        return 1.;
    }
    if (strcmp(scalar_type_as_string, "double"        ) == 0)
    {
        return 1.;
    }
    std::exit(0);
}
'''
