#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2016-2024                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

import math
import os
import shutil
import sys

import myPythonLibrary as mypy
import dolfin_warp     as dwarp

################################################################################

res_folder = sys.argv[0][:-3]
if not os.path.exists(res_folder): os.mkdir(res_folder)

# test = mypy.Test(
#     res_folder=res_folder,
#     perform_tests=0,
#     clean_after_tests=1)

n_dim_lst  = [ ]
n_dim_lst += [2]
# n_dim_lst += [3]

structure_deformation_type_lst = []
structure_deformation_type_lst += [["box" , "translation"]]
structure_deformation_type_lst += [["box" , "rotation"   ]]
structure_deformation_type_lst += [["box" , "compression"]]
structure_deformation_type_lst += [["box" , "shear"      ]]
structure_deformation_type_lst += [["ring", "heart"      ]]

texture_type_lst  = []
texture_type_lst += ["no"]
texture_type_lst += ["tagging"]
texture_type_lst += ["tagging-addComb"]
texture_type_lst += ["tagging-diffComb"]
texture_type_lst += ["tagging-signed"]
texture_type_lst += ["tagging-signed-addComb"]
texture_type_lst += ["tagging-signed-diffComb"]

upsampling_factor_lst  = [ ]
upsampling_factor_lst += [1]
upsampling_factor_lst += [2]

noise_level_lst  = []
noise_level_lst += [0]
noise_level_lst += [0.1]

for n_dim                            in n_dim_lst                     :
 for structure_type, deformation_type in structure_deformation_type_lst:
  for texture_type                     in texture_type_lst              :
   for upsampling_factor                in upsampling_factor_lst         :
    for noise_level                      in noise_level_lst               :

        images = {
            "n_dim":n_dim,
            "L":[1.]*n_dim,
            "n_voxels": [10]*n_dim,
            "upsampling_factors": [upsampling_factor]*n_dim,
            "T":1.,
            "n_frames":3,
            "data_type":"float",
            "folder":res_folder}

        images_basename = str(n_dim)+"D"
        images_basename += "-"+structure_type
        images_basename += "-"+deformation_type
        images_basename += "-"+texture_type
        images_basename += "-"+"x".join([str(n) for n in images["n_voxels"]])
        if (upsampling_factor != 1):
            images_basename += "-"+"x".join([str(n) for n in images["upsampling_factors"]])
        if (noise_level != 0):
            images_basename += "-noise="+str(noise_level)
        images["basename"] = images_basename

        if (structure_type == "box"):
            if (deformation_type == "translation"):
                Xmin = [0.1]+[0.2]*(n_dim-1)
                Xmax = [0.7]+[0.8]*(n_dim-1)
            elif (deformation_type in ("rotation", "compression", "shear")):
                Xmin = [0.2]*n_dim
                Xmax = [0.8]*n_dim
            else: assert (0)
            structure = {"type":"box", "Xmin":Xmin, "Xmax":Xmax}
        elif (structure_type == "disc"):
            if (deformation_type == "translation"):
                X0 = [0.4]+[0.5]*(n_dim-1)
            elif (deformation_type in ("rotation", "compression", "shear")):
                X0 = [0.5]*n_dim
            else: assert (0)
            R = 0.35
            structure = {"type":"disc", "X0":X0, "R":R}
        elif (structure_type == "ring"):
            if (deformation_type == "translation"):
                X0 = [0.4]+[0.5]*(n_dim-1)
            elif (deformation_type in ("rotation", "compression", "shear", "heart")):
                X0 = [0.5]*n_dim
            else: assert (0)
            Ri = 0.15
            Re = 0.35
            structure = {"type":"heart", "X0":X0, "Ri":Ri, "Re":Re}

        texture = {"type":texture_type, "s":0.5}

        if (noise_level == 0):
            noise = {"type":"no"}
        else:
            noise = {"type":"normal", "stdev":noise_level}

        if (deformation_type == "no"):
            deformation = {"type":"no"}
        elif (deformation_type == "translation"):
            deformation = {"type":"translation", "Dx":0.2, "Dy":0.}
        elif (deformation_type == "rotation"):
            deformation = {"type":"rotation", "Cx":0.5, "Cy":0.5, "Rz":90.}
        elif (deformation_type == "compression"):
            deformation = {"type":"homogeneous", "X0":0.5, "Y0":0.5, "Z0":0.5, "Exx":-0.20}
        elif (deformation_type == "shear"):
            deformation = {"type":"homogeneous", "X0":0.5, "Y0":0.5, "Z0":0.5, "Fxy":+0.20}
        elif (deformation_type == "heart"):
            deformation = {"type":"heart", "dRi":-0.10, "dRe":-0.05, "dTi":-math.pi/4, "dTe":-math.pi/8}

        evolution = {"type":"linear"}

        dwarp.generate_images(
            images=images,
            structure=structure,
            texture=texture,
            noise=noise,
            deformation=deformation,
            evolution=evolution,
            verbose=1)

shutil.rmtree(res_folder, ignore_errors=1)
