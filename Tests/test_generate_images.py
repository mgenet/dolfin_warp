#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2016-2022                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

import math
import sys

import myPythonLibrary as mypy
import dolfin_warp     as dwarp

################################################################################

images_folder = sys.argv[0][:-3]
test = mypy.Test(
    res_folder=images_folder,
    perform_tests=0,
    clean_after_tests=1)

n_dim_lst  = [ ]
n_dim_lst += [2]
n_dim_lst += [3]

upsampling_factor_lst  = [ ]
upsampling_factor_lst += [1]
upsampling_factor_lst += [2]

deformation_type_lst  = []
deformation_type_lst += ["translation"]
deformation_type_lst += ["rotation"]
deformation_type_lst += ["compression"]
deformation_type_lst += ["shear"]
deformation_type_lst += ["heart"]
deformation_type_lst += ["heart-notwist"]

texture_type_lst  = []
texture_type_lst += ["no"]
texture_type_lst += ["tagging"]
texture_type_lst += ["tagging-addComb"]
texture_type_lst += ["tagging-diffComb"]
texture_type_lst += ["tagging-signed"]
texture_type_lst += ["tagging-signed-addComb"]
texture_type_lst += ["tagging-signed-diffComb"]

noise_level_lst  = []
noise_level_lst += [0]
noise_level_lst += [0.1]

for n_dim             in n_dim_lst            :
 for upsampling_factor in upsampling_factor_lst:
  for deformation_type  in deformation_type_lst :
   for texture_type      in texture_type_lst     :
    for noise_level       in noise_level_lst      :

        images = {
            "n_dim":n_dim,
            "L":[1.]*n_dim,
            "n_voxels": [10]*n_dim,
            "upsampling_factors": [upsampling_factor]*n_dim,
            "T":1.,
            "n_frames":3,
            "data_type":"float",
            "folder":images_folder}

        if (n_dim == 2):
            images_basename = "square"
        elif (n_dim == 3):
            images_basename = "cube"
        images_basename += "-"+"x".join([str(n) for n in images["n_voxels"]])
        if (upsampling_factor != 1):
            images_basename += "-"+"x".join([str(n) for n in images["upsampling_factors"]])
        images_basename += "-"+deformation_type
        images_basename += "-"+texture_type
        if (noise_level != 0):
            images_basename += "-noise="+str(noise_level)
        images["basename"] = images_basename

        if (deformation_type in ("translation", "rotation", "compression", "shear")):
            structure = {
                "type":"box",
                "Xmin":[0.2]*n_dim,
                "Xmax":[0.8]*n_dim}
        elif (deformation_type in ("heart", "heart-notwist")):
            structure = {
                "type":"heart",
                "Ri":0.2,
                "Re":0.4}

        texture = {
            "type":texture_type,
            "s":0.5}

        if (noise_level == 0):
            noise = {
                "type":"no"}
        else:
            noise = {
                "type":"normal",
                "stdev":noise_level}

        if (deformation_type == "no"):
            deformation = {
                "type":"no"}
        elif (deformation_type == "translation"):
            deformation = {
                "type":"translation",
                "Dx":0.1,
                "Dy":0.0}
        elif (deformation_type == "rotation"):
            deformation = {
                "type":"rotation",
                "Cx":0.5,
                "Cy":0.5,
                "Rz":90.}
        elif (deformation_type == "compression"):
            deformation = {
                "type":"homogeneous",
                "X0":0.5,
                "Y0":0.5,
                "Z0":0.5,
                "Exx":-0.20}
        elif (deformation_type == "shear"):
            deformation = {
                "type":"homogeneous",
                "X0":0.5,
                "Y0":0.5,
                "Z0":0.5,
                "Fxy":+0.20}
        elif (deformation_type == "heart"):
            deformation = {
                "type":"heart",
                "dRi":-0.10,
                "dRe":-0.05,
                "dTi":-math.pi/4,
                "dTe":-math.pi/8}
        elif (deformation_type == "heart-notwist"):
            deformation = {
                "type":"heart",
                "dRi":-0.10,
                "dRe":-0.05,
                "dTi":0.,
                "dTe":0.}

        evolution = {
            "type":"linear"}

        dwarp.generate_images(
            images=images,
            structure=structure,
            texture=texture,
            noise=noise,
            deformation=deformation,
            evolution=evolution,
            verbose=1)
