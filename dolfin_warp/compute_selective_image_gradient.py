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

from builtins import range

import glob
import numpy
import vtk
import math

import myVTKPythonLibrary as myvtk

################################################################################

def compute_selective_image_gradient(
        images_folder,
        images_basename,
        images_ext,
        images_field_name,
        masks_folder,
        masks_basename,
        masks_ext,
        mask_field_name,
        suffix="",
        verbose=0):

    images_filenames = glob.glob(images_folder+"/"+images_basename+"_[0-9]*"+"."+images_ext)
    masks_filenames = glob.glob(masks_folder+"/"+masks_basename+"_[0-9]*"+"."+masks_ext)

    assert (len(images_filenames) == len(masks_filenames)),\
        "Number of masks ("+str(len(masks_filenames))+") and images ("+str(len(images_filenames))+") are not the same. Aborting."

    images_zfill = len(images_filenames[0].rsplit("_",1)[-1].split(".",1)[0])
    masks_zfill = len(masks_filenames[0].rsplit("_",1)[-1].split(".",1)[0])

    for k_frame in range(len(images_filenames)):
        image_filename = images_folder+"/"+images_basename+"_"+str(k_frame).zfill(images_zfill)+"."+images_ext
        image = myvtk.readImage(
            filename=image_filename,
            verbose=verbose-1)
        n_points = image.GetNumberOfPoints()
        intensity_field = image.GetPointData().GetArray(images_field_name)
        intensity_tuple = numpy.empty(1)

        dimensionality = myvtk.getImageDimensionality(
            image=image,
            verbose=verbose-1)
        dimensions = image.GetDimensions()
        spacing = image.GetSpacing()

        mask_filename = masks_folder+"/"+masks_basename+"_"+str(k_frame).zfill(masks_zfill)+"."+masks_ext
        mask = myvtk.readImage(
            filename=mask_filename,
            verbose=verbose-1)
        mask_field = mask.GetPointData().GetArray(mask_field_name)
        mask_tuple = numpy.empty(1)

        grad = myvtk.createFloatArray(
            name="ImageGradient",
            n_components=dimensionality,
            n_tuples=n_points,
            init_to_zero=1,
            verbose=verbose-1)
        grad_tuple = numpy.empty(dimensionality)
        image.GetPointData().AddArray(grad)

        for k_point in range(n_points):
            mask_field.GetTuple(k_point, mask_tuple)

            #if inside the mesh (= mask value is 1)
            if (mask_tuple[0] == 1):

                intensity_field.GetTuple(k_point, intensity_tuple)

                k_point_z = int(math.floor(k_point / (dimensions[0] * dimensions[1])))
                k_point_y = int(math.floor((k_point - k_point_z*dimensions[0] * dimensions[1]) / dimensions[0]))
                k_point_x = k_point - k_point_z*dimensions[0] * dimensions[1] - k_point_y*dimensions[0]

                if (dimensionality>=1):
                    maskValue_x_prev = mask_field.GetTuple1(k_point-1)
                    maskValue_x_next = mask_field.GetTuple1(k_point+1)

                    if   ((maskValue_x_prev == 1) and (maskValue_x_next != 1)) or (k_point_x == dimensions[0]-1):
                        #backward difference in x-direction (second condition is for boundary nodes- means that point is on the right border)
                        #if the mask covers the whole image, second condition is executed, otherwise not.
                        #Because gradient is computed only for the points having the mask value 1.
                        grad_tuple[0] = (intensity_field.GetTuple1(k_point) - intensity_field.GetTuple1(k_point-1))/spacing[0]

                    elif ((maskValue_x_prev != 1) and (maskValue_x_next == 1)) or (k_point_x == 0):
                        #forward difference in x-direction (second condition is for boundary nodes - means that point is on the left border)
                        grad_tuple[0] = (intensity_field.GetTuple1(k_point+1) - intensity_field.GetTuple1(k_point))/spacing[0]
                    elif ((maskValue_x_prev == 1) and (maskValue_x_next == 1)):
                        #central difference
                        grad_tuple[0] = (intensity_field.GetTuple1(k_point+1) - intensity_field.GetTuple1(k_point-1))/(2*spacing[0])

                if (dimensionality>=2):
                    maskValue_y_prev = mask_field.GetTuple1(k_point-dimensions[1])
                    maskValue_y_next = mask_field.GetTuple1(k_point+dimensions[1])
                    if   ((maskValue_y_prev == 1) and (maskValue_y_next != 1)) or (k_point_y == dimensions[1]-1):
                        #backward difference in y-direction (second condition is for boundary nodes)
                        grad_tuple[1] = (intensity_field.GetTuple1(k_point) - intensity_field.GetTuple1(k_point-dimensions[1]))/spacing[1]
                    elif ((maskValue_y_prev != 1) and (maskValue_y_next == 1)) or (k_point_y == 0):
                        #forward difference in y-direction (second condition is for boundary nodes)
                        grad_tuple[1] = (intensity_field.GetTuple1(k_point+dimensions[1]) - intensity_field.GetTuple1(k_point))/spacing[1]
                    elif ((maskValue_y_prev == 1) and (maskValue_y_next == 1)):
                        #central difference
                        grad_tuple[1] = (intensity_field.GetTuple1(k_point+dimensions[1]) - intensity_field.GetTuple1(k_point-dimensions[1]))/(2*spacing[1])

                if (dimensionality>=3):
                    if (k_point-dimensions[0]*dimensions[1]) < 0:
                        maskValue_z_prev = 0
                    else:
                        maskValue_z_prev = mask_field.GetTuple1(k_point-dimensions[0]*dimensions[1])

                    if (k_point+dimensions[0]*dimensions[1]) >=  dimensions[2]:
                        maskValue_z_next = 0
                    else:
                        maskValue_z_next = mask_field.GetTuple1(k_point+dimensions[0]*dimensions[1])

                    if   ((maskValue_z_prev == 1) and (maskValue_z_next != 1)) or (k_point_z == dimensions[2]-1):
                        #backward difference in z-direction (second condition is for boundary nodes)
                        grad_tuple[2] = (intensity_field.GetTuple1(k_point) - intensity_field.GetTuple1(k_point-dimensions[0]*dimensions[1]))/spacing[2]
                    elif ((maskValue_z_prev != 1) and (maskValue_z_next == 1)) or (k_point_z == 0):
                        #forward difference in z-direction (second condition is for boundary nodes)
                        grad_tuple[2] = (intensity_field.GetTuple1(k_point+dimensions[0]*dimensions[1]) - intensity_field.GetTuple1(k_point))/spacing[2]
                    elif ((maskValue_z_prev == 1) and (maskValue_z_next == 1)):
                        #central difference
                        grad_tuple[2] = (intensity_field.GetTuple1(k_point+dimensions[0]*dimensions[1]) - intensity_field.GetTuple1(k_point-dimensions[0]*dimensions[1]))/(2*spacing[2])

                grad.SetTuple(k_point, grad_tuple)

        myvtk.writeImage(
            image=image,
            filename=images_folder+"/"+images_basename+("_"+suffix)*(suffix!="")+"_"+str(k_frame).zfill(images_zfill)+"."+images_ext,
            verbose=verbose-1)
