#coding=utf8

################################################################################
###                                                                          ###
### Created by Ezgi Berberoğlu, 2017-2021                                    ###
###                                                                          ###
### Swiss Federal Institute of Technology (ETH), Zurich, Switzerland         ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
### And Martin Genet, 2016-2022                                              ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

import numpy

import myVTKPythonLibrary as myvtk

import dolfin_warp as dwarp

################################################################################

def compute_equalized_images(
        images_folder,
        images_basename,
        images_ext="vti",
        array_name="scalars",
        suffix=None,
        verbose=0):

    images_series = dwarp.ImagesSeries(
        folder=images_folder,
        basename=images_basename,
        ext=images_ext)

    for k_frame in range(images_series.n_frames):
        image = images_series.get_image(k_frame=k_frame)
        scalars = image.GetPointData().GetArray(array_name)
        n_points = scalars.GetNumberOfTuples()

        if (scalars.GetDataType() == 3):
            n_intensities = numpy.iinfo('uint8').max+1
        else:
            assert (0), "Not implemented. Aborting."
        intensity_count = numpy.zeros(n_intensities)

        for k_point in range(n_points):
            intensity_count[int(scalars.GetTuple1(k_point))] += 1

        intensity_count = intensity_count/n_points
        for k_intensity in range(1,n_intensities):
            intensity_count[k_intensity] += intensity_count[k_intensity-1]

        intensity_count = intensity_count*(n_intensities-1)
        for k_point in range(n_points):
            scalars.SetTuple1(k_point, int(round(intensity_count[int(scalars.GetTuple1(k_point))])))

        myvtk.writeImage(
            image=image,
            filename=images_series.get_image_filename(k_frame=k_frame, suffix=suffix))
