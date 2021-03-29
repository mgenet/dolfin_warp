#coding=utf8

################################################################################
###                                                                          ###
### Created by Ezgi Berberoğlu, 2018-2021                                    ###
###                                                                          ###
### Swiss Federal Institute of Technology (ETH), Zurich, Switzerland         ###
###                                                                          ###
### And Martin Genet, 2016-2021                                              ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

from builtins import range

import glob
import numpy

import myVTKPythonLibrary as myvtk

################################################################################

def compute_equalized_images(
        images_folder,
        images_basename,
        images_ext="vti",
        array_name="scalars",
        suffix="",
        verbose=0):

    image_filenames = glob.glob(images_folder+"/"+images_basename+"_[0-9]*"+"."+images_ext)
    n_frames = len(image_filenames)
    images_zfill = len(image_filenames[0].rsplit("_",1)[-1].split(".",1)[0])

    for k_frame in range(n_frames):
        image_filename = images_folder+"/"+images_basename+"_"+str(k_frame).zfill(images_zfill)+"."+images_ext
        image = myvtk.readImage(
            filename=image_filename,
            verbose=verbose-1)

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
            filename=images_folder+"/"+images_basename+("_"+suffix)*(suffix!="")+"_"+str(k_frame).zfill(images_zfill)+"."+images_ext)
