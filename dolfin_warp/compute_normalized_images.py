#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2016-2022                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

import numpy
import vtk

import myPythonLibrary as mypy

import dolfin_warp as dwarp

################################################################################

def compute_normalized_images(
        images_folder,
        images_basename,
        images_datatype,
        images_ext="vti",
        suffix=None,
        verbose=0):

    mypy.my_print(verbose, "*** compute_normalized_images ***")

    images_series = dwarp.ImagesSeries(
        folder=images_folder,
        basename=images_basename,
        ext=images_ext)
    image = images_series.get_image(k_frame=0)
    image_npoints = image.GetNumberOfPoints()

    if   (images_ext == "vtk"):
        reader = vtk.vtkImageReader()
        writer = vtk.vtkImageWriter()
    elif (images_ext == "vti"):
        reader = vtk.vtkXMLImageDataReader()
        writer = vtk.vtkXMLImageDataWriter()
    else:
        assert 0, "\"ext\" must be .vtk or .vti. Aborting."

    global_min = float("+Inf")
    global_max = float("-Inf")
    for k_frame in range(images_series.n_frames):
        reader.SetFileName(images_series.get_image_filename(k_frame=k_frame))
        reader.Update()

        image_scalars = reader.GetOutput().GetPointData().GetScalars()
        I = numpy.empty(image_scalars.GetNumberOfComponents())
        for k_point in range(image_npoints):
            image_scalars.GetTuple(k_point, I)
            global_min = min(global_min, I[0])
            global_max = max(global_max, I[0])
    mypy.my_print(verbose, "global_min = "+str(global_min))
    mypy.my_print(verbose, "global_max = "+str(global_max))

    shifter = vtk.vtkImageShiftScale()
    shifter.SetInputConnection(reader.GetOutputPort())
    shifter.SetShift(-global_min)
    if   (images_datatype in ("unsigned char", "uint8")):
        shifter.SetScale(float(2**8-1)/(global_max-global_min))
        shifter.SetOutputScalarTypeToUnsignedChar()
    elif (images_datatype in ("unsigned short", "uint16")):
        shifter.SetScale(float(2**16-1)/(global_max-global_min))
        shifter.SetOutputScalarTypeToUnsignedShort()
    elif (images_datatype in ("unsigned int", "uint32")):
        shifter.SetScale(float(2**32-1)/(global_max-global_min))
        shifter.SetOutputScalarTypeToUnsignedInt()
    elif (images_datatype in ("unsigned long", "uint64")):
        shifter.SetScale(float(2**64-1)/(global_max-global_min))
        shifter.SetOutputScalarTypeToUnsignedLong()
    elif (images_datatype in ("unsigned float", "ufloat", "float")):
        shifter.SetScale(1./(global_max-global_min))
        shifter.SetOutputScalarTypeToFloat()

    writer.SetInputConnection(shifter.GetOutputPort())

    for k_frame in range(images_series.n_frames):
        mypy.my_print(verbose, "k_frame = "+str(k_frame))

        reader.SetFileName(images_series.get_image_filename(k_frame=k_frame               ))
        writer.SetFileName(images_series.get_image_filename(k_frame=k_frame, suffix=suffix))
        writer.Write()
