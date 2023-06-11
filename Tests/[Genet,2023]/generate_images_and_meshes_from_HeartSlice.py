#coding=utf8

################################################################################

import math
import os

import myVTKPythonLibrary as myvtk
import dolfin_mech        as dmech
import dolfin_warp        as dwarp

################################################################################

def generate_images_and_meshes_from_HeartSlice(
        n_voxels         : int         ,
        deformation_type : str         ,
        texture_type     : str         ,
        noise_level      : float       ,
        k_run            : int   = None,
        run_model        : bool  = True,
        generate_images  : bool  = True,
        mesh_size        : float = None):

    images_folder = "generate_images"
    if not os.path.exists(images_folder):
        os.mkdir(images_folder)

    working_basename  = "heart"
    working_basename += "-"+deformation_type
    if (mesh_size is not None):
        working_basename += "-h="+str(mesh_size)

    if (mesh_size is None):
        mesh_size = 1./n_voxels
    mesh_params = {"X0":0.5, "Y0":0.5, "Ri":0.2, "Re":0.4, "l":mesh_size, "mesh_filebasename":images_folder+"/"+working_basename+"-mesh"}
    if   ("-coarse"    in deformation_type):
        mesh_params["l"] = 0.05
        mesh_params["mesh_filebasename"] = images_folder+"/"+working_basename+"-mesh-coarse"
    elif ("-fine"      in deformation_type):
        mesh_params["l"] = 0.01
        mesh_params["mesh_filebasename"] = images_folder+"/"+working_basename+"-mesh-fine"
    elif ("-ultrafine" in deformation_type):
        mesh_params["l"] = 0.005
        mesh_params["mesh_filebasename"] = images_folder+"/"+working_basename+"-mesh-ultrafine"

    load_params = {"type":"disp", "dRi":-0.10, "dRe":-0.05, "dTi":-math.pi/4, "dTe":-math.pi/8}
    if ("nocontract" in deformation_type):
        load_params["dRi"] = 0.
        load_params["dRe"] = 0.
    if ("notwist" in deformation_type):
        load_params["dTi"] = 0.
        load_params["dTe"] = 0.

    # print (run_model)
    if (run_model):
        dmech.HeartSlice_Hyperelasticity(
            incomp                                 = 0,
            mesh_params                            = mesh_params,
            mat_params                             = {"model":"CGNH", "parameters":{"E":1., "nu":0.3}},
            step_params                            = {"dt_ini":1/20},
            load_params                            = load_params,
            res_basename                           = images_folder+"/"+working_basename,
            write_vtus_with_preserved_connectivity = True,
            verbose                                = 1)

    if (generate_images):
        ref_image = myvtk.createImageFromSizeAndRes(
            dim  = 2       ,
            size = 1.      ,
            res  = n_voxels,
            up   = 1       )

        s = [0.1]*2
        ref_image_model = lambda X:math.sqrt(abs(math.sin(math.pi*X[0]/s[0]))
                                           * abs(math.sin(math.pi*X[1]/s[1])))

        if (noise_level == 0):
            noise_params = {"type":"no"}
        else:
            noise_params = {"type":"normal", "stdev":noise_level}

        dwarp.compute_warped_images(
            working_folder                  = images_folder,
            working_basename                = "heart-"+deformation_type,
            working_ext                     = "vtu",
            working_displacement_field_name = "U",
            ref_image                       = ref_image,
            ref_frame                       = 0,
            ref_image_model                 = ref_image_model,
            noise_params                    = noise_params,
            suffix                          = texture_type+"-noise="+str(noise_level)+(k_run is not None)*("-run="+str(k_run).zfill(2)),
            print_warped_mesh               = 0,
            verbose                         = 0)

########################################################################

if (__name__ == "__main__"):
    import fire
    fire.Fire(generate_images_and_meshes_from_HeartSlice)

    # import argparse
    # from distutils.util import strtobool
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--n_dim"                , type=int  , choices=[2]        )
    # parser.add_argument("--n_voxels"             , type=int                       )
    # parser.add_argument("--deformation_type"     , type=str                       )
    # parser.add_argument("--texture_type"         , type=str  , choices=["tagging"])
    # parser.add_argument("--noise_level"          , type=float                     )
    # parser.add_argument("--k_run"                , type=int                         , default=None)
    # parser.add_argument("--run_model"            , type=lambda x: bool(strtobool(x)), default=True) # MG20220901: Watch out! All non empty strings evaluate to True!
    # parser.add_argument("--generate_images"      , type=lambda x: bool(strtobool(x)), default=True) # MG20220901: Watch out! All non empty strings evaluate to True!

    # args = parser.parse_args()
    # # print (args.run_model)

    # generate_images_and_meshes_from_HeartSlice(
    #     n_dim            = args.n_dim,
    #     n_voxels         = args.n_voxels,
    #     deformation_type = args.deformation_type,
    #     texture_type     = args.texture_type,
    #     noise_level      = args.noise_level,
    #     k_run            = args.k_run,
    #     run_model        = args.run_model,
    #     generate_images  = args.generate_images)
