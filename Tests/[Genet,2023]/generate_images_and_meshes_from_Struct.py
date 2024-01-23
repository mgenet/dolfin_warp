#coding=utf8

################################################################################

import dolfin
import math
import typing

import dolfin_mech as dmech
import dolfin_warp as dwarp

################################################################################

def generate_images_and_meshes_from_Struct(
        n_dim            : int                        ,
        n_voxels         : int                        ,
        structure_type   : str                        ,
        deformation_type : str                        ,
        texture_type     : str                        ,
        noise_level      : float                      ,
        k_run            : typing.Optional[int] = None, # MG20220815: This can be written "int | None" starting with python 3.10
        mesh_size        : float                = 0.1 ,
        generate_images  : bool                 = True,
        compute_meshes   : bool                 = True):

    images = {
        "n_dim":n_dim,
        "L":[1.]*n_dim,
        "n_voxels":[n_voxels]*n_dim,
        "upsampling_factors":[1]*n_dim,
        "T":1.,
        "n_frames":21,
        "data_type":"float"}

    images["folder"] = "generate_images"

    images["basename"]  = structure_type
    images["basename"] += "-"+deformation_type
    images["basename"] += "-"+texture_type
    images["basename"] += "-noise="+str(noise_level)
    if (k_run is not None):
        images["basename"] += "-run="+str(k_run).zfill(2)
    images["ext"] = "vti"

    if (structure_type == "square"):
        if (deformation_type == "translation"):
            Xmin = [0.1,0.2]
            Xmax = [0.7,0.8]
        elif (deformation_type in ("no", "rotation", "compression", "shear")):
            Xmin = [0.2,0.2]
            Xmax = [0.8,0.8]
        else: assert (0)
        structure = {"type":"box", "Xmin":Xmin, "Xmax":Xmax}
    elif (structure_type == "disc"):
        if (deformation_type == "translation"):
            X0 = [0.4,0.5]
        elif (deformation_type in ("no", "rotation", "compression", "shear")):
            X0 = [0.5,0.5]
        else: assert (0)
        R = 0.35
        structure = {"type":"disc", "X0":X0, "R":R}
    elif (structure_type == "ring"):
        if (deformation_type == "translation"):
            X0 = [0.4,0.5]
        elif (deformation_type in ("no", "rotation", "compression", "shear", "inflate", "twist")):
            X0 = [0.5,0.5]
        else: assert (0)
        Ri = 0.15
        Re = 0.35
        structure = {"type":"heart", "X0":X0, "Ri":Ri, "Re":Re}

    if (texture_type == "tagging"):
        texture = {"type":"tagging", "s":0.1}

    if (noise_level == 0):
        noise = {"type":"no"}
    else:
        noise = {"type":"normal", "stdev":noise_level}

    if (deformation_type == "no"):
        deformation = {"type":"no"}
    elif (deformation_type == "translation"):
        deformation = {"type":"translation", "Dx":0.2, "Dy":0.}
    elif (deformation_type == "rotation"):
        deformation = {"type":"rotation", "Cx":0.5, "Cy":0.5, "Rz":45.} # 90 be require more time steps?
    elif (deformation_type == "compression"):
        deformation = {"type":"homogeneous", "X0":0.5, "Y0":0.5, "Exx":-0.20}
    elif (deformation_type == "shear"):
        deformation = {"type":"homogeneous", "X0":0.5, "Y0":0.5, "Fxy":+0.20}
    elif (deformation_type == "inflate"):
        deformation = {"type":"heart", "dRi":-0.10, "dRe":0., "dTi":0., "dTe":0.}
    elif (deformation_type == "twist"):
        deformation = {"type":"heart", "dRi":0., "dRe":0., "dTi":-math.pi/4, "dTe":0.}

    evolution = {"type":"linear"}

    if (generate_images):
        dwarp.generate_images(
            images           = images,
            structure        = structure,
            texture          = texture,
            noise            = noise,
            deformation      = deformation,
            evolution        = evolution,
            keep_temp_images = 0,
            verbose          = 1)

    if (compute_meshes):
        if (structure_type == "square"):
            n_cells = int((0.8-0.2)/mesh_size)
            if (n_dim == 2):
                mesh = dolfin.RectangleMesh(
                    dolfin.Point(Xmin),
                    dolfin.Point(Xmax),
                    n_cells, n_cells,
                    "crossed")
            elif (n_dim == 3):
                mesh = dolfin.BoxMesh(
                    dolfin.Point(Xmin),
                    dolfin.Point(Xmax),
                    n_cells, n_cells, n_cells)
        elif (structure_type == "disc"):
            # import mshr
            # geometry = mshr.Circle(dolfin.Point(X0[0], X0[1]), R)
            # mesh = mshr.generate_mesh(geometry, r/mesh_size)
            mesh, _, _, _, _, _, _, _ = dmech.run_Disc_Mesh(params={"X0":X0[0], "Y0":X0[1], "R":R, "l":mesh_size})
        elif (structure_type == "ring"):
            # import mshr
            # geometry = mshr.Circle(dolfin.Point(X0[0], X0[1]), Re)\
            #          - mshr.Circle(dolfin.Point(X0[0], X0[1]), Ri)
            # mesh = mshr.generate_mesh(geometry, (Re-Ri)/mesh_size)
            mesh, _, _, _, _, _, _, _, _ = dmech.run_HeartSlice_Mesh(params={"X0":X0[0], "Y0":X0[1], "Ri":Ri, "Re":Re, "l":mesh_size})

        working_folder = images["folder"]
        
        working_basename  = structure_type
        working_basename += "-"+deformation_type
        working_basename += "-h="+str(mesh_size)

        dolfin.File(working_folder+"/"+working_basename+".xml") << mesh

        dwarp.compute_warped_mesh(
            working_folder   = working_folder,
            working_basename = working_basename,
            images           = images,
            structure        = structure,
            deformation      = deformation,
            evolution        = evolution,
            mesh             = mesh,
            mesh_ext         = "vtu",
            verbose          = 1)

########################################################################

if (__name__ == "__main__"):
    import fire
    fire.Fire(generate_images_and_meshes_from_Struct)

    # import argparse
    # from distutils.util import strtobool
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--n_dim"           , type=int  , choices=[2,3]                                              )
    # parser.add_argument("--n_voxels"        , type=int                                                               )
    # parser.add_argument("--deformation_type", type=str  , choices=["translation", "rotation", "compression", "shear"])
    # parser.add_argument("--texture_type"    , type=str  , choices=["no", "tagging"]                                  )
    # parser.add_argument("--noise_level"     , type=float                                                             )
    # parser.add_argument("--k_run"           , type=int                                                 , default=None)
    # parser.add_argument("--generate_images" , type=lambda x: bool(strtobool(x))                        , default=True) # MG20220901: Watch out! All non empty strings evaluate to True!
    # parser.add_argument("--compute_meshes"  , type=lambda x: bool(strtobool(x))                        , default=True) # MG20220901: Watch out! All non empty strings evaluate to True!
    # args = parser.parse_args()

    # generate_images_and_meshes_from_Struct(
    #     n_dim            = args.n_dim,
    #     n_voxels         = args.n_voxels,
    #     deformation_type = args.deformation_type,
    #     texture_type     = args.texture_type,
    #     noise_level      = args.noise_level,
    #     k_run            = args.k_run,
    #     generate_images  = args.generate_images,
    #     compute_meshes   = args.compute_meshes)
