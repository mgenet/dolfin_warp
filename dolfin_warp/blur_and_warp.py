import dolfin

import dolfin_warp as dwarp

import numpy as np

################################################################################




def gaussian_windowing(
        working_folder              : str,
        image_name                  : str, 
        images_folder               : str,
        attenuation_factor          : float         = 2,                                # attenuation coef of the cut-off frequency
        image_ext                   : str           = '.vti',
        suffix                      : str           = "_downsampled=",
        verbose                     : bool          = False,
        ):

    import vtk

    import glob
    images_list = glob.glob(images_folder+"/"+image_name+"_??"+image_ext)


    suffix+=str(attenuation_factor)
    # Start by getting voxel_size

    for file in images_list:
        frame = file[-len(image_ext)-2:-len(image_ext)]
        print(f"blurring frame {frame} in progress")
        # file = working_folder+image_name+image_ext
        reader = vtk.vtkXMLImageDataReader()
        reader.SetFileName(file)
        reader.Update()

        image = reader.GetOutput()
        voxel_sizes = image.GetSpacing()                                                    # (dx, dy, dz)
        dimensions = image.GetDimensions()                                                  # (nx, ny, nz)

        #Compute the standard deviation associated with the attenuation factor
        sigma = np.sqrt(-(np.log(1/attenuation_factor))/(2*np.pi*np.array(voxel_sizes))**2)
        radius = np.ceil(6 * sigma)
        radius[radius % 2 == 0] += 1                                                        # Add 1 to even numbers to make them odd
        if verbose:
            print(f"* dimensions are {dimensions}")
            print(f"* voxel sizes are {voxel_sizes}")
            print(f"* standard deviation is {sigma}")
            print(f"* radius is {radius}")

        gaussian = vtk.vtkImageGaussianSmooth()
        gaussian.SetInputConnection(reader.GetOutputPort())
        gaussian.SetStandardDeviations(sigma)                                               # Standard deviations for the Gaussian in X, Y, Z
        gaussian.SetRadiusFactors(radius)                                                   # Radius factors 
        gaussian.Update()

        writer = vtk.vtkXMLImageDataWriter()
        writer.SetFileName(images_folder+"/"+image_name+suffix+"_"+frame+image_ext)
        writer.SetInputConnection(gaussian.GetOutputPort())
        writer.Write()
        print("Done downsampling. "+file)








def blur_and_warp(
        working_folder                              : str,
        working_basename                            : str,
        images_folder                               : str,
        images_basename                             : str,
        attenuation_factors             : list        = None                                   , # List of attenuation coefficients at the cut-off frequency
        images_grad_basename                        : str         = None                                ,
        images_ext                                  : str         = "vti"                               , # vti, vtk
        images_n_frames                             : int         = None                                ,
        images_ref_frame                            : int         = 0                                   ,
        images_quadrature                           : int         = None                                ,
        images_quadrature_from                      : str         = "points_count"                      , # points_count, integral
        images_static_scaling                       : bool        = False                               ,
        images_dynamic_scaling                      : bool        = False                               ,
        images_char_func                            : bool        = True                                ,
        images_is_cone                              : bool        = False                               ,
        mesh                                        : dolfin.Mesh = None                                ,
        mesh_folder                                 : str         = None                                ,
        mesh_basename                               : str         = None                                ,
        mesh_degree                                 : int         = 1                                   ,
        kinematics_type                             : str         = "full"                              , # full, reduced
        reduced_kinematics_model                    : str         = "translation+rotation+scaling+shear", # translation, rotation, scaling, shear, translation+rotation+scaling+shear, etc.
        regul_type                                  : str         = "continuous-equilibrated"           , # continuous-linear-equilibrated, continuous-linear-elastic, continuous-equilibrated, continuous-elastic, continuous-hyperelastic, discrete-simple-equilibrated, discrete-simple-elastic, discrete-linear-equilibrated, discrete-linear-tractions, discrete-linear-tractions-normal, discrete-linear-tractions-tangential, discrete-linear-tractions-normal-tangential, discrete-equilibrated, discrete-tractions, discrete-tractions-normal, discrete-tractions-tangential, discrete-tractions-normal-tangential
        regul_types                                 : list        = None                                ,
        regul_model                                 : str         = "ogdenciarletgeymonatneohookean"    , # hooke, kirchhoff, ogdenciarletgeymonatneohookean, ogdenciarletgeymonatneohookeanmooneyrivlin
        regul_models                                : list        = None                                ,
        regul_quadrature                            : int         = None                                ,
        regul_level                                 : float       = 0.                                  ,
        regul_levels                                : list        = None                                ,
        regul_poisson                               : float       = 0.                                  ,
        regul_b                                     : float       = None                                ,
        regul_volume_subdomain_data                               = None                                ,
        regul_volume_subdomain_id                                 = None                                ,
        regul_surface_subdomain_data                              = None                                ,
        regul_surface_subdomain_id                                = None                                ,
        relax_type                                  : str         = None                                , # None, constant, aitken, backtracking, gss
        relax_backtracking_factor                   : float       = None                                ,
        relax_tol                                   : float       = None                                ,
        relax_n_iter_max                            : int         = None                                ,
        relax_must_advance                          : bool        = None                                ,
        save_reduced_disp                           : bool        = False                               ,
        normalize_energies                          : bool        = False                               ,
        initialize_reduced_U_from_file              : bool        = False                               ,
        initialize_reduced_U_filename               : str         = None                                ,
        initialize_U_from_file                      : bool        = False                               ,
        initialize_U_folder                         : str         = None                                ,
        initialize_U_basename                       : str         = None                                ,
        initialize_U_ext                            : str         = "vtu"                               ,
        initialize_U_array_name                     : str         = "displacement"                      ,
        initialize_U_method                         : str         = "dofs_transfer"                     , # dofs_transfer, interpolation, projection
        register_ref_frame                          : bool        = False                               ,
        iteration_mode                              : str         = "normal"                            , # normal, loop
        gimic                                       : bool        = False                               ,
        gimic_texture                               : str         = "no"                                ,
        gimic_resample                              : int         = 1                                   ,
        nonlinearsolver                             : str         = "newton"                            , # None, newton, CMA
        tol_res_rel                                 : float       = None                                ,
        tol_dU                                      : float       = None                                ,
        tol_dU_rel                                  : float       = None                                ,
        n_iter_max                                  : int         = 100                                 ,
        continue_after_fail                         : bool        = False                               ,
        write_qois_limited_precision                : bool        = False                               ,
        write_VTU_files                             : bool        = True                                ,
        write_VTU_files_with_preserved_connectivity : bool        = False                               ,
        write_XML_files                             : bool        = False                               ,
        print_iterations                            : bool        = False                               ,
        silent                                      : bool        = False                               ):

        
        assert kinematics_type=="reduced", "blur_and_warp only defined for reduced kinematics. Aborting"


        initialize_reduced_U_filename_0 = initialize_reduced_U_filename                             # Save intial reduced dispalcement used for the first initialisation
        

        # Few integration point for blurry images with small spatial variations
        images_quadrature_progressive = np.linspace(1, images_quadrature, len(attenuation_factors))  # Generate m evenly spaced values
        images_quadrature_progressive = np.round(images_quadrature_progressive).astype(int)

        print(f"image quadrature list: {images_quadrature_progressive}") #DEBUG
        print(f"image quadrature init: {images_quadrature}") #DEBUG

        attenuation_factors.sort(reverse=True)                                                                   # 

        if attenuation_factors[-1] != 1:
            attenuation_factors.append(1)

        for i in range(len(attenuation_factors)):
            attenuation_factor = attenuation_factors[i]
            if attenuation_factor != 1:
                gaussian_windowing(
                    working_folder                              = working_folder,
                    images_folder                               = images_folder, 
                    image_name                                  = images_basename,
                    attenuation_factor                          = attenuation_factor,   
                    verbose                                     = True
                    )
                working_basename_current        = working_basename+"_downsampled="+str(attenuation_factor)
                images_basename_blur_factor     = images_basename + "_downsampled="+str(attenuation_factor)

            else:
                working_basename_current        = working_basename+"_sharp"
                images_basename_blur_factor     = images_basename 


            if i >=1 :
                import os
                initialize_reduced_U_filename = working_folder+"/"+working_basename+"_downsampled="+str(attenuation_factors[i-1])+"_reduced_kinematics.dat"
                assert os.path.isfile(initialize_reduced_U_filename), "initialize_reduced_U_filename not found. Aborting"
                    
            else:
                initialize_reduced_U_filename = initialize_reduced_U_filename_0

            dwarp.warp(
                working_folder                  = working_folder                ,
                working_basename                = working_basename_current      ,
                images_folder                   = images_folder                 ,
                images_basename                 = images_basename_blur_factor   ,
                # images_quadrature               = images_quadrature_progressive[i], # Issue with progressive quadrature image, DEBUG
                images_quadrature               = images_quadrature             ,
                mesh                            = mesh                          ,
                kinematics_type                 = kinematics_type               ,
                reduced_kinematics_model        = reduced_kinematics_model      ,
                normalize_energies              = normalize_energies            ,
                relax_type                      = relax_type                    ,
                tol_dU                          = tol_dU                        ,
                write_qois_limited_precision    = write_qois_limited_precision  , 
                initialize_reduced_U_from_file  = True                          ,
                initialize_reduced_U_filename   = initialize_reduced_U_filename ,
                print_iterations                = print_iterations              , 
                save_reduced_disp               = save_reduced_disp             ,
                n_iter_max                      = n_iter_max                    ,
                continue_after_fail             = continue_after_fail
                        )

