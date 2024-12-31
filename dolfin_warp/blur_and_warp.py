import dolfin

import dolfin_warp as dwarp

################################################################################




def gaussian_windowing(
        working_folder              : str,
        image_name                  : str, 
        images_folder               : str,
        attenuation_factor          : float         = 2,                                # attenuation coef of the cut-off frequency
        image_ext                   : str           = '.vti',
        suffix                      : str           = "_downsampled=",
        verbose                     : bool          = False
        ):

    import vtk
    suffix+=str(attenuation_factor)
    working_folder+="/"+images_folder+"/"
    # Start by getting voxel_size
    file = working_folder+image_name+image_ext
    reader = vtk.vtkXMLImageDataReader()
    reader.SetFileName(file)
    reader.Update()

    image = reader.GetOutput()
    voxel_sizes = image.GetSpacing()                                                    # (dx, dy, dz)
    dimensions = image.GetDimensions()                                                  # (nx, ny, nz)

    #Compute the standard deviation associated with the attenuation factor
    import numpy as np
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
    writer.SetFileName(working_folder+image_name+suffix+image_ext)
    writer.SetInputConnection(gaussian.GetOutputPort())
    writer.Write()
    print("Done downsampling. "+image_name)








def blur_and_warp(
        working_folder                  : str,
        working_basename                : str,
        images_folder                   : str,
        images_basename                 : str,
        attenuation_factors             : list        = None                                   , # List of attenuation coefficients at the cut-off frequency
        images_quadrature               : int         = None                                   ,
        images_quadrature_from          : str         = "points_count"                         , # points_count, integral
        mesh                            : dolfin.Mesh = None                                   ,
        kinematics_type                 : str         = "reduced"                              ,
        reduced_kinematics_model        : str         = "translation+rotation+scaling+shear"   ,
        refinement_levels               : list        = [0]                                    ,
        meshes                          : list        = None                                   ,
        mesh_folder                     : str         = None                                   ,
        mesh_basenames                  : list        = None                                   ,
        regul_type                      : str         = "continuous-equilibrated"              , # continuous-equilibrated, continuous-elastic, continuous-hyperelastic, discrete-linear-equilibrated, discrete-linear-elastic, discrete-equilibrated, discrete-tractions, discrete-tractions-normal, discrete-tractions-tangential, discrete-tractions-normal-tangential
        regul_types                     : list        = None                                   ,
        regul_model                     : str         = "ogdenciarletgeymonatneohookean"       , # hooke, kirchhoff, ogdenciarletgeymonatneohookean, ogdenciarletgeymonatneohookeanmooneyrivlin
        regul_models                    : list        = None                                   ,
        regul_level                     : float       = 0.                                     ,
        regul_levels                    : list        = None                                   ,
        regul_poisson                   : float       = 0.                                     ,
        regul_b                         : float       = None                                   ,
        regul_volume_subdomain_data                   = None                                   ,
        regul_volume_subdomain_id                     = None                                   ,
        regul_surface_subdomain_data                  = None                                   ,
        regul_surface_subdomain_id                    = None                                   ,
        relax_type                      : str         = "backtracking"                         , # constant, aitken, backtracking, gss
        relax_tol                       : float       = None                                   ,
        relax_n_iter_max                : int         = None                                   ,
        normalize_energies              : bool        = False                                  ,
        tol_dU                          : float       = None                                   ,
        n_iter_max                      : int         = 100                                    ,
        continue_after_fail             : bool        = False                                  ,
        write_qois_limited_precision    : bool        = False                                  ,
        print_iterations                : bool        = False                                  ,
        silent                          : bool        = False                                  ,
        initialize_reduced_U_from_file  : bool        = True                                   ,
        initialize_reduced_U_filename   : str         = None                                   ):

        
        assert kinematics_type=="reduced", "blur_and_warp only defined for reduced kinematics. Aborting"


        initialize_reduced_U_filename_0 = initialize_reduced_U_filename                             # Save intial reduced dispalcement used for the first initialisation
        
        if images is None:
            images = []

        # Few integration point for blurry images with small spatial variations
        images_quadrature_progressive = np.linspace(1, images_quadrature, len(attenuation_factors))  # Generate m evenly spaced values
        images_quadrature_progressive = np.round(images_quadrature_progressive).astype(int)

        attenuation_factors.sort(reverse=True)                                                                   # 

        if attenuation_factors[-1] != 1:
            attenuation_factors.apend(1)

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
                working_basename_current = working_basename+"_downsampled="+str(attenuation_factor)
            else:
                working_basename_current = working_basename+"_sharp"


            images_basename_blur_factor     = images_basename + "_downsampled="+str(attenuation_factor)
            if i >=1 :
                import os
                initialize_reduced_U_filename = working_basename+"_downsampled="+str(attenuation_factors[i-1])+"_reduced_kinematics.dat"
                assert os.path.isfile(initialize_reduced_U_filename), "initialize_reduced_U_filename not found. Aborting"
                    
            else:
                initialize_reduced_U_filename = initialize_reduced_U_filename_0
            
            dwarp.warp(
                working_folder                  = working_folder,
                working_basename                = working_basename_current,
                images_folder                   = images_folder,
                images_basename                 = images_basename_blur_factor,
                images_quadrature               = images_quadrature_progressive[i],
                mesh                            = mesh,
                kinematics_type                 = kinematics_type,
                reduced_kinematics_model        = reduced_kinematics_model,
                normalize_energies              = normalize_energies,
                relax_type                      = relax_type,
                tol_dU                          = tol_dU,
                write_qois_limited_precision    = write_qois_limited_precision, 
                initialize_reduced_U_from_file  = True,
                initialize_reduced_U_filename   = initialize_reduced_U_filename
                )

