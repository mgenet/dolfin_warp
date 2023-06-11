#coding=utf8

################################################################################

import os

import myPythonLibrary as mypy
import dolfin_warp     as dwarp

################################################################################

res_folder = "plot_disp_error_vs_regul_strength"

test = mypy.Test(
    res_folder=res_folder,
    perform_tests=1,
    stop_at_failure=1,
    clean_after_tests=1,
    qois_suffix="")

################################################################################

images_folder = "generate_images"

n_voxels = 100

deformation_type_lst  = [                  ]
deformation_type_lst += ["contractandtwist"]

texture_type_lst  = [         ]
texture_type_lst += ["tagging"]

noise_level_lst  = [   ]
noise_level_lst += [0.0]
noise_level_lst += [0.1]
noise_level_lst += [0.2]
noise_level_lst += [0.3]

n_runs_for_noisy_images = 10

working_folder  = "run_warp"

mesh_size_lst  = [        ]
mesh_size_lst += [0.1     ]
# mesh_size_lst += [0.1/2**1] # 0.05
# mesh_size_lst += [0.1/2**2] # 0.025
# mesh_size_lst += [0.1/2**3] # 0.0125
# mesh_size_lst += [0.1/2**4] # 0.00625

regul_type_lst  = []
regul_type_lst += ["continuous-linear-elastic"                               ]
regul_type_lst += ["continuous-linear-equilibrated"                          ]
regul_type_lst += ["continuous-elastic"                                      ]
regul_type_lst += ["continuous-equilibrated"                                 ]
regul_type_lst += ["discrete-simple-elastic"                                 ]
regul_type_lst += ["discrete-simple-equilibrated"                            ]
regul_type_lst += ["discrete-linear-equilibrated"                            ]
regul_type_lst += ["discrete-linear-equilibrated-tractions-normal"           ]
regul_type_lst += ["discrete-linear-equilibrated-tractions-tangential"       ]
regul_type_lst += ["discrete-linear-equilibrated-tractions-normal-tangential"]
regul_type_lst += ["discrete-equilibrated"                                   ]
regul_type_lst += ["discrete-equilibrated-tractions-normal"                  ]
regul_type_lst += ["discrete-equilibrated-tractions-tangential"              ]
regul_type_lst += ["discrete-equilibrated-tractions-normal-tangential"       ]

regul_level_lst  = [        ]
regul_level_lst += [0.99    ]
regul_level_lst += [0.1*2**3] # 0.8
regul_level_lst += [0.1*2**2] # 0.4
regul_level_lst += [0.1*2**1] # 0.2
regul_level_lst += [0.1     ]
regul_level_lst += [0.1/2**1] # 0.05
regul_level_lst += [0.1/2**2] # 0.025
regul_level_lst += [0.1/2**3] # 0.0125
regul_level_lst += [0.0     ]

do_generate_images                   = 1
do_generate_meshes                   = 1
do_run_warp                          = 1
do_plot_disp_error_vs_regul_strength = 1

use_subprocesses                   = 1
write_subprocesses_output_to_files = 1

################################################################################

if (use_subprocesses):
    subprocess_manager = mypy.SubprocessManager()
else:
    from generate_images_and_meshes_from_HeartSlice import generate_images_and_meshes_from_HeartSlice
    from plot_disp_error_vs_regul_strength          import plot_disp_error_vs_regul_strength

############################################################ generate_images ###

if (do_generate_images):
 for deformation_type in deformation_type_lst:

    # Need to run the model before generating all images in parallel
    print("*** running model ***"              )
    print("deformation_type:", deformation_type)

    texture_type    = "no"
    noise_level     = 0
    run_model       = True
    generate_images = False

    if (use_subprocesses):
        command_lst  = []
        command_lst += ["python", "generate_images_and_meshes_from_HeartSlice.py"]
        command_lst += ["--n_voxels"        , str(n_voxels)       ]
        command_lst += ["--deformation_type", deformation_type    ]
        command_lst += ["--texture_type"    , texture_type        ]
        command_lst += ["--noise_level"     , str(noise_level)    ]
        command_lst += ["--run_model"       , str(run_model)      ]
        command_lst += ["--generate_images" , str(generate_images)]

        if (write_subprocesses_output_to_files):
            stdout_folder = "generate_images"
            if not os.path.exists(stdout_folder): os.mkdir(stdout_folder)

            stdout_basename  = "heart"
            stdout_basename += "-"+deformation_type
            
            stdout_filename = stdout_folder+"/"+stdout_basename+".out"
        else:
            stdout_filename = None

        subprocess_manager.start_new_process_when_available(
            command_lst     = command_lst,
            stdout_filename = stdout_filename)
        subprocess_manager.wait_for_finished_processes()
    else:
        generate_images_and_meshes_from_HeartSlice(
            n_voxels         = n_voxels        ,
            deformation_type = deformation_type,
            texture_type     = texture_type    ,
            noise_level      = noise_level     ,
            run_model        = run_model       ,
            generate_images  = generate_images )

    for texture_type in texture_type_lst:
     for noise_level  in noise_level_lst :

        n_runs          = n_runs_for_noisy_images if (noise_level > 0) else 1
        run_model       = False
        generate_images = True

        for k_run in range(1, n_runs+1):

            print("*** generate_images ***"            )
            print("deformation_type:", deformation_type)
            print("texture_type:"    , texture_type    )
            print("noise_level:"     , noise_level     )
            print("k_run:"           , k_run           )

            if (use_subprocesses):
                command_lst  = []
                command_lst += ["python", "generate_images_and_meshes_from_HeartSlice.py"]
                command_lst += ["--n_voxels"        , str(n_voxels)       ]
                command_lst += ["--deformation_type", deformation_type    ]
                command_lst += ["--texture_type"    , texture_type        ]
                command_lst += ["--noise_level"     , str(noise_level)    ]
                if (n_runs > 1):
                    command_lst += ["--k_run"       , str(k_run)          ]
                command_lst += ["--run_model"       , str(run_model)      ]
                command_lst += ["--generate_images" , str(generate_images)]

                if (write_subprocesses_output_to_files):
                    stdout_folder = "generate_images"
                    if not os.path.exists(stdout_folder): os.mkdir(stdout_folder)

                    stdout_basename  = "heart"
                    stdout_basename += "-"+deformation_type
                    stdout_basename += "-"+texture_type
                    stdout_basename += "-noise="+str(noise_level)
                    if (n_runs > 1):
                        stdout_basename += "-run="+str(k_run).zfill(2)
                    
                    stdout_filename = stdout_folder+"/"+stdout_basename+".out"
                else:
                    stdout_filename = None

                subprocess_manager.start_new_process_when_available(
                    command_lst     = command_lst,
                    stdout_filename = stdout_filename)
            else:
                generate_images_and_meshes_from_HeartSlice(
                    n_voxels         = n_voxels                       ,
                    deformation_type = deformation_type               ,
                    texture_type     = texture_type                   ,
                    noise_level      = noise_level                    ,
                    k_run            = k_run if (n_runs > 1) else None,
                    run_model        = run_model                      ,
                    generate_images  = generate_images                )

if (use_subprocesses): subprocess_manager.wait_for_finished_processes()

############################################################ generate_meshes ###

if (do_generate_meshes):
 for deformation_type in deformation_type_lst:
  for mesh_size        in mesh_size_lst       :

    print("*** generate_meshes ***"            )
    print("deformation_type:", deformation_type)
    print("mesh_size:"       , mesh_size       )

    texture_type    = "no"
    noise_level     = 0
    run_model       = True
    generate_images = False

    if (use_subprocesses):
        command_lst  = []
        command_lst += ["python", "generate_images_and_meshes_from_HeartSlice.py"]
        command_lst += ["--n_voxels"        , str(n_voxels)       ]
        command_lst += ["--deformation_type", deformation_type    ]
        command_lst += ["--texture_type"    , texture_type        ]
        command_lst += ["--noise_level"     , str(noise_level)    ]
        command_lst += ["--run_model"       , str(run_model)      ]
        command_lst += ["--generate_images" , str(generate_images)]
        command_lst += ["--mesh_size"       , str(mesh_size)      ]

        if (write_subprocesses_output_to_files):
            stdout_folder = "generate_images"
            if not os.path.exists(stdout_folder): os.mkdir(stdout_folder)

            stdout_basename  = "heart"
            stdout_basename += "-"+deformation_type
            stdout_basename += "-h="+str(mesh_size)
            
            stdout_filename = stdout_folder+"/"+stdout_basename+".out"
        else:
            stdout_filename = None

        subprocess_manager.start_new_process_when_available(
            command_lst     = command_lst,
            stdout_filename = stdout_filename)
    else:
        generate_images_and_meshes_from_HeartSlice(
            n_voxels         = n_voxels        ,
            deformation_type = deformation_type,
            texture_type     = texture_type    ,
            noise_level      = noise_level     ,
            run_model        = run_model       ,
            generate_images  = generate_images ,
            mesh_size        = mesh_size       )

if (use_subprocesses): subprocess_manager.wait_for_finished_processes()

################################################################### run_warp ###

if (do_run_warp):
 for deformation_type in deformation_type_lst:
  for texture_type     in texture_type_lst    :
   for noise_level      in noise_level_lst     :

    n_runs = n_runs_for_noisy_images if (noise_level > 0) else 1

    for k_run       in range(1, n_runs+1):
     for mesh_size   in mesh_size_lst     :
      for regul_type  in regul_type_lst    :
       for regul_level in regul_level_lst   :

        if any([_ in regul_type for _ in ["linear", "simple"]]):
            regul_model = "hooke"
        else:
            regul_model = "ogdenciarletgeymonatneohookean"

        regul_poisson = 0.3

        print("*** run_warp ***"                   )
        print("deformation_type:", deformation_type)
        print("texture_type:"    , texture_type    )
        print("noise_level:"     , noise_level     )
        print("k_run:"           , k_run           )
        print("mesh_size:"       , mesh_size       )
        print("regul_type:"      , regul_type      )
        print("regul_model:"     , regul_model     )
        print("regul_level:"     , regul_level     )
        print("regul_poisson:"   , regul_poisson   )

        images_basename  = "heart"
        images_basename += "-"+deformation_type
        images_basename += "-"+texture_type
        images_basename += "-noise="+str(noise_level)
        if (n_runs > 1):
            images_basename += "-run="+str(k_run).zfill(2)

        mesh_folder = images_folder

        mesh_basename  = "heart"
        mesh_basename += "-"+deformation_type
        mesh_basename += "-h="+str(mesh_size)
        mesh_basename += "-mesh"

        working_basename = images_basename
        working_basename += "-h="+str(mesh_size)
        working_basename += "-"+regul_type
        working_basename += "-regul="+str(regul_level)

        relax_type                                  = "backtracking"
        tol_dU                                      = 1e-2
        n_iter_max                                  = 100
        normalize_energies                          = 1
        continue_after_fail                         = 1
        write_VTU_files                             = 1
        write_VTU_files_with_preserved_connectivity = 1

        if (use_subprocesses):
            command_lst  = []
            command_lst += ["python", "../../dolfin_warp/warp.py"]
            command_lst += ["--working_folder"                             , working_folder                                  ]
            command_lst += ["--working_basename"                           , working_basename                                ]
            command_lst += ["--images_folder"                              , images_folder                                   ]
            command_lst += ["--images_basename"                            , images_basename                                 ]
            command_lst += ["--mesh_folder"                                , mesh_folder                                     ]
            command_lst += ["--mesh_basename"                              , mesh_basename                                   ]
            command_lst += ["--regul_type"                                 , regul_type                                      ]
            command_lst += ["--regul_model"                                , regul_model                                     ]
            command_lst += ["--regul_level"                                , str(regul_level)                                ]
            command_lst += ["--regul_poisson"                              , str(regul_poisson)                              ]
            command_lst += ["--relax_type"                                 , relax_type                                      ]
            command_lst += ["--normalize_energies"                         , str(normalize_energies)                         ]
            command_lst += ["--tol_dU"                                     , str(tol_dU)                                     ]
            command_lst += ["--n_iter_max"                                 , str(n_iter_max)                                 ]
            command_lst += ["--continue_after_fail"                        , str(continue_after_fail)                        ]
            command_lst += ["--write_VTU_files"                            , str(write_VTU_files)                            ]
            command_lst += ["--write_VTU_files_with_preserved_connectivity", str(write_VTU_files_with_preserved_connectivity)]

            if (write_subprocesses_output_to_files):
                if not os.path.exists(working_folder): os.mkdir(working_folder)
                stdout_filename = working_folder+"/"+working_basename+".out"
            else:
                stdout_filename = None

            subprocess_manager.start_new_process_when_available(
                command_lst     = command_lst,
                stdout_filename = stdout_filename)
        else:
            dwarp.warp(
                working_folder                              = working_folder                             ,
                working_basename                            = working_basename                           ,
                images_folder                               = images_folder                              ,
                images_basename                             = images_basename                            ,
                mesh_folder                                 = mesh_folder                                ,
                mesh_basename                               = mesh_basename                              ,
                regul_type                                  = regul_type                                 ,
                regul_model                                 = regul_model                                ,
                regul_level                                 = regul_level                                ,
                regul_poisson                               = regul_poisson                              ,
                relax_type                                  = relax_type                                 ,
                normalize_energies                          = normalize_energies                         ,
                tol_dU                                      = tol_dU                                     ,
                n_iter_max                                  = n_iter_max                                 ,
                continue_after_fail                         = continue_after_fail                        ,
                write_VTU_files                             = write_VTU_files                            ,
                write_VTU_files_with_preserved_connectivity = write_VTU_files_with_preserved_connectivity)

if (use_subprocesses): subprocess_manager.wait_for_finished_processes()

########################################## plot_disp_error_vs_regul_strength ###

if (do_plot_disp_error_vs_regul_strength):
 for deformation_type in deformation_type_lst:
  for texture_type     in texture_type_lst    :
   for regul_type       in regul_type_lst      :

    print("*** plot_disp_error_vs_regul_strength ***")
    print("deformation_type:", deformation_type)
    print("texture_type:"    , texture_type    )
    print("regul_type:"      , regul_type      )

    structure_type                           = "heart"
    regul_level_for_zero                     = 1e-3
    generate_datafile                        = 1
    generate_datafile_with_limited_precision = 1
    generate_plotfile                        = 1
    generate_plot                            = 1

    if (use_subprocesses):
        command_lst  = []
        command_lst += ["python", "plot_disp_error_vs_regul_strength.py"]
        command_lst += ["--images_folder"                           , images_folder                                ]
        command_lst += ["--sol_folder"                              , working_folder                               ]
        command_lst += ["--structure_type"                          , structure_type                               ]
        command_lst += ["--deformation_type"                        , deformation_type                             ]
        command_lst += ["--texture_type"                            , texture_type                                 ]
        command_lst += ["--regul_type"                              , regul_type                                   ]
        command_lst += ["--noise_level_lst"                         , str(noise_level_lst).replace(" ", "")        ]
        command_lst += ["--n_runs_for_noisy_images"                 , str(n_runs_for_noisy_images)                 ]
        command_lst += ["--regul_level_lst"                         , str(regul_level_lst).replace(" ", "")        ]
        command_lst += ["--regul_level_for_zero"                    , str(regul_level_for_zero)                    ]
        command_lst += ["--generate_datafile"                       , str(generate_datafile)                       ]
        command_lst += ["--generate_datafile_with_limited_precision", str(generate_datafile_with_limited_precision)]
        command_lst += ["--generate_plotfile"                       , str(generate_plotfile)                       ]
        command_lst += ["--generate_plot"                           , str(generate_plot)                           ]

        if (write_subprocesses_output_to_files):
            stdout_folder = "plot_disp_error_vs_regul_strength"
            if not os.path.exists(stdout_folder): os.mkdir(stdout_folder)

            stdout_basename  = structure_type
            stdout_basename += "-"+deformation_type
            stdout_basename += "-"+texture_type
            stdout_basename += "-"+regul_type
            
            stdout_filename = stdout_folder+"/"+stdout_basename+".out"
        else:
            stdout_filename = None

        subprocess_manager.start_new_process_when_available(
            command_lst     = command_lst,
            stdout_filename = stdout_filename)
    else:
        plot_disp_error_vs_regul_strength(
            images_folder                            = images_folder                           ,
            sol_folder                               = working_folder                          ,
            structure_type                           = structure_type                          ,
            deformation_type                         = deformation_type                        ,
            texture_type                             = texture_type                            ,
            regul_type                               = regul_type                              ,
            noise_level_lst                          = noise_level_lst                         ,
            n_runs_for_noisy_images                  = n_runs_for_noisy_images                 ,
            regul_level_lst                          = regul_level_lst                         ,
            regul_level_for_zero                     = regul_level_for_zero                    ,
            generate_datafile                        = generate_datafile                       ,
            generate_datafile_with_limited_precision = generate_datafile_with_limited_precision,
            generate_plotfile                        = generate_plotfile                       ,
            generate_plot                            = generate_plot                           )

if (use_subprocesses): subprocess_manager.wait_for_finished_processes()

####################################################################### test ###

for deformation_type in deformation_type_lst:
 for texture_type     in texture_type_lst    :
  for regul_type       in regul_type_lst      :

    structure_type = "heart"

    res_basename = structure_type
    res_basename += "-"+deformation_type
    res_basename += "-"+texture_type
    res_basename += "-"+regul_type
    test.test(res_basename)
