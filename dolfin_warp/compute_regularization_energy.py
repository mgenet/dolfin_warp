#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2016-2022                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

import dolfin
import matplotlib
import matplotlib.pyplot                   as mpl
import meshio
import pandas
import typing
import vtk.numpy_interface.dataset_adapter as dsa

import dolfin_warp as dwarp

################################################################################

def compute_regularization_energy(
        dim: str,
        working_folder: str,
        working_basename: str,
        working_ext: str = "vtu",
        working_displacement_array_name: str = "displacement",
        regul_type: typing.Optional[str] = None, # MG20220815: This can be written "str | None" starting with python 3.10, but it is not readily available on the gitlab runners (Ubuntu 20.04)
        regul_types: typing.Optional["list[str]"] = None,
        regul_model: str = "ciarletgeymonatneohookean",
        regul_poisson: float = 0.,
        regul_quadrature: int = None,
        plot_regularization_energy: bool = True,
        verbose: bool = True):

    working_series = dwarp.MeshesSeries(
        folder=working_folder,
        basename=working_basename,
        ext=working_ext)

    meshio_mesh = meshio.read(working_series.get_mesh_filename(k_frame=0))
    if (dim == 2):
        meshio_mesh.points = meshio_mesh.points[:, :2]
    meshio.write(working_series.get_mesh_filename(k_frame=None, ext="xdmf"), meshio_mesh)

    mesh = dolfin.Mesh()
    dolfin.parameters['reorder_dofs_serial'] = False
    dolfin.XDMFFile(working_series.get_mesh_filename(k_frame=None, ext="xdmf")).read(mesh)
    # print (mesh)
    # print (mesh.num_vertices())
    # print (mesh.num_cells())

    problem = dwarp.WarpingProblem(
        mesh=mesh,
        U_family="Lagrange",
        U_degree=1)
    # print (len(problem.U.vector()))

    if (regul_types is None):
        if (regul_type is not None):
            regul_types = [regul_type]
        else:
            assert (0),\
                "Must provide regul_type or regul_types. Aborting."

    for regul_type in regul_types:
        if (regul_type in ("continuous-equilibrated", "continuous-elastic", "continuous-hyperelastic")):
            regularization_energy = dwarp.RegularizationContinuousEnergy(
                name=regul_type,
                problem=problem,
                w=1.,
                type=regul_type.split("-")[1],
                model=regul_model,
                poisson=regul_poisson,
                quadrature_degree=regul_quadrature)
        elif (regul_type in ("discrete-linear-equilibrated", "discrete-linear-elastic")):
            regularization_energy = dwarp.LinearRegularizationDiscreteEnergy(
                name=regul_type,
                problem=problem,
                w=1.,
                type=regul_type.split("-")[2],
                model="hooke",
                poisson=regul_poisson,
                quadrature_degree=regul_quadrature)
        elif (regul_type in ("discrete-equilibrated")):
            regularization_energy = dwarp.VolumeRegularizationDiscreteEnergy(
                name=regul_type,
                problem=problem,
                w=1.,
                type=regul_type.split("-")[1],
                model=regul_model,
                poisson=regul_poisson,
                quadrature_degree=regul_quadrature)
        elif (regul_type in ("discrete-tractions", "discrete-tractions-normal", "discrete-tractions-tangential", "discrete-tractions-normal-tangential")):
            regularization_energy = dwarp.SurfaceRegularizationDiscreteEnergy(
                name=regul_type,
                problem=problem,
                w=1.,
                type=regul_type.split("-",1)[1],
                model=regul_model,
                poisson=regul_poisson,
                quadrature_degree=regul_quadrature)
        problem.add_regul_energy(
            energy=regularization_energy,
            order_by_type=False)

    # print (regul_types)
    # print ([energy.name for energy in problem.energies])
    # print ([energy.type for energy in problem.energies])

    regul_ener_filebasename = working_folder+"/"+working_basename+"-regul_ener"
    regul_ener_file = open(regul_ener_filebasename+".dat", "w")
    regul_ener_file.write("#k_frame "+" ".join(regul_types)+"\n")

    for k_frame in range(working_series.n_frames):
        print("k_frame = "+str(k_frame))

        vtk_mesh = working_series.get_mesh(k_frame=k_frame)
        # print (vtk_mesh)
        # print (vtk_mesh.GetNumberOfPoints())
        # print (vtk_mesh.GetNumberOfCells())
        np_mesh = dsa.WrapDataObject(vtk_mesh)
        # print (np_mesh)

        np_disp = np_mesh.PointData[working_displacement_array_name]
        if (dim == 2):
            np_disp = np_disp[:,:2]
        # print (np_disp)
        # print (np_disp.shape)
        # print (np_disp.flatten().shape)

        problem.U.vector()[:] = np_disp.flatten(order="F")
        # dwarp.write_VTU_file(
        #     filebasename = working_series.get_mesh_filebasename(k_frame=None),
        #     function = problem.U,
        #     time = None)

        # for energy in problem.energies:
        #     print(energy.assemble_ener())

        regul_ener_file.write(" ".join([str(val) for val in [k_frame]+[energy.assemble_ener() for energy in problem.energies]])+"\n")

    regul_ener_file.close()

    if (plot_regularization_energy):
        ener_data = pandas.read_csv(
            regul_ener_filebasename+".dat",
            delim_whitespace=True,
            comment="#",
            names=open(regul_ener_filebasename+".dat").readline()[1:].split())

        # mpl.xkcd()
        ener_fig, ener_axes = mpl.subplots()
        # ener_data.plot(x="k_frame", y=regul_types, linestyle="dashed", ax=ener_axes, ylabel="regularization energy")
        # print(ener_axes.get_sketch_params())
        # ener_axes.set_sketch_params(1., 100., 2.) # Does nothing
        # print(ener_axes.get_sketch_params())
        # matplotlib.rcParams['path.sketch'] = (1., 100., 2.) # Applies to all lines!
        # print(ener_axes.get_sketch_params())
        for k, regul_type in enumerate(regul_types):
            ener_data.plot(x="k_frame", y=regul_type, dashes=[len(regul_types)-k, 2], ax=ener_axes, ylabel="regularization energy")
            # ener_data.plot(x="k_frame", y=regul_type, dashes=[len(regul_types)-k, 2], sketch_params=0.3, ax=ener_axes, ylabel="regularization energy")
        ener_axes.set_ylim([-0.001,0.1])
        # ener_axes.set_sketch_params(1.) # Does nothing
        ener_fig.savefig(regul_ener_filebasename+".pdf")

