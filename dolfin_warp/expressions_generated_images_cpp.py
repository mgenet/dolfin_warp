#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2016-2021                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

import dolfin_warp as dwarp

################################################################################

def get_ExprGenIm_cpp_pybind(
        im_dim, # 2, 3
        im_type="im", # im, grad
        im_is_def=1,
        im_texture="no", # no, tagging
        verbose=0):

    assert (im_dim in (2,3))
    assert (im_type in ("im", "grad"))
    assert (im_texture in ("no", "tagging", "tagging-diffComb", "tagging-signed", "tagging-signed-diffComb"))
    assert (not ((im_type=="grad") and (im_is_def)))

    name  = "Expr"
    name += str(im_dim)
    if   (im_type == "im"):
        name += "GenIm"
    elif (im_type == "grad"):
        name += "GenGradIm"
    if   (im_is_def == 0):
        name += "Ref"
    elif (im_is_def == 1):
        name += "Def"
    # print(name)

    if   (im_dim==2):
        n_points_per_cell = 3
    elif (im_dim==3):
        n_points_per_cell = 4

    cpp = '''\
#include <string.h>

#include <Eigen/Dense>

#include <dolfin/fem/DofMap.h>
#include <dolfin/function/Expression.h>
#include <dolfin/function/Function.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/la/Vector.h>
#include <dolfin/mesh/Mesh.h>

#include <vtkCellArray.h>
#include <vtkCellData.h>
#include <vtkDataArray.h>
#include <vtkDoubleArray.h>
#include <vtkImageData.h>
#include <vtkImageGradient.h>
#include <vtkImageInterpolator.h>
#include <vtkPointData.h>
#include <vtkPoints.h>
#include <vtkProbeFilter.h>
#include <vtkSmartPointer.h>
#include <vtkType.h>
#include <vtkUnstructuredGrid.h>
#include <vtkWarpVector.h>
#include <vtkXMLImageDataReader.h>
#include <vtkXMLImageDataWriter.h>
#include <vtkXMLUnstructuredGridWriter.h>

#include <vtkImageFFT.h>
#include <vtkImageRFFT.h>
#include <vtkImageExtractComponents.h>
#include <vtkImageMathematics.h>

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

class '''+name+''' : public dolfin::Expression
{
public:

    static constexpr unsigned int n_dim ='''+str(im_dim )+'''; // MG20200124: Why static?'''+('''

    mutable Eigen::Matrix<double, n_dim, 1> UX;''')*(im_is_def)+'''

    mutable Eigen::Matrix<double, 3, 1> X_3D, x_3D, ux_3D;

    vtkSmartPointer<vtkXMLImageDataReader> reader;
    vtkSmartPointer<vtkImageData>          image;
    vtkSmartPointer<vtkImageData>          image_output;
    vtkSmartPointer<vtkImageData>          image_up;'''+('''
    vtkSmartPointer<vtkImageGradient>      grad;
    vtkSmartPointer<vtkImageData>          grad_image;''')*(im_type=="grad")+'''
    vtkSmartPointer<vtkImageInterpolator>  interpolator;
    vtkSmartPointer<vtkUnstructuredGrid>   ugrid;
    vtkSmartPointer<vtkWarpVector>         warp;
    vtkSmartPointer<vtkUnstructuredGrid>   warp_ugrid;
    vtkSmartPointer<vtkProbeFilter>        probe;
    vtkSmartPointer<vtkImageData>          probe_image;

    vtkSmartPointer<vtkImageFFT>               fft;
    vtkSmartPointer<vtkImageData>              image_downsampled;
    vtkSmartPointer<vtkDoubleArray>            image_downsampled_scalars;
    vtkSmartPointer<vtkImageRFFT>              rfft;
    vtkSmartPointer<vtkImageExtractComponents> extract;

    std::shared_ptr<dolfin::Mesh>     mesh;
    std::shared_ptr<dolfin::Function> U;

    '''+name+'''
    (
        const char* image_interpol_mode="linear",
        const double &image_interpol_out_value=0.'''+(''',
        const double &Z=0.''')*(im_dim==2)+'''
    ) :
            dolfin::Expression('''+('''n_dim''')*(im_type=="grad")+'''),
            reader(vtkSmartPointer<vtkXMLImageDataReader>::New()),
            image_up(vtkSmartPointer<vtkImageData>::New()),'''+('''
            grad(vtkSmartPointer<vtkImageGradient>::New()),''')*(im_type=="grad")+'''
            interpolator(vtkSmartPointer<vtkImageInterpolator>::New()),
            ugrid(vtkSmartPointer<vtkUnstructuredGrid>::New()),
            warp(vtkSmartPointer<vtkWarpVector>::New()),
            probe(vtkSmartPointer<vtkProbeFilter>::New()),
            fft(vtkSmartPointer<vtkImageFFT>::New()),
            image_downsampled(vtkSmartPointer<vtkImageData>::New()),
            image_downsampled_scalars(vtkSmartPointer<vtkDoubleArray>::New()),
            rfft(vtkSmartPointer<vtkImageRFFT>::New()),
            extract(vtkSmartPointer<vtkImageExtractComponents>::New())
    {'''+('''
        std::cout << "constructor" << std::endl;''')*(verbose)+('''

        X_3D[2] = Z;
        x_3D[2] = Z;''')*(im_dim==2)+'''

        reader->UpdateDataObject();
        image = reader->GetOutput();'''+('''

        grad->SetInputDataObject(image_up); // FA20200217: image_up or image_output?
        grad->SetDimensionality(n_dim);
        grad->UpdateDataObject();
        grad_image = grad->GetOutput();''')*(im_type=="grad")+'''

        if (strcmp(image_interpol_mode, "nearest") == 0)
        {
            interpolator->SetInterpolationModeToNearest();
        }
        else if (strcmp(image_interpol_mode, "linear") == 0)
        {
            interpolator->SetInterpolationModeToLinear();
        }
        else if (strcmp(image_interpol_mode, "cubic") == 0)
        {
            interpolator->SetInterpolationModeToCubic();
        }
        else
        {
            std::cout << "Interpolator image_interpol_mode (" << image_interpol_mode << ") must be \\"nearest\\", \\"linear\\" or \\"cubic\\". Aborting." << std::endl;
            std::exit(0);
        }
        interpolator->SetOutValue(image_interpol_out_value);

        warp->SetInputDataObject(ugrid);
        warp->UpdateDataObject();
        warp_ugrid = warp->GetUnstructuredGridOutput();

        probe->SetInputDataObject(image_up);
        probe->SetSourceConnection(warp->GetOutputPort()); // probe->SetSourceDataObject(warp_ugrid); ?
        probe->UpdateDataObject();
        probe_image = probe->GetImageDataOutput();
    }

    void init_image
    (
        const char* filename,
        const double n_up=1
    )
    {'''+('''
        std::cout << "init_image" << std::endl;''')*(verbose)+'''

        reader->SetFileName(filename);
        reader->Update();

        int image_dimensions[3];
        image->GetDimensions(image_dimensions);
        // std::cout << "image_dimensions = " << image_dimensions[0] << " " << image_dimensions[1] << " " << image_dimensions[2] << std::endl;'''+('''
        image_up->SetDimensions(image_dimensions[0]*n_up, image_dimensions[1]*n_up,                    1    );''')*(im_dim==2)+('''
        image_up->SetDimensions(image_dimensions[0]*n_up, image_dimensions[1]*n_up, image_dimensions[2]*n_up);''')*(im_dim==3)+'''

        double image_spacing[3];
        image->GetSpacing(image_spacing);
        // std::cout << "image_spacing = " << image_spacing[0] << " " << image_spacing[1] << " " << image_spacing[2] << std::endl;'''+('''
        image_up->SetSpacing(image_spacing[0]/n_up, image_spacing[1]/n_up,                 1.   );''')*(im_dim==2)+('''
        image_up->SetSpacing(image_spacing[0]/n_up, image_spacing[1]/n_up, image_spacing[2]/n_up);''')*(im_dim==3)+'''

        double image_origin[3];
        image->GetOrigin(image_origin);
        // std::cout << "image_origin = " << image_origin[0] << " " << image_origin[1] << " " << image_origin[2] << std::endl;
        image_up->SetOrigin(image_origin);

        image_up->AllocateScalars(VTK_FLOAT, 1);
        // std::cout << image_up->GetPointData()->GetScalars()->GetName() << std::endl;
        // std::cout << image_up->GetPointData()->GetScalars()->GetNumberOfTuples() << std::endl;
        // std::cout << image_up->GetPointData()->GetScalars()->GetNumberOfComponents() << std::endl;'''+('''

        //interpolator->Initialize(image); // FA20200217: moved to generate_image, with Initialize(image_output)''')*(im_type=="im")+('''

        grad->Update();
        interpolator->Initialize(grad_image); // FA20200217: also this needs to be moved to generate_image?''')*(im_type=="grad")+'''
    }

    void init_ugrid
    (
        std::shared_ptr<dolfin::Mesh>     mesh_,
        std::shared_ptr<dolfin::Function> U_
    )
    {'''+('''
        std::cout << "init_ugrid" << std::endl;''')*(verbose)+'''

        mesh = mesh_;
        assert (mesh->geometry()->dim() == n_dim); // MG20190704: asserts are not executed…

        unsigned int n_points = mesh->num_vertices();
        unsigned int n_cells = mesh->num_cells();'''+('''
        std::cout << "n_points = "
                  <<  n_points
                  << std::endl;
        std::cout << "n_cells = "
                  <<  n_cells
                  << std::endl;''')*(verbose)+'''

        U = U_;
        assert (U->ufl_element()->value_size() == n_dim); // MG20190704: asserts are not executed…'''+('''

        unsigned int n_dofs = U->function_space()->dim();
        std::cout << "n_dofs = "
                  <<  n_dofs
                  << std::endl;
        std::cout << "n_dofs/n_dim = "
                  <<  n_dofs/n_dim
                  << std::endl;
        assert (n_dofs/n_dim == n_points); // MG20190704: asserts are not executed…''')*(verbose)+'''

        // Points
        std::vector<double> dofs_coordinates = U->function_space()->tabulate_dof_coordinates();
        // std::cout << "dofs_coordinates =";
        // for (double dof_coordinate: dofs_coordinates){
        //     std::cout << " " << dof_coordinate;}
        // std::cout << std::endl;

        vtkSmartPointer<vtkPoints> ugrid_points = vtkSmartPointer<vtkPoints>::New();
        ugrid_points->SetNumberOfPoints(n_points);'''+('''
        std::cout << "ugrid_points->GetNumberOfPoints() = "
                  <<  ugrid_points->GetNumberOfPoints()
                  << std::endl;''')*(verbose)+'''

        for (unsigned int k_point=0;
                          k_point<n_points;
                        ++k_point)
        {'''+('''
            ugrid_points->SetPoint(
                k_point,
                dofs_coordinates[4*k_point  ],
                dofs_coordinates[4*k_point+1],
                0.);
            // std::cout        << ugrid_points->GetPoint(k_point)[0]
            //           << " " << ugrid_points->GetPoint(k_point)[1]
            //           << std::endl;''')*(im_dim==2)+('''
            ugrid_points->SetPoint(
                k_point,
                dofs_coordinates[9*k_point  ],
                dofs_coordinates[9*k_point+1],
                dofs_coordinates[9*k_point+2]);
            // std::cout        << ugrid_points->GetPoint(k_point)[0]
            //           << " " << ugrid_points->GetPoint(k_point)[1]
            //           << " " << ugrid_points->GetPoint(k_point)[2]
            //           << std::endl;''')*(im_dim==3)+'''
        }
        ugrid->SetPoints(ugrid_points);'''+('''
        std::cout << "ugrid->GetNumberOfPoints() = "
                  <<  ugrid->GetNumberOfPoints()
                  << std::endl;''')*(verbose)+'''

        // Cells
        vtkSmartPointer<vtkIdTypeArray> ugrid_cells_ids = vtkSmartPointer<vtkIdTypeArray>::New();
        ugrid_cells_ids->SetNumberOfComponents(1);
        unsigned int n_points_per_cell = '''+str(n_points_per_cell)+''';
        ugrid_cells_ids->SetNumberOfTuples((1+n_points_per_cell)*n_cells);
        std::shared_ptr<const dolfin::GenericDofMap> dofmap = U->function_space()->dofmap();
        for (unsigned int k_cell=0;
                          k_cell<n_cells;
                        ++k_cell)
        {
            ugrid_cells_ids->SetTuple1(
                (1+n_points_per_cell)*k_cell,
                n_points_per_cell);
            // std::cout << "dofmap->cell_dofs(k_cell) = " << dofmap->cell_dofs(k_cell) << std::endl;
            auto cell_dofs = dofmap->cell_dofs(k_cell);
            // std::cout << "cell_dofs = " << cell_dofs << std::endl;'''+('''
            ugrid_cells_ids->SetTuple1(
                (1+n_points_per_cell)*k_cell+1,
                cell_dofs[0]/n_dim);
            ugrid_cells_ids->SetTuple1(
                (1+n_points_per_cell)*k_cell+2,
                cell_dofs[1]/n_dim);
            ugrid_cells_ids->SetTuple1(
                (1+n_points_per_cell)*k_cell+3,
                cell_dofs[2]/n_dim);''')*(im_dim==2)+('''
            ugrid_cells_ids->SetTuple1(
                (1+n_points_per_cell)*k_cell+1,
                cell_dofs[0]/n_dim);
            ugrid_cells_ids->SetTuple1(
                (1+n_points_per_cell)*k_cell+2,
                cell_dofs[1]/n_dim);
            ugrid_cells_ids->SetTuple1(
                (1+n_points_per_cell)*k_cell+3,
                cell_dofs[2]/n_dim);
            ugrid_cells_ids->SetTuple1(
                (1+n_points_per_cell)*k_cell+4,
                cell_dofs[3]/n_dim);''')*(im_dim==3)+'''
        }

        vtkSmartPointer<vtkCellArray> ugrid_cells = vtkSmartPointer<vtkCellArray>::New();
        ugrid_cells->SetCells(
            n_cells,
            ugrid_cells_ids);'''+('''
        std::cout << "ugrid_cells->GetNumberOfCells() = "
                  <<  ugrid_cells->GetNumberOfCells()
                  << std::endl;''')*(verbose)+('''

        ugrid->SetCells(
            VTK_TRIANGLE,
            ugrid_cells);''')*(im_dim==2)+('''

        ugrid->SetCells(
            VTK_TETRA,
            ugrid_cells);''')*(im_dim==3)+('''
        std::cout << "ugrid->GetNumberOfCells() = "
                  <<  ugrid->GetNumberOfCells()
                  << std::endl;''')*(verbose)+'''

        // Disp
        vtkSmartPointer<vtkDoubleArray> ugrid_disp = vtkSmartPointer<vtkDoubleArray>::New();
        ugrid_disp->SetName("U");
        ugrid_disp->SetNumberOfComponents(3);
        ugrid_disp->SetNumberOfTuples(n_points);
        ugrid->GetPointData()->AddArray(ugrid_disp);
        ugrid->GetPointData()->SetActiveVectors("U");'''+('''
        update_disp();''')*(im_is_def)+('''
        ugrid_disp->FillComponent(0, 0.);
        ugrid_disp->FillComponent(1, 0.);
        ugrid_disp->FillComponent(2, 0.);''')*(not im_is_def)+'''
    }

    void update_disp
    (
    )
    {'''+('''
        std::cout << "update_disp" << std::endl;''')*(verbose)+'''

        vtkSmartPointer<vtkDataArray> ugrid_disp = ugrid->GetPointData()->GetArray("U");
        unsigned int n_points = ugrid_disp->GetNumberOfTuples();
        for (unsigned int k_point=0;
                          k_point<n_points;
                        ++k_point)
        {'''+('''
            // std::cout << "U->vector() ="
            //           << " " << (*U->vector())[2*k_point  ]
            //           << " " << (*U->vector())[2*k_point+1]
            //           << std::endl;
            ugrid_disp->SetTuple3(
                k_point,
                (*U->vector())[2*k_point  ],
                (*U->vector())[2*k_point+1],
                0.)''')*(im_dim==2)+('''
            ugrid_disp->SetTuple3(
                k_point,
                (*U->vector())[3*k_point  ],
                (*U->vector())[3*k_point+1],
                (*U->vector())[3*k_point+2])''')*(im_dim==3)+''';
        }
        ugrid->Modified();
    }

    void compute_downsampled_image
    (
        double          downsampling_factor_input,
        unsigned int    write_temp_images=0,
        const char*     suffix = "" // FA20200217: suffix for temp_images
    )
    {'''+('''
        // FA20200214: Function to downsampled image_up, create "extract" object with the downsampled image
        std::cout << "compute_downsampled_images" << std::endl;''')*(verbose)+'''

        unsigned int image_up_ndim = n_dim;
        '''+('''std::cout << "image_up_ndim = " << image_up_ndim << std::endl;''')*(verbose)+'''

        int image_up_dimensions[3];
        image_up->GetDimensions(image_up_dimensions);
        '''+('''std::cout << "image_up_dimensions = " << image_up_dimensions[0] << " " << image_up_dimensions[1] << " " << image_up_dimensions[2] << std::endl;''')*(verbose)+'''

        double image_up_origin[3];
        image_up->GetOrigin(image_up_origin);
        '''+('''std::cout << "image_up_origin = " << image_up_origin[0] << " "<< image_up_origin[1] << " " << image_up_origin[2] << std::endl;''')*(verbose)+'''

        double image_up_spacing[3];
        image_up->GetSpacing(image_up_spacing);
        '''+('''std::cout << "image_up_spacing = " << image_up_spacing[0] << " " << image_up_spacing[1] << " " << image_up_spacing[2] << std::endl;''')*(verbose)+('''

        double downsampling_factors[3] = {downsampling_factor_input, downsampling_factor_input, 1};''')*(im_dim==2)+('''
        double downsampling_factors[3] = {downsampling_factor_input, downsampling_factor_input, downsampling_factor_input};''')*(im_dim==3)+('''

        std::cout << "downsampling_factors = ";
        std::copy(std::begin(downsampling_factors),std::end(downsampling_factors),std::ostream_iterator<int>(std::cout, "\n"));''')*(verbose)+'''

        double image_up_downsampled_dimensions_db[3];
        std::transform(image_up_dimensions, image_up_dimensions+3, downsampling_factors, image_up_downsampled_dimensions_db, std::divides<double>());
        image_up_downsampled_dimensions_db[0] = std::ceil(image_up_downsampled_dimensions_db[0]);
        image_up_downsampled_dimensions_db[1] = std::ceil(image_up_downsampled_dimensions_db[1]);
        image_up_downsampled_dimensions_db[2] = std::ceil(image_up_downsampled_dimensions_db[2]);
        int image_up_downsampled_dimensions[3];
        image_up_downsampled_dimensions[0] = int (image_up_downsampled_dimensions_db[0]);
        image_up_downsampled_dimensions[1] = int (image_up_downsampled_dimensions_db[1]);
        image_up_downsampled_dimensions[2] = int (image_up_downsampled_dimensions_db[2]);
        '''+('''std::cout << "image_up_downsampled_dimensions = " << image_up_downsampled_dimensions[0] << " " << image_up_downsampled_dimensions[1] << " " << image_up_downsampled_dimensions[2] << std::endl;''')*(verbose)+'''

        fft->SetDimensionality(image_up_ndim);
        fft->SetInputData(image_up);
        fft->UpdateDataObject();

        vtkSmartPointer<vtkXMLImageDataWriter> writer_fft;
        if (write_temp_images)
        {
            writer_fft = vtkSmartPointer<vtkXMLImageDataWriter>::New();
            writer_fft->SetInputData(fft->GetOutput());
        }

        int image_up_downsampled_npoints = image_up_downsampled_dimensions[0]*image_up_downsampled_dimensions[1]*image_up_downsampled_dimensions[2];
        '''+('''std::cout << "image_up_downsampled_npoints = " << image_up_downsampled_npoints << std::endl;''')*(verbose)+'''

        std::transform(image_up_dimensions, image_up_dimensions+3, image_up_downsampled_dimensions, downsampling_factors, std::divides<double>());
        '''+('''std::cout << "downsampling_factors = " << downsampling_factors[0] << " " << downsampling_factors[1] << " " << downsampling_factors[2] << std::endl;''')*(verbose)+'''

        double downsampling_factor = downsampling_factors[0]*downsampling_factors[1]*downsampling_factors[2];
        '''+('''std::cout << "downsampling_factor = " << downsampling_factor << std::endl;''')*(verbose)+'''

        double image_downsampled_origin[3];
        std::copy(std::begin(image_up_origin), std::end(image_up_origin), std::begin(image_downsampled_origin));
        '''+('''std::cout << "image_downsampled_origin = " << image_downsampled_origin[0] << " " << image_downsampled_origin[1] << " " << image_downsampled_origin[2] << std::endl;''')*(verbose)+'''

        double image_downsampled_spacing[3];
        std::transform(image_up_spacing, image_up_spacing+3, downsampling_factors, image_downsampled_spacing, std::multiplies<double>());
        '''+('''std::cout << "image_downsampled_spacing = " << image_downsampled_spacing[0] << " " << image_downsampled_spacing[1] << " " << image_downsampled_spacing[2] << std::endl;''')*(verbose)+'''

        image_downsampled->SetDimensions(image_up_downsampled_dimensions);
        image_downsampled->SetOrigin(image_downsampled_origin);
        image_downsampled->SetSpacing(image_downsampled_spacing);

        // NOTE: The following corresponds to the function myvtk.createDoubleArray (line 144 python version)
        int n_components = 2;
        int n_tuples = image_up_downsampled_npoints;
        int init_to_zero = 0;

        image_downsampled_scalars->SetName("ImageScalars");
        image_downsampled_scalars->SetNumberOfComponents(n_components);
        image_downsampled_scalars->SetNumberOfTuples(n_tuples);
        if (init_to_zero)
        {
            for (int k_component = 0; k_component<n_components; k_component++)
            {
                image_downsampled_scalars->FillComponent(k_component, 0.);
            }
        }

        image_downsampled->GetPointData()->SetScalars(image_downsampled_scalars);

        vtkSmartPointer<vtkXMLImageDataWriter> writer_sel;
        if (write_temp_images)
        {
            writer_sel = vtkSmartPointer<vtkXMLImageDataWriter>::New();
            writer_sel->SetInputData(image_downsampled);
        }

        rfft->SetDimensionality(image_up_ndim);
        rfft->SetInputData(image_downsampled);
        rfft->UpdateDataObject();

        extract->SetInputData(rfft->GetOutput());
        extract->SetComponents(0);
        extract->UpdateDataObject();

        // vtkSmartPointer<vtkXMLImageDataWriter> writer = vtkSmartPointer<vtkXMLImageDataWriter>::New();
        // writer->SetInputData(extract->GetOutput());

        fft->Update();
        if (write_temp_images)
        {
            std::string writer_fft_name_str = std::string(suffix)+"_fft.vti";
            char writer_fft_name[writer_fft_name_str.size()+1];

            writer_fft_name_str.copy(writer_fft_name, writer_fft_name_str.size()+1);
            writer_fft_name[writer_fft_name_str.size()] = '\0';

            writer_fft->SetFileName(writer_fft_name);
            writer_fft->Write();
        }

        vtkSmartPointer<vtkDataArray> image_scalars = fft->GetOutput()->GetPointData()->GetScalars();                 // FA20200217: CHECK: Is it well defined the type of this pointer?
        vtkSmartPointer<vtkDataArray> image_downsampled_scalars_1 = image_downsampled->GetPointData()->GetScalars();  // FA20200217: CHECK: Is it well defined the type of this pointer?
        int k_x, k_y, k_z;
        double I[2];
        for (int k_z_downsampled = 0; k_z_downsampled<image_up_downsampled_dimensions[2]; k_z_downsampled++)
        {
            if (k_z_downsampled <= image_up_downsampled_dimensions[2]/2) // FA20200217: NOTE: This should be equivalent to a//b (because it is dividing int/int)
            {
                k_z = k_z_downsampled;
            }
            else
            {
                k_z = k_z_downsampled+(image_up_dimensions[2]-image_up_downsampled_dimensions[2]);
            }

            for (int k_y_downsampled = 0; k_y_downsampled<image_up_downsampled_dimensions[1]; k_y_downsampled++)
            {
                if (k_y_downsampled <= image_up_downsampled_dimensions[1]/2) // FA20200217: NOTE: This should be equivalent to a//b (because it is dividing int/int)
                {
                    k_y = k_y_downsampled;
                }
                else
                {
                    k_y = k_y_downsampled+(image_up_dimensions[1]-image_up_downsampled_dimensions[1]);
                }

                for (int k_x_downsampled = 0; k_x_downsampled<image_up_downsampled_dimensions[0]; k_x_downsampled++)
                {
                    if (k_x_downsampled <= image_up_downsampled_dimensions[0]/2) // FA20200217: NOTE: This should be equivalent to a//b (because it is dividing int/int)
                    {
                        k_x = k_x_downsampled;
                    }
                    else
                    {
                        k_x = k_x_downsampled+(image_up_dimensions[0]-image_up_downsampled_dimensions[0]);
                    }

                    int k_point_downsampled = k_z_downsampled*image_up_downsampled_dimensions[1]*image_up_downsampled_dimensions[0] + k_y_downsampled*image_up_downsampled_dimensions[0] + k_x_downsampled;
                    int k_point             = k_z            *image_up_dimensions[1]            *image_up_dimensions[0]             + k_y            *image_up_dimensions[0]             + k_x;
                    image_scalars->GetTuple(k_point, I);
                    I[0] /= downsampling_factor;
                    I[1] /= downsampling_factor;
                    image_downsampled_scalars_1->SetTuple(k_point_downsampled, I);
                }
            }
            image_downsampled->Modified();

            if (write_temp_images)
            {
                std::string writer_sel_name_str = std::string(suffix)+"_sel.vti";
                char writer_sel_name[writer_sel_name_str.size()+1];

                writer_sel_name_str.copy(writer_sel_name, writer_sel_name_str.size()+1);
                writer_sel_name[writer_sel_name_str.size()] = '\0';

                writer_sel->SetFileName(writer_sel_name);
                writer_sel->Write();
            }
        }

        rfft->Update();

        extract->Update();

        // FA20200217: The downsampled image can be saved with write_image()
        // std::string writer_name_str = std::string(suffix)+".vti";
        // char writer_name[writer_name_str.size()+1];
        // writer_name_str.copy(writer_name, writer_name_str.size()+1);
        // writer_name[writer_name_str.size()] = '\0';
        // writer->SetFileName(writer_name);
        // writer->Write();

    }

    void generate_image
    (
        double n_down=1
    )
    {'''+('''
        std::cout << "generate_image" << std::endl;''')*(verbose)+'''

        warp->Update();

        probe->Update();

        unsigned int n_points = image_up->GetNumberOfPoints();
        vtkSmartPointer<vtkDataArray> image_up_scalars = image_up->GetPointData()->GetScalars();
        vtkSmartPointer<vtkDataArray> probe_mask = probe_image->GetPointData()->GetArray("vtkValidPointMask");
        vtkSmartPointer<vtkDataArray> probe_disp = probe_image->GetPointData()->GetArray("U");
        double m[1], I[1];
        for (unsigned int k_point=0;
                          k_point<n_points;
                        ++k_point)
        {
            probe_mask->GetTuple(k_point, m);
            if (m[0] == 0)
            {
                I[0] = 0.;
            }
            else
            {
                image_up->GetPoint(k_point, x_3D.data());
                probe_disp->GetTuple(k_point, ux_3D.data());
                X_3D = x_3D - ux_3D;'''+('''
                I[0] = 1.;''')*(im_texture=="no")+('''
                I[0] = pow(abs(sin(M_PI*X_3D[0]/0.1))
                         * abs(sin(M_PI*X_3D[1]/0.1)), 0.5);''')*(im_texture=="tagging")+('''
                I[0] = pow(1 + 3*abs(sin(M_PI*X_3D[0]/0.1))
                                *abs(sin(M_PI*X_3D[1]/0.1)), 0.5) - 1;''')*(im_texture=="tagging-diffComb")+('''
                I[0] = pow((1+sin(M_PI*X_3D[0]/0.1-M_PI/2))/2
                          *(1+sin(M_PI*X_3D[1]/0.1-M_PI/2))/2, 0.5);''')*(im_texture=="tagging-signed")+('''
                I[0] = pow(1 + 3*(1+sin(M_PI*X_3D[0]/0.1-M_PI/2))/2
                                *(1+sin(M_PI*X_3D[1]/0.1-M_PI/2))/2, 0.5) - 1;''')*(im_texture=="tagging-signed-diffComb")+'''
            }
            image_up_scalars->SetTuple(k_point, I);
        }
        image_up->Modified();

        compute_downsampled_image(n_down);
        image_output = extract->GetOutput(); // FA20200217: this has the information of the downsampled image

        interpolator->Initialize(image_output);

        '''+('''

        grad->Update();''')*(im_type=="grad")+'''
    }

    void write_image
    (
        const char* filename
    )
    {'''+('''
        std::cout << "write_image" << std::endl;''')*(verbose)+'''

        vtkSmartPointer<vtkXMLImageDataWriter> writer = vtkSmartPointer<vtkXMLImageDataWriter>::New();
        writer->SetInputData(image_output); // FA20200217: changed image_up to image_output
        writer->SetFileName(filename);
        writer->Write();
    }'''+('''

    void write_grad_image
    (
        const char* filename
    )
    {'''+('''
        std::cout << "write_image" << std::endl;''')*(verbose)+'''

        vtkSmartPointer<vtkXMLImageDataWriter> writer = vtkSmartPointer<vtkXMLImageDataWriter>::New();
        writer->SetInputData(grad_image);
        writer->SetFileName(filename);
        writer->Write();
    }''')*(im_type=="grad")+'''

    void write_probe_image
    (
        const char* filename
    )
    {'''+('''
        std::cout << "write_image" << std::endl;''')*(verbose)+'''

        vtkSmartPointer<vtkXMLImageDataWriter> writer = vtkSmartPointer<vtkXMLImageDataWriter>::New();
        writer->SetInputData(probe_image);
        writer->SetFileName(filename);
        writer->Write();
    }

    void write_ugrid
    (
        const char* filename
    )
    {'''+('''
        std::cout << "write_ugrid" << std::endl;''')*(verbose)+'''

        vtkSmartPointer<vtkXMLUnstructuredGridWriter> writer = vtkSmartPointer<vtkXMLUnstructuredGridWriter>::New();
        writer->SetInputData(ugrid);
        writer->SetFileName(filename);
        writer->Write();
    }

    void write_warp_ugrid
    (
        const char* filename
    )
    {'''+('''
        std::cout << "write_warp_ugrid" << std::endl;''')*(verbose)+'''

        vtkSmartPointer<vtkXMLUnstructuredGridWriter> writer = vtkSmartPointer<vtkXMLUnstructuredGridWriter>::New();
        writer->SetInputData(warp_ugrid);
        writer->SetFileName(filename);
        writer->Write();
    }

    void eval
    (
        Eigen::Ref<      Eigen::VectorXd> expr,
        Eigen::Ref<const Eigen::VectorXd> X
    ) const
    {'''+('''
        std::cout << "X = " << X << std::endl;''')*(verbose)+('''

        U->eval(UX, X);'''+('''
        std::cout << "UX = " << UX << std::endl;''')*(verbose)+('''

        x_3D.head<n_dim>() = X + UX;''')*(im_dim==2)+('''
        x_3D               = X + UX;''')*(im_dim==3)+('''
        std::cout << "x_3D = " << x_3D << std::endl;''')*(verbose)+'''

        interpolator->Interpolate(x_3D.data(), expr.data());''')*(im_is_def)+(('''

        X_3D.head<n_dim>() = X;'''+('''
        std::cout << "X_3D = " << X_3D << std::endl;''')*(verbose)+'''

        interpolator->Interpolate(X_3D.data(), expr.data());''')*(im_dim==2)+('''

        interpolator->Interpolate(X.data(), expr.data());''')*(im_dim==3))*(not im_is_def)+('''

        std::cout << "expr = " << expr << std::endl;''')*(verbose)+'''
    }
};

PYBIND11_MODULE(SIGNATURE, m)
{
    pybind11::class_<'''+name+''', std::shared_ptr<'''+name+'''>, dolfin::Expression>
    (m, "'''+name+'''")
    .def(pybind11::init<const char*, const double&, const double&>(), pybind11::arg("image_interpol_mode") = "linear", pybind11::arg("interpol_out_value") = 0.'''+(''', pybind11::arg("Z") = 0.''')*(im_dim==2)+''')
    .def("init_image", &'''+name+'''::init_image, pybind11::arg("filename"), pybind11::arg("n_up") = 1)
    .def("init_ugrid", &'''+name+'''::init_ugrid, pybind11::arg("mesh_"), pybind11::arg("U_"))
    .def("update_disp", &'''+name+'''::update_disp)
    .def("generate_image", &'''+name+'''::generate_image, pybind11::arg("n_down") = 1)
    .def("compute_downsampled_image", &'''+name+'''::compute_downsampled_image, pybind11::arg("downsampling_factor_input"), pybind11::arg("write_temp_images") = 0, pybind11::arg("suffix") = "")
    .def("write_image", &'''+name+'''::write_image, pybind11::arg("filename"))'''+('''
    .def("write_grad_image", &'''+name+'''::write_grad_image, pybind11::arg("filename"))''')*(im_type=="grad")+'''
    .def("write_probe_image", &'''+name+'''::write_probe_image, pybind11::arg("filename"))
    .def("write_ugrid", &'''+name+'''::write_ugrid, pybind11::arg("filename"))
    .def("write_warp_ugrid", &'''+name+'''::write_warp_ugrid, pybind11::arg("filename"));
}
'''
    # print(cpp)

    return name, cpp
