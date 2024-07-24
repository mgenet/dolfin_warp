#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2016-2024                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

################################################################################

def get_ExprGenIm_cpp_pybind(
        im_dim      : int        ,  # 2, 3
        im_type     : str  = "im",  # im, grad
        im_is_def   : bool = 1   ,  #
        im_texture  : str  = "no",  # no, tagging
        im_resample : bool = 1   ,  #
        verbose     : bool = 0   ): #

    assert (im_dim in (2,3))
    assert (im_type in ("im", "grad"))
    assert (not ((im_type=="grad") and (im_is_def)))
    assert (im_texture in ("no", "tagging", "tagging-diffComb", "tagging-signed", "tagging-signed-diffComb"))

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
    if   (im_resample):
        name += "Res"
    # print(name)

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

#include <vtkImageClip.h>
#include <vtkImageExtractComponents.h>
#include <vtkImageFFT.h>
#include <vtkImageFourierCenter.h>
#include <vtkImageMathematics.h>
#include <vtkImageRFFT.h>
#include <vtkImageShiftScale.h>

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

class '''+name+''' : public dolfin::Expression
{
public:

    static constexpr unsigned int n_dim = '''+str(im_dim)+''';

    vtkSmartPointer<vtkXMLImageDataReader>     measured_reader = vtkSmartPointer<vtkXMLImageDataReader>::New();
    vtkSmartPointer<vtkImageData>              measured_image = nullptr;
    double                                     measured_image_origin[3];
    double                                     measured_image_spacing[3];
    int                                        measured_image_dimensions[3];
    vtkSmartPointer<vtkImageFFT>               measured_fft_filter = vtkSmartPointer<vtkImageFFT>::New();
    vtkSmartPointer<vtkImageData>              measured_fft_image = nullptr;'''+('''
    int                                        resampling_factor;
    vtkSmartPointer<vtkImageData>              generated_upsampled_image = vtkSmartPointer<vtkImageData>::New();
    double                                     generated_upsampled_image_origin[3];
    double                                     generated_upsampled_image_spacing[3];
    int                                        generated_upsampled_image_dimensions[3];''')*(im_resample)+'''
    std::shared_ptr<dolfin::Mesh>              mesh = nullptr;
    std::shared_ptr<dolfin::Function>          U = nullptr;
    vtkSmartPointer<vtkUnstructuredGrid>       ugrid = vtkSmartPointer<vtkUnstructuredGrid>::New();'''+('''
    vtkSmartPointer<vtkWarpVector>             warp_filter = vtkSmartPointer<vtkWarpVector>::New();
    vtkSmartPointer<vtkUnstructuredGrid>       warp_ugrid = nullptr;''')*(im_is_def)+'''
    mutable Eigen::Matrix<double, 3, 1>        X_3D;'''+('''
    mutable Eigen::Matrix<double, 3, 1>        x_3D;
    Eigen::Matrix<double, 3, 1>                ux_3D;''')*(im_is_def)+'''
    vtkSmartPointer<vtkProbeFilter>            probe_filter = vtkSmartPointer<vtkProbeFilter>::New();'''+('''
    Eigen::Matrix<double, n_dim, 1>            X0;
    double                                     s;''')*(im_texture=="tagging")+('''
    vtkSmartPointer<vtkImageData>              generated_image = vtkSmartPointer<vtkImageData>::New();
    vtkSmartPointer<vtkImageFFT>               generated_fft_filter = vtkSmartPointer<vtkImageFFT>::New();
    vtkSmartPointer<vtkImageData>              generated_fft_image = nullptr;''')*(not im_resample)+('''
    vtkSmartPointer<vtkImageFFT>               generated_fft_filter = vtkSmartPointer<vtkImageFFT>::New();
    vtkSmartPointer<vtkImageData>              generated_fft_image = vtkSmartPointer<vtkImageData>::New();
    vtkSmartPointer<vtkImageRFFT>              generated_rfft_filter = vtkSmartPointer<vtkImageRFFT>::New();
    vtkSmartPointer<vtkImageExtractComponents> generated_extract_filter = vtkSmartPointer<vtkImageExtractComponents>::New();
    vtkSmartPointer<vtkImageData>              generated_image = nullptr;''')*(im_resample)+('''
    vtkSmartPointer<vtkImageGradient>          grad = vtkSmartPointer<vtkImageGradient>::New();
    vtkSmartPointer<vtkImageData>              grad_image = nullptr;''')*(im_type=="grad")+'''
    vtkSmartPointer<vtkImageInterpolator>      generated_interpolator = vtkSmartPointer<vtkImageInterpolator>::New();'''+('''
    mutable Eigen::Matrix<double, n_dim, 1>    UX;''')*(im_is_def)+'''

    '''+name+'''
    ('''+('''
        const double &Z=0.,''')*(im_dim==2)+('''
        const double &X0_=0.,
        const double &Y0_=0.,'''+('''
        const double &Z0_=0.,''')*(im_dim==3)+'''
        const double &s_=0.,''')*(im_texture=="tagging")+'''
        const char* image_interpol_mode="linear",
        const double &image_interpol_out_value=0.
    ) :
        dolfin::Expression('''+('''n_dim''')*(im_type=="grad")+''')
    {'''+('''
        std::cout << "constructor" << std::endl;''')*(verbose)+'''

        measured_reader->UpdateDataObject();
        measured_image = measured_reader->GetOutput();

        measured_fft_filter->SetInputDataObject(measured_image);
        measured_fft_filter->UpdateDataObject();
        measured_fft_image = measured_fft_filter->GetOutput();'''+('''

        warp_filter->SetInputDataObject(ugrid);
        warp_filter->UpdateDataObject();
        warp_ugrid = warp_filter->GetUnstructuredGridOutput();''')*(im_is_def)+('''

        X_3D[2] = Z;'''+('''
        x_3D[2] = Z;''')*(im_is_def))*(im_dim==2)+'''

        probe_filter->SetInputDataObject('''+('''measured_image''')*(not im_resample)+('''generated_upsampled_image''')*(im_resample)+''');
        probe_filter->SetSourceData('''+('''ugrid''')*(not im_is_def)+('''warp_ugrid''')*(im_is_def)+''');
        probe_filter->UpdateDataObject();'''+('''

        X0[0] = X0_;
        X0[1] = Y0_;'''+('''
        X0[2] = Z0_;''')*(im_dim==3)+'''
        s = s_;''')*(im_texture=="tagging")+('''

        generated_fft_filter->SetDimensionality(n_dim);
        generated_fft_filter->SetInputDataObject(generated_image);
        generated_fft_filter->UpdateDataObject();
        generated_fft_image = generated_fft_filter->GetOutput();''')*(not im_resample)+('''

        generated_fft_filter->SetDimensionality(n_dim);
        generated_fft_filter->SetInputDataObject(generated_upsampled_image);
        generated_fft_filter->UpdateDataObject();

        generated_rfft_filter->SetDimensionality(n_dim);
        generated_rfft_filter->SetInputDataObject(generated_fft_image);
        generated_rfft_filter->UpdateDataObject();

        generated_extract_filter->SetInputDataObject(generated_rfft_filter->GetOutput());
        generated_extract_filter->SetComponents(0);
        generated_extract_filter->UpdateDataObject();

        generated_image = generated_extract_filter->GetOutput();''')*(im_resample)+('''

        grad->SetDimensionality(n_dim);
        grad->SetInputDataObject(generated_image);
        grad->UpdateDataObject();
        grad_image = grad->GetOutput();''')*(im_type=="grad")+'''

        if (strcmp(image_interpol_mode, "nearest") == 0)
        {
            generated_interpolator->SetInterpolationModeToNearest();
        }
        else if (strcmp(image_interpol_mode, "linear") == 0)
        {
            generated_interpolator->SetInterpolationModeToLinear();
        }
        else if (strcmp(image_interpol_mode, "cubic") == 0)
        {
            generated_interpolator->SetInterpolationModeToCubic();
        }
        else
        {
            std::cout << "Interpolator image_interpol_mode (" << image_interpol_mode << ") must be \\"nearest\\", \\"linear\\" or \\"cubic\\". Aborting." << std::endl;
            std::exit(0);
        }
        generated_interpolator->SetOutValue(image_interpol_out_value);
        // generated_interpolator->Initialize('''+('''generated_image''')*(im_type=="im")+('''grad_image''')*(im_type=="grad")+'''); // MG20240524: Possible here? Nope! Apparently, after modifying the image content, the interpolator must be initialized again…
    }

    void init_image
    (
        const char* filename'''+(''',
        const int &resampling_factor_=1''')*(im_resample)+'''
    )
    {'''+('''
        std::cout << "init_image" << std::endl;''')*(verbose)+'''

        measured_reader->SetFileName(filename);
        measured_reader->Update();
        measured_fft_filter->Update();

        measured_image->GetOrigin(measured_image_origin);'''+('''
        std::cout << "measured_image_origin = "
                  <<  measured_image_origin[0] << " "
                  <<  measured_image_origin[1] << " "
                  <<  measured_image_origin[2] << std::endl;''')*(verbose)+'''

        measured_image->GetSpacing(measured_image_spacing);'''+('''
        std::cout << "measured_image_spacing = "
                  <<  measured_image_spacing[0] << " "
                  <<  measured_image_spacing[1] << " "
                  <<  measured_image_spacing[2] << std::endl;''')*(verbose)+'''

        measured_image->GetDimensions(measured_image_dimensions);'''+('''
        std::cout << "measured_image_dimensions = "
                  <<  measured_image_dimensions[0] << " "
                  <<  measured_image_dimensions[1] << " "
                  <<  measured_image_dimensions[2] << std::endl;''')*(verbose)+('''

        generated_image->SetOrigin(measured_image_origin);
        generated_image->SetSpacing(measured_image_spacing);
        generated_image->SetDimensions(measured_image_dimensions);
        generated_image->AllocateScalars(VTK_DOUBLE, 1);''')*(not im_resample)+('''

        resampling_factor = resampling_factor_;'''+('''
        std::cout << "resampling_factor = "
                  <<  resampling_factor << std::endl;''')*(verbose)+'''

        generated_upsampled_image_origin[0] = measured_image_origin[0];
        generated_upsampled_image_origin[1] = measured_image_origin[1];
        generated_upsampled_image_origin[2] = measured_image_origin[2];'''+('''
        std::cout << "generated_upsampled_image_origin = "
                  <<  generated_upsampled_image_origin[0] << " "
                  <<  generated_upsampled_image_origin[1] << " "
                  <<  generated_upsampled_image_origin[2] << std::endl;''')*(verbose)+'''

        generated_upsampled_image_spacing[0] = measured_image_spacing[0]/resampling_factor;
        generated_upsampled_image_spacing[1] = measured_image_spacing[1]/resampling_factor;
        generated_upsampled_image_spacing[2] = '''+('''1.''')*(im_dim==2)+('''measured_image_spacing[2]/resampling_factor''')*(im_dim==3)+''';'''+('''
        std::cout << "generated_upsampled_image_spacing = "
                  <<  generated_upsampled_image_spacing[0] << " "
                  <<  generated_upsampled_image_spacing[1] << " "
                  <<  generated_upsampled_image_spacing[2] << std::endl;''')*(verbose)+'''

        generated_upsampled_image_dimensions[0] = measured_image_dimensions[0]*resampling_factor;
        generated_upsampled_image_dimensions[1] = measured_image_dimensions[1]*resampling_factor;
        generated_upsampled_image_dimensions[2] = '''+('''1''')*(im_dim==2)+('''measured_image_dimensions[2]*resampling_factor''')*(im_dim==3)+''';'''+('''
        std::cout << "generated_upsampled_image_dimensions = "
                  <<  generated_upsampled_image_dimensions[0] << " "
                  <<  generated_upsampled_image_dimensions[1] << " "
                  <<  generated_upsampled_image_dimensions[2] << std::endl;''')*(verbose)+'''

        generated_upsampled_image->SetOrigin(generated_upsampled_image_origin);
        generated_upsampled_image->SetSpacing(generated_upsampled_image_spacing);
        generated_upsampled_image->SetDimensions(generated_upsampled_image_dimensions);'''+('''
        std::cout << "generated_upsampled_image->GetNumberOfPoints() = "
                  <<  generated_upsampled_image->GetNumberOfPoints() << std::endl;
        std::cout << "generated_upsampled_image->GetNumberOfCells() = "
                  <<  generated_upsampled_image->GetNumberOfCells() << std::endl;''')*(verbose)+'''

        generated_upsampled_image->AllocateScalars(VTK_DOUBLE, 1);'''+('''
        std::cout << "generated_upsampled_image->GetPointData()->GetScalars()->GetNumberOfTuples() = "
                  <<  generated_upsampled_image->GetPointData()->GetScalars()->GetNumberOfTuples() << std::endl;
        std::cout << "generated_upsampled_image->GetPointData()->GetScalars()->GetNumberOfComponents() = "
                  <<  generated_upsampled_image->GetPointData()->GetScalars()->GetNumberOfComponents() << std::endl;''')*(verbose)+'''

        generated_fft_image->SetOrigin(measured_image_origin);
        generated_fft_image->SetSpacing(measured_image_spacing);
        generated_fft_image->SetDimensions(measured_image_dimensions);'''+('''
        std::cout << "generated_fft_image->GetNumberOfPoints() = "
                  <<  generated_fft_image->GetNumberOfPoints() << std::endl;
        std::cout << "generated_fft_image->GetNumberOfCells() = "
                  <<  generated_fft_image->GetNumberOfCells() << std::endl;''')*(verbose)+'''

        generated_fft_image->AllocateScalars(VTK_DOUBLE, 2);'''+('''
        std::cout << "generated_fft_image->GetPointData()->GetScalars()->GetNumberOfTuples() = "
                  <<  generated_fft_image->GetPointData()->GetScalars()->GetNumberOfTuples() << std::endl;
        std::cout << "generated_fft_image->GetPointData()->GetScalars()->GetNumberOfComponents() = "
                  <<  generated_fft_image->GetPointData()->GetScalars()->GetNumberOfComponents() << std::endl;''')*(verbose))*(im_resample)+'''
    }

    void update_image
    (
        const char* filename
    )
    {'''+('''
        std::cout << "update_image" << std::endl;''')*(verbose)+'''

        measured_reader->SetFileName(filename);
        measured_reader->Update();
        measured_fft_filter->Update();
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
        std::cout << "n_points = " <<  n_points << std::endl;
        std::cout << "n_cells = " <<  n_cells << std::endl;''')*(verbose)+'''

        U = U_;
        assert (U->ufl_element()->value_size() == n_dim); // MG20190704: asserts are not executed…'''+('''

        unsigned int n_dofs = U->function_space()->dim();
        std::cout << "n_dofs = " <<  n_dofs << std::endl;
        std::cout << "n_dofs/n_dim = " <<  n_dofs/n_dim << std::endl;
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
                  <<  ugrid_points->GetNumberOfPoints() << std::endl;''')*(verbose)+'''

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
        ugrid_cells_ids->SetNumberOfComponents(1);'''+('''
        unsigned int n_points_per_cell = 3;''')*(im_dim==2)+('''
        unsigned int n_points_per_cell = 4;''')*(im_dim==3)+'''
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
    }'''+('''

    void update_disp()
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
                0.);''')*(im_dim==2)+('''
            // std::cout << "U->vector() ="
            //           << " " << (*U->vector())[3*k_point  ]
            //           << " " << (*U->vector())[3*k_point+1]
            //           << " " << (*U->vector())[3*k_point+2]
            //           << std::endl;
            ugrid_disp->SetTuple3(
                k_point,
                (*U->vector())[3*k_point  ],
                (*U->vector())[3*k_point+1],
                (*U->vector())[3*k_point+2]);''')*(im_dim==3)+'''
        }
        ugrid->Modified();
    }''')*(im_is_def)+'''

    void generate_upsampled_image()
    {'''+('''
        std::cout << "generate_upsampled_image" << std::endl;''')*(verbose)+('''

        warp_filter->Update();''')*(im_is_def)+'''

        probe_filter->Update();
        vtkSmartPointer<vtkDataArray> probe_filter_mask = probe_filter->GetImageDataOutput()->GetPointData()->GetArray("vtkValidPointMask");'''+('''
        vtkSmartPointer<vtkDataArray> probe_filter_disp = probe_filter->GetImageDataOutput()->GetPointData()->GetArray("U");''')*(im_is_def)+'''

        vtkSmartPointer<vtkImageData> ima = '''+('''generated_image''')*(not im_resample)+('''generated_upsampled_image''')*(im_resample)+''';
        vtkSmartPointer<vtkDataArray> sca = ima->GetPointData()->GetScalars();
        unsigned int n_points = ima->GetNumberOfPoints();
        double m[1], I[1];
        for (unsigned int k_point=0;
                          k_point<n_points;
                        ++k_point)
        {'''+('''
            // std::cout << "k_point = " << k_point << std::endl;''')*(verbose)+'''

            probe_filter_mask->GetTuple(k_point, m);
            if (m[0] == 0)
            {
                I[0] = 0.;
            }
            else
            {'''+('''
                ima->GetPoint(k_point, X_3D.data());''')*(not im_is_def)+('''
                ima->GetPoint(k_point, x_3D.data());
                probe_filter_disp->GetTuple(k_point, ux_3D.data());
                X_3D = x_3D - ux_3D;''')*(im_is_def)+('''

                I[0] = 1.;''')*(im_texture=="no")+('''
                I[0] = pow(abs(sin(M_PI*(X_3D[0]-X0[0])/s))
                         * abs(sin(M_PI*(X_3D[1]-X0[1])/s)), 0.5);''')*(im_texture=="tagging")+('''
                I[0] = pow(1 + 3*abs(sin(M_PI*(X_3D[0]-X0[0])/s))
                                *abs(sin(M_PI*(X_3D[1]-X0[1])/s)), 0.5) - 1;''')*(im_texture=="tagging-diffComb")+('''
                I[0] = pow((1+sin(M_PI*(X_3D[0]-X0[0])/s-M_PI/2))/2
                          *(1+sin(M_PI*(X_3D[1]-X0[1])/s-M_PI/2))/2, 0.5);''')*(im_texture=="tagging-signed")+('''
                I[0] = pow(1 + 3*(1+sin(M_PI*(X_3D[0]-X0[0])/s-M_PI/2))/2
                                *(1+sin(M_PI*(X_3D[1]-X0[1])/s-M_PI/2))/2, 0.5) - 1;''')*(im_texture=="tagging-signed-diffComb")+'''
            }
            sca->SetTuple(k_point, I);
        }
        ima->Modified();
    }'''+('''

    void compute_downsampled_image()
    {'''+('''
        std::cout << "compute_downsampled_images" << std::endl;''')*(verbose)+'''

        generated_fft_filter->Update();

        vtkSmartPointer<vtkDataArray> up_sca = generated_fft_filter->GetOutput()->GetPointData()->GetScalars();
        vtkSmartPointer<vtkDataArray> sca = generated_fft_image->GetPointData()->GetScalars();
        int up_k_x, up_k_y, up_k_z, up_k_point;
        int k_x, k_y, k_z, k_point;
        double I[2];
        for (k_z = 0; k_z<measured_image_dimensions[2]; ++k_z)
        {
            if (k_z <= measured_image_dimensions[2]/2) // FA20200217: NOTE: This should be equivalent to python's a//b (because it is dividing int/int)
            {
                up_k_z = k_z;
            }
            else
            {
                up_k_z = k_z + (generated_upsampled_image_dimensions[2] - measured_image_dimensions[2]);
            }

            for (k_y = 0; k_y<measured_image_dimensions[1]; ++k_y)
            {
                if (k_y <= measured_image_dimensions[1]/2) // FA20200217: NOTE: This should be equivalent to python's a//b (because it is dividing int/int)
                {
                    up_k_y = k_y;
                }
                else
                {
                    up_k_y = k_y + (generated_upsampled_image_dimensions[1] - measured_image_dimensions[1]);
                }

                for (k_x = 0; k_x<measured_image_dimensions[0]; ++k_x)
                {
                    if (k_x <= measured_image_dimensions[0]/2) // FA20200217: NOTE: This should be equivalent to python's a//b (because it is dividing int/int)
                    {
                        up_k_x = k_x;
                    }
                    else
                    {
                        up_k_x = k_x + (generated_upsampled_image_dimensions[0] - measured_image_dimensions[0]);
                    }

                       k_point =    k_z*           measured_image_dimensions[1]*           measured_image_dimensions[0] +    k_y*           measured_image_dimensions[0] +    k_x;
                    up_k_point = up_k_z*generated_upsampled_image_dimensions[1]*generated_upsampled_image_dimensions[0] + up_k_y*generated_upsampled_image_dimensions[0] + up_k_x;
                    up_sca->GetTuple(up_k_point, I);
                    // std::cout << "I = " << I[0] << " " << I[1] << std::endl;
                    I[0] /= pow(resampling_factor, n_dim);
                    I[1] /= pow(resampling_factor, n_dim);
                    // std::cout << "I = " << I[0] << " " << I[1] << std::endl;
                    sca->SetTuple(k_point, I);
                }
            }
            generated_fft_image->Modified();
        }

        generated_rfft_filter->Update();

        generated_extract_filter->Update();
    }''')*(im_resample)+'''

    void generate_image()
    {'''+('''
        std::cout << "generate_image" << std::endl;''')*(verbose)+'''

        generate_upsampled_image();'''+('''

        generated_fft_filter->Update();''')*(not im_resample)+('''

        compute_downsampled_image();''')*(im_resample)+('''

        grad->Update();''')*(im_type=="grad")+'''

        generated_interpolator->Initialize('''+('''generated_image''')*(im_type=="im")+('''grad_image''')*(im_type=="grad")+'''); // MG20240524: Not needed, right? Actually, it is! Apparently, after modifying the image content, the interpolator must be initialized again…
    }

    void write_image
    (
        const char* filename
    )
    {'''+('''
        std::cout << "write_image" << std::endl;''')*(verbose)+'''

        vtkSmartPointer<vtkXMLImageDataWriter> writer = vtkSmartPointer<vtkXMLImageDataWriter>::New();
        writer->SetInputData(generated_image);
        writer->SetFileName(filename);
        writer->Write();
    }'''+('''

    void write_grad_image
    (
        const char* filename
    )
    {'''+('''
        std::cout << "write_gard_image" << std::endl;''')*(verbose)+'''

        vtkSmartPointer<vtkXMLImageDataWriter> writer = vtkSmartPointer<vtkXMLImageDataWriter>::New();
        writer->SetInputData(grad_image);
        writer->SetFileName(filename);
        writer->Write();
    }''')*(im_type=="grad")+'''

    void write_probe_filter_image
    (
        const char* filename
    )
    {'''+('''
        std::cout << "write_probe_filter_image" << std::endl;''')*(verbose)+'''

        vtkSmartPointer<vtkXMLImageDataWriter> writer = vtkSmartPointer<vtkXMLImageDataWriter>::New();
        writer->SetInputData(probe_filter->GetImageDataOutput());
        writer->SetFileName(filename);
        writer->Write();
    }'''+('''

    void write_upsampled_image
    (
        const char* filename
    )
    {'''+('''
        std::cout << "write_upsampled_image" << std::endl;''')*(verbose)+'''

        vtkSmartPointer<vtkXMLImageDataWriter> writer = vtkSmartPointer<vtkXMLImageDataWriter>::New();
        writer->SetInputData(generated_upsampled_image);
        writer->SetFileName(filename);
        writer->Write();
    }''')*(im_resample)+'''

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
    }'''+('''

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
    }''')*(im_is_def)+'''

    double compute_measured_image_integral()
    {'''+('''
        std::cout << "compute_measured_image_integral" << std::endl;''')*(verbose)+'''

        double int = 0.;
        double val;
        for (int k_z = 0; k_z < measured_image_dimensions[2]; ++k_z) {
         for (int k_y = 0; k_y < measured_image_dimensions[1]; ++k_y) {
          for (int k_x = 0; k_x < measured_image_dimensions[0]; ++k_x) {
            val = measured_image->GetScalarComponentAsDouble(k_x, k_y, k_z, 0);
            int += val;
            // int += pow(val, 2.0);
          }
         }
        }
        int /= measured_image_dimensions[2]*measured_image_dimensions[1]*measured_image_dimensions[0];
        // int = pow(int, 0.5);'''+('''
        std::cout << "int = " << int << std::endl;''')*(verbose)+'''

        return int;
    }

    double compute_generated_image_integral()
    {'''+('''
        std::cout << "compute_generated_image_integral" << std::endl;''')*(verbose)+'''

        double int = 0.;
        double val;
        for (int k_z = 0; k_z < measured_image_dimensions[2]; ++k_z) {
         for (int k_y = 0; k_y < measured_image_dimensions[1]; ++k_y) {
          for (int k_x = 0; k_x < measured_image_dimensions[0]; ++k_x) {
            val = generated_image->GetScalarComponentAsDouble(k_x, k_y, k_z, 0);
            int += val;
            // int += pow(val, 2.0);
          }
         }
        }
        int /= measured_image_dimensions[2]*measured_image_dimensions[1]*measured_image_dimensions[0];
        // int = pow(int, 0.5);'''+('''
        std::cout << "int = " << int << std::endl;''')*(verbose)+'''

        return int;
    }

    double compute_image_energy()
    {'''+('''
        std::cout << "compute_image_energy" << std::endl;''')*(verbose)+'''

        double ener = 0., norm = 0.;
        double ima, gen;
        for (int k_z = 0; k_z < measured_image_dimensions[2]; ++k_z) {
         for (int k_y = 0; k_y < measured_image_dimensions[1]; ++k_y) {
          for (int k_x = 0; k_x < measured_image_dimensions[0]; ++k_x) {
            ima = measured_image->GetScalarComponentAsDouble(k_x, k_y, k_z, 0);
            gen = generated_image->GetScalarComponentAsDouble(k_x, k_y, k_z, 0);
            ener += pow(ima - gen, 2.0);
            norm += pow(ima, 2.0);
          }
         }
        }
        ener = pow(ener/norm, 0.5);'''+('''
        std::cout << "ener = " << ener << std::endl;''')*(verbose)+'''

        return ener;
    }

    double compute_fourier_energy()
    {'''+('''
        std::cout << "compute_image_energy" << std::endl;''')*(verbose)+'''

        double ener = 0., norm = 0.;
        double ima, gen;
        for (int k_z = 0; k_z < measured_image_dimensions[2]; ++k_z) {
         for (int k_y = 0; k_y < measured_image_dimensions[1]; ++k_y) {
          for (int k_x = 0; k_x < measured_image_dimensions[0]; ++k_x) {
            ima = measured_fft_image->GetScalarComponentAsDouble(k_x, k_y, k_z, 0);
            gen = generated_fft_image->GetScalarComponentAsDouble(k_x, k_y, k_z, 0);
            ener += pow(ima - gen, 2.0);
            norm += pow(ima, 2.0);

            ima = measured_fft_image->GetScalarComponentAsDouble(k_x, k_y, k_z, 1);
            gen = generated_fft_image->GetScalarComponentAsDouble(k_x, k_y, k_z, 1);
            ener += pow(ima - gen, 2.0);
            norm += pow(ima, 2.0);
          }
         }
        }
        ener = pow(ener/norm, 0.5);'''+('''
        std::cout << "ener = " << ener << std::endl;''')*(verbose)+'''

        return ener;
    }

    void eval
    (
        Eigen::Ref<      Eigen::VectorXd> expr,
        Eigen::Ref<const Eigen::VectorXd> X
    ) const
    {'''+('''
        // std::cout << "X = " << X << std::endl;''')*(verbose)+(('''
        X_3D.head<n_dim>() = X;'''+('''
        // std::cout << "X_3D = " << X_3D << std::endl;''')*(verbose)+'''

        generated_interpolator->Interpolate(X_3D.data(), expr.data());''')*(im_dim==2)+('''

        generated_interpolator->Interpolate(   X.data(), expr.data());''')*(im_dim==3))*(not im_is_def)+('''

        U->eval(UX, X);'''+('''
        // std::cout << "UX = " << UX << std::endl;''')*(verbose)+('''

        x_3D.head<n_dim>() = X + UX;''')*(im_dim==2)+('''
        x_3D               = X + UX;''')*(im_dim==3)+('''
        // std::cout << "x_3D = " << x_3D << std::endl;''')*(verbose)+'''

        generated_interpolator->Interpolate(x_3D.data(), expr.data());''')*(im_is_def)+('''

        // std::cout << "expr = " << expr << std::endl;''')*(verbose)+'''
    }
};

PYBIND11_MODULE(SIGNATURE, m)
{
    pybind11::class_<'''+name+''', std::shared_ptr<'''+name+'''>, dolfin::Expression>(m, "'''+name+'''")
    .def(pybind11::init<'''+('''const double&, ''')*(im_dim==2)+('''const double&, const double&, '''+('''const double&, ''')*(im_dim==3)+'''const double&, ''')*(im_texture=="tagging")+'''const char*, const double&>(), '''+('''pybind11::arg("Z") = 0., ''')*(im_dim==2)+('''pybind11::arg("X0") = 0., pybind11::arg("Y0") = 0., '''+('''pybind11::arg("Z0") = 0., ''')*(im_dim==3)+'''pybind11::arg("s") = 0.1, ''')*(im_texture=="tagging")+'''pybind11::arg("image_interpol_mode") = "linear", pybind11::arg("interpol_out_value") = 0.)
    .def("init_image", &'''+name+'''::init_image, pybind11::arg("filename")'''+(''', pybind11::arg("resampling_factor_") = 1''')*(im_resample)+''')
    .def("update_image", &'''+name+'''::update_image, pybind11::arg("filename"))
    .def("init_ugrid", &'''+name+'''::init_ugrid, pybind11::arg("mesh_"), pybind11::arg("U_"))'''+('''
    .def("update_disp", &'''+name+'''::update_disp)''')*(im_is_def)+'''
    .def("generate_upsampled_image", &'''+name+'''::generate_upsampled_image)'''+('''
    .def("compute_downsampled_image", &'''+name+'''::compute_downsampled_image)''')*(im_resample)+'''
    .def("generate_image", &'''+name+'''::generate_image)
    .def("write_image", &'''+name+'''::write_image, pybind11::arg("filename"))'''+('''
    .def("write_grad_image", &'''+name+'''::write_grad_image, pybind11::arg("filename"))''')*(im_type=="grad")+('''
    .def("write_upsampled_image", &'''+name+'''::write_upsampled_image, pybind11::arg("filename"))''')*(im_resample)+'''
    .def("write_probe_filter_image", &'''+name+'''::write_probe_filter_image, pybind11::arg("filename"))
    .def("write_ugrid", &'''+name+'''::write_ugrid, pybind11::arg("filename"))'''+('''
    .def("write_warp_ugrid", &'''+name+'''::write_warp_ugrid, pybind11::arg("filename"))''')*(im_is_def)+'''
    .def("compute_measured_image_integral", &'''+name+'''::compute_measured_image_integral)
    .def("compute_generated_image_integral", &'''+name+'''::compute_generated_image_integral)
    .def("compute_image_energy", &'''+name+'''::compute_image_energy)
    .def("compute_fourier_energy", &'''+name+'''::compute_fourier_energy);
}
'''
    # print(cpp)

    return name, cpp
