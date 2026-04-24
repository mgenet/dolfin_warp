#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2016-2026                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

def get_ExprGenContIm_cpp(
        im_dim      : int         ,  # 2, 3
        im_texture  : str  = "no" ,  # no, tagging
        verbose     : bool = 0    ): #

    assert (im_dim in (2,3))
    assert (im_texture in ("no", "tagging"))

    name  = "Expr"
    name += str(im_dim)
    name += "GenContImGrad"
    name += "Def"
    name += "Res"
    # print(name)

    cpp = '''\
#include <string.h>
#include <cmath>

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
    vtkSmartPointer<vtkImageData>              measured_fft_image = nullptr;
    vtkSmartPointer<vtkImageGradient>          measured_gradient_filter = vtkSmartPointer<vtkImageGradient>::New();
    vtkSmartPointer<vtkImageData>              measured_gradient_image = nullptr;
    vtkSmartPointer<vtkImageInterpolator>      measured_gradient_interpolator = vtkSmartPointer<vtkImageInterpolator>::New();

    double                                     resampling_factor;
    double                                     effective_resampling_factors[3];
    double                                     effective_resampling_factor;
    vtkSmartPointer<vtkImageData>              generated_upsampled_image = vtkSmartPointer<vtkImageData>::New();
    double                                     generated_upsampled_image_origin[3];
    double                                     generated_upsampled_image_spacing[3];
    int                                        generated_upsampled_image_dimensions[3];
    
    std::shared_ptr<dolfin::Mesh>              mesh = nullptr;
    std::shared_ptr<dolfin::Function>          U = nullptr;
    vtkSmartPointer<vtkUnstructuredGrid>       ugrid = vtkSmartPointer<vtkUnstructuredGrid>::New();
    vtkSmartPointer<vtkWarpVector>             warp_filter = vtkSmartPointer<vtkWarpVector>::New();
    vtkSmartPointer<vtkUnstructuredGrid>       warp_ugrid = nullptr;
    
    mutable Eigen::Matrix<double, 3, 1>        X_3D;
    mutable Eigen::Matrix<double, 3, 1>        x_3D;
    Eigen::Matrix<double, 3, 1>                ux_3D;
    
    vtkSmartPointer<vtkProbeFilter>            probe_filter = vtkSmartPointer<vtkProbeFilter>::New();'''+('''
    Eigen::Matrix<double, n_dim, 1>            X0;
    double                                     s;''')*(im_texture=="tagging")+'''
    
    vtkSmartPointer<vtkImageFFT>               generated_upsampled_fft_filter = vtkSmartPointer<vtkImageFFT>::New();
    vtkSmartPointer<vtkImageData>              generated_upsampled_fft_image = nullptr;
    vtkSmartPointer<vtkImageData>              generated_fft_image = vtkSmartPointer<vtkImageData>::New();
    vtkSmartPointer<vtkImageRFFT>              generated_rfft_filter = vtkSmartPointer<vtkImageRFFT>::New();
    vtkSmartPointer<vtkImageExtractComponents> generated_extract_filter = vtkSmartPointer<vtkImageExtractComponents>::New();
    vtkSmartPointer<vtkImageData>              generated_image = nullptr;
    vtkSmartPointer<vtkImageInterpolator>      generated_interpolator = vtkSmartPointer<vtkImageInterpolator>::New();
    
    mutable Eigen::Matrix<double, n_dim, 1>    UX;
    
    vtkSmartPointer<vtkImageMathematics>       residual_subtract_filter = vtkSmartPointer<vtkImageMathematics>::New();
    vtkSmartPointer<vtkImageData>              residual_image = nullptr;
    vtkSmartPointer<vtkImageInterpolator>      residual_interpolator = vtkSmartPointer<vtkImageInterpolator>::New();
    vtkSmartPointer<vtkImageFFT>               residual_fft_filter = vtkSmartPointer<vtkImageFFT>::New();
    vtkSmartPointer<vtkImageData>              residual_fft_image = nullptr;

    vtkSmartPointer<vtkImageData>              upsampled_residual_fft_image = vtkSmartPointer<vtkImageData>::New();
    vtkSmartPointer<vtkImageRFFT>              upsampled_residual_rfft_filter = vtkSmartPointer<vtkImageRFFT>::New();
    vtkSmartPointer<vtkImageExtractComponents> upsampled_residual_extract_filter = vtkSmartPointer<vtkImageExtractComponents>::New();
    vtkSmartPointer<vtkImageData>              upsampled_residual_image = nullptr;
    vtkSmartPointer<vtkImageInterpolator>      upsampled_residual_interpolator = vtkSmartPointer<vtkImageInterpolator>::New();

    '''+name+'''
    ('''+('''
        const double &Z=0.,''')*(im_dim==2)+('''
        const double &X0_=0.,
        const double &Y0_=0.,'''+('''
        const double &Z0_=0.,''')*(im_dim==3)+'''
        const double &s_=0.1,''')*(im_texture=="tagging")+'''
        const char* image_interpol_mode="linear",
        const double &image_interpol_out_value=0.
    ) :
        dolfin::Expression(4 + 2*n_dim)
    {'''+('''
        std::cout << "constructor" << std::endl;''')*(verbose)+'''

        measured_reader->UpdateDataObject();
        measured_image = measured_reader->GetOutput();

        measured_fft_filter->SetDimensionality(n_dim);
        measured_fft_filter->SetInputDataObject(measured_image);
        measured_fft_filter->UpdateDataObject();
        measured_fft_image = measured_fft_filter->GetOutput();

        measured_gradient_filter->SetDimensionality(n_dim);
        measured_gradient_filter->SetInputDataObject(measured_image);
        measured_gradient_filter->UpdateDataObject();
        measured_gradient_image = measured_gradient_filter->GetOutput();

        warp_filter->SetInputDataObject(ugrid);
        warp_filter->UpdateDataObject();
        warp_ugrid = warp_filter->GetUnstructuredGridOutput();'''+('''

        X_3D[2] = Z;
        x_3D[2] = Z;''')*(im_dim==2)+'''

        probe_filter->SetInputDataObject(generated_upsampled_image);
        probe_filter->SetSourceData(warp_ugrid);
        probe_filter->UpdateDataObject();'''+('''

        X0[0] = X0_;
        X0[1] = Y0_;'''+('''
        X0[2] = Z0_;''')*(im_dim==3)+'''
        s = s_;''')*(im_texture=="tagging")+'''

        generated_upsampled_fft_filter->SetDimensionality(n_dim);
        generated_upsampled_fft_filter->SetInputDataObject(generated_upsampled_image);
        generated_upsampled_fft_filter->UpdateDataObject();
        generated_upsampled_fft_image = generated_upsampled_fft_filter->GetOutput();

        generated_rfft_filter->SetDimensionality(n_dim);
        generated_rfft_filter->SetInputDataObject(generated_fft_image);
        generated_rfft_filter->UpdateDataObject();

        generated_extract_filter->SetInputDataObject(generated_rfft_filter->GetOutput());
        generated_extract_filter->SetComponents(0);
        generated_extract_filter->UpdateDataObject();

        generated_image = generated_extract_filter->GetOutput();

        residual_subtract_filter->SetOperationToSubtract();
        residual_subtract_filter->SetInput1Data(generated_image);
        residual_subtract_filter->SetInput2Data(measured_image);
        residual_subtract_filter->UpdateDataObject();
        residual_image = residual_subtract_filter->GetOutput();

        residual_fft_filter->SetDimensionality(n_dim);
        residual_fft_filter->SetInputDataObject(residual_image);
        residual_fft_filter->UpdateDataObject();
        residual_fft_image = residual_fft_filter->GetOutput();

        upsampled_residual_rfft_filter->SetDimensionality(n_dim);
        upsampled_residual_rfft_filter->SetInputDataObject(upsampled_residual_fft_image);
        upsampled_residual_extract_filter->SetInputDataObject(upsampled_residual_rfft_filter->GetOutput());
        upsampled_residual_extract_filter->SetComponents(0);
        upsampled_residual_extract_filter->UpdateDataObject();
        upsampled_residual_image = upsampled_residual_extract_filter->GetOutput();

        if (strcmp(image_interpol_mode, "nearest") == 0)
        {
            measured_gradient_interpolator->SetInterpolationModeToNearest();
            generated_interpolator->SetInterpolationModeToNearest();
            residual_interpolator->SetInterpolationModeToNearest();
            upsampled_residual_interpolator->SetInterpolationModeToNearest();
        }
        else if (strcmp(image_interpol_mode, "linear") == 0)
        {
            measured_gradient_interpolator->SetInterpolationModeToLinear();
            generated_interpolator->SetInterpolationModeToLinear();
            residual_interpolator->SetInterpolationModeToLinear();
            upsampled_residual_interpolator->SetInterpolationModeToLinear();
        }
        else if (strcmp(image_interpol_mode, "cubic") == 0)
        {
            measured_gradient_interpolator->SetInterpolationModeToCubic();
            generated_interpolator->SetInterpolationModeToCubic();
            residual_interpolator->SetInterpolationModeToCubic();
            upsampled_residual_interpolator->SetInterpolationModeToCubic();
        }
        else
        {
            std::cout << "Interpolator image_interpol_mode (" << image_interpol_mode << ") must be \\"nearest\\", \\"linear\\" or \\"cubic\\". Aborting." << std::endl;
            std::exit(0);
        }
        measured_gradient_interpolator->SetOutValue(0.0);
        generated_interpolator->SetOutValue(image_interpol_out_value);
        residual_interpolator->SetOutValue(0.0);
        upsampled_residual_interpolator->SetOutValue(0.0);
    }

    void init_images
    (
        const char* filename,
        const double &resampling_factor_=1.
    )
    {'''+('''
        std::cout << "init_images" << std::endl;''')*(verbose)+'''

        measured_reader->SetFileName(filename);
        measured_reader->Update();

        measured_fft_filter->Update();

        measured_gradient_filter->Update();
        measured_gradient_interpolator->Initialize(measured_gradient_image);

        measured_image->GetOrigin(measured_image_origin);
        measured_image->GetSpacing(measured_image_spacing);
        measured_image->GetDimensions(measured_image_dimensions);

        generated_upsampled_image_origin[0] = measured_image_origin[0];
        generated_upsampled_image_origin[1] = measured_image_origin[1];
        generated_upsampled_image_origin[2] = measured_image_origin[2];

        resampling_factor = resampling_factor_;

        generated_upsampled_image_dimensions[0] = std::ceil(measured_image_dimensions[0]*resampling_factor);
        generated_upsampled_image_dimensions[1] = std::ceil(measured_image_dimensions[1]*resampling_factor);
        generated_upsampled_image_dimensions[2] = '''+('''1''')*(im_dim==2)+('''std::ceil(measured_image_dimensions[2]*resampling_factor)''')*(im_dim==3)+''';

        effective_resampling_factors[0] = static_cast<double>(generated_upsampled_image_dimensions[0])/measured_image_dimensions[0];
        effective_resampling_factors[1] = static_cast<double>(generated_upsampled_image_dimensions[1])/measured_image_dimensions[1];
        effective_resampling_factors[2] = static_cast<double>(generated_upsampled_image_dimensions[2])/measured_image_dimensions[2];

        effective_resampling_factor = effective_resampling_factors[0]*effective_resampling_factors[1]*effective_resampling_factors[2];

        generated_upsampled_image_spacing[0] = measured_image_spacing[0]/effective_resampling_factors[0];
        generated_upsampled_image_spacing[1] = measured_image_spacing[1]/effective_resampling_factors[1];
        generated_upsampled_image_spacing[2] = '''+('''1.''')*(im_dim==2)+('''measured_image_spacing[2]/effective_resampling_factors[2]''')*(im_dim==3)+''';

        generated_upsampled_image->SetOrigin(generated_upsampled_image_origin);
        generated_upsampled_image->SetSpacing(generated_upsampled_image_spacing);
        generated_upsampled_image->SetDimensions(generated_upsampled_image_dimensions);
        generated_upsampled_image->AllocateScalars(VTK_DOUBLE, 1);

        generated_fft_image->SetOrigin(measured_image_origin);
        generated_fft_image->SetSpacing(measured_image_spacing);
        generated_fft_image->SetDimensions(measured_image_dimensions);
        generated_fft_image->AllocateScalars(VTK_DOUBLE, 2);

        upsampled_residual_fft_image->SetOrigin(generated_upsampled_image_origin);
        upsampled_residual_fft_image->SetSpacing(generated_upsampled_image_spacing);
        upsampled_residual_fft_image->SetDimensions(generated_upsampled_image_dimensions);
        upsampled_residual_fft_image->AllocateScalars(VTK_DOUBLE, 2);
    }

    void update_measured_image
    (
        const char* filename
    )
    {'''+('''
        std::cout << "update_measured_image" << std::endl;''')*(verbose)+'''

        measured_reader->SetFileName(filename);
        measured_reader->Update();

        measured_fft_filter->Update();

        measured_gradient_filter->Update();
        measured_gradient_interpolator->Initialize(measured_gradient_image);
    }

    void init_mesh_and_disp
    (
        std::shared_ptr<dolfin::Mesh>     mesh_,
        std::shared_ptr<dolfin::Function> U_
    )
    {'''+('''
        std::cout << "init_mesh_and_disp" << std::endl;''')*(verbose)+'''

        mesh = mesh_;

        unsigned int n_points = mesh->num_vertices();
        unsigned int n_cells = mesh->num_cells();

        U = U_;

        // Points
        std::vector<double> dofs_coordinates = U->function_space()->tabulate_dof_coordinates();

        vtkSmartPointer<vtkPoints> ugrid_points = vtkSmartPointer<vtkPoints>::New();
        ugrid_points->SetNumberOfPoints(n_points);

        for (unsigned int k_point=0;
                          k_point<n_points;
                        ++k_point)
        {'''+('''
            ugrid_points->SetPoint(
                k_point,
                dofs_coordinates[4*k_point  ],
                dofs_coordinates[4*k_point+1],
                0.);''')*(im_dim==2)+('''
            ugrid_points->SetPoint(
                k_point,
                dofs_coordinates[9*k_point  ],
                dofs_coordinates[9*k_point+1],
                dofs_coordinates[9*k_point+2]);''')*(im_dim==3)+'''
        }
        ugrid->SetPoints(ugrid_points);

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
            auto cell_dofs = dofmap->cell_dofs(k_cell);'''+('''
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

        ugrid->SetCells(
            VTK_TRIANGLE,
            ugrid_cells);''')*(im_dim==2)+('''

        ugrid->SetCells(
            VTK_TETRA,
            ugrid_cells);''')*(im_dim==3)+'''

        // Disp
        vtkSmartPointer<vtkDoubleArray> ugrid_disp = vtkSmartPointer<vtkDoubleArray>::New();
        ugrid_disp->SetName("U");
        ugrid_disp->SetNumberOfComponents(3);
        ugrid_disp->SetNumberOfTuples(n_points);
        ugrid->GetPointData()->AddArray(ugrid_disp);
        ugrid->GetPointData()->SetActiveVectors("U");

        update_disp();
    }

    void update_disp()
    {'''+('''
        std::cout << "update_disp" << std::endl;''')*(verbose)+'''

        vtkSmartPointer<vtkDataArray> ugrid_disp = ugrid->GetPointData()->GetArray("U");
        unsigned int n_points = ugrid_disp->GetNumberOfTuples();
        for (unsigned int k_point=0;
                          k_point<n_points;
                        ++k_point)
        {'''+('''
            ugrid_disp->SetTuple3(
                k_point,
                (*U->vector())[2*k_point  ],
                (*U->vector())[2*k_point+1],
                0.);''')*(im_dim==2)+('''
            ugrid_disp->SetTuple3(
                k_point,
                (*U->vector())[3*k_point  ],
                (*U->vector())[3*k_point+1],
                (*U->vector())[3*k_point+2]);''')*(im_dim==3)+'''
        }
        ugrid->Modified();
    }

    // ========================================================================
    // ANALYTICAL PURE IMAGE METHODS
    // ========================================================================
    inline double get_pure_image(const double* X) const
    {'''+('''
        return 1.0;''')*(im_texture=="no")+('''
        double val = 1.0;
        for (unsigned int d=0;
                         d<n_dim;
                       ++d)
        {
            val *= std::abs(sin(M_PI*(X[d]-X0[d])/s));
        }
        val = pow(val, 1./n_dim);
        return val;''')*(im_texture=="tagging")+'''
    }

    inline void get_pure_image_grad(const double* X, double* grad_X) const
    {'''+('''
        for(unsigned int d=0;
                         d<n_dim;
                       ++d)
        {
            grad_X[d] = 0.0;
        }''')*(im_texture=="no")+('''
        // 1. Get the current pure image intensity I(X)
        double I_val = get_pure_image(X);
        
        // 2. Safety threshold to prevent NaN explosions at the tagging cusps
        double eps = 1e-12;
        
        for (unsigned int d=0;
                         d<n_dim;
                       ++d)
        {
            double u = M_PI * (X[d] - X0[d]) / s;
            
            double sin_u = std::sin(u);
            double cos_u = std::cos(u);
            
            // 3. Preserve the sign, but cap the denominator away from absolute zero
            double safe_sin = (sin_u > 0) ? std::max(sin_u, eps) : std::min(sin_u, -eps);
            
            // 4. The Exact Logarithmic Gradient
            grad_X[d] = I_val * (M_PI / (n_dim * s)) * (cos_u / safe_sin);
        }''')*(im_texture=="tagging")+'''
    }
    // ========================================================================
    
    void generate_upsampled_image()
    {'''+('''
        std::cout << "generate_upsampled_image" << std::endl;''')*(verbose)+'''

        warp_filter->Update();
        probe_filter->Update();

        vtkSmartPointer<vtkDataArray> probe_filter_mask = probe_filter->GetImageDataOutput()->GetPointData()->GetArray("vtkValidPointMask");
        vtkSmartPointer<vtkDataArray> probe_filter_disp = probe_filter->GetImageDataOutput()->GetPointData()->GetArray("U");

        vtkSmartPointer<vtkDataArray> generated_upsampled_scalars = generated_upsampled_image->GetPointData()->GetScalars();
        unsigned int n_points = generated_upsampled_image->GetNumberOfPoints();
        
        double m[1], I[1];
        for (unsigned int k_point=0;
                          k_point<n_points;
                        ++k_point)
        {
            probe_filter_mask->GetTuple(k_point, m);
            if (m[0] == 0)
            {
                I[0] = 0.;
            }
            else
            {
                generated_upsampled_image->GetPoint(k_point, x_3D.data());
                probe_filter_disp->GetTuple(k_point, ux_3D.data());
                X_3D = x_3D - ux_3D;
                I[0] = get_pure_image(X_3D.data());
            }
            generated_upsampled_scalars->SetTuple(k_point, I);
        }
        generated_upsampled_image->Modified();
    }

    void compute_downsampled_image()
    {'''+('''
        std::cout << "compute_downsampled_image" << std::endl;''')*(verbose)+'''

        generated_upsampled_fft_filter->Update();

        int N_lx = measured_image_dimensions[0];
        int N_ly = measured_image_dimensions[1];
        int N_lz = measured_image_dimensions[2];

        bool has_nyq_x = (N_lx % 2 == 0);
        bool has_nyq_y = (N_ly % 2 == 0);
        bool has_nyq_z = (N_lz % 2 == 0);

        int N_hx = generated_upsampled_image_dimensions[0];
        int N_hy = generated_upsampled_image_dimensions[1];
        int N_hz = generated_upsampled_image_dimensions[2];

        // Loop over the coarse grid
        for (int k_z = 0; k_z < N_lz; ++k_z)
        {
            bool is_nyq_z = has_nyq_z && (k_z == N_lz / 2);
            int base_k_z = (k_z <= N_lz / 2) ? k_z : k_z + (N_hz - N_lz);
            int alias_k_z = (N_hz - base_k_z) % N_hz;
            int z_indices[2] = {base_k_z, alias_k_z};
            int z_count = is_nyq_z ? 2 : 1;
                        
            for (int k_y = 0; k_y < N_ly; ++k_y)
            {
                bool is_nyq_y = has_nyq_y && (k_y == N_ly / 2);
                int base_k_y = (k_y <= N_ly / 2) ? k_y : k_y + (N_hy - N_ly);
                int alias_k_y = (N_hy - base_k_y) % N_hy;
                int y_indices[2] = {base_k_y, alias_k_y};
                int y_count = is_nyq_y ? 2 : 1;

                for (int k_x = 0; k_x < N_lx; ++k_x)
                {
                    bool is_nyq_x = has_nyq_x && (k_x == N_lx / 2);
                    int base_k_x = (k_x <= N_lx / 2) ? k_x : k_x + (N_hx - N_lx);
                    int alias_k_x = (N_hx - base_k_x) % N_hx;
                    int x_indices[2] = {base_k_x, alias_k_x};
                    int x_count = is_nyq_x ? 2 : 1;

                    double sum_r = 0.0;
                    double sum_i = 0.0;

                    // Explicitly iterate over the valid source indices for each dimension
                    for (int iz = 0; iz < z_count; ++iz)
                    {
                        for (int iy = 0; iy < y_count; ++iy)
                        {
                            for (int ix = 0; ix < x_count; ++ix)
                            {
                                sum_r += generated_upsampled_fft_image->GetScalarComponentAsDouble(x_indices[ix], y_indices[iy], z_indices[iz], 0);
                                sum_i += generated_upsampled_fft_image->GetScalarComponentAsDouble(x_indices[ix], y_indices[iy], z_indices[iz], 1);
                            }
                        }
                    }

                    // Normalize
                    sum_r /= effective_resampling_factor;
                    sum_i /= effective_resampling_factor;

                    // Set Output
                    generated_fft_image->SetScalarComponentFromDouble(k_x, k_y, k_z, 0, sum_r);
                    generated_fft_image->SetScalarComponentFromDouble(k_x, k_y, k_z, 1, sum_i);
                }
            }
        }
        generated_fft_image->Modified();

        generated_rfft_filter->Update();

        generated_extract_filter->Update();
    }

    void compute_upsampled_residual()
    {'''+('''
        std::cout << "compute_upsampled_residual" << std::endl;''')*(verbose)+'''

        // Computes the global Fourier adjoint swap (R -> R_tilde)
        int n_tuples_fine_fft = upsampled_residual_fft_image->GetPointData()->GetScalars()->GetNumberOfTuples();
        vtkDataArray* up_fft_ptr = upsampled_residual_fft_image->GetPointData()->GetScalars();
        for (int i = 0; i < n_tuples_fine_fft; ++i)
        {
            up_fft_ptr->SetComponent(i, 0, 0.0);
            up_fft_ptr->SetComponent(i, 1, 0.0);
        }

        int N_lx = measured_image_dimensions[0];
        int N_ly = measured_image_dimensions[1];
        int N_lz = measured_image_dimensions[2];
        bool has_nyq_x = (N_lx % 2 == 0);
        bool has_nyq_y = (N_ly % 2 == 0);
        bool has_nyq_z = (N_lz % 2 == 0);
        int N_hx = generated_upsampled_image_dimensions[0];
        int N_hy = generated_upsampled_image_dimensions[1];
        int N_hz = generated_upsampled_image_dimensions[2];

        for (int k_z = 0; k_z < N_lz; ++k_z)
        {
            bool is_nyq_z = has_nyq_z && (k_z == N_lz / 2);
            int base_k_z = (k_z <= N_lz / 2) ? k_z : k_z + (N_hz - N_lz);
            int alias_k_z = (N_hz - base_k_z) % N_hz;
            int z_indices[2] = {base_k_z, alias_k_z};
            int z_count = is_nyq_z ? 2 : 1;
            for (int k_y = 0; k_y < N_ly; ++k_y)
            {
                bool is_nyq_y = has_nyq_y && (k_y == N_ly / 2);
                int base_k_y = (k_y <= N_ly / 2) ? k_y : k_y + (N_hy - N_ly);
                int alias_k_y = (N_hy - base_k_y) % N_hy;
                int y_indices[2] = {base_k_y, alias_k_y};
                int y_count = is_nyq_y ? 2 : 1;
                for (int k_x = 0; k_x < N_lx; ++k_x)
                {
                    bool is_nyq_x = has_nyq_x && (k_x == N_lx / 2);
                    int base_k_x = (k_x <= N_lx / 2) ? k_x : k_x + (N_hx - N_lx);
                    int alias_k_x = (N_hx - base_k_x) % N_hx;
                    int x_indices[2] = {base_k_x, alias_k_x};
                    int x_count = is_nyq_x ? 2 : 1;
                    double r = residual_fft_image->GetScalarComponentAsDouble(k_x, k_y, k_z, 0) / effective_resampling_factor;
                    double i = residual_fft_image->GetScalarComponentAsDouble(k_x, k_y, k_z, 1) / effective_resampling_factor;
                    for (int iz = 0; iz < z_count; ++iz)
                    {
                        for (int iy = 0; iy < y_count; ++iy)
                        {
                            for (int ix = 0; ix < x_count; ++ix)
                            {
                                upsampled_residual_fft_image->SetScalarComponentFromDouble(x_indices[ix], y_indices[iy], z_indices[iz], 0, r);
                                upsampled_residual_fft_image->SetScalarComponentFromDouble(x_indices[ix], y_indices[iy], z_indices[iz], 1, i);
                            }
                        }
                    }
                }
            }
        }
        upsampled_residual_fft_image->Modified();

        // 2. IFFT -> R_tilde
        upsampled_residual_rfft_filter->Update();
        upsampled_residual_extract_filter->Update();
    }

    void update_generated_image()
    {'''+('''
        std::cout << "update_generated_image" << std::endl;''')*(verbose)+'''

        generate_upsampled_image();
        compute_downsampled_image();

        residual_subtract_filter->Update();
        residual_fft_filter->Update();

        compute_upsampled_residual();
        
        generated_interpolator->Initialize(generated_image);
        residual_interpolator->Initialize(residual_image);
        upsampled_residual_interpolator->Initialize(upsampled_residual_image);
    }

    vtkSmartPointer<vtkImageData> get_image_from_name
    (
        const char* image_name
    )
    {
        vtkSmartPointer<vtkImageData> image = vtkSmartPointer<vtkImageData>::New();
        if (strcmp(image_name, "measured") == 0)
        {
            image = measured_image;
        }
        else if (strcmp(image_name, "measured_fft") == 0)
        {
            image = measured_fft_image;
        }
        else if (strcmp(image_name, "upsampled") == 0)
        {
            image = generated_upsampled_image;
        }
        else if (strcmp(image_name, "upsampled_fft") == 0)
        {
            image = generated_upsampled_fft_image;
        }
        else if (strcmp(image_name, "generated_fft") == 0)
        {
            image = generated_fft_image;
        }
        else if (strcmp(image_name, "generated") == 0)
        {
            image = generated_image;
        }
        else if (strcmp(image_name, "probe") == 0)
        {
            image = probe_filter->GetImageDataOutput();
        }
        else
        {
            std::cout << "image_name (" << image_name << ") must be \\"measured\\", \\"upsampled\\", \\"generated\\" or their fft. Aborting." << std::endl;
            std::exit(0);
        }
        return image;
    }

    void write_image
    (
        const char* image_name,
        const char* filename
    )
    {'''+('''
        std::cout << "write_" << image_name << "_image" << std::endl;''')*(verbose)+'''

        vtkSmartPointer<vtkXMLImageDataWriter> writer = vtkSmartPointer<vtkXMLImageDataWriter>::New();
        writer->SetInputData(get_image_from_name(image_name));
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

    double compute_image_integral
    (
        const char* image_name
    )
    {'''+('''
        std::cout << "compute_" << image_name << "_image_integral" << std::endl;''')*(verbose)+'''

        vtkSmartPointer<vtkImageData> image = get_image_from_name(image_name);
        
        int* image_dimensions = image->GetDimensions();

        double sum = 0.;
        double val;
        for (int k_z = 0; k_z < image_dimensions[2]; ++k_z)
        {
         for (int k_y = 0; k_y < image_dimensions[1]; ++k_y)
         {
          for (int k_x = 0; k_x < image_dimensions[0]; ++k_x)
          {
            val = image->GetScalarComponentAsDouble(k_x, k_y, k_z, 0);
            sum += val;
          }
         }
        }
        // int N_voxels = image_dimensions[0] * image_dimensions[1] * image_dimensions[2];
        // sum /= N_voxels;

        double* image_spacing = image->GetSpacing();
        double voxel_volume = image_spacing[0] * image_spacing[1] * image_spacing[2];
        sum *= voxel_volume;'''+('''
        std::cout << "sum = " << sum << std::endl;''')*(verbose)+'''

        return sum;
    }

    double compute_image_norm
    (
        const char* image_name
    )
    {'''+('''
        std::cout << "compute " << image_name << " image_norm" << std::endl;''')*(verbose)+'''

        vtkSmartPointer<vtkImageData> image = get_image_from_name(image_name);'''+('''
        std::cout << "image = " << image << std::endl;''')*(verbose)+'''
        
        int* image_dimensions = image->GetDimensions();'''+('''
        std::cout << "image_dimensions = "
                  <<  image_dimensions[0] << " "
                  <<  image_dimensions[1] << " "
                  <<  image_dimensions[2] << std::endl;''')*(verbose)+'''

        double sum = 0.;
        double val;
        for (int k_z = 0; k_z < image_dimensions[2]; ++k_z)
        {
         for (int k_y = 0; k_y < image_dimensions[1]; ++k_y)
         {
          for (int k_x = 0; k_x < image_dimensions[0]; ++k_x)
          {
            val = image->GetScalarComponentAsDouble(k_x, k_y, k_z, 0);
            sum += pow(val, 2.0);
          }
         }
        }
        int N_voxels = image_dimensions[0] * image_dimensions[1] * image_dimensions[2];
        sum /= N_voxels;
        sum = pow(sum, 0.5);'''+('''
        std::cout << "sum = " << sum << std::endl;''')*(verbose)+'''

        return sum;
    }

    double compute_fourier_norm
    (
        const char* image_name
    )
    {'''+('''
        std::cout << "compute_" << image_name << "_image_integral" << std::endl;''')*(verbose)+'''

        vtkSmartPointer<vtkImageData> image = get_image_from_name(image_name);
        
        int* image_dimensions = image->GetDimensions();

        double sum = 0.;
        double val;
        for (int k_z = 0; k_z < image_dimensions[2]; ++k_z)
        {
         for (int k_y = 0; k_y < image_dimensions[1]; ++k_y)
         {
          for (int k_x = 0; k_x < image_dimensions[0]; ++k_x)
          {
            val = image->GetScalarComponentAsDouble(k_x, k_y, k_z, 0);
            sum += pow(val, 2.0);
            val = image->GetScalarComponentAsDouble(k_x, k_y, k_z, 1);
            sum += pow(val, 2.0);
          }
         }
        }
        int N_voxels = image_dimensions[0] * image_dimensions[1] * image_dimensions[2];
        sum /= N_voxels;
        sum /= N_voxels;
        sum = pow(sum, 0.5);'''+('''
        std::cout << "sum = " << sum << std::endl;''')*(verbose)+'''

        return sum;
    }

    double compute_image_energy()
    {'''+('''
        std::cout << "compute_image_energy" << std::endl;''')*(verbose)+'''

        double ener = 0., norm = 0.;
        double gen, mes, res;
        for (int k_z = 0; k_z < measured_image_dimensions[2]; ++k_z)
        {
         for (int k_y = 0; k_y < measured_image_dimensions[1]; ++k_y)
         {
          for (int k_x = 0; k_x < measured_image_dimensions[0]; ++k_x)
          {
            res = residual_image->GetScalarComponentAsDouble(k_x, k_y, k_z, 0);
            mes = measured_image->GetScalarComponentAsDouble(k_x, k_y, k_z, 0);
            ener += pow(res, 2.0);
            norm += pow(mes, 2.0);
            // gen = generated_image->GetScalarComponentAsDouble(k_x, k_y, k_z, 0);
            // mes = measured_image->GetScalarComponentAsDouble(k_x, k_y, k_z, 0);
            // ener += pow(gen - mes, 2.0);
            // norm += pow(mes      , 2.0);
          }
         }
        }
        // ener /= norm;

        // ener /= 2;
        // ener = pow(ener, 0.5);

        double voxel_volume = measured_image_spacing[0] * measured_image_spacing[1] * measured_image_spacing[2];
        ener *= voxel_volume/2;'''+('''
        std::cout << "ener = " << ener << std::endl;''')*(verbose)+'''

        return ener;
    }

    double compute_fourier_energy()
    {'''+('''
        std::cout << "compute_fourier_energy" << std::endl;''')*(verbose)+'''

        double ener = 0., norm = 0.;
        double gen, mes, res;
        for (int k_z = 0; k_z < measured_image_dimensions[2]; ++k_z)
        {
         for (int k_y = 0; k_y < measured_image_dimensions[1]; ++k_y)
         {
          for (int k_x = 0; k_x < measured_image_dimensions[0]; ++k_x)
          {
            res = residual_fft_image->GetScalarComponentAsDouble(k_x, k_y, k_z, 0);
            mes = measured_fft_image->GetScalarComponentAsDouble(k_x, k_y, k_z, 0);
            ener += pow(res, 2.0);
            norm += pow(mes, 2.0);

            res = residual_fft_image->GetScalarComponentAsDouble(k_x, k_y, k_z, 1);
            mes = measured_fft_image->GetScalarComponentAsDouble(k_x, k_y, k_z, 1);
            ener += pow(res, 2.0);
            norm += pow(mes, 2.0);

            // mes = measured_fft_image->GetScalarComponentAsDouble(k_x, k_y, k_z, 0);
            // gen = generated_fft_image->GetScalarComponentAsDouble(k_x, k_y, k_z, 0);
            // ener += pow(gen - mes, 2.0);
            // norm += pow(mes      , 2.0);

            // mes = measured_fft_image->GetScalarComponentAsDouble(k_x, k_y, k_z, 1);
            // gen = generated_fft_image->GetScalarComponentAsDouble(k_x, k_y, k_z, 1);
            // ener += pow(gen - mes, 2.0);
            // norm += pow(mes      , 2.0);
          }
         }
        }
        // ener /= norm;

        // ener /= 2;
        // ener = pow(ener, 0.5);

        double voxel_volume = measured_image_spacing[0] * measured_image_spacing[1] * measured_image_spacing[2];
        double N_voxels = measured_image_dimensions[0] * measured_image_dimensions[1] * measured_image_dimensions[2];
        ener *= voxel_volume / 2 / N_voxels;'''+('''
        std::cout << "ener = " << ener << std::endl;''')*(verbose)+'''

        return ener;
    }

    void eval
    (
        Eigen::Ref<      Eigen::VectorXd> expr,
        Eigen::Ref<const Eigen::VectorXd> X
    ) const override
    {'''+('''
        // std::cout << "X = " << X << std::endl;''')*(verbose)+'''
        U->eval(UX, X);'''+('''
        // std::cout << "UX = " << UX << std::endl;''')*(verbose)+('''

        x_3D.head<n_dim>() = X + UX;''')*(im_dim==2)+('''
        x_3D               = X + UX;''')*(im_dim==3)+('''
        // std::cout << "x_3D = " << x_3D << std::endl;''')*(verbose)+'''

        // Output Layout: [ I_gen(x(X)), R(x(X)), R_tilde(x(X)), I_gen0(X), Grad_Igen0(X), grad_Imes(x(X)) ]
        generated_interpolator->Interpolate(x_3D.data(), &expr[0]);
        residual_interpolator->Interpolate(x_3D.data(), &expr[1]);
        upsampled_residual_interpolator->Interpolate(x_3D.data(), &expr[2]);

        expr[3] = get_pure_image(X.data());
        get_pure_image_grad(X.data(), &expr[4]);
        
        double grad_Imes[3];
        measured_gradient_interpolator->Interpolate(x_3D.data(), grad_Imes);
        for(unsigned int d=0; d<n_dim; ++d) expr[4 + n_dim + d] = grad_Imes[d];'''+('''
        std::cout << "expr = " << expr << std::endl;''')*(verbose)+'''
    }
};

PYBIND11_MODULE(SIGNATURE, m)
{
    pybind11::class_<'''+name+''', std::shared_ptr<'''+name+'''>, dolfin::Expression>(m, "'''+name+'''")
    .def(pybind11::init<'''+('''const double&, ''')*(im_dim==2)+('''const double&, const double&, '''+('''const double&, ''')*(im_dim==3)+'''const double&, ''')*(im_texture=="tagging")+'''const char*, const double&>(), '''+('''pybind11::arg("Z") = 0., ''')*(im_dim==2)+('''pybind11::arg("X0") = 0., pybind11::arg("Y0") = 0., '''+('''pybind11::arg("Z0") = 0., ''')*(im_dim==3)+'''pybind11::arg("s") = 0.1, ''')*(im_texture=="tagging")+'''pybind11::arg("image_interpol_mode") = "linear", pybind11::arg("interpol_out_value") = 0.)
    .def("init_images", &'''+name+'''::init_images, pybind11::arg("filename"), pybind11::arg("resampling_factor_") = 1.)
    .def("update_measured_image", &'''+name+'''::update_measured_image, pybind11::arg("filename"))
    .def("init_mesh_and_disp", &'''+name+'''::init_mesh_and_disp, pybind11::arg("mesh_"), pybind11::arg("U_"))
    .def("update_disp", &'''+name+'''::update_disp)
    .def("update_generated_image", &'''+name+'''::update_generated_image)
    .def("write_image", &'''+name+'''::write_image, pybind11::arg("image_name"), pybind11::arg("filename"))
    .def("write_ugrid", &'''+name+'''::write_ugrid, pybind11::arg("filename"))
    .def("write_warp_ugrid", &'''+name+'''::write_warp_ugrid, pybind11::arg("filename"))
    .def("compute_image_integral", &'''+name+'''::compute_image_integral, pybind11::arg("image_name"))
    .def("compute_image_norm", &'''+name+'''::compute_image_norm, pybind11::arg("image_name"))
    .def("compute_fourier_norm", &'''+name+'''::compute_fourier_norm, pybind11::arg("image_name"))
    .def("compute_image_energy", &'''+name+'''::compute_image_energy)
    .def("compute_fourier_energy", &'''+name+'''::compute_fourier_energy);
}
'''
    # print(cpp)

    return name, cpp
