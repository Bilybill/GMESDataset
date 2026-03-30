#include <torch/extension.h>
#include <vector>
#include <complex>
#include <cmath>
#include <iostream>
#include <chrono>
#include <algorithm>
#include <cuda_runtime.h>
#include <cuComplex.h>

#include "include/mesh3d.h"
#include "include/femAssemble3d.h"
#include "include/boundaryConditions3d.h"
#include "include/gpuComplexSolver.h"
#include "include/gpuIterativeSolver.h"

struct SurfaceField {
    std::complex<double> Ex, Ey, Hx, Hy;
};

std::vector<SurfaceField> computeSurfaceFields(const HostMesh3D& mesh, double freq,
                                               const std::vector<std::complex<double>>& x,
                                               int core_NX, int core_NY, int npad) {
    double mu = 4e-7 * M_PI;
    double w = 2.0 * M_PI * freq;
    std::complex<double> neg_iwmu(0.0, -w * mu);
    
    std::vector<SurfaceField> fields(core_NX * core_NY); // 只返回核心区大小！

    int k = 0; // 地表层
    for (int j_core = 0; j_core < core_NY; ++j_core) {
        for (int i_core = 0; i_core < core_NX; ++i_core) {
            
            // 映射到加了 Padding 后的全局网格索引
            int i = i_core + npad;
            int j = j_core + npad;
            int elemIdx = i + j * mesh.NX + k * mesh.NX * mesh.NY;
            
            // 提取当地的网格尺寸
            double dx = mesh.hx[i];
            double dy = mesh.hy[j];
            double dz = mesh.hz[k];
            
            int e0 = mesh.ME_Edges[12 * elemIdx + 0]; 
            int e1 = mesh.ME_Edges[12 * elemIdx + 1]; 
            int e2 = mesh.ME_Edges[12 * elemIdx + 2]; 
            int e3 = mesh.ME_Edges[12 * elemIdx + 3]; 
            int e4 = mesh.ME_Edges[12 * elemIdx + 4]; 
            int e5 = mesh.ME_Edges[12 * elemIdx + 5]; 
            int e6 = mesh.ME_Edges[12 * elemIdx + 6]; 
            int e7 = mesh.ME_Edges[12 * elemIdx + 7]; 
            int e8  = mesh.ME_Edges[12 * elemIdx + 8];  
            int e9  = mesh.ME_Edges[12 * elemIdx + 9];  
            int e10 = mesh.ME_Edges[12 * elemIdx + 10]; 
            int e11 = mesh.ME_Edges[12 * elemIdx + 11]; 

            std::complex<double> Ex_z0 = (x[e0] + x[e2]) / 2.0;
            std::complex<double> Ey_z0 = (x[e1] + x[e3]) / 2.0;
            std::complex<double> Ex_z1 = (x[e4] + x[e6]) / 2.0;
            std::complex<double> Ey_z1 = (x[e5] + x[e7]) / 2.0;
            std::complex<double> Ez_y0 = (x[e8] + x[e9]) / 2.0;
            std::complex<double> Ez_y1 = (x[e10] + x[e11]) / 2.0;
            std::complex<double> Ez_x0 = (x[e8] + x[e11]) / 2.0;
            std::complex<double> Ez_x1 = (x[e9] + x[e10]) / 2.0;

            std::complex<double> Ex_center = (Ex_z0 + Ex_z1) / 2.0;
            std::complex<double> Ey_center = (Ey_z0 + Ey_z1) / 2.0;

            std::complex<double> dEy_dz = (Ey_z1 - Ey_z0) / dz;
            std::complex<double> dEx_dz = (Ex_z1 - Ex_z0) / dz;
            std::complex<double> dEz_dy = (Ez_y1 - Ez_y0) / dy;
            std::complex<double> dEz_dx = (Ez_x1 - Ez_x0) / dx;

            std::complex<double> Hx = (dEz_dy - dEy_dz) / neg_iwmu;
            std::complex<double> Hy = (dEx_dz - dEz_dx) / neg_iwmu;

            fields[i_core + j_core * core_NX] = {Ex_center, Ey_center, Hx, Hy};
        }
    }
    return fields;
}

extern "C" void gpu_femAssemble3D_Matrix(
    int NE, int NX, int NY,
    const double* d_hx, const double* d_hy, const double* d_hz,
    const double* d_rho, double freq,
    const int* d_ME_Edges,
    const int* d_csrRowPtr, const int* d_csrColInd, 
    cuDoubleComplex* d_csrVal, int nnz);

std::tuple<torch::Tensor, torch::Tensor> compute_mt_3d(torch::Tensor rho_tensor, double dx, double dy, double dz, std::vector<double> freqs) {
    TORCH_CHECK(rho_tensor.is_contiguous(), "rho_tensor must be contiguous");
    int device_count = 0;
    cudaError_t cuda_status = cudaGetDeviceCount(&device_count);
    TORCH_CHECK(
        cuda_status == cudaSuccess && device_count > 0,
        "mt_forward_cuda requires a working CUDA runtime and at least one visible GPU. "
        "cudaGetDeviceCount failed with: ", cudaGetErrorString(cuda_status)
    );
    double* rho_ptr = rho_tensor.data_ptr<double>();
    
    int NX = rho_tensor.size(0);
    int NY = rho_tensor.size(1);
    int NZ = rho_tensor.size(2);
    
    int NE = NX * NY * NZ;
    std::vector<double> rho_vec(NE);
    for(int x=0; x<NX; ++x) {
        for(int y=0; y<NY; ++y) {
            for(int z=0; z<NZ; ++z) {
                rho_vec[x + y*NX + z*NX*NY] = rho_ptr[x * (NY * NZ) + y * NZ + z];
            }
        }
    }

    double bg_rho_for_bc = -1.0;
    double sum_rho = 0.0;
    int count_boundary = 0;
    for (int k = 0; k < NZ; ++k) {
        for (int j = 0; j < NY; ++j) {
            for (int i = 0; i < NX; ++i) {
                if (i == 0 || i == NX - 1 || j == 0 || j == NY - 1 || k == 0 || k == NZ - 1) {
                    int idx = i + j * NX + k * NX * NY;
                    sum_rho += rho_vec[idx];
                    count_boundary++;
                }
            }
        }
    }
    bg_rho_for_bc = sum_rho / count_boundary;

    int npad = 10;
    double alpha = 1.4;
    
    int NX_pad = NX + 2 * npad;
    int NY_pad = NY + 2 * npad;
    int NZ_pad = NZ + npad; 
    
    std::vector<double> hx(NX_pad), hy(NY_pad), hz(NZ_pad);
    for(int i=0; i<npad; i++) hx[npad - 1 - i] = dx * std::pow(alpha, i+1); 
    for(int i=0; i<NX; i++) hx[npad + i] = dx;                              
    for(int i=0; i<npad; i++) hx[npad + NX + i] = dx * std::pow(alpha, i+1);
    
    for(int i=0; i<npad; i++) hy[npad - 1 - i] = dy * std::pow(alpha, i+1); 
    for(int i=0; i<NY; i++) hy[npad + i] = dy;                              
    for(int i=0; i<npad; i++) hy[npad + NY + i] = dy * std::pow(alpha, i+1);

    for(int i=0; i<NZ; i++) hz[i] = dz;                                     
    for(int i=0; i<npad; i++) hz[NZ + i] = dz * std::pow(alpha, i+1);       
    
    int NE_pad = NX_pad * NY_pad * NZ_pad;
    std::vector<double> rho_pad(NE_pad, bg_rho_for_bc);
    for(int k=0; k<NZ_pad; k++) {
        int orig_k = std::min(k, NZ - 1); 
        for(int j=0; j<NY_pad; j++) {
            int orig_j = std::min(std::max(j - npad, 0), NY - 1);
            for(int i=0; i<NX_pad; i++) {
                int orig_i = std::min(std::max(i - npad, 0), NX - 1);
                rho_pad[i + j*NX_pad + k*NX_pad*NY_pad] = rho_vec[orig_i + orig_j*NX + orig_k*NX*NY];
            }
        }
    }

    HostMesh3D mesh = generateMesh3D(NX_pad, NY_pad, NZ_pad, hx, hy, hz);
    int NDoF = mesh.NEdges;
    
    if (freqs.empty()) {
        double skin_depth_min = dz * 2.0;
        double total_depth = NZ * dz;
        double skin_depth_max = total_depth / 3.0;
        double f_max = bg_rho_for_bc * std::pow(503.0 / skin_depth_min, 2.0);
        double f_min = bg_rho_for_bc * std::pow(503.0 / skin_depth_max, 2.0);

        std::vector<double> phoenix_coeffs = {8.0, 6.0, 4.0, 3.0, 2.0, 1.5, 1.0};
        int max_power = std::ceil(std::log10(f_max));
        int min_power = std::floor(std::log10(f_min));

        for (int p = max_power; p >= min_power; --p) {
            double power_of_10 = std::pow(10.0, p);
            for (double coeff : phoenix_coeffs) {
                double current_f = coeff * power_of_10;
                if (current_f <= f_max && current_f >= f_min) {
                    freqs.push_back(current_f);
                }
            }
        }

        if (freqs.empty()) {
            freqs.push_back(f_max);
            freqs.push_back(f_min);
        }
    }

    int n_freqs = (int)freqs.size();
    
    auto options = torch::TensorOptions().dtype(torch::kFloat64);
    // Keep GMESDataset's repository-wide convention: [n_freq, NX, NY, 2].
    torch::Tensor app_res = torch::empty({n_freqs, NX, NY, 2}, options);
    torch::Tensor phase = torch::empty({n_freqs, NX, NY, 2}, options);
    
    double* res_ptr = app_res.data_ptr<double>();
    double* phs_ptr = phase.data_ptr<double>();

    std::cout << "3D MT Forward Modeling Started (Right-Preconditioned BiCGStab)...\n";
    std::cout << "Mesh: " << NX << "x" << NY << "x" << NZ << ", Cell: " << dx << "x" << dy << "x" << dz << " m\n";
    std::cout << "-> Auto-calculated Background Rho: " << bg_rho_for_bc << " Ohm-m\n";
    std::cout << "-> Auto-Padding Applied: Mesh expanded to " << NX_pad << "x" << NY_pad << "x" << NZ_pad << std::endl;
    std::cout << "   Range: [" << freqs.front() << " Hz  ...  " << freqs.back() << " Hz], Points = " << n_freqs << "\n";

    std::vector<std::complex<double>> x_pol1(NDoF, {0.0, 0.0});
    std::vector<std::complex<double>> x_pol2(NDoF, {0.0, 0.0});

    // 提前获取静态的 CSR 结构
    std::vector<std::complex<double>> dummy_csrVal;
    std::vector<int> h_csrColInd, h_csrRowPtr;
    femAssemble3D_Matrix(mesh, 1.0, rho_pad, dummy_csrVal, h_csrColInd, h_csrRowPtr);
    int nnz = dummy_csrVal.size();

    // ==========================================
    // 分配 GPU 长驻显存：坐标、拓扑与 CSR 结构
    // ==========================================
    double *d_hx, *d_hy, *d_hz, *d_rho;
    int *d_ME_Edges, *d_csrRowPtr, *d_csrColInd;
    cuDoubleComplex *d_csrVal;

    cudaMalloc(&d_hx, mesh.NX * sizeof(double));
    cudaMalloc(&d_hy, mesh.NY * sizeof(double));
    cudaMalloc(&d_hz, mesh.NZ * sizeof(double));
    cudaMalloc(&d_rho, mesh.NE * sizeof(double));
    cudaMalloc(&d_ME_Edges, mesh.ME_Edges.size() * sizeof(int));
    cudaMalloc(&d_csrRowPtr, h_csrRowPtr.size() * sizeof(int));
    cudaMalloc(&d_csrColInd, h_csrColInd.size() * sizeof(int));
    cudaMalloc(&d_csrVal, nnz * sizeof(cuDoubleComplex));

    cudaMemcpy(d_hx, mesh.hx.data(), mesh.NX * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_hy, mesh.hy.data(), mesh.NY * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_hz, mesh.hz.data(), mesh.NZ * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rho, rho_pad.data(), mesh.NE * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ME_Edges, mesh.ME_Edges.data(), mesh.ME_Edges.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_csrRowPtr, h_csrRowPtr.data(), h_csrRowPtr.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_csrColInd, h_csrColInd.data(), h_csrColInd.size() * sizeof(int), cudaMemcpyHostToDevice);

    GpuBiCGStabContext* solverContext = create_gpu_bicgstab_context(NDoF, nnz);
    BoundaryConditionPlan* boundaryPlan = createBoundaryConditionPlan3D(mesh, h_csrColInd, h_csrRowPtr);
    std::vector<int> modifiedValueIndices;
    getBoundaryConditionModifiedValueIndices3D(boundaryPlan, modifiedValueIndices);
    std::vector<std::complex<double>> csrVal_work(nnz);
    std::vector<std::complex<double>> b_work(NDoF);

    for (int ff = 0; ff < n_freqs; ff++) {
        double freq = freqs[ff];
        std::cout << "\n=== Processing Frequency " << ff + 1 << "/" << n_freqs << ": " << freq << " Hz ===" << std::endl;
        
        double w = 2.0 * M_PI * freq;
        double mu = 4e-7 * M_PI;

        // 调用 GPU 组装
        gpu_femAssemble3D_Matrix(mesh.NE, mesh.NX, mesh.NY, d_hx, d_hy, d_hz, d_rho, freq, d_ME_Edges, d_csrRowPtr, d_csrColInd, d_csrVal, nnz);
        
        std::vector<std::complex<double>> csrVal_base(nnz);
        cudaMemcpy(csrVal_base.data(), d_csrVal, nnz * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
        csrVal_work = csrVal_base;
        
        // 我们利用上面 CPU 版本算好的 h_csrColInd, h_csrRowPtr
        std::vector<int>& csrColInd = h_csrColInd;
        std::vector<int>& csrRowPtr = h_csrRowPtr;
        
        applyBoundaryConditions3D(mesh, freq, bg_rho_for_bc, csrVal_work, csrColInd, csrRowPtr, b_work, 1, boundaryPlan);
        std::cout << "  -> Solving Polarization 1 (Ex) using BiCGStab..." << std::endl;
        SolverStats pol1Stats;
        gpu_complex_solver_bicgstab(NDoF, csrRowPtr, csrColInd, csrVal_work, b_work, x_pol1, 1e-7, 50000, solverContext, &pol1Stats);
        std::cout << "     Polarization 1 stats: iterations=" << pol1Stats.iterations
                  << ", residual=" << pol1Stats.finalResidual << std::endl;
        auto fields_pol1 = computeSurfaceFields(mesh, freq, x_pol1, NX, NY, npad);
        
        for (int idx : modifiedValueIndices) {
            csrVal_work[idx] = csrVal_base[idx];
        }
        applyBoundaryConditions3D(mesh, freq, bg_rho_for_bc, csrVal_work, csrColInd, csrRowPtr, b_work, 2, boundaryPlan);
        std::cout << "  -> Solving Polarization 2 (Ey) using BiCGStab..." << std::endl;
        SolverStats pol2Stats;
        gpu_complex_solver_bicgstab(NDoF, csrRowPtr, csrColInd, csrVal_work, b_work, x_pol2, 1e-7, 50000, solverContext, &pol2Stats);
        std::cout << "     Polarization 2 stats: iterations=" << pol2Stats.iterations
                  << ", residual=" << pol2Stats.finalResidual << std::endl;
        auto fields_pol2 = computeSurfaceFields(mesh, freq, x_pol2, NX, NY, npad);
        
        for (int j = 0; j < NY; ++j) {
            for (int i = 0; i < NX; ++i) {
                int flat_idx = i + j * NX;
                auto f1 = fields_pol1[flat_idx];
                auto f2 = fields_pol2[flat_idx];

                std::complex<double> detH = f1.Hx * f2.Hy - f2.Hx * f1.Hy;
                double rho_xy = 0, rho_yx = 0, phi_xy = 0, phi_yx = 0;
                
                if (std::abs(detH) > 1e-30) {
                    std::complex<double> Zxy = (f2.Ex * f1.Hx - f1.Ex * f2.Hx) / detH;
                    std::complex<double> Zyx = (f1.Ey * f2.Hy - f2.Ey * f1.Hy) / detH;
                    
                    rho_xy = std::norm(Zxy) / (w * mu); 
                    rho_yx = std::norm(Zyx) / (w * mu);
                    phi_xy = std::atan2(Zxy.imag(), Zxy.real()) * 180.0 / M_PI;
                    phi_yx = std::atan2(Zyx.imag(), Zyx.real()) * 180.0 / M_PI;
                }
                
                int out_idx = (ff * NX * NY + i * NY + j) * 2;
                res_ptr[out_idx + 0] = rho_xy;
                res_ptr[out_idx + 1] = rho_yx;
                phs_ptr[out_idx + 0] = phi_xy;
                phs_ptr[out_idx + 1] = phi_yx;
            }
        }
    }

    cudaFree(d_hx);
    cudaFree(d_hy);
    cudaFree(d_hz);
    cudaFree(d_rho);
    cudaFree(d_ME_Edges);
    cudaFree(d_csrRowPtr);
    cudaFree(d_csrColInd);
    cudaFree(d_csrVal);
    destroyBoundaryConditionPlan3D(boundaryPlan);
    destroy_gpu_bicgstab_context(solverContext);

    return std::make_tuple(app_res, phase);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("compute_mt_3d", &compute_mt_3d, "MT 3D Forward Modeling");
}
