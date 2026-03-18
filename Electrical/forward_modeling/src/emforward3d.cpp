// src/emforward3d.cpp
#include <iostream>
#include <fstream>
#include <vector>
#include <complex>
#include <cmath>
#include <chrono>
#include <algorithm>

#include "../include/mesh3d.h"
#include "../include/femAssemble3d.h"
#include "../include/boundaryConditions3d.h"
#include "../include/gpuIterativeSolver.h" 

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

            // 💡 自由度已经是电场强度 V/m，绝对不要除以 dx/dy/dz ！
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

            // 求旋度时除以局部的距离
            std::complex<double> dEy_dz = (Ey_z1 - Ey_z0) / dz;
            std::complex<double> dEx_dz = (Ex_z1 - Ex_z0) / dz;
            std::complex<double> dEz_dy = (Ez_y1 - Ez_y0) / dy;
            std::complex<double> dEz_dx = (Ez_x1 - Ez_x0) / dx;

            std::complex<double> Hx = (dEz_dy - dEy_dz) / neg_iwmu;
            std::complex<double> Hy = (dEx_dz - dEz_dx) / neg_iwmu;

            // 写入纯净的核心区数组
            fields[i_core + j_core * core_NX] = {Ex_center, Ey_center, Hx, Hy};
        }
    }
    return fields;
}

int main(int argc, char *argv[]) {
    if (argc < 8) {
        std::cerr << "Usage: " << argv[0] << " <rhofilename> <NX> <NY> <NZ> <dx> <dy> <dz> [f_min] [f_max] [bg_rho]\n";
        return 1;
    }
    
    std::string rhofilename = argv[1];
    int NX = std::atoi(argv[2]);
    int NY = std::atoi(argv[3]);
    int NZ = std::atoi(argv[4]);
    double dx = std::atof(argv[5]);
    double dy = std::atof(argv[6]);
    double dz = std::atof(argv[7]);
    
    double f_min = -1.0;
    double f_max = -1.0;
    double bg_rho_for_bc = -1.0; 

    if (argc >= 10) {
        f_min = std::atof(argv[8]);
        f_max = std::atof(argv[9]);
    }
    if (argc >= 11) bg_rho_for_bc = std::atof(argv[10]);

    std::cout << "3D MT Forward Modeling Started (Right-Preconditioned BiCGStab)...\n";
    std::cout << "Mesh: " << NX << "x" << NY << "x" << NZ << ", Cell: " << dx << "x" << dy << "x" << dz << " m\n";

    int NE = NX * NY * NZ;
    std::vector<double> rho_vec(NE, 100.0);

    FILE *rhoFile = fopen(rhofilename.c_str(), "rb");
    if (rhoFile) {
        std::vector<float> temp_rho(NE);
        size_t read_elements = fread(temp_rho.data(), sizeof(float), NE, rhoFile);
        if (read_elements == NE) {
            for(int i=0; i<NE; ++i) rho_vec[i] = temp_rho[i];
            std::cout << "Loaded 3D rho model successfully.\n";
        }
        fclose(rhoFile);
    } else {
        std::cerr << "Warning: Could not open " << rhofilename << ", using uniform 100 Ohm-m.\n";
        if (bg_rho_for_bc < 0.0) bg_rho_for_bc = 100.0;
    }

    if (bg_rho_for_bc < 0.0) {
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
        std::cout << "-> Auto-calculated Background Rho: " << bg_rho_for_bc << " Ohm-m\n";
    }

    // ========================================================
    // 💡 工业级黑魔法：在内存中自动构建外延 Padding
    // ========================================================
    int npad = 10;          // 往四周扩展 10 层
    double alpha = 1.4;     // 每层放大 1.4 倍
    
    int NX_pad = NX + 2 * npad;
    int NY_pad = NY + 2 * npad;
    int NZ_pad = NZ + npad; // 地表之上无空气层，只往下扩展 Z
    
    std::vector<double> hx(NX_pad), hy(NY_pad), hz(NZ_pad);
    for(int i=0; i<npad; i++) hx[npad - 1 - i] = dx * std::pow(alpha, i+1); // 左侧
    for(int i=0; i<NX; i++) hx[npad + i] = dx;                              // 核心区
    for(int i=0; i<npad; i++) hx[npad + NX + i] = dx * std::pow(alpha, i+1);// 右侧

    for(int i=0; i<npad; i++) hy[npad - 1 - i] = dy * std::pow(alpha, i+1); 
    for(int i=0; i<NY; i++) hy[npad + i] = dy;                              
    for(int i=0; i<npad; i++) hy[npad + NY + i] = dy * std::pow(alpha, i+1);

    for(int i=0; i<NZ; i++) hz[i] = dz;                                     // 核心区深度
    for(int i=0; i<npad; i++) hz[NZ + i] = dz * std::pow(alpha, i+1);       // 底部深层扩展

    // 填充 Padded 版本的电阻率模型，使用边界最外层的值向外延伸
    int NE_pad = NX_pad * NY_pad * NZ_pad;
    std::vector<double> rho_pad(NE_pad, bg_rho_for_bc);
    for(int k=0; k<NZ_pad; k++) {
        int orig_k = std::min(k, NZ - 1); // 超过底部的全部沿用最底层的值
        for(int j=0; j<NY_pad; j++) {
            int orig_j = std::min(std::max(j - npad, 0), NY - 1);
            for(int i=0; i<NX_pad; i++) {
                int orig_i = std::min(std::max(i - npad, 0), NX - 1);
                rho_pad[i + j*NX_pad + k*NX_pad*NY_pad] = rho_vec[orig_i + orig_j*NX + orig_k*NX*NY];
            }
        }
    }

    std::cout << "-> Auto-Padding Applied: Mesh expanded to " << NX_pad << "x" << NY_pad << "x" << NZ_pad << std::endl;
    HostMesh3D mesh = generateMesh3D(NX_pad, NY_pad, NZ_pad, hx, hy, hz);
    int NDoF = mesh.NEdges;

    if (f_min <= 0.0 || f_max <= 0.0) {
        double skin_depth_min = dz * 2.0;                
        double total_depth = NZ * dz;
        double skin_depth_max = total_depth * 10;         

        f_max = bg_rho_for_bc * std::pow(503.0 / skin_depth_min, 2.0);
        f_min = bg_rho_for_bc * std::pow(503.0 / skin_depth_max, 2.0);

        std::cout << "-> Auto-calculated Frequency Range:\n";
        std::cout << "   f_max: " << f_max << " Hz, f_min: " << f_min << " Hz\n";
    } else {
        std::cout << "-> User-provided Frequency Range:\n";
        if (f_min > f_max) std::swap(f_min, f_max); 
    }

    std::vector<double> freqs;
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
    
    int n_freqs = (int)freqs.size();
    std::cout << "   Range: [" << freqs.front() << " Hz  ...  " << freqs.back() << " Hz], Points = " << n_freqs << "\n";

    std::ofstream fout_res("apparent_res_3d.txt");
    std::ofstream fout_phs("phase_3d.txt");

    auto start_time = std::chrono::high_resolution_clock::now();

    // 依然保留极其重要的频率级联 (Warm Start)
    std::vector<std::complex<double>> x_pol1(NDoF, {0.0, 0.0});
    std::vector<std::complex<double>> x_pol2(NDoF, {0.0, 0.0});

    for (int ff = 0; ff < n_freqs; ff++) {
        double freq = freqs[ff];
        std::cout << "\n=== Processing Frequency " << ff + 1 << "/" << n_freqs << ": " << freq << " Hz ===" << std::endl;
        
        double w = 2.0 * M_PI * freq;
        double mu = 4e-7 * M_PI;

        std::vector<std::complex<double>> csrVal_base;
        std::vector<int> csrColInd, csrRowPtr;
        femAssemble3D_Matrix(mesh, freq, rho_pad, csrVal_base, csrColInd, csrRowPtr); // 💡 传入 rho_pad

        // --- 极化 1 (Ex主导) ---
        std::vector<std::complex<double>> csrVal_pol1 = csrVal_base; 
        std::vector<std::complex<double>> b_pol1(NDoF);
        applyBoundaryConditions3D(mesh, freq, bg_rho_for_bc, csrVal_pol1, csrColInd, csrRowPtr, b_pol1, 1);
        
        std::cout << "  -> Solving Polarization 1 (Ex) using BiCGStab..." << std::endl;
        // 核心修改：Tolerance 调整为 1e-7，截断无效震荡
        gpu_complex_solver_bicgstab(NDoF, csrRowPtr, csrColInd, csrVal_pol1, b_pol1, x_pol1, 1e-7, 50000);
        auto fields_pol1 = computeSurfaceFields(mesh, freq, x_pol1, NX, NY, npad);

        // --- 极化 2 (Ey主导) ---
        std::vector<std::complex<double>> csrVal_pol2 = csrVal_base; 
        std::vector<std::complex<double>> b_pol2(NDoF);
        applyBoundaryConditions3D(mesh, freq, bg_rho_for_bc, csrVal_pol2, csrColInd, csrRowPtr, b_pol2, 2);
        
        std::cout << "  -> Solving Polarization 2 (Ey) using BiCGStab..." << std::endl;
        gpu_complex_solver_bicgstab(NDoF, csrRowPtr, csrColInd, csrVal_pol2, b_pol2, x_pol2, 1e-7, 50000);
        auto fields_pol2 = computeSurfaceFields(mesh, freq, x_pol2, NX, NY, npad);

        // 计算张量阻抗与视电阻率
        for (int i = 0; i < NX * NY; ++i) {
            auto f1 = fields_pol1[i];
            auto f2 = fields_pol2[i];

            std::complex<double> detH = f1.Hx * f2.Hy - f2.Hx * f1.Hy;
            if (std::abs(detH) < 1e-30) continue;

            std::complex<double> Zxy = (f2.Ex * f1.Hx - f1.Ex * f2.Hx) / detH;
            std::complex<double> Zyx = (f1.Ey * f2.Hy - f2.Ey * f1.Hy) / detH;

            double rho_xy = std::norm(Zxy) / (w * mu); 
            double rho_yx = std::norm(Zyx) / (w * mu);
            double phi_xy = std::atan2(Zxy.imag(), Zxy.real()) * 180.0 / M_PI;
            double phi_yx = std::atan2(Zyx.imag(), Zyx.real()) * 180.0 / M_PI;

            fout_res << freq << " " << i << " " << rho_xy << " " << rho_yx << "\n";
            fout_phs << freq << " " << i << " " << phi_xy << " " << phi_yx << "\n";
        }
    }

    fout_res.close();
    fout_phs.close();

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    std::cout << "\nAll done! Total time: " << elapsed.count() << " s\n";
    return 0;
}