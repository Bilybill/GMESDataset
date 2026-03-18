// src/boundaryConditions3d.cpp
#include "../include/boundaryConditions3d.h"
#include <iostream>
#include <cmath>

void applyBoundaryConditions3D(const HostMesh3D &mesh, double freq, double rho_bg,
                               std::vector<std::complex<double>> &csrVal, 
                               std::vector<int> &csrColInd, 
                               const std::vector<int> &csrRowPtr,
                               std::vector<std::complex<double>> &b,
                               int polarization) 
{
    int NX = mesh.NX;
    int NY = mesh.NY;
    int NZ = mesh.NZ;

    int nEx = NX * (NY + 1) * (NZ + 1); 
    int nEy = (NX + 1) * NY * (NZ + 1); 

    int offset_X = 0;
    int offset_Y = nEx;
    int offset_Z = nEx + nEy;

    double mu = 4e-7 * M_PI;
    double w = 2.0 * M_PI * freq;
    double sigma = 1.0 / rho_bg;
    std::complex<double> neg_i(0.0, -1.0);
    std::complex<double> k_num = std::sqrt(neg_i * w * mu * sigma);

    b.assign(mesh.NEdges, std::complex<double>(0.0, 0.0));

    // ==========================================================
    // 核心修复：对称行-列消元法 (Symmetric Row-Col Elimination)
    // 完美保持 A = A^T，且不虚增 right-hand side (b) 的范数！
    // ==========================================================

    // 💡 预计算张量网格的累积 Z 坐标 (深度)
    std::vector<double> z_coords(NZ + 1, 0.0);
    for(int k=0; k<NZ; ++k) z_coords[k+1] = z_coords[k] + mesh.hz[k];

    std::vector<bool> is_boundary(mesh.NEdges, false);
    std::vector<std::complex<double>> bc_values(mesh.NEdges, {0.0, 0.0});

    // 1. 扫描并标记所有边界边，计算解析解
    // X 族边
    for (int k_idx = 0; k_idx <= NZ; ++k_idx) {
        for (int j = 0; j <= NY; ++j) {
            for (int i = 0; i < NX; ++i) {
                if (k_idx == 0 || k_idx == NZ || j == 0 || j == NY) {
                    int edge_id = offset_X + i + j * NX + k_idx * NX * (NY + 1);
                    is_boundary[edge_id] = true;
                    // 💡 使用累积深度 z_coords[k_idx]
                    if (polarization == 1) bc_values[edge_id] = std::exp(neg_i * k_num * z_coords[k_idx]);
                }
            }
        }
    }

    // Y 族边
    for (int k_idx = 0; k_idx <= NZ; ++k_idx) {
        for (int j = 0; j < NY; ++j) {
            for (int i = 0; i <= NX; ++i) {
                if (k_idx == 0 || k_idx == NZ || i == 0 || i == NX) {
                    int edge_id = offset_Y + i + j * (NX + 1) + k_idx * (NX + 1) * NY;
                    is_boundary[edge_id] = true;
                    // 💡 使用累积深度 z_coords[k_idx]
                    if (polarization == 2) bc_values[edge_id] = std::exp(neg_i * k_num * z_coords[k_idx]);
                }
            }
        }
    }

    // Z 族边
    for (int k_idx = 0; k_idx < NZ; ++k_idx) {
        for (int j = 0; j <= NY; ++j) {
            for (int i = 0; i <= NX; ++i) {
                if (i == 0 || i == NX || j == 0 || j == NY) {
                    int edge_id = offset_Z + i + j * (NX + 1) + k_idx * (NX + 1) * (NY + 1);
                    is_boundary[edge_id] = true;
                    bc_values[edge_id] = {0.0, 0.0}; // 1D 无垂直电场
                }
            }
        }
    }

    // 2. 消元步骤 A：处理内部节点，将边界值移至右端项，并清零边界列
    for (int i = 0; i < mesh.NEdges; ++i) {
        if (!is_boundary[i]) {
            for (int idx = csrRowPtr[i]; idx < csrRowPtr[i+1]; ++idx) {
                int col = csrColInd[idx];
                if (is_boundary[col]) {
                    b[i] -= csrVal[idx] * bc_values[col]; // b_I = b_I - A_IB * x_B
                    csrVal[idx] = {0.0, 0.0};             // 将列元素清零
                }
            }
        }
    }

    // 3. 消元步骤 B：处理边界节点，对角线赋1，非对角线（行）清零
    for (int i = 0; i < mesh.NEdges; ++i) {
        if (is_boundary[i]) {
            b[i] = bc_values[i];
            for (int idx = csrRowPtr[i]; idx < csrRowPtr[i+1]; ++idx) {
                if (csrColInd[idx] == i) {
                    csrVal[idx] = {1.0, 0.0}; // 对角线
                } else {
                    csrVal[idx] = {0.0, 0.0}; // 行清零
                }
            }
        }
    }

    // std::cout << "Applied 3D Boundary Conditions for Polarization " << polarization << " (Symmetric Elimination)" << std::endl;
}