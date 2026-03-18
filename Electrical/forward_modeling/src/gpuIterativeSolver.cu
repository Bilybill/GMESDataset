// src/gpuIterativeSolver.cu
#include "../include/gpuIterativeSolver.h"
#include <cuda_runtime.h>
#include <cusparse.h>
#include <cublas_v2.h>
#include <cuComplex.h>
#include <iostream>
#include <cmath>

__global__ void complexElementWiseMul(int n, const cuDoubleComplex* a, const cuDoubleComplex* b, cuDoubleComplex* c) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = cuCmul(a[idx], b[idx]);
    }
}

inline cuDoubleComplex to_cuDoubleComplex(const std::complex<double>& c) {
    return make_cuDoubleComplex(c.real(), c.imag());
}

bool gpu_complex_solver_bicgstab(int N,
                                 const std::vector<int> &h_csrRowPtr,
                                 const std::vector<int> &h_csrColInd,
                                 const std::vector<std::complex<double>> &h_csrVal,
                                 const std::vector<std::complex<double>> &h_b,
                                 std::vector<std::complex<double>> &x,
                                 double tol,
                                 int maxIter)
{
    // ==========================================================
    // 工业级 Right-Preconditioned BiCGStab
    // 利用混合内积 (Zdotu / Zdotc) 完美镇压低频复对称矩阵的震荡！
    // ==========================================================
    
    int nnz = static_cast<int>(h_csrVal.size());

    cublasHandle_t cublasH = NULL;
    cusparseHandle_t cusparseH = NULL;
    cublasCreate(&cublasH);
    cusparseCreate(&cusparseH);

    // 1. 提取对角线，计算 Jacobi 预条件子 (M_inv = 1 / diag(A))
    std::vector<cuDoubleComplex> h_M_inv(N);
    for (int i = 0; i < N; ++i) {
        cuDoubleComplex diag = make_cuDoubleComplex(1.0, 0.0);
        for (int j = h_csrRowPtr[i]; j < h_csrRowPtr[i+1]; ++j) {
            if (h_csrColInd[j] == i) {
                diag = to_cuDoubleComplex(h_csrVal[j]);
                break;
            }
        }
        double mag_sq = cuCreal(diag)*cuCreal(diag) + cuCimag(diag)*cuCimag(diag);
        if (mag_sq > 1e-30) {
            h_M_inv[i] = make_cuDoubleComplex(cuCreal(diag)/mag_sq, -cuCimag(diag)/mag_sq);
        } else {
            h_M_inv[i] = make_cuDoubleComplex(1.0, 0.0);
        }
    }

    // 2. 分配显存
    int *d_csrRowPtr, *d_csrColInd;
    cuDoubleComplex *d_csrVal, *d_b, *d_x, *d_Minv;
    cuDoubleComplex *d_r, *d_r0_hat, *d_p, *d_v, *d_s, *d_t, *d_yp, *d_ys;

    cudaMalloc((void**)&d_csrRowPtr, (N + 1) * sizeof(int));
    cudaMalloc((void**)&d_csrColInd, nnz * sizeof(int));
    cudaMalloc((void**)&d_csrVal, nnz * sizeof(cuDoubleComplex));
    cudaMalloc((void**)&d_b, N * sizeof(cuDoubleComplex));
    cudaMalloc((void**)&d_x, N * sizeof(cuDoubleComplex));
    cudaMalloc((void**)&d_Minv, N * sizeof(cuDoubleComplex));
    
    cudaMalloc((void**)&d_r, N * sizeof(cuDoubleComplex));
    cudaMalloc((void**)&d_r0_hat, N * sizeof(cuDoubleComplex));
    cudaMalloc((void**)&d_p, N * sizeof(cuDoubleComplex));
    cudaMalloc((void**)&d_v, N * sizeof(cuDoubleComplex));
    cudaMalloc((void**)&d_s, N * sizeof(cuDoubleComplex));
    cudaMalloc((void**)&d_t, N * sizeof(cuDoubleComplex));
    cudaMalloc((void**)&d_yp, N * sizeof(cuDoubleComplex));
    cudaMalloc((void**)&d_ys, N * sizeof(cuDoubleComplex));

    cudaMemcpy(d_csrRowPtr, h_csrRowPtr.data(), (N + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_csrColInd, h_csrColInd.data(), nnz * sizeof(int), cudaMemcpyHostToDevice);
    
    std::vector<cuDoubleComplex> c_csrVal(nnz), c_b(N), c_x(N);
    for (int i = 0; i < nnz; i++) c_csrVal[i] = to_cuDoubleComplex(h_csrVal[i]);
    for (int i = 0; i < N; i++) {
        c_b[i] = to_cuDoubleComplex(h_b[i]);
        c_x[i] = to_cuDoubleComplex(x[i]); 
    }

    cudaMemcpy(d_csrVal, c_csrVal.data(), nnz * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, c_b.data(), N * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, c_x.data(), N * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice); 
    cudaMemcpy(d_Minv, h_M_inv.data(), N * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);

    cuDoubleComplex alpha_c = make_cuDoubleComplex(1.0, 0.0);
    cuDoubleComplex beta_c  = make_cuDoubleComplex(0.0, 0.0);

    // 3. 配置 cuSPARSE SpMV
    cusparseSpMatDescr_t matA;
    cusparseCreateCsr(&matA, N, N, nnz, d_csrRowPtr, d_csrColInd, d_csrVal,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_C_64F);

    size_t bufferSize = 0;
    cusparseDnVecDescr_t vecX, vecY;
    cusparseCreateDnVec(&vecX, N, d_p, CUDA_C_64F);
    cusparseCreateDnVec(&vecY, N, d_t, CUDA_C_64F);
    cusparseSpMV_bufferSize(cusparseH, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha_c, matA, vecX, &beta_c, vecY, CUDA_C_64F, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize);
    
    void* dBuffer = NULL;
    cudaMalloc(&dBuffer, bufferSize);
    cusparseDestroyDnVec(vecX);
    cusparseDestroyDnVec(vecY);

    auto SpMV_A = [&](cuDoubleComplex* in, cuDoubleComplex* out) {
        cusparseDnVecDescr_t vIn, vOut;
        cusparseCreateDnVec(&vIn, N, in, CUDA_C_64F);
        cusparseCreateDnVec(&vOut, N, out, CUDA_C_64F);
        cusparseSpMV(cusparseH, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha_c, matA, vIn, &beta_c, vOut, CUDA_C_64F, CUSPARSE_SPMV_ALG_DEFAULT, dBuffer);
        cusparseDestroyDnVec(vIn);
        cusparseDestroyDnVec(vOut);
    };

    // ==========================================
    // 4. 执行 Right-Preconditioned BiCGStab
    // ==========================================
    double norm_b, res_norm;
    cublasDznrm2(cublasH, N, d_b, 1, &norm_b);
    if(norm_b == 0.0) norm_b = 1.0;

    cuDoubleComplex rho_prev = make_cuDoubleComplex(1.0, 0.0);
    cuDoubleComplex alpha = make_cuDoubleComplex(1.0, 0.0);
    cuDoubleComplex omega = make_cuDoubleComplex(1.0, 0.0);

    cudaMemset(d_p, 0, N * sizeof(cuDoubleComplex));
    cudaMemset(d_v, 0, N * sizeof(cuDoubleComplex));

    // 计算初始残差: r0 = b - A * x0
    cudaMemcpy(d_r, d_b, N * sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice);
    cublasDznrm2(cublasH, N, d_x, 1, &res_norm);
    if (res_norm > 0.0) { // Warm Start
        SpMV_A(d_x, d_t);
        cuDoubleComplex neg_one = make_cuDoubleComplex(-1.0, 0.0);
        cublasZaxpy(cublasH, N, &neg_one, d_t, 1, d_r, 1);
    }
    
    // r0_hat = r0 (固定投影向量)
    cudaMemcpy(d_r0_hat, d_r, N * sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice);

    cublasDznrm2(cublasH, N, d_r, 1, &res_norm);
    if (res_norm / norm_b < tol) {
        std::cout << "  -> BiCGStab converged immediately (Warm Start). Final res: " << res_norm / norm_b << std::endl;
        return true; 
    }

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    int iter = 0;
    bool converged = false;

    for (iter = 1; iter <= maxIter; ++iter) {
        cuDoubleComplex rho;
        // ⚠️【极其核心】：BiCG 投影步骤，必须无共轭 (Zdotu) 维持复对称性！
        cublasZdotu(cublasH, N, d_r0_hat, 1, d_r, 1, &rho); 
        
        if (cuCabs(rho) < 1e-30) break; // Breakdown

        cuDoubleComplex beta = cuCmul(cuCdiv(rho, rho_prev), cuCdiv(alpha, omega));

        // p = r + beta * (p - omega * v)
        cuDoubleComplex neg_omega = make_cuDoubleComplex(-cuCreal(omega), -cuCimag(omega));
        cublasZaxpy(cublasH, N, &neg_omega, d_v, 1, d_p, 1);
        cublasZscal(cublasH, N, &beta, d_p, 1);
        cuDoubleComplex one_c = make_cuDoubleComplex(1.0, 0.0);
        cublasZaxpy(cublasH, N, &one_c, d_r, 1, d_p, 1);

        // 预处理: yp = M_inv * p
        complexElementWiseMul<<<blocks, threads>>>(N, d_Minv, d_p, d_yp);
        
        // v = A * yp
        SpMV_A(d_yp, d_v);

        // alpha = rho / (r0_hat, v) 
        cuDoubleComplex r0_v;
        cublasZdotu(cublasH, N, d_r0_hat, 1, d_v, 1, &r0_v); 
        if (cuCabs(r0_v) < 1e-30) break;
        alpha = cuCdiv(rho, r0_v);

        // s = r - alpha * v
        cudaMemcpy(d_s, d_r, N * sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice);
        cuDoubleComplex neg_alpha = make_cuDoubleComplex(-cuCreal(alpha), -cuCimag(alpha));
        cublasZaxpy(cublasH, N, &neg_alpha, d_v, 1, d_s, 1);

        cublasDznrm2(cublasH, N, d_s, 1, &res_norm);
        if (res_norm / norm_b < tol) {
            cublasZaxpy(cublasH, N, &alpha, d_yp, 1, d_x, 1);
            converged = true;
            break;
        }

        // 预处理: ys = M_inv * s
        complexElementWiseMul<<<blocks, threads>>>(N, d_Minv, d_s, d_ys);
        
        // t = A * ys
        SpMV_A(d_ys, d_t);

        // omega = (t, s) / (t, t) 
        // ⚠️【极其核心】：MR 残差最小化步骤，必须共轭 (Zdotc) 求模长！
        cuDoubleComplex ts, tt;
        cublasZdotc(cublasH, N, d_t, 1, d_s, 1, &ts);
        cublasZdotc(cublasH, N, d_t, 1, d_t, 1, &tt);
        if (cuCabs(tt) < 1e-30) break;
        omega = cuCdiv(ts, tt);

        // x = x + alpha * yp + omega * ys
        cublasZaxpy(cublasH, N, &alpha, d_yp, 1, d_x, 1);
        cublasZaxpy(cublasH, N, &omega, d_ys, 1, d_x, 1);

        // r = s - omega * t
        cudaMemcpy(d_r, d_s, N * sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice);
        neg_omega = make_cuDoubleComplex(-cuCreal(omega), -cuCimag(omega));
        cublasZaxpy(cublasH, N, &neg_omega, d_t, 1, d_r, 1);

        rho_prev = rho;

        cublasDznrm2(cublasH, N, d_r, 1, &res_norm);
        if (res_norm / norm_b < tol) {
            converged = true;
            break;
        }

        if (iter % 500 == 0) {
            std::cout << "  [Jacobi-BiCGStab] Iter " << iter << " - Residual: " << res_norm / norm_b << std::endl;
        }
    }

    if (converged) {
        std::cout << "  -> BiCGStab converged in " << iter << " iterations. Final res: " << res_norm / norm_b << std::endl;
    } else {
        std::cerr << "  -> BiCGStab failed to converge within maxIter. Final res: " << res_norm / norm_b << std::endl;
    }

    cudaMemcpy(c_x.data(), d_x, N * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
    for (int i = 0; i < N; i++) {
        x[i] = std::complex<double>(cuCreal(c_x[i]), cuCimag(c_x[i]));
    }

    // 释放资源
    cudaFree(d_csrRowPtr); cudaFree(d_csrColInd); cudaFree(d_csrVal);
    cudaFree(d_b); cudaFree(d_x); cudaFree(d_Minv);
    cudaFree(d_r); cudaFree(d_r0_hat); cudaFree(d_p); cudaFree(d_v);
    cudaFree(d_s); cudaFree(d_t); cudaFree(d_yp); cudaFree(d_ys);
    cudaFree(dBuffer);
    cusparseDestroySpMat(matA);
    cusparseDestroy(cusparseH); cublasDestroy(cublasH);

    return converged;
}