// include/gpuIterativeSolver.h
#ifndef GPU_ITERATIVE_SOLVER_H
#define GPU_ITERATIVE_SOLVER_H

#include <limits>
#include <vector>
#include <complex>

struct SolverStats {
    int iterations = 0;
    double finalResidual = std::numeric_limits<double>::infinity();
    bool converged = false;
};

struct GpuBiCGStabContext;

GpuBiCGStabContext* create_gpu_bicgstab_context(int N, int nnz);
void destroy_gpu_bicgstab_context(GpuBiCGStabContext* context);

// 函数声明：使用 cuBLAS 和 cuSPARSE 实现的复数 BiCGSTAB 迭代求解器
// 求解 A x = b
bool gpu_complex_solver_bicgstab(int N,
                                 const std::vector<int> &csrRowPtr,
                                 const std::vector<int> &csrColInd,
                                 const std::vector<std::complex<double>> &csrVal,
                                 const std::vector<std::complex<double>> &b,
                                 std::vector<std::complex<double>> &x,
                                 double tol = 1e-8,
                                 int maxIter = 5000,
                                 GpuBiCGStabContext* context = nullptr,
                                 SolverStats* stats = nullptr);

#endif // GPU_ITERATIVE_SOLVER_H
