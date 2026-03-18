// include/gpuIterativeSolver.h
#ifndef GPU_ITERATIVE_SOLVER_H
#define GPU_ITERATIVE_SOLVER_H

#include <vector>
#include <complex>

// 函数声明：使用 cuBLAS 和 cuSPARSE 实现的复数 BiCGSTAB 迭代求解器
// 求解 A x = b
bool gpu_complex_solver_bicgstab(int N,
                                 const std::vector<int> &csrRowPtr,
                                 const std::vector<int> &csrColInd,
                                 const std::vector<std::complex<double>> &csrVal,
                                 const std::vector<std::complex<double>> &b,
                                 std::vector<std::complex<double>> &x,
                                 double tol = 1e-8,
                                 int maxIter = 5000);

#endif // GPU_ITERATIVE_SOLVER_H