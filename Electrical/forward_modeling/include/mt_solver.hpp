#pragma once

#include "mt_types.hpp"

#include <memory>

namespace gmes::mt
{

class MT3DForwardSolver
{
public:
    explicit MT3DForwardSolver(const MTInput& input);
    ~MT3DForwardSolver();

    MTResponse Run();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

MTResponse ComputeMT3D(const MTInput& input);

} // namespace gmes::mt
