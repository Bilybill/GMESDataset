// src/boundaryConditions3d.cpp
#include "../include/boundaryConditions3d.h"
#include <iostream>
#include <cmath>
#include <memory>
#include <algorithm>

namespace {

struct BoundaryCouplingEntry {
    int row;
    int csrIndex;
    int boundaryCol;
};

struct BoundaryRowEntry {
    int row;
    int rowStart;
    int rowEnd;
    int diagIndex;
};

struct BoundaryConditionPlanImpl {
    std::vector<double> zCoords;
    std::vector<unsigned char> isBoundary;
    std::vector<int> boundaryEdges;
    std::vector<int> boundaryLookup;
    std::vector<BoundaryCouplingEntry> boundaryCouplings;
    std::vector<BoundaryRowEntry> boundaryRows;
    std::vector<int> modifiedValueIndices;
};

BoundaryConditionPlanImpl buildBoundaryConditionPlanImpl(const HostMesh3D &mesh,
                                                         const std::vector<int> &csrColInd,
                                                         const std::vector<int> &csrRowPtr)
{
    BoundaryConditionPlanImpl plan;

    int NX = mesh.NX;
    int NY = mesh.NY;
    int NZ = mesh.NZ;

    int nEx = NX * (NY + 1) * (NZ + 1);
    int nEy = (NX + 1) * NY * (NZ + 1);

    int offset_X = 0;
    int offset_Y = nEx;
    int offset_Z = nEx + nEy;

    plan.zCoords.assign(NZ + 1, 0.0);
    for (int k = 0; k < NZ; ++k) {
        plan.zCoords[k + 1] = plan.zCoords[k] + mesh.hz[k];
    }

    plan.isBoundary.assign(mesh.NEdges, 0);
    plan.boundaryLookup.assign(mesh.NEdges, -1);

    auto markBoundary = [&](int edgeId) {
        if (!plan.isBoundary[edgeId]) {
            plan.isBoundary[edgeId] = 1;
            plan.boundaryLookup[edgeId] = static_cast<int>(plan.boundaryEdges.size());
            plan.boundaryEdges.push_back(edgeId);
        }
    };

    for (int k_idx = 0; k_idx <= NZ; ++k_idx) {
        for (int j = 0; j <= NY; ++j) {
            for (int i = 0; i < NX; ++i) {
                if (k_idx == 0 || k_idx == NZ || j == 0 || j == NY) {
                    markBoundary(offset_X + i + j * NX + k_idx * NX * (NY + 1));
                }
            }
        }
    }

    for (int k_idx = 0; k_idx <= NZ; ++k_idx) {
        for (int j = 0; j < NY; ++j) {
            for (int i = 0; i <= NX; ++i) {
                if (k_idx == 0 || k_idx == NZ || i == 0 || i == NX) {
                    markBoundary(offset_Y + i + j * (NX + 1) + k_idx * (NX + 1) * NY);
                }
            }
        }
    }

    for (int k_idx = 0; k_idx < NZ; ++k_idx) {
        for (int j = 0; j <= NY; ++j) {
            for (int i = 0; i <= NX; ++i) {
                if (i == 0 || i == NX || j == 0 || j == NY) {
                    markBoundary(offset_Z + i + j * (NX + 1) + k_idx * (NX + 1) * (NY + 1));
                }
            }
        }
    }

    for (int row = 0; row < mesh.NEdges; ++row) {
        if (plan.isBoundary[row]) {
            int diagIndex = -1;
            for (int idx = csrRowPtr[row]; idx < csrRowPtr[row + 1]; ++idx) {
                if (csrColInd[idx] == row) {
                    diagIndex = idx;
                    break;
                }
            }
            plan.boundaryRows.push_back({row, csrRowPtr[row], csrRowPtr[row + 1], diagIndex});
            continue;
        }

        for (int idx = csrRowPtr[row]; idx < csrRowPtr[row + 1]; ++idx) {
            int col = csrColInd[idx];
            if (plan.isBoundary[col]) {
                plan.boundaryCouplings.push_back({row, idx, col});
            }
        }
    }

    plan.modifiedValueIndices.reserve(plan.boundaryCouplings.size() + plan.boundaryRows.size() * 8);
    for (const auto& coupling : plan.boundaryCouplings) {
        plan.modifiedValueIndices.push_back(coupling.csrIndex);
    }
    for (const auto& rowEntry : plan.boundaryRows) {
        for (int idx = rowEntry.rowStart; idx < rowEntry.rowEnd; ++idx) {
            plan.modifiedValueIndices.push_back(idx);
        }
    }
    std::sort(plan.modifiedValueIndices.begin(), plan.modifiedValueIndices.end());
    plan.modifiedValueIndices.erase(
        std::unique(plan.modifiedValueIndices.begin(), plan.modifiedValueIndices.end()),
        plan.modifiedValueIndices.end()
    );

    return plan;
}

const BoundaryConditionPlanImpl* unwrapPlan(const BoundaryConditionPlan* plan)
{
    return reinterpret_cast<const BoundaryConditionPlanImpl*>(plan);
}

} // namespace

struct BoundaryConditionPlan {};

BoundaryConditionPlan* createBoundaryConditionPlan3D(const HostMesh3D &mesh,
                                                     const std::vector<int> &csrColInd,
                                                     const std::vector<int> &csrRowPtr)
{
    auto impl = buildBoundaryConditionPlanImpl(mesh, csrColInd, csrRowPtr);
    return reinterpret_cast<BoundaryConditionPlan*>(new BoundaryConditionPlanImpl(std::move(impl)));
}

void destroyBoundaryConditionPlan3D(BoundaryConditionPlan* plan)
{
    delete reinterpret_cast<BoundaryConditionPlanImpl*>(plan);
}

void getBoundaryConditionModifiedValueIndices3D(const BoundaryConditionPlan* plan,
                                                std::vector<int> &indices)
{
    indices = unwrapPlan(plan)->modifiedValueIndices;
}

void applyBoundaryConditions3D(const HostMesh3D &mesh, double freq, double rho_bg,
                               std::vector<std::complex<double>> &csrVal,
                               std::vector<int> &csrColInd,
                               const std::vector<int> &csrRowPtr,
                               std::vector<std::complex<double>> &b,
                               int polarization,
                               const BoundaryConditionPlan* planHandle)
{
    std::unique_ptr<BoundaryConditionPlan, void(*)(BoundaryConditionPlan*)> ownedPlan(nullptr, destroyBoundaryConditionPlan3D);
    if (!planHandle) {
        ownedPlan.reset(createBoundaryConditionPlan3D(mesh, csrColInd, csrRowPtr));
        planHandle = ownedPlan.get();
    }
    const auto* plan = unwrapPlan(planHandle);

    double mu = 4e-7 * M_PI;
    double w = 2.0 * M_PI * freq;
    double sigma = 1.0 / rho_bg;
    std::complex<double> neg_i(0.0, -1.0);
    std::complex<double> k_num = std::sqrt(neg_i * w * mu * sigma);

    if (static_cast<int>(b.size()) != mesh.NEdges) {
        b.resize(mesh.NEdges);
    }
    std::fill(b.begin(), b.end(), std::complex<double>(0.0, 0.0));

    std::vector<std::complex<double>> bc_values(plan->boundaryEdges.size(), {0.0, 0.0});

    int NX = mesh.NX;
    int NY = mesh.NY;
    int NZ = mesh.NZ;
    int nEx = NX * (NY + 1) * (NZ + 1);
    int nEy = (NX + 1) * NY * (NZ + 1);
    int offset_Y = nEx;
    int offset_Z = nEx + nEy;

    for (int edgeId : plan->boundaryEdges) {
        int boundaryPos = plan->boundaryLookup[edgeId];
        if (edgeId < offset_Y) {
            if (polarization == 1) {
                int local = edgeId;
                int k_idx = local / (NX * (NY + 1));
                bc_values[boundaryPos] = std::exp(neg_i * k_num * plan->zCoords[k_idx]);
            }
        } else if (edgeId < offset_Z) {
            if (polarization == 2) {
                int local = edgeId - offset_Y;
                int k_idx = local / ((NX + 1) * NY);
                bc_values[boundaryPos] = std::exp(neg_i * k_num * plan->zCoords[k_idx]);
            }
        }
    }

    for (const auto& entry : plan->boundaryCouplings) {
        b[entry.row] -= csrVal[entry.csrIndex] * bc_values[plan->boundaryLookup[entry.boundaryCol]];
        csrVal[entry.csrIndex] = {0.0, 0.0};
    }

    for (const auto& rowEntry : plan->boundaryRows) {
        b[rowEntry.row] = bc_values[plan->boundaryLookup[rowEntry.row]];
        for (int idx = rowEntry.rowStart; idx < rowEntry.rowEnd; ++idx) {
            csrVal[idx] = {0.0, 0.0};
        }
        if (rowEntry.diagIndex >= 0) {
            csrVal[rowEntry.diagIndex] = {1.0, 0.0};
        }
    }

    std::cout << "Applied 3D Boundary Conditions for Polarization " << polarization << " (Symmetric Elimination)" << std::endl;
}
