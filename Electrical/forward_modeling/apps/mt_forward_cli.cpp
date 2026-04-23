#include "mt_solver.hpp"

#include <mpi.h>

#include <cstdlib>
#include <iostream>
#include <string>

using namespace gmes::mt;

namespace
{

int ArgInt(char** argv, int& i, int argc, int fallback)
{
    if (i + 1 >= argc)
    {
        return fallback;
    }
    return std::atoi(argv[++i]);
}

double ArgDouble(char** argv, int& i, int argc, double fallback)
{
    if (i + 1 >= argc)
    {
        return fallback;
    }
    return std::atof(argv[++i]);
}

} // namespace

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int nx = 8;
    int ny = 8;
    int nz = 8;
    double rho = 100.0;
    double freq = 1.0;
    std::string device = "cpu";

    for (int i = 1; i < argc; ++i)
    {
        const std::string arg(argv[i]);
        if (arg == "--nx")
        {
            nx = ArgInt(argv, i, argc, nx);
        }
        else if (arg == "--ny")
        {
            ny = ArgInt(argv, i, argc, ny);
        }
        else if (arg == "--nz")
        {
            nz = ArgInt(argv, i, argc, nz);
        }
        else if (arg == "--rho")
        {
            rho = ArgDouble(argv, i, argc, rho);
        }
        else if (arg == "--freq")
        {
            freq = ArgDouble(argv, i, argc, freq);
        }
        else if (arg == "--device" && i + 1 < argc)
        {
            device = argv[++i];
        }
    }

    try
    {
        MTInput input;
        input.model.nx = nx;
        input.model.ny = ny;
        input.model.nz = nz;
        input.model.dx = 100.0;
        input.model.dy = 100.0;
        input.model.dz = 100.0;
        input.model.rho_ohm_m.assign(static_cast<std::size_t>(nx) *
                                         static_cast<std::size_t>(ny) *
                                         static_cast<std::size_t>(nz),
                                     rho);
        input.freqs_hz = {freq};
        input.solver.device = device;
        input.solver.verbose = false;

        MTResponse output = ComputeMT3D(input);
        std::cout << "MT response shape: (" << output.nfreq << ", "
                  << output.nx << ", " << output.ny << ", 2)\n";
    }
    catch (const std::exception& exc)
    {
        std::cerr << "mt_forward_cli failed: " << exc.what() << "\n";
        MPI_Finalize();
        return 1;
    }

    MPI_Finalize();
    return 0;
}
