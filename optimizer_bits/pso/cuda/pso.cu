#include "cuda_funcs.cuh"
#include <stdio.h>
#include <curand_kernel.h>


__global__ void updateParticles(double* position, double* velocity, double* pBestPosition, double* pBestCost, double* gBestPosition, size_t swarmSize, size_t vecSpace, double inertia, double c1, double c2, double gBestCost, size_t maxIter, double inertiaDrop)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < swarmSize)
    {
        curandState state;
        curand_init(1234, i, 0, &state); // Initialize RNG state with a seed (1234) and thread ID

        for (size_t itr = 0; itr < maxIter; ++itr)
        {
            for (size_t j = 0; j < vecSpace; ++j)
            {
                // Generate pseudorandom numbers uniformly distributed between 0 and 1
                double r1 = curand_uniform(&state);
                double r2 = curand_uniform(&state);

                velocity[i*vecSpace+j] = (inertia * velocity[i*vecSpace+j]) +
                                         (r1 * c1 * (pBestPosition[i*vecSpace+j] - position[i*vecSpace+j])) +
                                         (r2 * c2 * (gBestPosition[j] - position[i*vecSpace+j]));

                position[i*vecSpace+j] += velocity[i*vecSpace+j];
            }

            double cost = 2; // function.Evaluate(position);

            // Update personal best.
            if (cost < pBestCost[i])
            {
                pBestCost[i] = cost;

                for (size_t j = 0; j < vecSpace; j++)
                {
                    pBestPosition[i*vecSpace+j] = position[i*vecSpace+j];
                }
            }

            inertia *= inertiaDrop;
        }
    }
}

namespace Wrapper
{
    void wrapper(
        int* particlePositions[][vecSpace],
        double* particleVelocities[][vecSpace],
        int* pBestPositions[][vecSpace],
        double particleCosts[],
        double pBestCosts[],
        int* gBestPosition,
        double gBestCost,
        size_t maxIter,
        double inertia,
        double c1,
        double c2,
        double inertiaDrop
    ) {
        // CUDA kernel invocation and other CUDA operations here...
    }
}