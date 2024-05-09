#include "cuda_funcs.cuh"
#include <stdio.h>
#include <curand_kernel.h>

__global__ void updateParticles(int *position, double *velocity, double *pBestPosition, double *pBestCost, double *gBestPosition, size_t swarmSize, size_t vecSpace, double inertia, double c1, double c2, double gBestCost, size_t maxIter, double inertiaDrop)
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

                velocity[i * vecSpace + j] = (inertia * velocity[i * vecSpace + j]) +
                                             (r1 * c1 * (pBestPosition[i * vecSpace + j] - position[i * vecSpace + j])) +
                                             (r2 * c2 * (gBestPosition[j] - position[i * vecSpace + j]));

                position[i * vecSpace + j] += velocity[i * vecSpace + j];
            }

            double cost = 2; // function.Evaluate(position);

            // Update personal best.
            if (cost < pBestCost[i])
            {
                pBestCost[i] = cost;

                for (size_t j = 0; j < vecSpace; j++)
                {
                    pBestPosition[i * vecSpace + j] = position[i * vecSpace + j];
                }
            }

            inertia *= inertiaDrop;
        }
    }
}

__global__ void testKernel(int *particlePositions, double *particleVelocities, int *particleBestPositions, double *particleCosts)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    // Check if the index is within the array bounds
    if (idx < vecSpace * 10)
    {
        // Set the position to 0
        particlePositions[idx] = 0;
        particleVelocities[idx] = 0.0;
        particleBestPositions[idx] = 0;
    }

    if (idx < 10)
    {
        // Set the cost to 0
        particleCosts[idx] = 0.0;
    }
}

namespace Wrapper
{
    void wrapper(
        int *particlePositions[][vecSpace],
        double *particleVelocities[][vecSpace],
        int *pBestPositions[][vecSpace],
        double particleCosts[],
        double pBestCosts[],
        int *gBestPosition,
        double gBestCost,
        size_t maxIter,
        double inertia,
        double c1,
        double c2,
        double inertiaDrop)
    {
        // Print particle positions
        for (size_t i = 0; i < vecSpace; ++i)
        {
            for (size_t j = 0; j < 10; ++j)
            {
                //printf("%d ", *(particlePositions[i][j]));
                printf("%f ", (particleCosts[j]));
            }
            printf("\n");
        }

        // Step 1: Flatten the 2D matrix into a 1D array
        int *flatParticlePositions = new int[vecSpace * 10];
        double *flatParticleVelocities = new double[vecSpace * 10];
        int *flatPBestPositions = new int[vecSpace * 10];
        for (int i = 0; i < vecSpace; ++i)
        {
            for (int j = 0; j < 10; ++j)
            {
                flatParticlePositions[i * 10 + j] = *(particlePositions[i][j]);
                flatParticleVelocities[i * 10 + j] = *(particleVelocities[i][j]);
                flatPBestPositions[i * 10 + j] = *(pBestPositions[i][j]);
            }
        }

        // Step 2: Allocate memory on the GPU for the 1D array
        int *d_particlePositions;
        double *d_particleVelocities;
        int *d_pBestPositions;
        double *d_particleCosts;
        cudaMalloc(&d_particlePositions, vecSpace * 10 * sizeof(int));
        cudaMalloc(&d_particleVelocities, vecSpace * 10 * sizeof(double));
        cudaMalloc(&d_pBestPositions, vecSpace * 10 * sizeof(int));
        cudaMalloc(&d_particleCosts, 10 * sizeof(double));

        // Step 3: Copy the 1D array from host to device
        cudaMemcpy(d_particlePositions, flatParticlePositions, vecSpace * 10 * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_particleVelocities, flatParticleVelocities, vecSpace * 10 * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_pBestPositions, flatPBestPositions, vecSpace * 10 * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_particleCosts, particleCosts, 10 * sizeof(double), cudaMemcpyHostToDevice);

        // Step 4: Launch the kernel to process the 1D array
        int blockSize = 256;
        int numBlocks = (vecSpace * 10 + blockSize - 1) / blockSize;
        testKernel<<<numBlocks, blockSize>>>(d_particlePositions, d_particleVelocities, d_pBestPositions, d_particleCosts);

        // Step 5: Copy the processed 1D array from device to host
        cudaMemcpy(flatParticlePositions, d_particlePositions, vecSpace * 10 * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(flatParticleVelocities, d_particleVelocities, vecSpace * 10 * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(flatPBestPositions, d_pBestPositions, vecSpace * 10 * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(particleCosts, d_particleCosts, 10 * sizeof(double), cudaMemcpyDeviceToHost);

        // Step 6: Reshape the 1D array back into a 2D matrix
        for (int i = 0; i < vecSpace; ++i)
        {
            for (int j = 0; j < 10; ++j)
            {
                *(particlePositions[i][j]) = flatParticlePositions[i * 10 + j];
                *(particleVelocities[i][j]) = flatParticleVelocities[i * 10 + j];
                *(pBestPositions[i][j]) = flatPBestPositions[i * 10 + j];
            }
        }

        // Free the memory
        delete[] flatParticlePositions;
        delete[] flatParticleVelocities;
        delete[] flatPBestPositions;
        cudaFree(d_particlePositions);
        cudaFree(d_particleVelocities);
        cudaFree(d_pBestPositions);
        cudaFree(d_particleCosts);

        // Print the updated particle positions
        for (size_t i = 0; i < 2; ++i)
        {
            for (size_t j = 0; j < 10; ++j)
            {
                printf("%d ", *(particlePositions[i][j]));
                printf("%f ", (particleCosts[j]));
            }
            printf("\n");
        }

    }
}