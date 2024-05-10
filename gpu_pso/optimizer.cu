
/*
Use CUDA functions to calculate block size
*/

#include <iostream>
#include <stdlib.h>
#include <cmath>
#include <string>
#include <ctime>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <fstream>

const unsigned int ARR_LEN = 20;

// Position struct contains x and y coordinates
struct Sol_arr
{
    int array[ARR_LEN];

    std::string toString()
    {
        std::string str = "[";
        for (int i = 0; i < ARR_LEN; i++)
        {
            str += std::to_string(array[i]);
            if (i < ARR_LEN - 1)
            {
                str += ", ";
            }
        }
        str += "]";
        return str;
    }

    __device__ __host__ void operator+=(const Sol_arr &a)
    {
        for (size_t i = 0; i < ARR_LEN; i++)
        {
            array[i] += a.array[i];
        }
    }

    __device__ __host__ void operator=(const Sol_arr &a)
    {
        for (size_t i = 0; i < ARR_LEN; i++)
        {
            array[i] = a.array[i];
        }
    }
};

// Particle struct has current location, best location and velocity
struct Particle
{
    Sol_arr best_position;
    Sol_arr current_position;
    Sol_arr velocity;
    int best_value;
};

const unsigned int N = 50000;
const unsigned int ITERATIONS = 10000;
const int SEARCH_MIN = -100;
const int SEARCH_MAX = 100;
const float w = 0.9f;
const float c_ind = 1.0f;
const float c_team = 2.0f;

// return a random int between low and high
int randInt(int min, int max)
{
    return min + rand() % (max - min + 1);
}

// function to optimize
__device__ __host__ int calcValue(Sol_arr p)
{
    int sum = 500;
    for (int i = 0; i < ARR_LEN; i++)
    {
        sum += p.array[i] * p.array[i] * p.array[i];
    }
    return abs(sum);
}

// Initialize state for random numbers
__global__ void init_kernel(curandState *state, long seed)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(seed, idx, 0, state);
}

// Returns the index of the particle with the best position
__global__ void updateTeamBestIndex(Particle *d_particles, int *d_team_best_value, int *d_team_best_index, int N)
{
    __shared__ int best_value;
    __shared__ int best_index;
    best_value = d_particles[0].best_value;
    best_index = 0;
    __syncthreads();

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N)
    {
        if (d_particles[idx].best_value < best_value)
        {
            best_value = d_particles[idx].best_value;
            best_index = idx;
            __syncthreads();
        }
    }
    *d_team_best_value = best_value;
    *d_team_best_index = best_index;
}

// Update velocity for all particles
__global__ void updateVelocity(Particle *d_particles, int *d_team_best_index, float w, float c_ind, float c_team, int N, curandState *state)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    __shared__ int best[ARR_LEN];
    for (int i = 0; i < ARR_LEN; i++)
    {
        best[i] = d_particles[*d_team_best_index].best_position.array[i];
    }
    __syncthreads();

    if (idx < N)
    {
        float r_ind = curand_uniform(state);
        float r_team = curand_uniform(state);
        for (int i = 0; i < ARR_LEN; i++)
        {
            d_particles[idx].velocity.array[i] = (int)w * d_particles[idx].velocity.array[i] +
                                                 r_ind * c_ind * (d_particles[idx].best_position.array[i] - d_particles[idx].current_position.array[i]) +
                                                 r_team * c_team * (best[i] - d_particles[idx].current_position.array[i]);
        }
    }
}

__global__ void updatePosition(Particle *d_particles, int N)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N)
    {
        d_particles[idx].current_position += d_particles[idx].velocity;
        int newValue = calcValue(d_particles[idx].current_position);
        if (newValue < d_particles[idx].best_value)
        {
            d_particles[idx].best_value = newValue;
            d_particles[idx].best_position = d_particles[idx].current_position;
        }
    }
}

int main(void)
{
    // Open the CSV file
    std::ofstream file("timing_results.csv");
    file << "Swarm Size,Execution Time\n"; // Write the header

    for (int N = 10; N <= 15000; N *= 2) // Change this line to control the swarm sizes
    {
        // for timing
        long start = std::clock();

        // Random seed for cpu
        std::srand(std::time(NULL));
        // Random seed for gpu
        curandState *state;
        cudaMalloc(&state, sizeof(curandState));
        init_kernel<<<1, 1>>>(state, clock());

        // Initialize particles
        Particle *h_particles = new Particle[N];
        Particle *d_particles; // for the gpu

        //  Initialize particles on host
        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < ARR_LEN; j++)
            {
                h_particles[i].current_position.array[j] = randInt(SEARCH_MIN, SEARCH_MAX);
                h_particles[i].best_position.array[j] = h_particles[i].current_position.array[j];
                h_particles[i].velocity.array[j] = randInt(SEARCH_MIN, SEARCH_MAX);
                h_particles[i].best_value = calcValue(h_particles[i].best_position);
            }
        }

        // Allocate memory + copy data to gpu
        size_t particleSize = sizeof(Particle) * N;
        cudaMalloc((void **)&d_particles, particleSize);
        cudaMemcpy(d_particles, h_particles, particleSize, cudaMemcpyHostToDevice); // dest, source, size, direction

        // initialize variables for gpu
        int *d_team_best_index;
        int *d_team_best_value;

        // Allocate gpu memory
        cudaMalloc((void **)&d_team_best_index, sizeof(int));
        cudaMalloc((void **)&d_team_best_value, sizeof(int));

        // Initialize team best index and value
        updateTeamBestIndex<<<1, 1>>>(d_particles, d_team_best_value, d_team_best_index, N);

        // assign thread and blockcount
        int blockSize = 32;
        int gridSize = (N + blockSize - 1) / blockSize;

        // For i in interations
        for (int i = 0; i < ITERATIONS; i++)
        {
            updateVelocity<<<gridSize, blockSize>>>(d_particles, d_team_best_index, w, c_ind, c_team, N, state);
            updatePosition<<<gridSize, blockSize>>>(d_particles, N);
            updateTeamBestIndex<<<gridSize, blockSize>>>(d_particles, d_team_best_value, d_team_best_index, N);
        }

        // copy best particle back to host
        int team_best_index;
        cudaMemcpy(&team_best_index, d_team_best_index, sizeof(int), cudaMemcpyDeviceToHost);

        // copy particle data back to host
        cudaMemcpy(h_particles, d_particles, particleSize, cudaMemcpyDeviceToHost);

        long stop = std::clock();
        long elapsed = (stop - start) * 1000 / CLOCKS_PER_SEC;

        file << N << "," << elapsed << "\n"; // Write the data to the CSV file

        // print results
        // std::cout << "Ending Best: " << std::endl;
        // std::cout << "Team best value: " << h_particles[team_best_index].best_value << std::endl;
        // std::cout << "Team best position: " << h_particles[team_best_index].best_position.toString() << std::endl;

        // std::cout << "Run time: " << elapsed << "ms" << std::endl;

        cudaFree(d_particles);
        cudaFree(d_team_best_index);
        cudaFree(d_team_best_value);
        cudaFree(state);
    }

    return 0;
}