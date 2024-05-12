
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

const unsigned int ARR_LEN = 2;

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

const unsigned int N = 1024;
const unsigned int ITERATIONS = 1;
const int SEARCH_MIN = 0;
const int SEARCH_MAX = 25;
const float w = 0.9f;
const float c_ind = 1.0f;
const float c_team = 2.0f;

// return a random int between low and high
int randInt(int min, int max)
{
    return min + rand() % (max - min + 1);
}

// function to optimize
__device__ __host__ int calcValue(int* device_1, int* device_2, int* d_prices, Sol_arr p)
{
    int price = 0;
    int offset_device_1 = p.array[0];
    int offset_device_2 = p.array[1];
    
    if (offset_device_1 < 0 || offset_device_1 > 19) {
        return 1000000;
    }

    if (offset_device_2 < 0 || offset_device_2 > 19) {
        return 1000000;
    }

    for (int i = 0; i < 5; i++)
    {
        price += device_1[i] * d_prices[offset_device_1 + i];
        price += device_2[i] * d_prices[offset_device_2 + i];
    }

    return price;
}

// Initialize state for random numbers
__global__ void init_kernel(curandState *state, long seed)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(seed, idx, 0, state);
}

__global__ void updateTeamBestIndex(Particle *d_particles, int *d_team_best_value, int *d_team_best_index, int N)
{
    __shared__ int best_value;
    __shared__ int best_index;

    // Each block has its own copy of best value and index
    if (threadIdx.x == 0)
    {
        best_value = d_particles[0].best_value;
        best_index = 0;
    }

    __syncthreads();

    // Reduction loop
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;

        if (idx < N)
        {
            int other_idx = idx + stride;
            if (other_idx < N)
            {
                int min_value = min(d_particles[idx].best_value, d_particles[other_idx].best_value);
                int min_index = (d_particles[idx].best_value < d_particles[other_idx].best_value) ? idx : other_idx;
                
                atomicMin(&best_value, min_value);
                if (min_value == best_value)
                    atomicExch(&best_index, min_index);
            }
        }

        __syncthreads();
    }

    // Write back team's best value and index
    if (threadIdx.x == 0)
    {
        d_team_best_value[blockIdx.x] = best_value;
        d_team_best_index[blockIdx.x] = best_index;
    }
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

__global__ void updatePosition(int* d_device_1, int* d_device_2, int* d_prices, Particle *d_particles, int N)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N)
    {
        d_particles[idx].current_position += d_particles[idx].velocity;
        int newValue = calcValue(d_device_1, d_device_2, d_prices, d_particles[idx].current_position);
        if (newValue < d_particles[idx].best_value)
        {
            d_particles[idx].best_value = newValue;
            d_particles[idx].best_position = d_particles[idx].current_position;
        }
    }
}

int main(void)
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

    // Initialize pricing array
    int *h_prices = new int[24]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                           11, 12, 13, 14, 15, 1, 1, 1, 1, 1,
                           21, 22, 23, 24};
    int *d_prices;

    int *h_device_1 = new int[5]{1,5,1,2,3};
    int *d_device_1;

    int *h_device_2 = new int[5]{100, 1, 200, 3, 1};
    int *d_device_2;

    //  Initialize particles on host
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < ARR_LEN; j++)
        {
            h_particles[i].current_position.array[j] = randInt(SEARCH_MIN, SEARCH_MAX);
            h_particles[i].best_position.array[j] = h_particles[i].current_position.array[j];
            h_particles[i].velocity.array[j] = randInt(SEARCH_MIN, SEARCH_MAX);
            h_particles[i].best_value = calcValue(h_device_1, h_device_2, h_prices, h_particles[i].best_position);
        }
    }

    // Allocate memory + copy data to gpu
    size_t particleSize = sizeof(Particle) * N;
    cudaMalloc((void **)&d_particles, particleSize);
    cudaMemcpy(d_particles, h_particles, particleSize, cudaMemcpyHostToDevice); // dest, source, size, direction

    cudaMalloc((void **)&d_prices, sizeof(int) * 24);
    cudaMemcpy(d_prices, h_prices, sizeof(int) * 24, cudaMemcpyHostToDevice);

    cudaMalloc((void **)&d_device_1, sizeof(int) * 5);
    cudaMemcpy(d_device_1, h_device_1, sizeof(int) * 5, cudaMemcpyHostToDevice);

    cudaMalloc((void **)&d_device_2, sizeof(int) * 5);
    cudaMemcpy(d_device_2, h_device_2, sizeof(int) * 5, cudaMemcpyHostToDevice);

    // initialize variables for gpu
    int *d_team_best_index;
    int *d_team_best_value;

    // Allocate gpu memory
    cudaMalloc((void **)&d_team_best_index, sizeof(int));
    cudaMalloc((void **)&d_team_best_value, sizeof(int));

    // Initialize team best index and value
    updateTeamBestIndex<<<1, 1>>>(d_particles, d_team_best_value, d_team_best_index, N);

    // assign thread and blockcount
    int blockSize = 512;
    int gridSize = (N + blockSize - 1) / blockSize;

    // For i in interations
    for (int i = 0; i < ITERATIONS; i++)
    {
        updateVelocity<<<gridSize, blockSize>>>(d_particles, d_team_best_index, w, c_ind, c_team, N, state);
        updatePosition<<<gridSize, blockSize>>>(d_device_1, d_device_2, d_prices,d_particles, N);
        updateTeamBestIndex<<<gridSize, blockSize>>>(d_particles, d_team_best_value, d_team_best_index, N);
    }

    // copy best particle back to host
    int team_best_index;
    cudaMemcpy(&team_best_index, d_team_best_index, sizeof(int), cudaMemcpyDeviceToHost);

    // copy particle data back to host
    cudaMemcpy(h_particles, d_particles, particleSize, cudaMemcpyDeviceToHost);

    long stop = std::clock();
    long elapsed = (stop - start) * 1000 / CLOCKS_PER_SEC;

    // print results
    std::cout << "Ending Best: " << std::endl;
    std::cout << "Swarm best value: " << h_particles[team_best_index].best_value << std::endl;
    std::cout << "Swarm best position: " << h_particles[team_best_index].best_position.toString() << std::endl;

    std::cout << "Run time: " << elapsed << "ms" << std::endl;

    cudaFree(d_particles);
    cudaFree(d_team_best_index);
    cudaFree(d_team_best_value);
    cudaFree(state);

    return 0;
}