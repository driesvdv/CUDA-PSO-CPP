
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

const unsigned int ARR_LEN = 1;
const unsigned int N = 500; 
const unsigned int ITERATIONS = 25; 
const float w = 0.9f; 
const float c_ind = 1.0f; 
const float c_team = 2.0f; 


// Position struct contains x and y coordinates 
struct Sol_arr {
    int array[ARR_LEN];

    std::string toString() {
        std::string str = "["; 
        for (int i = 0; i < ARR_LEN; i++) {
            str += std::to_string(array[i]); 
            if (i < ARR_LEN - 1) {
                str += ", "; 
            }
        }
        str += "]"; 
        return str; 
    }

    __device__ __host__ void operator+=(const Sol_arr& a) {
        for (size_t i = 0; i < ARR_LEN; i++)
        {
            array[i] += a.array[i];
        }
        
    }

    __device__ __host__ void operator=(const Sol_arr& a) {
        for (size_t i = 0; i < ARR_LEN; i++)
        {
            array[i] = a.array[i];
        }
    }
}; 

// Particle struct has current location, best location and velocity 
struct Particle {
    Sol_arr best_position; 
    Sol_arr current_position; 
    Sol_arr velocity; 
    float best_value; 
};

int randomInt() {
    int number = std::rand() % 10001; // rand() % (max - min + 1) + min
    return number; 
}

// function to optimize 
__device__ __host__ int calcValue(Sol_arr p) {
    int sum = 0;
    for (int i = 0; i < ARR_LEN; i++) {
        sum += p.array[i] * p.array[i]; 

        if (p.array[i] > 1000 || p.array[i] < 0){
            return 9999999; 
        }
    }
    return sum;
}

// Initialize state for random numbers 
__global__ void init_kernel(curandState *state, long seed) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x; 
    curand_init(seed, idx, 0, state);
}

// Returns the index of the particle with the best position
__global__ void updateTeamBestIndex(Particle *d_particles, float *team_best_value, int *team_best_index, int N) {
    __shared__ float best_value; 
    __shared__ int best_index; 
    best_value = d_particles[0].best_value;
    best_index = 0; 
    __syncthreads(); 
    
    int idx = threadIdx.x + blockIdx.x * blockDim.x; 
    if (idx < N) {
        if (d_particles[idx].best_value < best_value) {
            best_value = d_particles[idx].best_value; 
            best_index = idx; 
            __syncthreads(); 
        }
    }
    *team_best_value = best_value; 
    *team_best_index = best_index; 
}


// Update velocity for all particles 
__global__ void updateVelocity(Particle* d_particles, int *team_best_index, float w, float c_ind, float c_team, int N, curandState *state) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x; 

    __shared__ float best[ARR_LEN]; 
    for (int i = 0; i < ARR_LEN; i++) {
        best[i] = d_particles[*team_best_index].best_position.array[i]; 
    }
    __syncthreads(); 

    if (idx < N) {
        for (int i = 0; i < ARR_LEN; i++) {
            float r_ind = curand_uniform(state);
            float r_team = curand_uniform(state);
            d_particles[idx].velocity.array[i] = w * d_particles[idx].velocity.array[i] + 
                           r_ind * c_ind * (d_particles[idx].best_position.array[i] - d_particles[idx].current_position.array[i]) + 
                           r_team * c_team * (best[i] - d_particles[idx].current_position.array[i]); 
        }
    }
}

__global__ void updatePosition(Particle *d_particles, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x; 
    if (idx < N) {
        d_particles[idx].current_position += d_particles[idx].velocity; 
        float newValue = calcValue(d_particles[idx].current_position); 
        if (newValue < d_particles[idx].best_value) {
            d_particles[idx].best_value = newValue; 
            d_particles[idx].best_position = d_particles[idx].current_position; 
        }
    }
}


int main(void) {
    // for timing 
    long start = std::clock();

    // Random seed for cpu 
    std::srand(std::time(NULL)); 
    // Random seed for gpu 
    curandState *state; 
    cudaMalloc(&state, sizeof(curandState)); 
    init_kernel<<<1,1>>>(state, clock()); 

    // Initialize particles 
    Particle *particles; 
    size_t particleSize = sizeof(Particle) * N; 

    // initialize variables for team best 
    int *team_best_index; 
    float *team_best_value; 

    // Allocate particles in unified memory 
    cudaMallocManaged(&particles, particleSize);
    cudaMallocManaged(&team_best_index, sizeof(int)); 
    
    // Allocate team_best_value for gpu only 
    cudaMalloc(&team_best_value, sizeof(float)); 

    // Prefetch data to the GPU 
    int device = cudaGetDevice(&device); 
    cudaMemPrefetchAsync(team_best_index, sizeof(int), device, NULL); // ptr, size_t, device, stream 

    // Memory hints 
    cudaMemAdvise(particles, particleSize, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId); // start on cpu

    //  Initialize particles on host 
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < ARR_LEN; j++) {
            particles[i].current_position.array[j] = randomInt(); 
            particles[i].best_position.array[j] = particles[i].current_position.array[j];
            particles[i].velocity.array[j] = randomInt();
            particles[i].best_value = calcValue(particles[i].best_position);
        }
    }

    // Prefetch particles to gpu 
    cudaMemPrefetchAsync(particles, particleSize, device, NULL); 

    // Initialize team best index and value 
    updateTeamBestIndex<<<1,1>>>(particles, team_best_value, team_best_index, N); 

    // assign thread and blockcount 
    int blockSize = 32; 
    int gridSize = (N + blockSize - 1) / blockSize; 

    // For i in interations 
    for (int i = 0; i < ITERATIONS; i++) {
        updateVelocity<<<gridSize, blockSize>>>(particles, team_best_index, w, c_ind, c_team, N, state); 
        updatePosition<<<gridSize, blockSize>>>(particles, N); 
        updateTeamBestIndex<<<gridSize, blockSize>>>(particles, team_best_value, team_best_index, N); 
    }

    // Wait for gpu to finish computation 
    cudaDeviceSynchronize(); 

    // Prefetch particles and best index back 
    cudaMemPrefetchAsync(particles, particleSize, cudaCpuDeviceId); 
    cudaMemPrefetchAsync(team_best_index, sizeof(int), cudaCpuDeviceId); 

    // Stop clock
    long stop = std::clock(); 
    long elapsed = (stop - start) * 1000 / CLOCKS_PER_SEC;

    // print results 
    std::cout << "Ending Best: " << std::endl;
    std::cout << "Team best value: " << particles[*team_best_index].best_value << std::endl;
    std::cout << "Team best position: " << particles[*team_best_index].best_position.toString() << std::endl; 
    
    std::cout << "Run time: " << elapsed << "ms" << std::endl;

    cudaFree(particles); 
    cudaFree(team_best_index); 
    cudaFree(team_best_value); 
    cudaFree(state);
    return 0; 
}