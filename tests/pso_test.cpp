#include "../optimizer.hpp"
#include <iostream>
#include <chrono>
#include "../optimizer_bits/pso/cuda/cuda_funcs.cuh"

using namespace std;
using namespace optimization;
using namespace optimization::testFunction;
using namespace std::chrono;


int main()
{
    srand(time(0));

    const size_t vecSpace = size_t(2);
    int *minFuncRange = new int[vecSpace];
    int *maxFuncRange = new int[vecSpace];

    for (size_t i = 0; i < vecSpace; ++i)
    {
        minFuncRange[i] = -10;
        maxFuncRange[i] = 10;
    }

    PSOTestFunction func(vecSpace, minFuncRange, maxFuncRange);

    // array of variables needed to be optimized.
    double *iterate = func.GetInitialPoint();


    // Start timer
    auto start = high_resolution_clock::now();

    PSO optimizer(25, 10, 1.0, 0.99, 1.5, 1.5);
    double cost = optimizer.Optimize(func, iterate);

    // Stop timer
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);

    cout << "\nTime taken by function: "
         << duration.count() << " microseconds" << endl;

    cout << "\nminimum value : \n [ ";
    for (size_t j = 0; j < vecSpace; j++)
    {
        cout << iterate[j] << " ,";
    }
    cout << "]\n final cost : " << cost;

    return 0;
}
