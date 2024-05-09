#ifndef WRAPPER_CUH
#define WRAPPER_CUH

const int vecSpace = 2;

namespace Wrapper {
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
    );
}

#endif // WRAPPER_CUH
