#include <iostream>
#include <stdlib.h>
#include <cmath> 
#include <string>
#include <ctime> 

const unsigned int ARR_LEN = 2;

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

    void operator+=(const Sol_arr& a) {
        for (size_t i = 0; i < ARR_LEN; i++)
        {
            array[i] += a.array[i];
        }
        
    }

    void operator=(const Sol_arr& a) {
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
    int current_value; 
};

int randInt(int low, int high); 
int calcValue(Sol_arr p, int* prices, int* device_1, int* device_2); 
int getTeamBestIndex(Particle* particles, int N);
void updateVelocity(Particle &p, Sol_arr team_best_position, float w, float c_ind, float c_team);
void updatePosition(Particle &p);

const unsigned int N = 5000; 
const unsigned int ITERATIONS = 1000; 
const int SEARCH_MIN = 0; 
const int SEARCH_MAX = 20; 
const float w = 0.9f; 
const float c_ind = 1.0f; 
const float c_team = 2.0f; 

// return a random int between low and high 
int randInt(int min, int max) {
    return min + rand() % (max - min + 1);
}

// function to optimize 
int calcValue(Sol_arr p, int* prices, int* device_1, int* device_2) {
    int price = 0;
    int offset_device_1 = p.array[0];
    int offset_device_2 = p.array[1];
    if (offset_device_1 < 0 || offset_device_1 > 19) {
        return 1000000;
    }
    if (offset_device_2 < 0 || offset_device_2 > 19) {
        return 1000000;
    }
    
    for (int i = 0; i < 5; i++) {
        price += (prices[offset_device_1 + i] * device_1[i]);
        price += (prices[offset_device_2 + i] * device_2[i]);
    }
    
    return price;
}

// Returns the index of the particle with the best position
int getTeamBestIndex(Particle* particles, int N) {
    int best_index = 0; 
    float current_team_best = particles[0].current_value; 
    for (int i = 1; i < N; i++) {
        if (particles[i].current_value < current_team_best) {
            best_index = i; 
            current_team_best = particles[i].current_value; 
        }
    }
    return best_index; 
}

// Calculate velocity for a particle 
void updateVelocity(Particle &p, Sol_arr team_best_position, float w, float c_ind, float c_team) {
    float r_ind = (float)rand() / RAND_MAX; 
    float r_team = (float)rand() / RAND_MAX;

    for (int i = 0; i < ARR_LEN; i++) {
        p.velocity.array[i] = (int) w * p.velocity.array[i] + 
                   r_ind * c_ind * (p.best_position.array[i] - p.current_position.array[i]) + 
                   r_team * c_team * (team_best_position.array[i] - p.current_position.array[i]); 
    }
}

// Updates current position, checks if best position and value need to be updated
void updatePosition(Particle &p, int* prices, int* device_1, int* device_2) {
    p.current_position += p.velocity; 
    int newValue = calcValue(p.current_position, prices, device_1, device_2); 
    if (newValue < p.current_value) {
        p.current_value = newValue; 
        p.best_position = p.current_position; 
    }
}


int main(void) {
    // for timing 
    long start = std::clock(); 

    // Random seed 
    std::srand(std::time(NULL)); 

    // Initialize particles 
    Particle* h_particles = new Particle[N]; 

    // Initialize pricing array
    int prices[24] = {100, 200, 300, 400, 500, 600, 700, 800, 900, 1000,
                      1100, 1200, 1, 50, 50, 1, 1, 1800, 1900,
                      2000, 2100, 2200, 2300, 2400};

    int device_1[5] = {1, 1, 2, 3, 4};
    int device_2[5] = {100, 1, 200, 3, 1};
    

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < ARR_LEN; j++) {
            h_particles[i].current_position.array[j] = randInt(SEARCH_MIN, SEARCH_MAX);
            h_particles[i].best_position.array[j] = h_particles[i].current_position.array[j];
            h_particles[i].current_value = calcValue(h_particles[i].best_position, prices, device_1, device_2);
            h_particles[i].velocity.array[j] = randInt(SEARCH_MIN, SEARCH_MAX); 
        }
    }

    // Calculate team best position and team best value 
    int team_best_index = getTeamBestIndex(h_particles, N); 
    Sol_arr team_best_position = h_particles[team_best_index].best_position; 
    int team_best_value = h_particles[team_best_index].current_value; 
    std::cout << "Starting Best: " << std::endl;
    std::cout << "Best Particle: " << team_best_index << std::endl; 
    std::cout << "Best value: " << team_best_value << std::endl; 
    std::cout << "Best position" << team_best_position.toString() << std::endl;

    // For i in interations 
    for (int i = 0; i < ITERATIONS; i++) {
        // for each particle 
        for (int j = 0; j < N; j++) {
            // For each particle calculate velocity 
            updateVelocity(h_particles[i], team_best_position, w, c_ind, c_team);
            // Update position and particle best value + position
            updatePosition(h_particles[i], prices, device_1, device_2); 
        }
        // Calculate team best 
        team_best_index = getTeamBestIndex(h_particles, N); 
        team_best_position = h_particles[team_best_index].best_position; 
        team_best_value = h_particles[team_best_index].current_value; 
    }

    long stop = std::clock(); 
    long elapsed = (stop - start) * 1000 / CLOCKS_PER_SEC;

    // print results 
    std::cout << "Ending Best: " << std::endl;
    std::cout << "Team best value: " << team_best_value << std::endl;
    std::cout << "Team best position: " << team_best_position.toString() << std::endl; 
    
    std::cout << "Run time: " << elapsed << "ms" << std::endl;
    return 0; 
}