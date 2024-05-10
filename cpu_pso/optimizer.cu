#include <iostream>
#include <stdlib.h>
#include <cmath> 
#include <string>
#include <ctime> 

const unsigned int ARR_LEN = 20;

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
int calcValue(Sol_arr p); 
int getTeamBestIndex(Particle* particles, int N);
void updateVelocity(Particle &p, Sol_arr team_best_position, float w, float c_ind, float c_team);
void updatePosition(Particle &p);

const unsigned int N = 10; 
const unsigned int ITERATIONS = 10000; 
const int SEARCH_MIN = -100; 
const int SEARCH_MAX = 100; 
const float w = 0.9f; 
const float c_ind = 1.0f; 
const float c_team = 2.0f; 

// return a random int between low and high 
int randInt(int min, int max) {
    return min + rand() % (max - min + 1);
}

// function to optimize 
int calcValue(Sol_arr p) {
    int sum = 500;
    for (int i = 0; i < ARR_LEN; i++) {
        // convert p.array[i] to int and add to sum
        sum += p.array[i] * p.array[i] * p.array[i];
    }
    return abs(sum);
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
void updatePosition(Particle &p) {
    p.current_position += p.velocity; 
    int newValue = calcValue(p.current_position); 
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

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < ARR_LEN; j++) {
            h_particles[i].current_position.array[j] = randInt(SEARCH_MIN, SEARCH_MAX);
            h_particles[i].best_position.array[j] = h_particles[i].current_position.array[j];
            h_particles[i].current_value = calcValue(h_particles[i].best_position);
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
            updatePosition(h_particles[i]); 
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