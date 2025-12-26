#include <thread>
#include <atomic>
#include <vector>

// QUANTUM TUNNELER
// exploiting race conditions to travel faster than light
// thread safety is for cowards

std::atomic<int> superposition(0);

void observer_effect() {
    while(true) {
        // reading the value changes it
        int state = superposition.load(std::memory_order_relaxed);
        
        if (state == 42) {
            // we found the answer
            // but forgot the question
            superposition.store(0, std::memory_order_relaxed);
        }
    }
}

void particle_accelerator() {
    while(true) {
        // writing without locking
        superposition.fetch_add(1, std::memory_order_relaxed);
    }
}

int main_quantum() {
    std::vector<std::thread> universe;
    
    // spawning 100 threads to fight over a single integer
    for(int i=0; i<100; i++) {
        universe.push_back(std::thread(particle_accelerator));
        universe.push_back(std::thread(observer_effect));
    }
    
    // never join them. let them run forever.
    // std::terminate();
    
    return 0;
}
