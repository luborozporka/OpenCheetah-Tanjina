/* ENERGY CONSUMPTION.cpp
 *   by Lut99
 *
 * Created:
 *   10 Oct 2024, 10:32:20
 * Last edited:
 *   11 Oct 2024, 14:48:59
 * Auto updated?
 *   Yes
 *
 * Description:
 *   A small library that can be used to measure energy consumption in the
 *   background of other C++ code running.
**/

#include <cerrno>
#include <chrono>
#include <cstring>
#include <iostream>
#include <fstream>

#include "energy_consumption.hpp"


/***** THREAD *****/
/* Defines the code running in the background thread.
 * 
 * # Arguments
 * - `running`: Pointer to some value that determines how long we should run.
 * - `input`: The handle to the file to read the measurements from.
 * - `results`: Pointer to the array to write the results to.
 */
void measurement_thread(bool* running, std::string input, std::vector<std::pair<uint64_t, int64_t>>* results) {
    // Loop
    for (uint64_t i = 0; *running; i++) {
        // Open the file
        std::ifstream input_h(input);
        if (input_h.fail()) {
            std::cerr << "ERROR: Failed to open measurement file '" << input << "': " << std::strerror(errno) << " (MEASUREMENT STOPPED)" << std::endl;
            return;
        }

        // Read the value
        uint64_t value;
        if (!(input_h >> value)) {
            if (input_h.eof()) {
                std::cerr << "WARNING: Measurement file '" << input << "' closed before we could read it" << std::endl;
            } else {
                std::cerr << "ERROR: Failed to read measurement file '" << input << "': " << std::strerror(errno) << std::endl;
            }
        }
        input_h.close();

        // Next, write to the output to the results file
        (*results).push_back(std::make_pair(value, std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count()));

        // Wait a second for the file to update
        // std::this_thread::sleep_for(std::chrono::seconds(1));
        std::this_thread::sleep_for(std::chrono::milliseconds(10)); // Wait 10ms for next power reading
    }
}





/***** IMPLEMENTATIONS *****/
EnergyMeasurement::EnergyMeasurement(EnergyMeasurement&& other) :
    thread(other.thread),
    results(other.results),
    measuring(other.measuring)
{
    // Ensure nothing gets deallocated
    other.thread = nullptr;
    other.results = nullptr;
    other.measuring = nullptr;
}

EnergyMeasurement::~EnergyMeasurement() {
    // Deallocate all things left to deallocate
    if (this->thread != nullptr) {
        // First, stop the thread before killing the object
        *(this->measuring) = false;
        this->thread->join();
        delete this->thread;
    }
    if (this->results != nullptr) {
        delete this->results;
    }
    if (this->measuring != nullptr) {
        delete this->measuring;
    }
}



EnergyMeasurement::EnergyMeasurement(const std::string& measurement_file):
    thread(nullptr),
    results(new std::vector<std::pair<uint64_t, int64_t>>()),
    measuring(new bool(false))
{
    // Launch the thread
    *this->measuring = true;
    this->thread = new std::thread(measurement_thread, this->measuring, measurement_file, this->results);
}

std::vector<std::pair<uint64_t, int64_t>> EnergyMeasurement::stop() {
    if (this->thread == nullptr || this->results == nullptr) { return std::vector<std::pair<uint64_t, int64_t>>(); }

    // First, stop the thread
    *(this->measuring) = false;
    this->thread->join();

    // Destroy the thread
    delete this->thread;
    this->thread = nullptr;

    // Get the results list out of ourselves
    std::vector<std::pair<uint64_t, int64_t>> res = std::move(*this->results);
    delete this->results;
    this->results = nullptr;

    // OK, return it
    return res;
}
