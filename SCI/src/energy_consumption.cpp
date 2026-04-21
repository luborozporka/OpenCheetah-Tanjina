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


// Tanjina's layer-wise power instrumentation
std::string power_usage_path = "/sys/class/hwmon/hwmon3/device/power1_average";

double computeAveragePower(uint64_t totalPower, int layerCount,
                           const std::string& layerName) {
    if (layerCount != 0) {
        // Convert from microwatts to watts
        return (static_cast<double>(totalPower) / 1000000.0) / layerCount;
    }
    std::cerr << "Error: " << layerName
              << " layer count is 0, can not divide by zero!" << std::endl;
    return 0.0;
}

uint64_t integrate_energy_uj(
    const std::vector<std::pair<uint64_t, int64_t>>& samples) {
    if (samples.size() < 2) return 0;
    double energy_nj = 0.0;
    for (size_t i = 1; i < samples.size(); ++i) {
        const double avg_p_uw =
            (static_cast<double>(samples[i - 1].first) +
             static_cast<double>(samples[i].first)) *
            0.5;
        const double dt_ms =
            static_cast<double>(samples[i].second - samples[i - 1].second);
        if (dt_ms > 0.0) energy_nj += avg_p_uw * dt_ms;
    }
    return static_cast<uint64_t>(energy_nj / 1000.0);
}

double measure_idle_power_w(const std::string& hwmon_path, int duration_ms, size_t* samples_out, int64_t* effective_duration_ms_out) {
    if (duration_ms <= 0) return 0.0;
    EnergyMeasurement em(hwmon_path);
    std::this_thread::sleep_for(std::chrono::milliseconds(duration_ms));
    auto samples = em.stop();
    if (samples_out) *samples_out = samples.size();
    if (effective_duration_ms_out) {
        *effective_duration_ms_out = samples.size() >= 2
            ? samples.back().second - samples.front().second
            : 0;
    }
    if (samples.empty()) return 0.0;
    long double sum_uw = 0.0L;
    for (const auto& s : samples) sum_uw += static_cast<long double>(s.first);
    const long double mean_uw = sum_uw / static_cast<long double>(samples.size());
    return static_cast<double>(mean_uw / 1.0e6L);
}

std::string ConvOutputFile = "Output/conv_output.csv";
std::vector<std::string> ConvHeaders = {
    "index", "layer_name", "layer_number", "timestamp_power_reading",
    "avg_power_usage_mcW", "conv_start_timestamp", "conv_end_timestamp",
    "execution_time_ms", "conv_N", "conv_H", "conv_W", "conv_CI",
    "conv_FH", "conv_FW", "conv_CO", "conv_ zPadHLeft", "conv_zPadHRight",
    "conv_zPadWLeft", "conv_zPadWRight", "conv_strideH", "conv_strideW"};
WriteToCSV writeConvCSV(ConvOutputFile, ConvHeaders);
