/* ENERGY CONSUMPTION.hpp
 *   by Lut99
 *
 * Created:
 *   09 Oct 2024, 17:17:49
 * Last edited:
 *   11 Oct 2024, 14:45:37
 * Auto updated?
 *   Yes
 *
 * Description:
 *   A small library that can be used to measure energy consumption in the
 *   background of other C++ code running.
**/

#ifndef _ENERGY_CONSUMPTION_HPP
#define _ENERGY_CONSUMPTION_HPP

#include <cstdint>
#include <string>
#include <thread>
#include <vector>

#include "csv_writer.hpp"


/* Represents a single measurement of energy in the background. */
class EnergyMeasurement {
private:
    /* The thread that does the measurement in the background. May be NULL. */
    std::thread* thread;
    /* The results collected during measuring. */
    std::vector<std::pair<uint64_t, int64_t>>* results;
    /* The flag telling the thread to continue. */
    bool* measuring;

public:
    /* Constructor for the EnergyMeasurement.
     * 
     * # Arguments
     * - `measurement_file`: The input path where the measurement is taken from.
     */
    EnergyMeasurement(const std::string& measurement_file);
    /* Copy constructor for the EnergyMeasurement, which doesn't exist. */
    EnergyMeasurement(const EnergyMeasurement&) = delete;
    /* Move constructor for the EnergyMeasurement. */
    EnergyMeasurement(EnergyMeasurement&&);
    /* Destructor for the EnergyMeasurement. */
    ~EnergyMeasurement();



    /* Constructor for the EnergyMeasurement.
     * 
     * # Arguments
     * - `measurement_file`: The input path where the measurement is taken from.
     * 
     * # Returns
     * A new EnergyMeasurement that will measure as long as it exists. Call
     * `EnergyMeasurement::stop()` to stop it.
     */
    inline static EnergyMeasurement start(const std::string& measurement_file) { return EnergyMeasurement(measurement_file); }

    /* Stops the measurement.
     * 
     * This will block the current thread until one more result is read. This ensures that there
     * are always at least two measurements.
     * 
     * Note that the object is left in a legal, but unusable state. It won't measure anything
     * anymore.
     * 
     * # Returns
     * The list of results as obtained during the measurement.
     */
    std::vector<std::pair<uint64_t, int64_t>> stop();



    /* Copy assignment operator for the EnergyMeasurement, which doesn't exist. */
    EnergyMeasurement& operator=(const EnergyMeasurement&) = delete;
    /* Move assignment operator for the EnergyMeasurement. */
    inline EnergyMeasurement& operator=(EnergyMeasurement&& other) { if (this != &other) { swap(*this, other); } return *this; }
    /* Swap operator for the EnergyMeasurement class. */
    friend void swap(EnergyMeasurement& em1, EnergyMeasurement& em2) {
        using std::swap;

        swap(em1.thread, em2.thread);
        swap(em1.results, em2.results);
        swap(em1.measuring, em2.measuring);
    }
};

// Tanjina: hwmon sysfs path sampled by EnergyMeasurement
// TODO: configure per cheetah-server host at startup
extern std::string power_usage_path;

// Tanjina: average power per layer reported in per-layer summaries
double computeAveragePower(uint64_t totalPower, int layerCount,
                           const std::string& layerName);

// Estimates total energy used from a series of power readings
uint64_t integrate_energy_uj(const std::vector<std::pair<uint64_t, int64_t>>& samples);

// Tanjina: per-layer CSV for Conv layers (layer index, timestamps, avg power, geometry)
extern std::string ConvOutputFile;
extern std::vector<std::string> ConvHeaders;
extern WriteToCSV writeConvCSV;

#endif
