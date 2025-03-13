/** CSV_WRITER.hpp
 * 
 * Created by Tanjina
 * Creation date:
 * 15 Oct 2024
 * Last Edit on:
 * 21 Oct 2024
 * 
 * Description:
 * Header file for the CSV Writer library
 **/

#ifndef CSV_WRITER_HPP
#define CSV_WRITER_HPP

#include <fstream>
#include <string>
#include <vector>
#include <variant> // For multiple data types
// Define a varaint to handle different data type for output CSV column
using csv_column_type = std::variant<int, int64_t, uint64_t, std::string>; 

class WriteToCSV{
private:
    std:: ofstream csvFile;    
public:
    /* Constructor for the WriteToCSV Class
     * 
     * # Arguments
     * - `output_file_name`: The name of the output CSV file.
     */
    WriteToCSV(const std::string& output_file_name, const std::vector<std::string>& headers);

    /* Function to write a data row to the output CSV file
     * 
     * # Arguments
     * - `i`: Iteration number
     * - `layer_name`: The name of the current computation layer of a NN
     * - `layer_number`: Layer counter for a computing layer
     * - `timestamp_power_reading`: Timestamp when the avg_power_usage value is measured
     * - `avg_power_usage`: Average power usage value that is read from the system's power_usage_path
     */
    void insertDataRow(int i, const std::string& layer_name, int layer_counter, int64_t timestamp_power_reading, double avg_power_usage);

    // /* Function to define custom headers to the CSV file
    //  * 
    //  * # Arguments
    //  * - `headers` : A vector of column names 
    // */
    // void writeCSVHeaders(const std::vector<std::string>& headers);

    /* Function to write a data row into the output CSV file with mixed column types
     * # Arguments
     * - `dataRow_values`: Row values to insert into the CSV file using csv_column_type variant
     */
    void insertDataRow(const std::vector<csv_column_type>& dataRow_values); 

    // /* Function to write a data row into the output CSV file (for flexible columns)
    //  * 
    //  * # Arguments
    //  * - `dataRow_values`: Row values to insert into the CSV file (currenly all values are passed as string type)
    //  */
    // void WriteToCSV::insertDataRow(const std::vector<std::string>& dataRow_values);
    // /* Function to write a data row into the output CSV file (for flexible columns)
    //  * 
    //  * # Arguments
    //  * - `dataRow_values`: Row values to insert into the CSV file with int64_t data type
    //  */
    // void WriteToCSV::insertDataRow(const std::vector<int64_t>& dataRow_values);
    // /* Function to write a data row into the output CSV file (for flexible columns)
    //  * 
    //  * # Arguments
    //  * - `dataRow_values`: Row values to insert into the CSV file with int32_t data type
    //  */
    // void WriteToCSV::insertDataRow(const std::vector<int32_t>& dataRow_values);

    /* Destructor for the WriteToCSV Class */ 
    ~WriteToCSV();

};

#endif // CSV_WRITER_HPP
