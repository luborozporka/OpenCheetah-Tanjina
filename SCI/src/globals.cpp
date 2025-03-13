/*
Authors: Nishant Kumar, Deevashwer Rathee
Copyright:
Copyright (c) 2021 Microsoft Research
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include "globals.h"
#include "csv_writer.hpp"

sci::NetIO *io;
sci::OTPack<sci::NetIO> *otpack;

#ifdef SCI_OT
LinearOT *mult;
AuxProtocols *aux;
Truncation *truncation;
XTProtocol *xt;
MathFunctions *math;
#endif
ArgMaxProtocol<sci::NetIO, intType> *argmax;
ReLUProtocol<sci::NetIO, intType> *relu;
MaxPoolProtocol<sci::NetIO, intType> *maxpool;
// Additional classes for Athos
#ifdef SCI_OT
MatMulUniform<sci::NetIO, intType, sci::IKNP<sci::NetIO>> *multUniform;
#endif

#ifdef SCI_HE
FCField *he_fc;
ElemWiseProdField *he_prod;
#endif

#if USE_CHEETAH
gemini::CheetahLinear *cheetah_linear;
bool kIsSharedInput;
#elif defined(SCI_HE)
ConvField *he_conv;
#endif

sci::IKNP<sci::NetIO> *iknpOT;
sci::IKNP<sci::NetIO> *iknpOTRoleReversed;
sci::KKOT<sci::NetIO> *kkot;
sci::PRG128 *prg128Instance;

sci::NetIO *ioArr[MAX_THREADS];
sci::OTPack<sci::NetIO> *otpackArr[MAX_THREADS];
#ifdef SCI_OT
LinearOT *multArr[MAX_THREADS];
AuxProtocols *auxArr[MAX_THREADS];
Truncation *truncationArr[MAX_THREADS];
XTProtocol *xtArr[MAX_THREADS];
MathFunctions *mathArr[MAX_THREADS];
#endif
ReLUProtocol<sci::NetIO, intType> *reluArr[MAX_THREADS];
MaxPoolProtocol<sci::NetIO, intType> *maxpoolArr[MAX_THREADS];
// Additional classes for Athos
#ifdef SCI_OT
MatMulUniform<sci::NetIO, intType, sci::IKNP<sci::NetIO>> *multUniformArr[MAX_THREADS];
#endif
sci::IKNP<sci::NetIO> *otInstanceArr[MAX_THREADS];
sci::KKOT<sci::NetIO> *kkotInstanceArr[MAX_THREADS];
sci::PRG128 *prgInstanceArr[MAX_THREADS];

std::chrono::time_point<std::chrono::high_resolution_clock> start_time;
uint64_t comm_threads[MAX_THREADS];
uint64_t num_rounds;

#ifdef LOG_LAYERWISE
uint64_t ConvTimeInMilliSec = 0;
uint64_t MatAddTimeInMilliSec = 0;
uint64_t BatchNormInMilliSec = 0;
uint64_t TruncationTimeInMilliSec = 0;
uint64_t ReluTimeInMilliSec = 0;
uint64_t MaxpoolTimeInMilliSec = 0;
uint64_t AvgpoolTimeInMilliSec = 0;
uint64_t MatMulTimeInMilliSec = 0;
uint64_t MatAddBroadCastTimeInMilliSec = 0;
uint64_t MulCirTimeInMilliSec = 0;
uint64_t ScalarMulTimeInMilliSec = 0;
uint64_t SigmoidTimeInMilliSec = 0;
uint64_t TanhTimeInMilliSec = 0;
uint64_t SqrtTimeInMilliSec = 0;
uint64_t NormaliseL2TimeInMilliSec = 0;
uint64_t ArgMaxTimeInMilliSec = 0;

uint64_t ConvCommSent = 0;
uint64_t MatAddCommSent = 0;
uint64_t BatchNormCommSent = 0;
uint64_t TruncationCommSent = 0;
uint64_t ReluCommSent = 0;
uint64_t MaxpoolCommSent = 0;
uint64_t AvgpoolCommSent = 0;
uint64_t MatMulCommSent = 0;
uint64_t MatAddBroadCastCommSent = 0;
uint64_t MulCirCommSent = 0;
uint64_t ScalarMulCommSent = 0;
uint64_t SigmoidCommSent = 0;
uint64_t TanhCommSent = 0;
uint64_t SqrtCommSent = 0;
uint64_t NormaliseL2CommSent = 0;
uint64_t ArgMaxCommSent = 0;

/**
 * Added by Tanjina - starts
 */

// for Power readings in microwatts
uint64_t ConvTotalPowerConsumption = 0;
uint64_t ReluTotalPowerConsumption = 0;
uint64_t MaxPoolTotalPowerConsumption = 0;
uint64_t BatchNormTotalPowerConsumption = 0;
uint64_t MatMulTotalPowerConsumption = 0;
uint64_t AvgPoolTotalPowerConsumption = 0;
uint64_t ArgMaxTotalPowerConsumption = 0;

// for layer counter
int Conv_layer_count = 0;
int Relu_layer_count = 0;
int MaxPool_layer_count = 0;
int BatchNorm_layer_count = 0;
int MatMul_layer_count = 0;
int AvgPool_layer_count = 0;
int ArgMax_layer_count = 0;

// Path to the power usage
string power_usage_path = "/sys/class/hwmon/hwmon3/device/power1_average"; 

// Added by Tanjina
double computeAveragePower(uint64_t totalPower, int layerCount, const std::string& layerName){
  if(layerCount != 0){
    return (static_cast<double>(totalPower) / 1000000.0) / layerCount; // Convert from micro watts to watts
  }else{
    std::cerr << "Error: " << layerName << " layer count is 0, can not divide by zero!" << std::endl;
    return 0.0;
  }
}
/* for execution time/duration */
uint64_t ConvStartTime = 0;
uint64_t ConvEndTime = 0;
uint64_t ConvExecutionTime = 0;

uint64_t ReluStartTime = 0;
uint64_t ReluEndTime = 0;
double ReluExecutionTime = 0.0;

uint64_t MaxPoolStartTime = 0;
uint64_t MaxPoolEndTime = 0;
double MaxPoolExecutionTime = 0.0;

uint64_t BatchNormStartTime = 0;
uint64_t BatchNormEndTime = 0;
double BatchNormExecutionTime = 0.0;

uint64_t MatMulStartTime = 0;
uint64_t MatMulEndTime = 0;
double MatMulExecutionTime = 0.0;

uint64_t AvgPoolStartTime = 0;
uint64_t AvgPoolEndTime = 0;
double AvgPoolExecutionTime = 0.0;

uint64_t ArgMaxStartTime = 0;
uint64_t ArgMaxEndTime = 0;
double ArgMaxExecutionTime = 0.0;

//std::string layerType = "Conv";
std::string ConvOutputFile = "/home/tanjina/OpenCheetah-Tanjina/Output/conv_output.csv";
std::vector<std::string> ConvHeaders = {"index", "layer_name", "layer_number", "timestamp_power_reading", "avg_power_usage_mcW", "conv_start_timestamp", "conv_end_timestamp", "execution_time_ms", "conv_N", "conv_H", "conv_W", "conv_CI", "conv_FH", "conv_FW", "conv_CO", "conv_ zPadHLeft", "conv_zPadHRight", "conv_zPadWLeft", "conv_zPadWRight", "conv_strideH", "conv_strideW"};
WriteToCSV writeConvCSV(ConvOutputFile, ConvHeaders);

//std::string layerType = "Relu";
// std::string ReluOutputFile = "/home/tanjina/OpenCheetah-Tanjina/Output/relu_output.csv";
// std::vector<std::string> ReluHeaders = {"index", "layer_name", "layer_number", "timestamp_power_reading", "avg_power_usage", "execution time", "relu_coeff"};
// WriteToCSV writeReluCSV(ReluOutputFile, ReluHeaders);

//std::string layerType = "MaxPool";
// std::string MaxPoolOutputFile = "/home/tanjina/OpenCheetah-Tanjina/Output/maxpool_output.csv";
// std::vector<std::string> MaxPoolHeaders = {"index", "layer_name", "layer_number", "timestamp_power_reading", "avg_power_usage", "execution time", "maxpool_N", "maxpool_H", "maxpool_W", "maxpool_C", "maxpool_ksizeH", "maxpool_ksizeW", "maxpool_zPadHLeft", "maxpool_zPadHRight", "maxpool_zPadWLeft", "maxpool_zPadWRight", "maxpool_strideH", "maxpool_strideW", "maxpool_N1", "maxpool_imgH", "maxpool_imgW", "maxpool_C1"};
// WriteToCSV writeMaxPoolCSV(MaxPoolOutputFile, MaxPoolHeaders);

//std::string layerType = "BatchNorm 1";
// std::string BatchNorm1OutputFile = "/home/tanjina/OpenCheetah-Tanjina/Output/batchnorm1_output.csv";
// std::vector<std::string> BatchNorm1Headers = {"index", "layer_name", "layer_number", "timestamp_power_reading", "avg_power_usage", "execution time", "BN1_C", "BN1_H", "BN1_W"};
// WriteToCSV writeBatchNorm1CSV(BatchNorm1OutputFile, BatchNorm1Headers);

/**
 * Added by Tanjina - ends
 * 
 */

#endif
