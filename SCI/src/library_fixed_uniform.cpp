/*
Authors: Nishant Kumar, Deevashwer Rathee
Modified by Wen-jie Lu
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

#include "library_fixed_uniform.h"

#include "cleartext_library_fixed_uniform.h"
#include "functionalities_uniform.h"
#include "library_fixed_common.h"
#include "session.h"

#include "energy_consumption.hpp"
#include "csv_writer.hpp" // Added by Tanjina for writing the measurement values into a csv file

#define LOG_LAYERWISE
#define VERIFY_LAYERWISE
#undef VERIFY_LAYERWISE // undefine this to turn OFF the verifcation
// #undef LOG_LAYERWISE // undefine this to turn OFF the log

#ifdef SCI_HE
uint64_t prime_mod = sci::default_prime_mod.at(41);
#elif SCI_OT
uint64_t prime_mod = (bitlength == 64 ? 0ULL : 1ULL << bitlength);
uint64_t moduloMask = prime_mod - 1;
uint64_t moduloMidPt = prime_mod / 2;
#endif

#if !USE_CHEETAH
void MatMul2D(int32_t s1, int32_t s2, int32_t s3, const intType *A,
              const intType *B, intType *C, bool modelIsA) {
  MatMul2D(*sci::CurrentSession(), s1, s2, s3, A, B, C, modelIsA);
}

void MatMul2D(sci::Session &s, int32_t s1, int32_t s2, int32_t s3,
              const intType *A, const intType *B, intType *C, bool modelIsA) {
  // Alias session-owned resources
  const int party = s.party_value();
#if defined(SCI_OT)
  const int bitlength = s.bitlength_value();
  const int num_threads = s.num_threads_value();
  auto *multUniform = s.multUniform();
  auto *iknpOT = s.iknpOT();
  auto *iknpOTRoleReversed = s.iknpOTRoleReversed();
#elif defined(SCI_HE)
  auto *he_fc = s.he_fc();
#endif

#ifdef LOG_LAYERWISE
  INIT_ALL_IO_DATA_SENT;
  INIT_TIMER;
#endif

  std::cout << "Matmul called s1,s2,s3 = " << s1 << " " << s2 << " " << s3
            << std::endl;

  // By default, the model is A and server/Alice has it
  // So, in the AB mult, party with A = server and party with B = client.
  int partyWithAInAB_mul = sci::ALICE;
  int partyWithBInAB_mul = sci::BOB;
  if (!modelIsA) {
    // Model is B
    partyWithAInAB_mul = sci::BOB;
    partyWithBInAB_mul = sci::ALICE;
  }

#if defined(SCI_OT)
#ifndef MULTITHREADED_MATMUL
#ifdef USE_LINEAR_UNIFORM
  if (partyWithAInAB_mul == sci::ALICE) {
    if (party == sci::ALICE) {
      multUniform->funcOTSenderInputA(s1, s2, s3, A, C, iknpOT);
    } else {
      multUniform->funcOTReceiverInputB(s1, s2, s3, B, C, iknpOT);
    }
  } else {
    if (party == sci::BOB) {
      multUniform->funcOTSenderInputA(s1, s2, s3, A, C, iknpOTRoleReversed);
    } else {
      multUniform->funcOTReceiverInputB(s1, s2, s3, B, C, iknpOTRoleReversed);
    }
  }
#else  // USE_LINEAR_UNIFORM
  if (modelIsA) {
    mult->matmul_cross_terms(s1, s2, s3, A, B, C, bitlength, bitlength,
                             bitlength, true, MultMode::Alice_has_A);
  } else {
    mult->matmul_cross_terms(s1, s2, s3, A, B, C, bitlength, bitlength,
                             bitlength, true, MultMode::Alice_has_B);
  }
#endif  // USE_LINEAR_UNIFORM

  if (party == sci::ALICE) {
    // Now irrespective of whether A is the model or B is the model and whether
    //	server holds A or B, server should add locally A*B.
    //
    // Add also A*own share of B
    intType *CTemp = new intType[s1 * s3];
#ifdef USE_LINEAR_UNIFORM
    multUniform->ideal_func(s1, s2, s3, A, B, CTemp);
#else  // USE_LINEAR_UNIFORM
    mult->matmul_cleartext(s1, s2, s3, A, B, CTemp, true);
#endif  // USE_LINEAR_UNIFORM
    sci::elemWiseAdd<intType>(s1 * s3, C, CTemp, C);
    delete[] CTemp;
  } else {
    // For minionn kind of hacky runs, switch this off
#ifndef HACKY_RUN
    if (modelIsA) {
      for (int i = 0; i < s1 * s2; i++) assert(A[i] == 0);
    } else {
      for (int i = 0; i < s1 * s2; i++) assert(B[i] == 0);
    }
#endif
  }

#else  // MULTITHREADED_MATMUL is ON
  int required_num_threads = num_threads;
  if (s2 < num_threads) {
    required_num_threads = s2;
  }
  intType *C_ans_arr[required_num_threads];
  std::thread matmulThreads[required_num_threads];
  for (int i = 0; i < required_num_threads; i++) {
    C_ans_arr[i] = new intType[s1 * s3];
    matmulThreads[i] = std::thread(funcMatmulThread, i, required_num_threads,
                                   s1, s2, s3, (intType *)A, (intType *)B,
                                   (intType *)C_ans_arr[i], partyWithAInAB_mul);
  }
  for (int i = 0; i < required_num_threads; i++) {
    matmulThreads[i].join();
  }
  for (int i = 0; i < s1 * s3; i++) {
    C[i] = 0;
  }
  for (int i = 0; i < required_num_threads; i++) {
    for (int j = 0; j < s1 * s3; j++) {
      C[j] += C_ans_arr[i][j];
    }
    delete[] C_ans_arr[i];
  }

  if (party == sci::ALICE) {
    intType *CTemp = new intType[s1 * s3];
#ifdef USE_LINEAR_UNIFORM
    multUniform->ideal_func(s1, s2, s3, A, B, CTemp);
#else  // USE_LINEAR_UNIFORM
    mult->matmul_cleartext(s1, s2, s3, A, B, CTemp, true);
#endif  // USE_LINEAR_UNIFORM
    sci::elemWiseAdd<intType>(s1 * s3, C, CTemp, C);
    delete[] CTemp;
  } else {
    // For minionn kind of hacky runs, switch this off
#ifndef HACKY_RUN
    if (modelIsA) {
      for (int i = 0; i < s1 * s2; i++) assert(A[i] == 0);
    } else {
      for (int i = 0; i < s1 * s2; i++) assert(B[i] == 0);
    }
#endif
  }
#endif
  intType moduloMask = (1ULL << bitlength) - 1;
  if (bitlength == 64) moduloMask = -1;
  for (int i = 0; i < s1 * s3; i++) {
    C[i] = C[i] & moduloMask;
  }

#elif defined(SCI_HE)
  // We only support matrix vector multiplication.
  assert(modelIsA == false &&
         "Assuming code generated by compiler produces B as the model.");
  std::vector<std::vector<intType>> At(s2);
  std::vector<std::vector<intType>> Bt(s3);
  std::vector<std::vector<intType>> Ct(s3);
  for (int i = 0; i < s2; i++) {
    At[i].resize(s1);
    for (int j = 0; j < s1; j++) {
      At[i][j] = getRingElt(Arr2DIdxRowM(A, s1, s2, j, i));
    }
  }
  for (int i = 0; i < s3; i++) {
    Bt[i].resize(s2);
    Ct[i].resize(s1);
    for (int j = 0; j < s2; j++) {
      Bt[i][j] = getRingElt(Arr2DIdxRowM(B, s2, s3, j, i));
    }
  }
  he_fc->matrix_multiplication(s3, s2, s1, Bt, At, Ct);
  for (int i = 0; i < s1; i++) {
    for (int j = 0; j < s3; j++) {
      Arr2DIdxRowM(C, s1, s3, i, j) = getRingElt(Ct[j][i]);
    }
  }
#endif

#ifdef LOG_LAYERWISE
  auto temp = TIMER_TILL_NOW;
  MatMulTimeInMilliSec += temp;
  s.matmul_time_ms += temp;
  std::cout << "Time in sec for current matmul = " << (temp / 1000.0)
            << std::endl;
  uint64_t curComm;
  FIND_ALL_IO_TILL_NOW(curComm);
  MatMulCommSent += curComm;
  s.matmul_comm_sent += curComm;
#endif

#ifdef VERIFY_LAYERWISE
#ifdef SCI_HE
  for (int i = 0; i < s1; i++) {
    for (int j = 0; j < s3; j++) {
      assert(Arr2DIdxRowM(C, s1, s3, i, j) < prime_mod);
    }
  }
#endif
  if (party == SERVER) {
    funcReconstruct2PCCons(nullptr, A, s1 * s2);
    funcReconstruct2PCCons(nullptr, B, s2 * s3);
    funcReconstruct2PCCons(nullptr, C, s1 * s3);
  } else {
    signedIntType *VA = new signedIntType[s1 * s2];
    funcReconstruct2PCCons(VA, A, s1 * s2);
    signedIntType *VB = new signedIntType[s2 * s3];
    funcReconstruct2PCCons(VB, B, s2 * s3);
    signedIntType *VC = new signedIntType[s1 * s3];
    funcReconstruct2PCCons(VC, C, s1 * s3);

    std::vector<std::vector<uint64_t>> VAvec;
    std::vector<std::vector<uint64_t>> VBvec;
    std::vector<std::vector<uint64_t>> VCvec;
    VAvec.resize(s1, std::vector<uint64_t>(s2, 0));
    VBvec.resize(s2, std::vector<uint64_t>(s3, 0));
    VCvec.resize(s1, std::vector<uint64_t>(s3, 0));

    for (int i = 0; i < s1; i++) {
      for (int j = 0; j < s2; j++) {
        VAvec[i][j] = getRingElt(Arr2DIdxRowM(VA, s1, s2, i, j));
      }
    }
    for (int i = 0; i < s2; i++) {
      for (int j = 0; j < s3; j++) {
        VBvec[i][j] = getRingElt(Arr2DIdxRowM(VB, s2, s3, i, j));
      }
    }

    MatMul2D_pt(s1, s2, s3, VAvec, VBvec, VCvec, 0);

    bool pass = true;
    for (int i = 0; i < s1; i++) {
      for (int j = 0; j < s3; j++) {
        if (Arr2DIdxRowM(VC, s1, s3, i, j) != getSignedVal(VCvec[i][j])) {
          pass = false;
        }
      }
    }
    if (pass == true)
      std::cout << GREEN << "MatMul Output Matches" << RESET << std::endl;
    else
      std::cout << RED << "MatMul Output Mismatch" << RESET << std::endl;

    delete[] VA;
    delete[] VB;
    delete[] VC;
  }
#endif
}
#endif

static void Conv2D(int32_t N, int32_t H, int32_t W, int32_t CI, int32_t FH,
                   int32_t FW, int32_t CO, int32_t zPadHLeft,
                   int32_t zPadHRight, int32_t zPadWLeft, int32_t zPadWRight,
                   int32_t strideH, int32_t strideW, uint64_t *inputArr,
                   uint64_t *filterArr, uint64_t *outArr) {
  int32_t reshapedFilterRows = CO;

  int32_t reshapedFilterCols = ((FH * FW) * CI);

  int32_t reshapedIPRows = ((FH * FW) * CI);

  int32_t newH =
      ((((H + (zPadHLeft + zPadHRight)) - FH) / strideH) + (int32_t)1);

  int32_t newW =
      ((((W + (zPadWLeft + zPadWRight)) - FW) / strideW) + (int32_t)1);

  int32_t reshapedIPCols = ((N * newH) * newW);

  uint64_t *filterReshaped =
      make_array<uint64_t>(reshapedFilterRows, reshapedFilterCols);

  uint64_t *inputReshaped =
      make_array<uint64_t>(reshapedIPRows, reshapedIPCols);

  uint64_t *matmulOP = make_array<uint64_t>(reshapedFilterRows, reshapedIPCols);
  Conv2DReshapeFilter(FH, FW, CI, CO, filterArr, filterReshaped);
  Conv2DReshapeInput(N, H, W, CI, FH, FW, zPadHLeft, zPadHRight, zPadWLeft,
                     zPadWRight, strideH, strideW, reshapedIPRows,
                     reshapedIPCols, inputArr, inputReshaped);
  MatMul2D(reshapedFilterRows, reshapedFilterCols, reshapedIPCols,
           filterReshaped, inputReshaped, matmulOP, 1);
  Conv2DReshapeMatMulOP(N, newH, newW, CO, matmulOP, outArr);
  ClearMemSecret2(reshapedFilterRows, reshapedFilterCols, filterReshaped);
  ClearMemSecret2(reshapedIPRows, reshapedIPCols, inputReshaped);
  ClearMemSecret2(reshapedFilterRows, reshapedIPCols, matmulOP);
}

#if !USE_CHEETAH
void Conv2DWrapper(signedIntType N, signedIntType H, signedIntType W,
                   signedIntType CI, signedIntType FH, signedIntType FW,
                   signedIntType CO, signedIntType zPadHLeft,
                   signedIntType zPadHRight, signedIntType zPadWLeft,
                   signedIntType zPadWRight, signedIntType strideH,
                   signedIntType strideW, intType *inputArr, intType *filterArr,
                   intType *outArr) {
  Conv2DWrapper(*sci::CurrentSession(), N, H, W, CI, FH, FW, CO, zPadHLeft,
                zPadHRight, zPadWLeft, zPadWRight, strideH, strideW, inputArr,
                filterArr, outArr);
}

void Conv2DWrapper(sci::Session &s, signedIntType N, signedIntType H,
                   signedIntType W, signedIntType CI, signedIntType FH,
                   signedIntType FW, signedIntType CO, signedIntType zPadHLeft,
                   signedIntType zPadHRight, signedIntType zPadWLeft,
                   signedIntType zPadWRight, signedIntType strideH,
                   signedIntType strideW, intType *inputArr, intType *filterArr,
                   intType *outArr) {
  // Alias session-owned resources
  const int party = s.party_value();
#ifdef SCI_HE
  auto *he_conv = s.he_conv();
#endif

#ifdef LOG_LAYERWISE
  sleep(1); // Added by Tanjina to adjust the first power reading timestamp
  INIT_ALL_IO_DATA_SENT;
  INIT_TIMER;

  // Add by Eloise
  std::cout << "*******************" << std::endl;
  auto cur_start = CURRENT_TIME;
  std::cout << "Current time of start for current conv = " << cur_start
            << std::endl;
  ConvStartTime = cur_start; // Added by Tanjina to calculate the duration/execution time
  s.conv_start_time = cur_start;
#endif

/** 
  * Code block for power measurement in Conv layer starts
  * Added by - Tanjina
**/
#ifdef LOG_LAYERWISE
  // conv layer counter
  Conv_layer_count++;
  s.conv_layer_count++;

  std::cout << "STARTING ENERGY MEASUREMENT" << std::endl;
  // Pass the the Power usage file path to the Energy measurement library 
  EnergyMeasurement measurement(power_usage_path);

#endif

  static int ctr = 1;
  std::cout << "Conv2DCSF " << ctr << " called N=" << N << ", H=" << H
            << ", W=" << W << ", CI=" << CI << ", FH=" << FH << ", FW=" << FW
            << ", CO=" << CO << ", S=" << strideH << std::endl;
  ctr++;

  signedIntType newH = (((H + (zPadHLeft + zPadHRight) - FH) / strideH) + 1);
  signedIntType newW = (((W + (zPadWLeft + zPadWRight) - FW) / strideW) + 1);

#ifdef SCI_OT
  // If its a ring, then its a OT based -- use the default Conv2DCSF
  // implementation that comes from the EzPC library
  Conv2D(N, H, W, CI, FH, FW, CO, zPadHLeft, zPadHRight, zPadWLeft, zPadWRight,
         strideH, strideW, inputArr, filterArr, outArr);
#endif

#ifdef SCI_HE
  // If its a field, then its a HE based -- use the HE based conv implementation
  std::vector<std::vector<std::vector<std::vector<intType>>>> inputVec;
  inputVec.resize(N, std::vector<std::vector<std::vector<intType>>>(
                         H, std::vector<std::vector<intType>>(
                                W, std::vector<intType>(CI, 0))));

  std::vector<std::vector<std::vector<std::vector<intType>>>> filterVec;
  filterVec.resize(FH, std::vector<std::vector<std::vector<intType>>>(
                           FW, std::vector<std::vector<intType>>(
                                   CI, std::vector<intType>(CO, 0))));

  std::vector<std::vector<std::vector<std::vector<intType>>>> outputVec;
  outputVec.resize(N, std::vector<std::vector<std::vector<intType>>>(
                          newH, std::vector<std::vector<intType>>(
                                    newW, std::vector<intType>(CO, 0))));

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < H; j++) {
      for (int k = 0; k < W; k++) {
        for (int p = 0; p < CI; p++) {
          inputVec[i][j][k][p] =
              getRingElt(Arr4DIdxRowM(inputArr, N, H, W, CI, i, j, k, p));
        }
      }
    }
  }
  for (int i = 0; i < FH; i++) {
    for (int j = 0; j < FW; j++) {
      for (int k = 0; k < CI; k++) {
        for (int p = 0; p < CO; p++) {
          filterVec[i][j][k][p] =
              getRingElt(Arr4DIdxRowM(filterArr, FH, FW, CI, CO, i, j, k, p));
        }
      }
    }
  }

  he_conv->convolution(N, H, W, CI, FH, FW, CO, zPadHLeft, zPadHRight,
                       zPadWLeft, zPadWRight, strideH, strideW, inputVec,
                       filterVec, outputVec);

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < newH; j++) {
      for (int k = 0; k < newW; k++) {
        for (int p = 0; p < CO; p++) {
          Arr4DIdxRowM(outArr, N, newH, newW, CO, i, j, k, p) =
              getRingElt(outputVec[i][j][k][p]);
        }
      }
    }
  }

#endif

#ifdef LOG_LAYERWISE
  auto temp = TIMER_TILL_NOW;
  ConvTimeInMilliSec += temp;
  s.conv_time_ms += temp;
  std::cout << "Time in sec for current conv = " << (temp / 1000.0)
            << std::endl;
  uint64_t curComm;
  FIND_ALL_IO_TILL_NOW(curComm);
  ConvCommSent += curComm;
  s.conv_comm_sent += curComm;
#endif

#ifdef VERIFY_LAYERWISE
#ifdef SCI_HE
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < newH; j++) {
      for (int k = 0; k < newW; k++) {
        for (int p = 0; p < CO; p++) {
          assert(Arr4DIdxRowM(outArr, N, newH, newW, CO, i, j, k, p) <
                 prime_mod);
        }
      }
    }
  }
#endif
  if (party == SERVER) {
    funcReconstruct2PCCons(nullptr, inputArr, N * H * W * CI);
    funcReconstruct2PCCons(nullptr, filterArr, FH * FW * CI * CO);
    funcReconstruct2PCCons(nullptr, outArr, N * newH * newW * CO);
  } else {
    signedIntType *VinputArr = new signedIntType[N * H * W * CI];
    funcReconstruct2PCCons(VinputArr, inputArr, N * H * W * CI);
    signedIntType *VfilterArr = new signedIntType[FH * FW * CI * CO];
    funcReconstruct2PCCons(VfilterArr, filterArr, FH * FW * CI * CO);
    signedIntType *VoutputArr = new signedIntType[N * newH * newW * CO];
    funcReconstruct2PCCons(VoutputArr, outArr, N * newH * newW * CO);

    std::vector<std::vector<std::vector<std::vector<uint64_t>>>> VinputVec;
    VinputVec.resize(N, std::vector<std::vector<std::vector<uint64_t>>>(
                            H, std::vector<std::vector<uint64_t>>(
                                   W, std::vector<uint64_t>(CI, 0))));

    std::vector<std::vector<std::vector<std::vector<uint64_t>>>> VfilterVec;
    VfilterVec.resize(FH, std::vector<std::vector<std::vector<uint64_t>>>(
                              FW, std::vector<std::vector<uint64_t>>(
                                      CI, std::vector<uint64_t>(CO, 0))));

    std::vector<std::vector<std::vector<std::vector<uint64_t>>>> VoutputVec;
    VoutputVec.resize(N, std::vector<std::vector<std::vector<uint64_t>>>(
                             newH, std::vector<std::vector<uint64_t>>(
                                       newW, std::vector<uint64_t>(CO, 0))));

    for (int i = 0; i < N; i++) {
      for (int j = 0; j < H; j++) {
        for (int k = 0; k < W; k++) {
          for (int p = 0; p < CI; p++) {
            VinputVec[i][j][k][p] =
                getRingElt(Arr4DIdxRowM(VinputArr, N, H, W, CI, i, j, k, p));
          }
        }
      }
    }
    for (int i = 0; i < FH; i++) {
      for (int j = 0; j < FW; j++) {
        for (int k = 0; k < CI; k++) {
          for (int p = 0; p < CO; p++) {
            VfilterVec[i][j][k][p] = getRingElt(
                Arr4DIdxRowM(VfilterArr, FH, FW, CI, CO, i, j, k, p));
          }
        }
      }
    }

    Conv2DWrapper_pt(N, H, W, CI, FH, FW, CO, zPadHLeft, zPadHRight, zPadWLeft,
                     zPadWRight, strideH, strideW, VinputVec, VfilterVec,
                     VoutputVec);  // consSF = 0

    bool pass = true;
    for (int i = 0; i < N; i++) {
      for (int j = 0; j < newH; j++) {
        for (int k = 0; k < newW; k++) {
          for (int p = 0; p < CO; p++) {
            if (Arr4DIdxRowM(VoutputArr, N, newH, newW, CO, i, j, k, p) !=
                getSignedVal(VoutputVec[i][j][k][p])) {
              pass = false;
            }
          }
        }
      }
    }
    if (pass == true)
      std::cout << GREEN << "Convolution Output Matches" << RESET << std::endl;
    else
      std::cout << RED << "Convolution Output Mismatch" << RESET << std::endl;

    delete[] VinputArr;
    delete[] VfilterArr;
    delete[] VoutputArr;
  }
#endif

// Add by Eloise
#ifdef LOG_LAYERWISE
  auto cur_end = CURRENT_TIME;
  std::cout << "Current time of end for current conv = " << cur_end
            << std::endl;
  ConvEndTime = cur_end; // Added by Tanjina to calculate the duration/execution time
  s.conv_end_time = cur_end;
  // sleep(1); // Added by Tanjina 
# endif
  
/** 
 * Code block for power measurement in Conv layer ends
 * Added by - Tanjina
**/  
#ifdef LOG_LAYERWISE
  std::vector<std::pair<uint64_t, int64_t>> power_readings = measurement.stop();
  // ConvExecutionTime = (ConvEndTime - ConvStartTime) / 1000.0; // Added by Tanjina to calculate the duration/execution time (Convert from milliseconds to seconds)
  ConvExecutionTime = (ConvEndTime - ConvStartTime); // Note-Tanjina: Keep in milliseconds, need to do the conversion later
  s.conv_execution_time = ConvExecutionTime;
 
  for(int i = 0; i < power_readings.size(); ++i){
    uint64_t avgPower = power_readings[i].first; // Note-Tanjina: Keep in microwatts, need to do the conversion later
    int64_t timestampPower = power_readings[i].second;
    // double avgPowerUsage = avgPower / 1000000.0;

    ConvTotalPowerConsumption += avgPower;
    s.conv_total_power_uw += avgPower;
    std::cout << "Tanjina-Power usage values from the power_reading for HomConv #" << Conv_layer_count << " : " << avgPower << " microwatts " << "Timestamp of the current power reading: " << timestampPower << " Conv layer start Timestamp: " << ConvStartTime << " Conv layer end Timestamp: " << ConvEndTime <<  " Execution time: " << ConvExecutionTime << " milliseconds" << std::endl;
    // std::cout <<  "Tanjina-NN architecture info: " << "Conv_N = " << N << " Conv_H = " << H << " Conv_W = " << W << " Conv_CI = " << CI << " Conv_FH = " << FH << " Conv_FW = " << FW << " Conv_CO = " << CO << " Conv_ zPadHLeft = " << zPadHLeft << " Conv_zPadHRight = " << zPadHRight << " Conv_zPadWLeft = " << zPadWLeft  << " Conv_zPadWRight = " << zPadWRight << " Conv_strideH = " << strideH << " Conv_strideW = " << strideW << std::endl;
    
    std::vector<csv_column_type> conv_data;
    conv_data.push_back(i);
    conv_data.push_back("Conv");
    conv_data.push_back(Conv_layer_count);
    conv_data.push_back(timestampPower);
    conv_data.push_back(avgPower);
    conv_data.push_back(ConvStartTime);
    conv_data.push_back(ConvEndTime);
    conv_data.push_back(ConvExecutionTime);
    conv_data.push_back(N);
    conv_data.push_back(H);
    conv_data.push_back(W);
    conv_data.push_back(CI);
    conv_data.push_back(FH);
    conv_data.push_back(FW);
    conv_data.push_back(CO);
    conv_data.push_back(zPadHLeft);
    conv_data.push_back(zPadHRight);
    conv_data.push_back(zPadWLeft);
    conv_data.push_back(zPadWRight);
    conv_data.push_back(strideH);
    conv_data.push_back(strideW);

    writeConvCSV.insertDataRow(conv_data);
  }
  
#endif

}
#endif

#ifdef SCI_OT
void Conv2DGroup(int32_t N, int32_t H, int32_t W, int32_t CI, int32_t FH,
                 int32_t FW, int32_t CO, int32_t zPadHLeft, int32_t zPadHRight,
                 int32_t zPadWLeft, int32_t zPadWRight, int32_t strideH,
                 int32_t strideW, int32_t G, intType *inputArr,
                 intType *filterArr, intType *outArr);
#endif

void Conv2DGroupWrapper(signedIntType N, signedIntType H, signedIntType W,
                        signedIntType CI, signedIntType FH, signedIntType FW,
                        signedIntType CO, signedIntType zPadHLeft,
                        signedIntType zPadHRight, signedIntType zPadWLeft,
                        signedIntType zPadWRight, signedIntType strideH,
                        signedIntType strideW, signedIntType G,
                        intType *inputArr, intType *filterArr,
                        intType *outArr) {
  Conv2DGroupWrapper(*sci::CurrentSession(), N, H, W, CI, FH, FW, CO,
                     zPadHLeft, zPadHRight, zPadWLeft, zPadWRight, strideH,
                     strideW, G, inputArr, filterArr, outArr);
}

void Conv2DGroupWrapper(sci::Session &s, signedIntType N, signedIntType H,
                        signedIntType W, signedIntType CI, signedIntType FH,
                        signedIntType FW, signedIntType CO,
                        signedIntType zPadHLeft, signedIntType zPadHRight,
                        signedIntType zPadWLeft, signedIntType zPadWRight,
                        signedIntType strideH, signedIntType strideW,
                        signedIntType G, intType *inputArr, intType *filterArr,
                        intType *outArr) {
#ifdef LOG_LAYERWISE
  INIT_ALL_IO_DATA_SENT;
  INIT_TIMER;
#endif

  static int ctr = 1;
  std::cout << "Conv2DGroupCSF " << ctr << " called N=" << N << ", H=" << H
            << ", W=" << W << ", CI=" << CI << ", FH=" << FH << ", FW=" << FW
            << ", CO=" << CO << ", S=" << strideH << ",G=" << G << std::endl;
  ctr++;

#ifdef SCI_OT
  // If its a ring, then its a OT based -- use the default Conv2DGroupCSF
  // implementation that comes from the EzPC library
  Conv2DGroup(N, H, W, CI, FH, FW, CO, zPadHLeft, zPadHRight, zPadWLeft,
              zPadWRight, strideH, strideW, G, inputArr, filterArr, outArr);
#endif

#ifdef SCI_HE
  if (G == 1)
    Conv2DWrapper(s, N, H, W, CI, FH, FW, CO, zPadHLeft, zPadHRight, zPadWLeft,
                  zPadWRight, strideH, strideW, inputArr, filterArr, outArr);
  else
    assert(false && "Grouped conv not implemented in HE");
#endif

#ifdef LOG_LAYERWISE
  auto temp = TIMER_TILL_NOW;
  ConvTimeInMilliSec += temp;
  s.conv_time_ms += temp;
  std::cout << "Time in sec for current conv = " << (temp / 1000.0)
            << std::endl;
  uint64_t curComm;
  FIND_ALL_IO_TILL_NOW(curComm);
  ConvCommSent += curComm;
  s.conv_comm_sent += curComm;
#endif
}

#if !USE_CHEETAH
void ElemWiseActModelVectorMult(int32_t size, intType *inArr,
                                intType *multArrVec, intType *outputArr) {
  ElemWiseActModelVectorMult(*sci::CurrentSession(), size, inArr, multArrVec,
                             outputArr);
}

void ElemWiseActModelVectorMult(sci::Session &s, int32_t size, intType *inArr,
                                intType *multArrVec, intType *outputArr) {
  // Alias session-owned resources
  const int party = s.party_value();
#ifdef SCI_OT
  const int num_threads = s.num_threads_value();
#endif
#ifdef SCI_HE
  auto *he_prod = s.he_prod();
#endif

#ifdef LOG_LAYERWISE
  INIT_ALL_IO_DATA_SENT;
  INIT_TIMER;
#endif

  if (party == CLIENT) {
    for (int i = 0; i < size; i++) {
      assert((multArrVec[i] == 0) &&
             "The semantics of ElemWiseActModelVectorMult dictate multArrVec "
             "should be the model and client share should be 0 for it.");
    }
  }

  static int batchNormCtr = 1;
  std::cout << "Starting fused batchNorm #" << batchNormCtr << std::endl;
  batchNormCtr++;

#ifdef SCI_OT
#ifdef MULTITHREADED_DOTPROD
  std::thread dotProdThreads[num_threads];
  int chunk_size = ceil(size / double(num_threads));
  intType *inputArrPtr;
  if (party == SERVER) {
    inputArrPtr = multArrVec;
  } else {
    inputArrPtr = inArr;
  }
  for (int i = 0; i < num_threads; i++) {
    int offset = i * chunk_size;
    int curSize;
    curSize =
        ((i + 1) * chunk_size > size ? std::max(0, size - offset) : chunk_size);
    /*
    if (i == (num_threads - 1)) {
        curSize = size - offset;
    }
    else{
        curSize = chunk_size;
    }
    */
    dotProdThreads[i] = std::thread(funcDotProdThread, i, num_threads, curSize,
                                    multArrVec + offset, inArr + offset,
                                    outputArr + offset, false);
  }
  for (int i = 0; i < num_threads; ++i) {
    dotProdThreads[i].join();
  }
#else
  matmul->hadamard_cross_terms(size, multArrVec, inArr, outputArr, bitlength,
                               bitlength, bitlength, MultMode::Alice_has_A);
#endif

  if (party == SERVER) {
    for (int i = 0; i < size; i++) {
      outputArr[i] += (inArr[i] * multArrVec[i]);
    }
  } else {
    for (int i = 0; i < size; i++) {
      assert(multArrVec[i] == 0 && "Client's share of model is non-zero.");
    }
  }
#endif  // SCI_OT

#ifdef SCI_HE
  std::vector<uint64_t> tempInArr(size);
  std::vector<uint64_t> tempOutArr(size);
  std::vector<uint64_t> tempMultArr(size);

  for (int i = 0; i < size; i++) {
    tempInArr[i] = getRingElt(inArr[i]);
    tempMultArr[i] = getRingElt(multArrVec[i]);
  }

  he_prod->elemwise_product(size, tempInArr, tempMultArr, tempOutArr);

  for (int i = 0; i < size; i++) {
    outputArr[i] = getRingElt(tempOutArr[i]);
  }
#endif

#ifdef LOG_LAYERWISE
  auto temp = TIMER_TILL_NOW;
  BatchNormInMilliSec += temp;
  s.batch_norm_time_ms += temp;
  uint64_t curComm;
  FIND_ALL_IO_TILL_NOW(curComm);
  BatchNormCommSent += curComm;
  s.batch_norm_comm_sent += curComm;
  std::cout << "Time in sec for current BN = [" << (temp / 1000.0)
            << "] sent [" << (curComm / 1024. / 1024.) << "] MB"
            << std::endl;
#endif

#ifdef VERIFY_LAYERWISE
#ifdef SCI_HE
  for (int i = 0; i < size; i++) {
    assert(outputArr[i] < prime_mod);
  }
#endif
  if (party == SERVER) {
    funcReconstruct2PCCons(nullptr, inArr, size);
    funcReconstruct2PCCons(nullptr, multArrVec, size);
    funcReconstruct2PCCons(nullptr, outputArr, size);
  } else {
    signedIntType *VinArr = new signedIntType[size];
    funcReconstruct2PCCons(VinArr, inArr, size);
    signedIntType *VmultArr = new signedIntType[size];
    funcReconstruct2PCCons(VmultArr, multArrVec, size);
    signedIntType *VoutputArr = new signedIntType[size];
    funcReconstruct2PCCons(VoutputArr, outputArr, size);

    std::vector<uint64_t> VinVec(size);
    std::vector<uint64_t> VmultVec(size);
    std::vector<uint64_t> VoutputVec(size);

    for (int i = 0; i < size; i++) {
      VinVec[i] = getRingElt(VinArr[i]);
      VmultVec[i] = getRingElt(VmultArr[i]);
    }

    ElemWiseActModelVectorMult_pt(size, VinVec, VmultVec, VoutputVec);

    bool pass = true;
    for (int i = 0; i < size; i++) {
      if (VoutputArr[i] != getSignedVal(VoutputVec[i])) {
        pass = false;
      }
    }
    if (pass == true)
      std::cout << GREEN << "ElemWiseSecretVectorMult Output Matches" << RESET
                << std::endl;
    else
      std::cout << RED << "ElemWiseSecretVectorMult Output Mismatch" << RESET
                << std::endl;

    delete[] VinArr;
    delete[] VmultArr;
    delete[] VoutputArr;
  }
#endif
}
#endif

void ArgMax(int32_t s1, int32_t s2, intType *inArr, intType *outArr) {
  ArgMax(*sci::CurrentSession(), s1, s2, inArr, outArr);
}

void ArgMax(sci::Session &s, int32_t s1, int32_t s2, intType *inArr,
            intType *outArr) {
  // Alias session-owned resources
  const int party = s.party_value();
  auto *argmax = s.argmax();

#ifdef LOG_LAYERWISE
  INIT_ALL_IO_DATA_SENT;
  INIT_TIMER;
  // Add by Eloise >> Tanjina-Note: I moved them here. I could still found them in the log before when it was residing outside of this #ifdef LOG_LAYERWISE block, but to be consistent I placed them here!
  std::cout << "*******************" << std::endl;
  auto cur_start = CURRENT_TIME;
  std::cout << "Current time of start for current ArgMax = " << cur_start
            << std::endl;
  ArgMaxStartTime = cur_start; // Added by Tanjina to calculate the duration/execution time
  s.argmax_start_time = cur_start;
#endif

/** 
  * Code block for power measurement in ArgMax layer starts
  * Added by - Tanjina
**/
#ifdef LOG_LAYERWISE
  // ArgMax layer counter
  ArgMax_layer_count++;
  s.argmax_layer_count++;

  std::cout << "STARTING ENERGY MEASUREMENT" << std::endl;
  // Pass the the Power usage file path to the Energy measurement library 
  EnergyMeasurement measurement(power_usage_path);         

#endif

  static int ctr = 1;

  // Add by Eloise // Tanjina - Need to check if I found this in the log because it is outside the #ifdef LOG_LAYERWISE block! >> Tanjina-Note: Move it to #ifdef LOG_LAYERWISE block???
  // std::cout << "*******************" << std::endl;
  // auto cur_start = CURRENT_TIME;
  // std::cout << "Current time of start for current ArgMax = " << cur_start
  //           << std::endl;
  std::cout << "ArgMax #" << ctr << " called, s1=" << s1 << ", s2=" << s2
            << std::endl;
  ctr++;

  assert(s1 == 1 && "ArgMax impl right now assumes s1==1");
  argmax->ArgMaxMPC(s2, inArr, outArr);


#ifdef LOG_LAYERWISE
  auto temp = TIMER_TILL_NOW;
  ArgMaxTimeInMilliSec += temp;
  s.argmax_time_ms += temp;
  uint64_t curComm;
  FIND_ALL_IO_TILL_NOW(curComm);
  ArgMaxCommSent += curComm;
  s.argmax_comm_sent += curComm;
#endif

#ifdef VERIFY_LAYERWISE
  if (party == SERVER) {
    funcReconstruct2PCCons(nullptr, inArr, s1 * s2);
    funcReconstruct2PCCons(nullptr, outArr, s1);
  } else {
    signedIntType *VinArr = new signedIntType[s1 * s2];
    funcReconstruct2PCCons(VinArr, inArr, s1 * s2);
    signedIntType *VoutArr = new signedIntType[s1];
    funcReconstruct2PCCons(VoutArr, outArr, s1);

    std::vector<std::vector<uint64_t>> VinVec;
    VinVec.resize(s1, std::vector<uint64_t>(s2, 0));
    std::vector<uint64_t> VoutVec(s1);

    for (int i = 0; i < s1; i++) {
      for (int j = 0; j < s2; j++) {
        VinVec[i][j] = getRingElt(Arr2DIdxRowM(VinArr, s1, s2, i, j));
      }
    }

    ArgMax_pt(s1, s2, VinVec, VoutVec);

    bool pass = true;
    for (int i = 0; i < s1; i++) {
      std::cout << VoutArr[i] << " =? " << getSignedVal(VoutVec[i])
                << std::endl;
      if (VoutArr[i] != getSignedVal(VoutVec[i])) {
        pass = false;
      }
    }

    if (pass == true) {
      std::cout << GREEN << "ArgMax1 Output Matches" << RESET << std::endl;
    } else {
      std::cout << RED << "ArgMax1 Output Mismatch" << RESET << std::endl;
    }

    delete[] VinArr;
    delete[] VoutArr;
  }
#endif

// Add by Eloise >> Tanjina-Note: I moved them here. I could still found them in the log before when it was residing outside of this #ifdef LOG_LAYERWISE block, but to be consistent I placed them here!
#ifdef LOG_LAYERWISE
  auto cur_end = CURRENT_TIME;
  std::cout << "Current time of end for current ArgMax = " << cur_end
          << std::endl;
  ArgMaxEndTime = cur_end; // Added by Tanjina to calculate the duration/execution time
  s.argmax_end_time = cur_end;
#endif

/** 
  * Code block for power measurement in ArgMax layer ends
  * Added by - Tanjina
**/
#ifdef LOG_LAYERWISE
  std::vector<std::pair<uint64_t, int64_t>> power_readings = measurement.stop();
  // ArgMaxExecutionTime = (ArgMaxEndTime - ArgMaxStartTime) / 1000.0; // Added by Tanjina to calculate the duration/execution time (Convert from milliseconds to seconds)
  ArgMaxExecutionTime = (ArgMaxEndTime - ArgMaxStartTime); // Note-Tanjina: Keep in milliseconds, need to do the conversion later
  s.argmax_execution_time = ArgMaxExecutionTime;
  
  for(int i = 0; i < power_readings.size(); ++i){
    uint64_t avgPower = power_readings[i].first;
    int64_t timestampPower = power_readings[i].second;
    double avgPowerUsage = avgPower / 1000000.0;

    ArgMaxTotalPowerConsumption += avgPower;
    s.argmax_total_power_uw += avgPower;
    std::cout << "Tanjina-Power usage values from the power_reading for ArgMax #" << ArgMax_layer_count << " : " << avgPowerUsage << " watts " << "Timestamp of the current power reading: " << timestampPower << " Execution time: " << ArgMaxExecutionTime << " seconds" << std::endl; 
  }
        
#endif
}

void Relu(int32_t size, intType *inArr, intType *outArr, int sf, bool doTruncation) {
  Relu(*sci::CurrentSession(), size, inArr, outArr, sf, doTruncation);
}

void Relu(sci::Session &s, int32_t size, intType *inArr, intType *outArr,
          int sf, bool doTruncation) {
  // Alias session-owned resources
  const int party = s.party_value();
  const int bitlength = s.bitlength_value();
  const int num_threads = s.num_threads_value();
  auto *relu = s.relu();

#ifdef LOG_LAYERWISE
  INIT_ALL_IO_DATA_SENT;
  INIT_TIMER;

  // Add by Eloise
  std::cout << "*******************" << std::endl;
  auto cur_start = CURRENT_TIME;
  std::cout << "Current time of start for current relu = " << cur_start
            << std::endl;
  ReluStartTime = cur_start; // Added by Tanjina to calculate the duration/execution time
  s.relu_start_time = cur_start;
#endif

/** 
  * Code block for power measurement in Relu layer starts
  * Added by - Tanjina
**/
#ifdef LOG_LAYERWISE
  // Relu layer counter
  Relu_layer_count++;
  s.relu_layer_count++;

  std::cout << "STARTING ENERGY MEASUREMENT" << std::endl;
  // Pass the the Power usage file path to the Energy measurement library 
  EnergyMeasurement measurement(power_usage_path);         

#endif

  static int ctr = 1;
  printf("Relu #%d on %d points, truncate=%d by %d bits\n", ctr++, size, doTruncation, sf);
  ctr++;

  intType moduloMask = sci::all1Mask(bitlength);
  int eightDivElemts = ((size + 8 - 1) / 8) * 8;  //(ceil of s1*s2/8.0)*8
  uint8_t *msbShare = new uint8_t[eightDivElemts];
  intType *tempInp = new intType[eightDivElemts];
  intType *tempOutp = new intType[eightDivElemts];
  sci::copyElemWisePadded(size, inArr, eightDivElemts, tempInp, 0);

#ifndef MULTITHREADED_NONLIN
  relu->relu(tempOutp, tempInp, eightDivElemts, nullptr);
#else
  std::thread relu_threads[num_threads];
  int chunk_size = (eightDivElemts / (8 * num_threads)) * 8;
  for (int i = 0; i < num_threads; ++i) {
    int offset = i * chunk_size;
    int lnum_relu;
    if (i == (num_threads - 1)) {
      lnum_relu = eightDivElemts - offset;
    } else {
      lnum_relu = chunk_size;
    }
    relu_threads[i] = std::thread(funcReLUThread, i, tempOutp + offset, tempInp + offset, lnum_relu, nullptr, false);
  }
  for (int i = 0; i < num_threads; ++i) {
    relu_threads[i].join();
  }
#endif

#ifdef LOG_LAYERWISE
  auto temp = TIMER_TILL_NOW;
  ReluTimeInMilliSec += temp;
  s.relu_time_ms += temp;
  std::cout << "Time in sec for current relu = " << (temp / 1000.0) << std::endl;
  uint64_t curComm;
  FIND_ALL_IO_TILL_NOW(curComm);
  ReluCommSent += curComm;
  s.relu_comm_sent += curComm;

  // Add by Eloise
  // auto cur_end = CURRENT_TIME;
  // std::cout << "Current time of end for current relu = " << cur_end
  //           << std::endl;

#endif

  if (doTruncation) {
#ifdef LOG_LAYERWISE
    INIT_ALL_IO_DATA_SENT;
    INIT_TIMER;
#endif
    for (int i = 0; i < eightDivElemts; i++) {
      msbShare[i] = 0;  // After relu, all numbers are +ve
    }

    intType *tempTruncOutp = new intType[eightDivElemts];
#ifdef SCI_OT
    for (int i = 0; i < eightDivElemts; i++) {
      tempOutp[i] = tempOutp[i] & moduloMask;
    }

#if USE_CHEETAH == 0
    funcTruncateTwoPowerRingWrapper(eightDivElemts, tempOutp, tempTruncOutp, sf, bitlength, true, msbShare);
#else
    funcReLUTruncateTwoPowerRingWrapper(eightDivElemts, tempOutp, tempTruncOutp, sf, bitlength, true);
#endif

#else
    funcFieldDivWrapper<intType>(eightDivElemts, tempOutp, tempTruncOutp, 1ULL << sf, msbShare);
#endif
    memcpy(outArr, tempTruncOutp, size * sizeof(intType));
    delete[] tempTruncOutp;

#ifdef LOG_LAYERWISE
    auto temp = TIMER_TILL_NOW;
    TruncationTimeInMilliSec += temp;
    s.truncation_time_ms += temp;
    uint64_t curComm;
    FIND_ALL_IO_TILL_NOW(curComm);
    TruncationCommSent += curComm;
    s.truncation_comm_sent += curComm;
#endif
  } else {
    for (int i = 0; i < size; i++) {
      outArr[i] = tempOutp[i];
    }
  }

#ifdef SCI_OT
  for (int i = 0; i < size; i++) {
    outArr[i] = outArr[i] & moduloMask;
  }
#endif

#ifdef VERIFY_LAYERWISE
#ifdef SCI_HE
  for (int i = 0; i < size; i++) {
    assert(tempOutp[i] < prime_mod);
    assert(outArr[i] < prime_mod);
  }
#endif

  if (party == SERVER) {
    funcReconstruct2PCCons(nullptr, inArr, size);
    funcReconstruct2PCCons(nullptr, tempOutp, size);
    funcReconstruct2PCCons(nullptr, outArr, size);
  } else {
    signedIntType *VinArr = new signedIntType[size];
    funcReconstruct2PCCons(VinArr, inArr, size);
    signedIntType *VtempOutpArr = new signedIntType[size];
    funcReconstruct2PCCons(VtempOutpArr, tempOutp, size);
    signedIntType *VoutArr = new signedIntType[size];
    funcReconstruct2PCCons(VoutArr, outArr, size);

    std::vector<uint64_t> VinVec;
    VinVec.resize(size, 0);

    std::vector<uint64_t> VoutVec;
    VoutVec.resize(size, 0);

    for (int i = 0; i < size; i++) {
      VinVec[i] = getRingElt(VinArr[i]);
    }

    Relu_pt(size, VinVec, VoutVec, 0, false);  // sf = 0

    bool pass = true;
    for (int i = 0; i < size; i++) {
      if (VtempOutpArr[i] != getSignedVal(VoutVec[i])) {
        pass = false;
      }
    }
    if (pass == true)
      std::cout << GREEN << "ReLU Output Matches" << RESET << std::endl;
    else
      std::cout << RED << "ReLU Output Mismatch" << RESET << std::endl;

    ScaleDown_pt(size, VoutVec, sf);

    pass = true;
#if USE_CHEETAH
    constexpr signedIntType error_upper = 1;
#else
    constexpr signedIntType error_upper = 0;
#endif
    for (int i = 0; i < size; i++) {
      if (std::abs(VoutArr[i] - getSignedVal(VoutVec[i])) > error_upper) {
        pass = false;
      }
    }
    if (pass == true)
      std::cout << GREEN << "Truncation (after ReLU) Output Matches" << RESET
                << std::endl;
    else
      std::cout << RED << "Truncation (after ReLU) Output Mismatch" << RESET
                << std::endl;

    delete[] VinArr;
    delete[] VtempOutpArr;
    delete[] VoutArr;
  }
#endif

// Add by Eloise
#ifdef LOG_LAYERWISE
  auto cur_end = CURRENT_TIME;
  std::cout << "Current time of end for current relu = " << cur_end
            << std::endl;
  ReluEndTime = cur_end; // Added by Tanjina to calculate the duration/execution time
  s.relu_end_time = cur_end;
#endif
/** 
  * Code block for power measurement in Relu layer ends
  * Added by - Tanjina
**/
#ifdef LOG_LAYERWISE
  std::vector<std::pair<uint64_t, int64_t>> power_readings = measurement.stop();
  // ReluExecutionTime = (ReluEndTime - ReluStartTime) / 1000.0; // Added by Tanjina to calculate the duration/execution time (Convert from milliseconds to seconds)
  ReluExecutionTime = (ReluEndTime - ReluStartTime); // Note-Tanjina: Keep in milliseconds, need to do the conversion later
  s.relu_execution_time = ReluExecutionTime;

  for(int i = 0; i < power_readings.size(); ++i){
    uint64_t avgPower = power_readings[i].first;
    int64_t timestampPower = power_readings[i].second;
    double avgPowerUsage = avgPower / 1000000.0;

    ReluTotalPowerConsumption += avgPower;
    s.relu_total_power_uw += avgPower;
    std::cout << "Tanjina-Power usage values from the power_reading for Relu #" << Relu_layer_count << " : " << avgPowerUsage << " watts " << "Timestamp of the current power reading: " << timestampPower << " Execution time: " << ReluExecutionTime << " seconds" << " relu_coeff = " << size << std::endl; 

    // std::vector<csv_column_type> relu_data;
    // relu_data.push_back(i);
    // relu_data.push_back("Relu");
    // relu_data.push_back(Relu_layer_count);
    // relu_data.push_back(timestampPower);
    // relu_data.push_back(avgPowerUsage);
    // relu_data.push_back(ReluExecutionTime);
    // relu_data.push_back(size);

    // writeReluCSV.insertDataRow(relu_data);
  }       
#endif

  delete[] tempInp;
  delete[] tempOutp;
  delete[] msbShare;
}

void MaxPool(int32_t N, int32_t H, int32_t W, int32_t C, int32_t ksizeH,
             int32_t ksizeW, int32_t zPadHLeft, int32_t zPadHRight,
             int32_t zPadWLeft, int32_t zPadWRight, int32_t strideH,
             int32_t strideW, int32_t N1, int32_t imgH, int32_t imgW,
             int32_t C1, intType *inArr, intType *outArr) {
  MaxPool(*sci::CurrentSession(), N, H, W, C, ksizeH, ksizeW, zPadHLeft,
          zPadHRight, zPadWLeft, zPadWRight, strideH, strideW, N1, imgH, imgW,
          C1, inArr, outArr);
}

void MaxPool(sci::Session &s, int32_t N, int32_t H, int32_t W, int32_t C,
             int32_t ksizeH, int32_t ksizeW, int32_t zPadHLeft,
             int32_t zPadHRight, int32_t zPadWLeft, int32_t zPadWRight,
             int32_t strideH, int32_t strideW, int32_t N1, int32_t imgH,
             int32_t imgW, int32_t C1, intType *inArr, intType *outArr) {
  // Alias session-owned resources
  const int party = s.party_value();
  const int bitlength = s.bitlength_value();
  const int num_threads = s.num_threads_value();
  auto *maxpool = s.maxpool();

#ifdef LOG_LAYERWISE
  INIT_ALL_IO_DATA_SENT;
  INIT_TIMER;

  // Add by Eloise
  std::cout << "*******************" << std::endl;
  auto cur_start = CURRENT_TIME;
  std::cout << "Current time of start for current maxpool = " << cur_start
            << std::endl;
  MaxPoolStartTime = cur_start; // Added by Tanjina to calculate the duration/execution time
  s.maxpool_start_time = cur_start;
#endif
/** 
  * Code block for power measurement in MaxPool layer starts
  * Added by - Tanjina
**/
#ifdef LOG_LAYERWISE
  // MaxPool layer counter
  MaxPool_layer_count++;
  s.maxpool_layer_count++;

  std::cout << "STARTING ENERGY MEASUREMENT" << std::endl;
  // Pass the the Power usage file path to the Energy measurement library 
  EnergyMeasurement measurement(power_usage_path);       

#endif

  static int ctr = 1;
  std::cout << "Maxpool #" << ctr << " called N=" << N << ", H=" << H
            << ", W=" << W << ", C=" << C << ", ksizeH=" << ksizeH
            << ", ksizeW=" << ksizeW << ", zPadHLeft=" << zPadHLeft << ", zPadHRight=" << zPadHRight 
            << ", zPadWLeft=" << zPadWLeft << ", zPadWRight=" << zPadWRight 
            << ", strideH=" << strideH << ", strideW=" << strideW 
            << ", N1=" << N1 << ", imgH=" << imgH 
            << ", imgW=" << imgW << ", C1=" << C1 << std::endl;
  ctr++;

  uint64_t moduloMask = sci::all1Mask(bitlength);
  int rowsOrig = N * H * W * C;
  int rows = ((rowsOrig + 8 - 1) / 8) * 8;  //(ceil of rows/8.0)*8
  int cols = ksizeH * ksizeW;

  intType *reInpArr = new intType[rows * cols];
  intType *maxi = new intType[rows];
  intType *maxiIdx = new intType[rows];

  int rowIdx = 0;
  for (int n = 0; n < N; n++) {
    for (int c = 0; c < C; c++) {
      int32_t leftTopCornerH = -zPadHLeft;
      int32_t extremeRightBottomCornerH = imgH - 1 + zPadHRight;
      while ((leftTopCornerH + ksizeH - 1) <= extremeRightBottomCornerH) {
        int32_t leftTopCornerW = -zPadWLeft;
        int32_t extremeRightBottomCornerW = imgW - 1 + zPadWRight;
        while ((leftTopCornerW + ksizeW - 1) <= extremeRightBottomCornerW) {
          for (int fh = 0; fh < ksizeH; fh++) {
            for (int fw = 0; fw < ksizeW; fw++) {
              int32_t colIdx = fh * ksizeW + fw;
              int32_t finalIdx = rowIdx * (ksizeH * ksizeW) + colIdx;

              int32_t curPosH = leftTopCornerH + fh;
              int32_t curPosW = leftTopCornerW + fw;

              intType temp = 0;
              if ((((curPosH < 0) || (curPosH >= imgH)) ||
                   ((curPosW < 0) || (curPosW >= imgW)))) {
                temp = 0;
              } else {
                temp = Arr4DIdxRowM(inArr, N, imgH, imgW, C, n, curPosH,
                                    curPosW, c);
              }
              reInpArr[finalIdx] = temp;
            }
          }

          rowIdx += 1;
          leftTopCornerW = leftTopCornerW + strideW;
        }

        leftTopCornerH = leftTopCornerH + strideH;
      }
    }
  }

  for (int i = rowsOrig; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      reInpArr[i * cols + j] = 0;  // The extra padded values
    }
  }

#ifndef MULTITHREADED_NONLIN
  maxpool->funcMaxMPC(rows, cols, reInpArr, maxi, maxiIdx);
#else
  std::thread maxpool_threads[num_threads];
  int chunk_size = (rows / (8 * num_threads)) * 8;
  for (int i = 0; i < num_threads; ++i) {
    int offset = i * chunk_size;
    int lnum_rows;
    if (i == (num_threads - 1)) {
      lnum_rows = rows - offset;
    } else {
      lnum_rows = chunk_size;
    }
    maxpool_threads[i] =
        std::thread(funcMaxpoolThread, i, lnum_rows, cols,
                    reInpArr + offset * cols, maxi + offset, maxiIdx + offset);
  }
  for (int i = 0; i < num_threads; ++i) {
    maxpool_threads[i].join();
  }
#endif

  for (int n = 0; n < N; n++) {
    for (int c = 0; c < C; c++) {
      for (int h = 0; h < H; h++) {
        for (int w = 0; w < W; w++) {
          int iidx = n * C * H * W + c * H * W + h * W + w;
          Arr4DIdxRowM(outArr, N, H, W, C, n, h, w, c) = getRingElt(maxi[iidx]);
        }
      }
    }
  }

  delete[] reInpArr;
  delete[] maxi;
  delete[] maxiIdx;

#ifdef LOG_LAYERWISE
  auto temp = TIMER_TILL_NOW;
  MaxpoolTimeInMilliSec += temp;
  s.maxpool_time_ms += temp;
  std::cout << "Time in sec for current maxpool = " << (temp / 1000.0)
            << std::endl;
  uint64_t curComm;
  FIND_ALL_IO_TILL_NOW(curComm);
  MaxpoolCommSent += curComm;
  s.maxpool_comm_sent += curComm;
#endif

#ifdef VERIFY_LAYERWISE
#ifdef SCI_HE
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < H; j++) {
      for (int k = 0; k < W; k++) {
        for (int p = 0; p < C; p++) {
          assert(Arr4DIdxRowM(outArr, N, H, W, C, i, j, k, p) < prime_mod);
        }
      }
    }
  }
#endif
  if (party == SERVER) {
    funcReconstruct2PCCons(nullptr, inArr, N * imgH * imgW * C);
    funcReconstruct2PCCons(nullptr, outArr, N * H * W * C);
  } else {
    signedIntType *VinArr = new signedIntType[N * imgH * imgW * C];
    funcReconstruct2PCCons(VinArr, inArr, N * imgH * imgW * C);
    signedIntType *VoutArr = new signedIntType[N * H * W * C];
    funcReconstruct2PCCons(VoutArr, outArr, N * H * W * C);

    std::vector<std::vector<std::vector<std::vector<uint64_t>>>> VinVec;
    VinVec.resize(N, std::vector<std::vector<std::vector<uint64_t>>>(
                         imgH, std::vector<std::vector<uint64_t>>(
                                   imgW, std::vector<uint64_t>(C, 0))));

    std::vector<std::vector<std::vector<std::vector<uint64_t>>>> VoutVec;
    VoutVec.resize(N, std::vector<std::vector<std::vector<uint64_t>>>(
                          H, std::vector<std::vector<uint64_t>>(
                                 W, std::vector<uint64_t>(C, 0))));

    for (int i = 0; i < N; i++) {
      for (int j = 0; j < imgH; j++) {
        for (int k = 0; k < imgW; k++) {
          for (int p = 0; p < C; p++) {
            VinVec[i][j][k][p] =
                getRingElt(Arr4DIdxRowM(VinArr, N, imgH, imgW, C, i, j, k, p));
          }
        }
      }
    }

    MaxPool_pt(N, H, W, C, ksizeH, ksizeW, zPadHLeft, zPadHRight, zPadWLeft,
               zPadWRight, strideH, strideW, N1, imgH, imgW, C1, VinVec,
               VoutVec);

    bool pass = true;
    for (int i = 0; i < N; i++) {
      for (int j = 0; j < H; j++) {
        for (int k = 0; k < W; k++) {
          for (int p = 0; p < C; p++) {
            if (Arr4DIdxRowM(VoutArr, N, H, W, C, i, j, k, p) !=
                getSignedVal(VoutVec[i][j][k][p])) {
              pass = false;
              // std::cout << i << "\t" << j << "\t" << k << "\t" << p << "\t"
              // << Arr4DIdxRowM(VoutArr,N,H,W,C,i,j,k,p) << "\t" <<
              // getSignedVal(VoutVec[i][j][k][p]) << std::endl;
            }
          }
        }
      }
    }
    if (pass == true)
      std::cout << GREEN << "Maxpool Output Matches" << RESET << std::endl;
    else
      std::cout << RED << "Maxpool Output Mismatch" << RESET << std::endl;

    delete[] VinArr;
    delete[] VoutArr;
  }
#endif

// Add by Eloise
#ifdef LOG_LAYERWISE
  auto cur_end = CURRENT_TIME;
  std::cout << "Current time of end for current maxpool = " << cur_end
            << std::endl;
  MaxPoolEndTime = cur_end; // Added by Tanjina to calculate the duration/execution time
  s.maxpool_end_time = cur_end;
#endif
/** 
   * Code block for power measurement in MaxPool layer ends
   * Added by - Tanjina
**/
#ifdef LOG_LAYERWISE
  std::vector<std::pair<uint64_t, int64_t>> power_readings = measurement.stop();
  // MaxPoolExecutionTime = (MaxPoolEndTime - MaxPoolStartTime) / 1000.0; // Added by Tanjina to calculate the duration/execution time (Convert from milliseconds to seconds)
  MaxPoolExecutionTime = (MaxPoolEndTime - MaxPoolStartTime); // Note-Tanjina: Keep in milliseconds, need to do the conversion later
  s.maxpool_execution_time = MaxPoolExecutionTime;
  
  for(int i = 0; i < power_readings.size(); ++i){
    uint64_t avgPower = power_readings[i].first;
    int64_t timestampPower = power_readings[i].second;
    double avgPowerUsage = avgPower / 1000000.0;

    MaxPoolTotalPowerConsumption += avgPower;
    s.maxpool_total_power_uw += avgPower;
    std::cout << "Tanjina-Power usage values from the power_reading for MaxPool #" << MaxPool_layer_count << " : " << avgPowerUsage << " watts " << "Timestamp of the current power reading: " << timestampPower << " Execution time: " << MaxPoolExecutionTime << " seconds" << std::endl; 
    // std::cout << "Tanjina-NN architecture info: " << "MaxPool_N = " << N << " MaxPool_H = " << H << " MaxPool_W = " << W << " MaxPool_C = " << C << " MaxPool_ksizeH = " << ksizeH << " MaxPool_ksizeW = " << ksizeW << " MaxPool_zPadHLeft = " << zPadHLeft << " MaxPool_zPadHRight = " << zPadHRight << " MaxPool_zPadWLeft = " << zPadWLeft  << " MaxPool_zPadWRight = " << zPadWRight << " MaxPool_strideH = " << strideH << " MaxPool_strideW = " << strideW << " MaxPool_N1 = " << N1 << " MaxPool_imgH = " << imgH << " MaxPool_imgW = " << imgW << " MaxPool_C1 = " << C1 << std::endl;

    // std::vector<csv_column_type> maxpool_data;
    // maxpool_data.push_back(i);
    // maxpool_data.push_back("MaxPool");
    // maxpool_data.push_back(MaxPool_layer_count);
    // maxpool_data.push_back(timestampPower);
    // maxpool_data.push_back(avgPowerUsage);
    // maxpool_data.push_back(MaxPoolExecutionTime);
    // maxpool_data.push_back(N);
    // maxpool_data.push_back(H);
    // maxpool_data.push_back(W);
    // maxpool_data.push_back(C);
    // maxpool_data.push_back(ksizeH);
    // maxpool_data.push_back(ksizeW);
    // maxpool_data.push_back(zPadHLeft);
    // maxpool_data.push_back(zPadHRight);
    // maxpool_data.push_back(zPadWLeft);
    // maxpool_data.push_back(zPadWRight);
    // maxpool_data.push_back(strideH);
    // maxpool_data.push_back(strideW);
    // maxpool_data.push_back(N1);
    // maxpool_data.push_back(imgH);
    // maxpool_data.push_back(imgW);
    // maxpool_data.push_back(C1);

    // writeMaxPoolCSV.insertDataRow(maxpool_data);
  }        
#endif

}

void AvgPool(int32_t N, int32_t H, int32_t W, int32_t C, int32_t ksizeH,
             int32_t ksizeW, int32_t zPadHLeft, int32_t zPadHRight,
             int32_t zPadWLeft, int32_t zPadWRight, int32_t strideH,
             int32_t strideW, int32_t N1, int32_t imgH, int32_t imgW,
             int32_t C1, intType *inArr, intType *outArr) {
  AvgPool(*sci::CurrentSession(), N, H, W, C, ksizeH, ksizeW, zPadHLeft,
          zPadHRight, zPadWLeft, zPadWRight, strideH, strideW, N1, imgH, imgW,
          C1, inArr, outArr);
}

void AvgPool(sci::Session &s, int32_t N, int32_t H, int32_t W, int32_t C,
             int32_t ksizeH, int32_t ksizeW, int32_t zPadHLeft,
             int32_t zPadHRight, int32_t zPadWLeft, int32_t zPadWRight,
             int32_t strideH, int32_t strideW, int32_t N1, int32_t imgH,
             int32_t imgW, int32_t C1, intType *inArr, intType *outArr) {
  const int party = s.party_value();
  const int bitlength = s.bitlength_value();

#ifdef LOG_LAYERWISE
  INIT_ALL_IO_DATA_SENT;
  INIT_TIMER;

  // Add by Eloise
  std::cout << "*******************" << std::endl;
  auto cur_start = CURRENT_TIME;
  std::cout << "Current time of start for current avgpool = " << cur_start
            << std::endl;
  AvgPoolStartTime = cur_start; // Added by Tanjina to calculate the duration/execution time
  s.avgpool_start_time = cur_start;
#endif
/** 
  * Code block for power measurement in AvgPool layer starts
  * Added by - Tanjina
**/
#ifdef LOG_LAYERWISE
  // AvgPool layer counter
  AvgPool_layer_count++;
  s.avgpool_layer_count++;

  std::cout << "STARTING ENERGY MEASUREMENT" << std::endl;
  // Pass the the Power usage file path to the Energy measurement library 
  EnergyMeasurement measurement(power_usage_path);          

#endif

  static int ctr = 1;
  std::cout << "AvgPool #" << ctr << " called N=" << N << ", H=" << H
            << ", W=" << W << ", C=" << C << ", ksizeH=" << ksizeH
            << ", ksizeW=" << ksizeW << ", zPadHLeft=" << zPadHLeft << ", zPadHRight=" << zPadHRight 
            << ", zPadWLeft=" << zPadWLeft << ", zPadWRight=" << zPadWRight 
            << ", strideH=" << strideH << ", strideW=" << strideW 
            << ", N1=" << N1 << ", imgH=" << imgH 
            << ", imgW=" << imgW << ", C1=" << C1 << std::endl;
  ctr++;

  uint64_t moduloMask = sci::all1Mask(bitlength);
  int rows = N * H * W * C;
  int rowsPadded = ((rows + 8 - 1) / 8) * 8;
  intType *filterSum = new intType[rowsPadded];
  intType *filterAvg = new intType[rowsPadded];

  int rowIdx = 0;
  for (int n = 0; n < N; n++) {
    for (int c = 0; c < C; c++) {
      int32_t leftTopCornerH = -zPadHLeft;
      int32_t extremeRightBottomCornerH = imgH - 1 + zPadHRight;
      while ((leftTopCornerH + ksizeH - 1) <= extremeRightBottomCornerH) {
        int32_t leftTopCornerW = -zPadWLeft;
        int32_t extremeRightBottomCornerW = imgW - 1 + zPadWRight;
        while ((leftTopCornerW + ksizeW - 1) <= extremeRightBottomCornerW) {
          intType curFilterSum = 0;
          for (int fh = 0; fh < ksizeH; fh++) {
            for (int fw = 0; fw < ksizeW; fw++) {
              int32_t curPosH = leftTopCornerH + fh;
              int32_t curPosW = leftTopCornerW + fw;

              intType temp = 0;
              if ((((curPosH < 0) || (curPosH >= imgH)) ||
                   ((curPosW < 0) || (curPosW >= imgW)))) {
                temp = 0;
              } else {
                temp = Arr4DIdxRowM(inArr, N, imgH, imgW, C, n, curPosH,
                                    curPosW, c);
              }
#ifdef SCI_OT
              curFilterSum += temp;
#else
              curFilterSum =
                  sci::neg_mod(curFilterSum + temp, (int64_t)prime_mod);
#endif
            }
          }

          filterSum[rowIdx] = curFilterSum;
          rowIdx += 1;
          leftTopCornerW = leftTopCornerW + strideW;
        }

        leftTopCornerH = leftTopCornerH + strideH;
      }
    }
  }

  for (int i = rows; i < rowsPadded; i++) {
    filterSum[i] = 0;
  }

#ifdef SCI_OT
  for (int i = 0; i < rowsPadded; i++) {
    filterSum[i] = filterSum[i] & moduloMask;
  }
  funcAvgPoolTwoPowerRingWrapper(rowsPadded, filterSum, filterAvg,
                                 ksizeH * ksizeW);
#else
  for (int i = 0; i < rowsPadded; i++) {
    filterSum[i] = sci::neg_mod(filterSum[i], (int64_t)prime_mod);
  }
  funcFieldDivWrapper<intType>(rowsPadded, filterSum, filterAvg,
                               ksizeH * ksizeW, nullptr);
#endif

  for (int n = 0; n < N; n++) {
    for (int c = 0; c < C; c++) {
      for (int h = 0; h < H; h++) {
        for (int w = 0; w < W; w++) {
          int iidx = n * C * H * W + c * H * W + h * W + w;
          Arr4DIdxRowM(outArr, N, H, W, C, n, h, w, c) = filterAvg[iidx];
#ifdef SCI_OT
          Arr4DIdxRowM(outArr, N, H, W, C, n, h, w, c) =
              Arr4DIdxRowM(outArr, N, H, W, C, n, h, w, c) & moduloMask;
#endif
        }
      }
    }
  }

  delete[] filterSum;
  delete[] filterAvg;

#ifdef LOG_LAYERWISE
  auto temp = TIMER_TILL_NOW;
  AvgpoolTimeInMilliSec += temp;
  s.avgpool_time_ms += temp;
  std::cout << "Time in sec for current avgpool = " << (temp / 1000.0)
            << std::endl;
  uint64_t curComm;
  FIND_ALL_IO_TILL_NOW(curComm);
  AvgpoolCommSent += curComm;
  s.avgpool_comm_sent += curComm;
#endif

#ifdef VERIFY_LAYERWISE
#ifdef SCI_HE
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < H; j++) {
      for (int k = 0; k < W; k++) {
        for (int p = 0; p < C; p++) {
          assert(Arr4DIdxRowM(outArr, N, H, W, C, i, j, k, p) < prime_mod);
        }
      }
    }
  }
#endif
  if (party == SERVER) {
    funcReconstruct2PCCons(nullptr, inArr, N * imgH * imgW * C);
    funcReconstruct2PCCons(nullptr, outArr, N * H * W * C);
  } else {
    signedIntType *VinArr = new signedIntType[N * imgH * imgW * C];
    funcReconstruct2PCCons(VinArr, inArr, N * imgH * imgW * C);
    signedIntType *VoutArr = new signedIntType[N * H * W * C];
    funcReconstruct2PCCons(VoutArr, outArr, N * H * W * C);

    std::vector<std::vector<std::vector<std::vector<uint64_t>>>> VinVec;
    VinVec.resize(N, std::vector<std::vector<std::vector<uint64_t>>>(
                         imgH, std::vector<std::vector<uint64_t>>(
                                   imgW, std::vector<uint64_t>(C, 0))));

    std::vector<std::vector<std::vector<std::vector<uint64_t>>>> VoutVec;
    VoutVec.resize(N, std::vector<std::vector<std::vector<uint64_t>>>(
                          H, std::vector<std::vector<uint64_t>>(
                                 W, std::vector<uint64_t>(C, 0))));

    for (int i = 0; i < N; i++) {
      for (int j = 0; j < imgH; j++) {
        for (int k = 0; k < imgW; k++) {
          for (int p = 0; p < C; p++) {
            VinVec[i][j][k][p] =
                getRingElt(Arr4DIdxRowM(VinArr, N, imgH, imgW, C, i, j, k, p));
          }
        }
      }
    }

    AvgPool_pt(N, H, W, C, ksizeH, ksizeW, zPadHLeft, zPadHRight, zPadWLeft,
               zPadWRight, strideH, strideW, N1, imgH, imgW, C1, VinVec,
               VoutVec);

    bool pass = true;
    for (int i = 0; i < N; i++) {
      for (int j = 0; j < H; j++) {
        for (int k = 0; k < W; k++) {
          for (int p = 0; p < C; p++) {
            if (Arr4DIdxRowM(VoutArr, N, H, W, C, i, j, k, p) !=
                getSignedVal(VoutVec[i][j][k][p])) {
              pass = false;
            }
          }
        }
      }
    }

    if (pass == true)
      std::cout << GREEN << "AvgPool Output Matches" << RESET << std::endl;
    else
      std::cout << RED << "AvgPool Output Mismatch" << RESET << std::endl;

    delete[] VinArr;
    delete[] VoutArr;
  }
#endif

// Add by Eloise
#ifdef LOG_LAYERWISE
  auto cur_end = CURRENT_TIME;
  std::cout << "Current time of end for current avgpool = " << cur_end
            << std::endl;
  AvgPoolEndTime = cur_end; // Added by Tanjina to calculate the duration/execution time
  s.avgpool_end_time = cur_end;
#endif
/** 
   * Code block for power measurement in AvgPool layer ends
   * Added by - Tanjina
**/
#ifdef LOG_LAYERWISE
  std::vector<std::pair<uint64_t, int64_t>> power_readings = measurement.stop();
  // AvgPoolExecutionTime = (AvgPoolEndTime - AvgPoolStartTime) / 1000.0; // Added by Tanjina to calculate the duration/execution time (Convert from milliseconds to seconds)
  AvgPoolExecutionTime = (AvgPoolEndTime - AvgPoolStartTime); // Note-Tanjina: Keep in milliseconds, need to do the conversion later
  s.avgpool_execution_time = AvgPoolExecutionTime;
  
  for(int i = 0; i < power_readings.size(); ++i){
    uint64_t avgPower = power_readings[i].first;
    int64_t timestampPower = power_readings[i].second;
    double avgPowerUsage = avgPower / 1000000.0;

    AvgPoolTotalPowerConsumption += avgPower;
    s.avgpool_total_power_uw += avgPower;
    std::cout << "Tanjina-Power usage values from the power_reading for AvgPool #" << AvgPool_layer_count << " : " << avgPowerUsage << " watts " << "Timestamp of the current power reading: " << timestampPower << " Execution time: " << AvgPoolExecutionTime << " seconds" << std::endl;
  }       
#endif

}

void ScaleDown(int32_t size, intType *inArr, int32_t sf) {
  ScaleDown(*sci::CurrentSession(), size, inArr, sf);
}

void ScaleDown(sci::Session &s, int32_t size, intType *inArr, int32_t sf) {
  // Alias session-owned resources
  const int party = s.party_value();
  const int bitlength = s.bitlength_value();

#ifdef LOG_LAYERWISE
  INIT_ALL_IO_DATA_SENT;
  INIT_TIMER;
#endif
  static int ctr = 1;
  printf("Truncate #%d on %d points by %d bits\n", ctr++, size, sf);

  int eightDivElemts = ((size + 8 - 1) / 8) * 8; //(ceil of s1*s2/8.0)*8
  intType *tempInp;
  if (size != eightDivElemts) {
    tempInp = new intType[eightDivElemts];
    memcpy(tempInp, inArr, sizeof(intType) * size);
  } else {
    tempInp = inArr;
  }
  intType *outp = new intType[eightDivElemts];

#ifdef SCI_OT
  uint64_t moduloMask = sci::all1Mask(bitlength);
  for (int i = 0; i < eightDivElemts; i++) {
    tempInp[i] = tempInp[i] & moduloMask;
  }

  funcTruncateTwoPowerRingWrapper(eightDivElemts, tempInp, outp, sf, bitlength, true, nullptr);
#else
  for (int i = 0; i < eightDivElemts; i++) {
    tempInp[i] = sci::neg_mod(tempInp[i], (int64_t)prime_mod);
  }
  funcFieldDivWrapper<intType>(eightDivElemts, tempInp, outp, 1ULL << sf, nullptr);
#endif

#ifdef LOG_LAYERWISE
  auto temp = TIMER_TILL_NOW;
  TruncationTimeInMilliSec += temp;
  uint64_t curComm;
  FIND_ALL_IO_TILL_NOW(curComm);
  TruncationCommSent += curComm;
#endif

#ifdef VERIFY_LAYERWISE
#ifdef SCI_HE
  for (int i = 0; i < size; i++) {
    assert(outp[i] < prime_mod);
  }
#endif

  if (party == SERVER) {
    funcReconstruct2PCCons(nullptr, inArr, size);
    funcReconstruct2PCCons(nullptr, outp, size);
  } else {
    signedIntType *VinArr = new signedIntType[size];
    funcReconstruct2PCCons(VinArr, inArr, size);
    signedIntType *VoutpArr = new signedIntType[size];
    funcReconstruct2PCCons(VoutpArr, outp, size);

    std::vector<uint64_t> VinVec;
    VinVec.resize(size, 0);

    for (int i = 0; i < size; i++) {
      VinVec[i] = getRingElt(VinArr[i]);
    }

    ScaleDown_pt(size, VinVec, sf);

    bool pass = true;
#if USE_CHEETAH
    constexpr signedIntType error_upper = 1;
#else
    constexpr signedIntType error_upper = 0;
#endif
    for (int i = 0; i < size; i++) {
      if (std::abs(VoutpArr[i] - getSignedVal(VinVec[i])) > error_upper) {
        pass = false;
      }
    }

    if (pass == true)
      std::cout << GREEN << "Truncation4 Output Matches" << RESET << std::endl;
    else
      std::cout << RED << "Truncation4 Output Mismatch" << RESET << std::endl;

    delete[] VinArr;
    delete[] VoutpArr;
  }
#endif

  std::memcpy(inArr, outp, sizeof(intType) * size);
  delete[] outp;
  if (size != eightDivElemts) delete[] tempInp;
}

void ScaleUp(int32_t size, intType *arr, int32_t sf) {
  ScaleUp(*sci::CurrentSession(), size, arr, sf);
}

void ScaleUp(sci::Session &s, int32_t size, intType *arr, int32_t sf) {
  (void)s;
  for (int i = 0; i < size; i++) {
#ifdef SCI_OT
    arr[i] = (arr[i] << sf);
#else
    arr[i] = sci::neg_mod(arr[i] << sf, (int64_t)prime_mod);
#endif
  }
}

// Process-singleton session populated by StartComputation and torn down by EndComputation
static sci::Session g_main_session;

// Copies every pointer owned by s into the corresponding file-scope global declared in globals.cpp
static void PublishSessionAliases(sci::Session& s) {
  io = s.io();
  otpack = s.otpack();
  iknpOT = s.iknpOT();
  iknpOTRoleReversed = s.iknpOTRoleReversed();
  kkot = s.kkot();
  prg128Instance = s.prg128();

  relu = s.relu();
  maxpool = s.maxpool();
  argmax = s.argmax();

  for (int i = 0; i < MAX_THREADS; i++) {
    ioArr[i] = s.ioArr()[i];
    otpackArr[i] = s.otpackArr()[i];
    otInstanceArr[i] = s.otInstanceArr()[i];
    kkotInstanceArr[i] = s.kkotInstanceArr()[i];
    prgInstanceArr[i] = s.prgInstanceArr()[i];
    reluArr[i] = s.reluArr()[i];
    maxpoolArr[i] = s.maxpoolArr()[i];
  }

#ifdef SCI_OT
  mult = s.mult();
  aux = s.aux();
  truncation = s.truncation();
  xt = s.xt();
  math = s.math();
  multUniform = s.multUniform();
  for (int i = 0; i < MAX_THREADS; i++) {
    multArr[i] = s.multArr()[i];
    auxArr[i] = s.auxArr()[i];
    truncationArr[i] = s.truncationArr()[i];
    xtArr[i] = s.xtArr()[i];
    mathArr[i] = s.mathArr()[i];
    multUniformArr[i] = s.multUniformArr()[i];
  }
#endif

#ifdef SCI_HE
  he_fc = s.he_fc();
  he_prod = s.he_prod();
#endif

#if USE_CHEETAH
  cheetah_linear = s.cheetah_linear();
#elif defined(SCI_HE)
  he_conv = s.he_conv();
#endif
}

static void ClearSessionAliases() {
  io = nullptr;
  otpack = nullptr;
  iknpOT = nullptr;
  iknpOTRoleReversed = nullptr;
  kkot = nullptr;
  prg128Instance = nullptr;

  relu = nullptr;
  maxpool = nullptr;
  argmax = nullptr;

  for (int i = 0; i < MAX_THREADS; i++) {
    ioArr[i] = nullptr;
    otpackArr[i] = nullptr;
    otInstanceArr[i] = nullptr;
    kkotInstanceArr[i] = nullptr;
    prgInstanceArr[i] = nullptr;
    reluArr[i] = nullptr;
    maxpoolArr[i] = nullptr;
  }

#ifdef SCI_OT
  mult = nullptr;
  aux = nullptr;
  truncation = nullptr;
  xt = nullptr;
  math = nullptr;
  multUniform = nullptr;
  for (int i = 0; i < MAX_THREADS; i++) {
    multArr[i] = nullptr;
    auxArr[i] = nullptr;
    truncationArr[i] = nullptr;
    xtArr[i] = nullptr;
    mathArr[i] = nullptr;
    multUniformArr[i] = nullptr;
  }
#endif

#ifdef SCI_HE
  he_fc = nullptr;
  he_prod = nullptr;
#endif

#if USE_CHEETAH
  cheetah_linear = nullptr;
#elif defined(SCI_HE)
  he_conv = nullptr;
#endif
}

void StartComputation() {
  assert(bitlength < 64 && bitlength > 0);
  assert(num_threads <= MAX_THREADS);

  std::string backend;

#ifdef SCI_HE
  backend = "PrimeField";
  auto kv = sci::default_prime_mod.find(bitlength);
  if (kv == sci::default_prime_mod.end()) {
    bitlength = 41;
    prime_mod = sci::default_prime_mod.at(bitlength);
  } else {
    prime_mod = kv->second;
  }
#elif SCI_OT
  prime_mod = (bitlength == 64 ? 0ULL : 1ULL << bitlength);
  moduloMask = prime_mod - 1;
  moduloMidPt = prime_mod / 2;
  backend = "Ring";
#endif

#if USE_CHEETAH
  backend += "-SilentOT";
#else
  backend += "-OT";
#endif

  checkIfUsingEigen();
  printf("Doing BaseOT ...\n");

  g_main_session.setup(party, port, address, num_threads, bitlength, kScale);
  PublishSessionAliases(g_main_session);
  sci::SetCurrentSession(&g_main_session);

#if USE_CHEETAH
  backend += "-Cheetah";
#elif defined(SCI_HE)
  backend += "-SCI_HE";
  assertFieldRun();
#elif defined(SCI_OT)
  backend += "-SCI_OT";
#endif

  g_main_session.start_base_ot();

  std::cout << "After one-time setup, communication" << std::endl;
  start_time = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < num_threads; i++) {
    auto temp = ioArr[i]->counter;
    comm_threads[i] = temp;
    std::cout << "Thread i = " << i << ", total data sent till now = " << temp
              << std::endl;
  }
  std::cout << "-----------Syncronizing-----------" << std::endl;
  io->sync();
  num_rounds = io->num_rounds;
  std::cout << "secret_share_mod: " << prime_mod << " bitlength: " << bitlength << std::endl;
  std::cout << "backend: " << backend << std::endl;
  std::cout << "-----------Syncronized - now starting execution-----------"
            << std::endl;
}

void EndComputation() {
  auto endTimer = std::chrono::high_resolution_clock::now();
  auto execTimeInMilliSec =
      std::chrono::duration_cast<std::chrono::milliseconds>(endTimer -
                                                            start_time)
          .count();
  uint64_t totalComm = 0;
  for (int i = 0; i < num_threads; i++) {
    auto temp = ioArr[i]->counter;
    std::cout << "Thread i = " << i << ", total data sent till now = " << temp
              << std::endl;
    totalComm += (temp - comm_threads[i]);
  }
  uint64_t totalCommClient;
  std::cout << "------------------------------------------------------\n";
  std::cout << "------------------------------------------------------\n";
  std::cout << "------------------------------------------------------\n";
  std::cout << "Total time taken = " << execTimeInMilliSec
            << " milliseconds.\n";
  std::cout << "Total data sent = " << (totalComm / (1.0 * (1ULL << 20)))
            << " MiB." << std::endl;
  std::cout << "Number of rounds = " << ioArr[0]->num_rounds - num_rounds
            << std::endl;
  if (party == SERVER) {
    io->recv_data(&totalCommClient, sizeof(uint64_t));
    std::cout << "Total comm (sent+received) = "
              << ((totalComm + totalCommClient) / (1.0 * (1ULL << 20)))
              << " MiB." << std::endl;
  } else if (party == CLIENT) {
    io->send_data(&totalComm, sizeof(uint64_t));
    std::cout << "Total comm (sent+received) = (see SERVER OUTPUT)"
              << std::endl;
  }
  std::cout << "------------------------------------------------------\n";

#ifdef LOG_LAYERWISE
  std::cout << "Total time in Conv = " << (ConvTimeInMilliSec / 1000.0)
            << " seconds." << std::endl;
  std::cout << "Total time in MatMul = " << (MatMulTimeInMilliSec / 1000.0)
            << " seconds." << std::endl;
  std::cout << "Total time in BatchNorm = " << (BatchNormInMilliSec / 1000.0)
            << " seconds." << std::endl;
  std::cout << "Total time in Truncation = "
            << (TruncationTimeInMilliSec / 1000.0) << " seconds." << std::endl;
  std::cout << "Total time in Relu = " << (ReluTimeInMilliSec / 1000.0)
            << " seconds." << std::endl;
  std::cout << "Total time in MaxPool = " << (MaxpoolTimeInMilliSec / 1000.0)
            << " seconds." << std::endl;
  std::cout << "Total time in AvgPool = " << (AvgpoolTimeInMilliSec / 1000.0)
            << " seconds." << std::endl;
  std::cout << "Total time in ArgMax = " << (ArgMaxTimeInMilliSec / 1000.0)
            << " seconds." << std::endl;
  std::cout << "Total time in MatAdd = " << (MatAddTimeInMilliSec / 1000.0)
            << " seconds." << std::endl;
  std::cout << "Total time in MatAddBroadCast = "
            << (MatAddBroadCastTimeInMilliSec / 1000.0) << " seconds."
            << std::endl;
  std::cout << "Total time in MulCir = " << (MulCirTimeInMilliSec / 1000.0)
            << " seconds." << std::endl;
  std::cout << "Total time in ScalarMul = "
            << (ScalarMulTimeInMilliSec / 1000.0) << " seconds." << std::endl;
  std::cout << "Total time in Sigmoid = " << (SigmoidTimeInMilliSec / 1000.0)
            << " seconds." << std::endl;
  std::cout << "Total time in Tanh = " << (TanhTimeInMilliSec / 1000.0)
            << " seconds." << std::endl;
  std::cout << "Total time in Sqrt = " << (SqrtTimeInMilliSec / 1000.0)
            << " seconds." << std::endl;
  std::cout << "Total time in NormaliseL2 = "
            << (NormaliseL2TimeInMilliSec / 1000.0) << " seconds." << std::endl;
  std::cout << "------------------------------------------------------\n";
  std::cout << "Conv data sent = " << ((ConvCommSent) / (1.0 * (1ULL << 20)))
            << " MiB." << std::endl;
  std::cout << "MatMul data sent = "
            << ((MatMulCommSent) / (1.0 * (1ULL << 20))) << " MiB."
            << std::endl;
  std::cout << "BatchNorm data sent = "
            << ((BatchNormCommSent) / (1.0 * (1ULL << 20))) << " MiB."
            << std::endl;
  std::cout << "Truncation data sent = "
            << ((TruncationCommSent) / (1.0 * (1ULL << 20))) << " MiB."
            << std::endl;
  std::cout << "Relu data sent = " << ((ReluCommSent) / (1.0 * (1ULL << 20)))
            << " MiB." << std::endl;
  std::cout << "Maxpool data sent = "
            << ((MaxpoolCommSent) / (1.0 * (1ULL << 20))) << " MiB."
            << std::endl;
  std::cout << "Avgpool data sent = "
            << ((AvgpoolCommSent) / (1.0 * (1ULL << 20))) << " MiB."
            << std::endl;
  std::cout << "ArgMax data sent = "
            << ((ArgMaxCommSent) / (1.0 * (1ULL << 20))) << " MiB."
            << std::endl;
  std::cout << "MatAdd data sent = "
            << ((MatAddCommSent) / (1.0 * (1ULL << 20))) << " MiB."
            << std::endl;
  std::cout << "MatAddBroadCast data sent = "
            << ((MatAddBroadCastCommSent) / (1.0 * (1ULL << 20))) << " MiB."
            << std::endl;
  std::cout << "MulCir data sent = "
            << ((MulCirCommSent) / (1.0 * (1ULL << 20))) << " MiB."
            << std::endl;
  std::cout << "Sigmoid data sent = "
            << ((SigmoidCommSent) / (1.0 * (1ULL << 20))) << " MiB."
            << std::endl;
  std::cout << "Tanh data sent = " << ((TanhCommSent) / (1.0 * (1ULL << 20)))
            << " MiB." << std::endl;
  std::cout << "Sqrt data sent = " << ((SqrtCommSent) / (1.0 * (1ULL << 20)))
            << " MiB." << std::endl;
  std::cout << "NormaliseL2 data sent = "
            << ((NormaliseL2CommSent) / (1.0 * (1ULL << 20))) << " MiB."
            << std::endl;
  std::cout << "------------------------------------------------------\n";
  // Added by Tanjina - for power readings (total)
  std::cout << "Total power consumption in Conv layer = " << (ConvTotalPowerConsumption / 1000000.0) // Convert from micro watts to watts
            << " watts." << std::endl;
  std::cout << "Total power consumption in Relu layer = " << (ReluTotalPowerConsumption / 1000000.0) // Convert from micro watts to watts
            << " watts." << std::endl;
  std::cout << "Total power consumption in MaxPool layer = " << (MaxPoolTotalPowerConsumption / 1000000.0) // Convert from micro watts to watts
            << " watts." << std::endl;
  std::cout << "Total power consumption in BatchNorm layer = " << (BatchNormTotalPowerConsumption / 1000000.0) // Convert from micro watts to watts
            << " watts." << std::endl;
  std::cout << "Total power consumption in MatMul layer = " << (MatMulTotalPowerConsumption / 1000000.0) // Convert from micro watts to watts
            << " watts." << std::endl;
  std::cout << "Total power consumption in AvgPool layer = " << (AvgPoolTotalPowerConsumption / 1000000.0) // Convert from micro watts to watts
            << " watts." << std::endl;
  std::cout << "Total power consumption in ArgMax layer = " << (ArgMaxTotalPowerConsumption / 1000000.0) // Convert from micro watts to watts
            << " watts." << std::endl;
  std::cout << "------------------------------------------------------\n";
  // Added by Tanjina - for layer counts
  std::cout << "Total number of Conv layer = " << Conv_layer_count
            << " layers" << std::endl;
  std::cout << "Total number of Relu layer = " << Relu_layer_count
            << " layers" << std::endl;
  std::cout << "Total number of MaxPool layer = " << MaxPool_layer_count
            << " layers" << std::endl;
  std::cout << "Total number of BatchNorm layer = " << BatchNorm_layer_count
            << " layers" << std::endl;
  std::cout << "Total number of MatMul layer = " << MatMul_layer_count
            << " layers" << std::endl;
  std::cout << "Total number of AvgPool layer = " << AvgPool_layer_count
            << " layers" << std::endl;
  std::cout << "Total number of ArgMax layer = " << ArgMax_layer_count
            << " layers" << std::endl;
  std::cout << "------------------------------------------------------\n";

  if (party == SERVER) {
    uint64_t ConvCommSentClient = 0;
    uint64_t MatMulCommSentClient = 0;
    uint64_t BatchNormCommSentClient = 0;
    uint64_t TruncationCommSentClient = 0;
    uint64_t ReluCommSentClient = 0;
    uint64_t MaxpoolCommSentClient = 0;
    uint64_t AvgpoolCommSentClient = 0;
    uint64_t ArgMaxCommSentClient = 0;
    uint64_t MatAddCommSentClient = 0;
    uint64_t MatAddBroadCastCommSentClient = 0;
    uint64_t MulCirCommSentClient = 0;
    uint64_t ScalarMulCommSentClient = 0;
    uint64_t SigmoidCommSentClient = 0;
    uint64_t TanhCommSentClient = 0;
    uint64_t SqrtCommSentClient = 0;
    uint64_t NormaliseL2CommSentClient = 0;

    io->recv_data(&ConvCommSentClient, sizeof(uint64_t));
    io->recv_data(&MatMulCommSentClient, sizeof(uint64_t));
    io->recv_data(&BatchNormCommSentClient, sizeof(uint64_t));
    io->recv_data(&TruncationCommSentClient, sizeof(uint64_t));
    io->recv_data(&ReluCommSentClient, sizeof(uint64_t));
    io->recv_data(&MaxpoolCommSentClient, sizeof(uint64_t));
    io->recv_data(&AvgpoolCommSentClient, sizeof(uint64_t));
    io->recv_data(&ArgMaxCommSentClient, sizeof(uint64_t));
    io->recv_data(&MatAddCommSentClient, sizeof(uint64_t));
    io->recv_data(&MatAddBroadCastCommSentClient, sizeof(uint64_t));
    io->recv_data(&MulCirCommSentClient, sizeof(uint64_t));
    io->recv_data(&ScalarMulCommSentClient, sizeof(uint64_t));
    io->recv_data(&SigmoidCommSentClient, sizeof(uint64_t));
    io->recv_data(&TanhCommSentClient, sizeof(uint64_t));
    io->recv_data(&SqrtCommSentClient, sizeof(uint64_t));
    io->recv_data(&NormaliseL2CommSentClient, sizeof(uint64_t));

    std::cout << "Conv data (sent+received) = "
              << ((ConvCommSent + ConvCommSentClient) / (1.0 * (1ULL << 20)))
              << " MiB." << std::endl;
    std::cout << "MatMul data (sent+received) = "
              << ((MatMulCommSent + MatMulCommSentClient) /
                  (1.0 * (1ULL << 20)))
              << " MiB." << std::endl;
    std::cout << "BatchNorm data (sent+received) = "
              << ((BatchNormCommSent + BatchNormCommSentClient) /
                  (1.0 * (1ULL << 20)))
              << " MiB." << std::endl;
    std::cout << "Truncation data (sent+received) = "
              << ((TruncationCommSent + TruncationCommSentClient) /
                  (1.0 * (1ULL << 20)))
              << " MiB." << std::endl;
    std::cout << "Relu data (sent+received) = "
              << ((ReluCommSent + ReluCommSentClient) / (1.0 * (1ULL << 20)))
              << " MiB." << std::endl;
    std::cout << "Maxpool data (sent+received) = "
              << ((MaxpoolCommSent + MaxpoolCommSentClient) /
                  (1.0 * (1ULL << 20)))
              << " MiB." << std::endl;
    std::cout << "Avgpool data (sent+received) = "
              << ((AvgpoolCommSent + AvgpoolCommSentClient) /
                  (1.0 * (1ULL << 20)))
              << " MiB." << std::endl;
    std::cout << "ArgMax data (sent+received) = "
              << ((ArgMaxCommSent + ArgMaxCommSentClient) /
                  (1.0 * (1ULL << 20)))
              << " MiB." << std::endl;
    std::cout << "MatAdd data (sent+received) = "
              << ((MatAddCommSent + MatAddCommSentClient) /
                  (1.0 * (1ULL << 20)))
              << " MiB." << std::endl;
    std::cout << "MatAddBroadCast data (sent+received) = "
              << ((MatAddBroadCastCommSent + MatAddBroadCastCommSentClient) /
                  (1.0 * (1ULL << 20)))
              << " MiB." << std::endl;
    std::cout << "MulCir data (sent+received) = "
              << ((MulCirCommSent + MulCirCommSentClient) /
                  (1.0 * (1ULL << 20)))
              << " MiB." << std::endl;
    std::cout << "ScalarMul data (sent+received) = "
              << ((ScalarMulCommSent + ScalarMulCommSentClient) /
                  (1.0 * (1ULL << 20)))
              << " MiB." << std::endl;
    std::cout << "Sigmoid data (sent+received) = "
              << ((SigmoidCommSent + SigmoidCommSentClient) /
                  (1.0 * (1ULL << 20)))
              << " MiB." << std::endl;
    std::cout << "Tanh data (sent+received) = "
              << ((TanhCommSent + TanhCommSentClient) / (1.0 * (1ULL << 20)))
              << " MiB." << std::endl;
    std::cout << "Sqrt data (sent+received) = "
              << ((SqrtCommSent + SqrtCommSentClient) / (1.0 * (1ULL << 20)))
              << " MiB." << std::endl;
    std::cout << "NormaliseL2 data (sent+received) = "
              << ((NormaliseL2CommSent + NormaliseL2CommSentClient) /
                  (1.0 * (1ULL << 20)))
              << " MiB." << std::endl;

#ifdef WRITE_LOG
    std::string file_addr = "results-Porthos2PC-server.csv";
    bool write_title = true;
    {
      std::fstream result(file_addr.c_str(), std::fstream::in);
      if (result.is_open()) write_title = false;
      result.close();
    }
    std::fstream result(file_addr.c_str(),
                        std::fstream::out | std::fstream::app);
    if (write_title) {
      result << "Algebra,Bitlen,Base,#Threads,Total Time,Total Comm,Conv "
                "Time,Conv Comm,MatMul Time,MatMul Comm,BatchNorm "
                "Time,BatchNorm Comm,Truncation Time,Truncation Comm,ReLU "
                "Time,ReLU Comm,MaxPool Time,MaxPool Comm,AvgPool Time,AvgPool "
                "Comm,ArgMax Time,ArgMax Comm"
             << std::endl;
    }
    result << (isNativeRing ? "Ring" : "Field") << "," << bitlength << ","
           << MILL_PARAM << "," << num_threads << ","
           << execTimeInMilliSec / 1000.0 << ","
           << (totalComm + totalCommClient) / (1.0 * (1ULL << 20)) << ","
           << ConvTimeInMilliSec / 1000.0 << ","
           << (ConvCommSent + ConvCommSentClient) / (1.0 * (1ULL << 20)) << ","
           << MatMulTimeInMilliSec / 1000.0 << ","
           << (MatMulCommSent + MatMulCommSentClient) / (1.0 * (1ULL << 20))
           << "," << BatchNormInMilliSec / 1000.0 << ","
           << (BatchNormCommSent + BatchNormCommSentClient) /
                  (1.0 * (1ULL << 20))
           << "," << TruncationTimeInMilliSec / 1000.0 << ","
           << (TruncationCommSent + TruncationCommSentClient) /
                  (1.0 * (1ULL << 20))
           << "," << ReluTimeInMilliSec / 1000.0 << ","
           << (ReluCommSent + ReluCommSentClient) / (1.0 * (1ULL << 20)) << ","
           << MaxpoolTimeInMilliSec / 1000.0 << ","
           << (MaxpoolCommSent + MaxpoolCommSentClient) / (1.0 * (1ULL << 20))
           << "," << AvgpoolTimeInMilliSec / 1000.0 << ","
           << (AvgpoolCommSent + AvgpoolCommSentClient) / (1.0 * (1ULL << 20))
           << "," << ArgMaxTimeInMilliSec / 1000.0 << ","
           << (ArgMaxCommSent + ArgMaxCommSentClient) / (1.0 * (1ULL << 20))
           << std::endl;
    result.close();
#endif
  } else if (party == CLIENT) {
    io->send_data(&ConvCommSent, sizeof(uint64_t));
    io->send_data(&MatMulCommSent, sizeof(uint64_t));
    io->send_data(&BatchNormCommSent, sizeof(uint64_t));
    io->send_data(&TruncationCommSent, sizeof(uint64_t));
    io->send_data(&ReluCommSent, sizeof(uint64_t));
    io->send_data(&MaxpoolCommSent, sizeof(uint64_t));
    io->send_data(&AvgpoolCommSent, sizeof(uint64_t));
    io->send_data(&ArgMaxCommSent, sizeof(uint64_t));
    io->send_data(&MatAddCommSent, sizeof(uint64_t));
    io->send_data(&MatAddBroadCastCommSent, sizeof(uint64_t));
    io->send_data(&MulCirCommSent, sizeof(uint64_t));
    io->send_data(&ScalarMulCommSent, sizeof(uint64_t));
    io->send_data(&SigmoidCommSent, sizeof(uint64_t));
    io->send_data(&TanhCommSent, sizeof(uint64_t));
    io->send_data(&SqrtCommSent, sizeof(uint64_t));
    io->send_data(&NormaliseL2CommSent, sizeof(uint64_t));
  }
#endif

  sci::SetCurrentSession(nullptr);
  ClearSessionAliases();
  g_main_session.teardown();
}

intType SecretAdd(intType x, intType y) {
  return SecretAdd(*sci::CurrentSession(), x, y);
}

intType SecretAdd(sci::Session &s, intType x, intType y) {
  (void)s;
#ifdef SCI_OT
  return (x + y);
#else
  return sci::neg_mod(x + y, (int64_t)prime_mod);
#endif
}

intType SecretSub(intType x, intType y) {
  return SecretSub(*sci::CurrentSession(), x, y);
}

intType SecretSub(sci::Session &s, intType x, intType y) {
  (void)s;
#ifdef SCI_OT
  return (x - y);
#else
  return sci::neg_mod(x - y, (int64_t)prime_mod);
#endif
}

intType SecretMult(intType x, intType y) {
  return SecretMult(*sci::CurrentSession(), x, y);
}

intType SecretMult(sci::Session &s, intType x, intType y) {
  (void)s;
  // assert(false);
  return x * y;
}

void ElemWiseVectorPublicDiv(int32_t s1, intType *arr1, int32_t divisor,
                             intType *outArr) {
  ElemWiseVectorPublicDiv(*sci::CurrentSession(), s1, arr1, divisor, outArr);
}

void ElemWiseVectorPublicDiv(sci::Session &s, int32_t s1, intType *arr1,
                             int32_t divisor, intType *outArr) {
  (void)s;
  intType *inp;
  intType *out;
  const int alignment = 8;
  size_t aligned_size = (s1 + alignment - 1) &
                        -alignment;  // rounding up to multiple of alignment

  if ((size_t)s1 != aligned_size) {
    inp = new intType[aligned_size];
    out = new intType[aligned_size];
    memcpy(inp, arr1, s1 * sizeof(intType));
    memset(inp + s1, 0, (aligned_size - s1) * sizeof(intType));
  } else {
    inp = arr1;
    out = outArr;
  }
  assert(divisor > 0 && "No support for division by a negative divisor.");

#ifdef SCI_OT
  funcAvgPoolTwoPowerRingWrapper(aligned_size, inp, out, (intType)divisor);
#else
  funcFieldDivWrapper(aligned_size, inp, out, (intType)divisor, nullptr);
#endif

  if ((size_t)s1 != aligned_size) {
    memcpy(outArr, out, s1 * sizeof(intType));
    delete[] inp;
    delete[] out;
  }

  return;
}

void ElemWiseSecretSharedVectorMult(int32_t size, intType *inArr,
                                    intType *multArrVec, intType *outputArr) {
  ElemWiseSecretSharedVectorMult(*sci::CurrentSession(), size, inArr,
                                 multArrVec, outputArr);
}

void ElemWiseSecretSharedVectorMult(sci::Session &s, int32_t size,
                                    intType *inArr, intType *multArrVec,
                                    intType *outputArr) {
  // Alias session-owned resources
  const int party = s.party_value();
#ifdef SCI_OT
  const int num_threads = s.num_threads_value();
#endif

#ifdef LOG_LAYERWISE
  INIT_ALL_IO_DATA_SENT;
  INIT_TIMER;
#endif
  static int batchNormCtr = 1;
  std::cout << "Starting fused batchNorm #" << batchNormCtr << std::endl;
  batchNormCtr++;

#ifdef SCI_OT
#ifdef MULTITHREADED_DOTPROD
  std::thread dotProdThreads[num_threads];
  int chunk_size = (size / num_threads);
  for (int i = 0; i < num_threads; i++) {
    int offset = i * chunk_size;
    int curSize;
    if (i == (num_threads - 1)) {
      curSize = size - offset;
    } else {
      curSize = chunk_size;
    }
    dotProdThreads[i] = std::thread(funcDotProdThread, i, num_threads, curSize,
                                    multArrVec + offset, inArr + offset,
                                    outputArr + offset, true);
  }
  for (int i = 0; i < num_threads; ++i) {
    dotProdThreads[i].join();
  }
#else
  matmul->hadamard_cross_terms(size, multArrVec, inArr, outputArr, bitlength,
                               bitlength, bitlength, MultMode::None);
#endif

  for (int i = 0; i < size; i++) {
    outputArr[i] += (inArr[i] * multArrVec[i]);
  }
#endif

#ifdef LOG_LAYERWISE
  auto temp = TIMER_TILL_NOW;
  BatchNormInMilliSec += temp;
  s.batch_norm_time_ms += temp;
  uint64_t curComm;
  FIND_ALL_IO_TILL_NOW(curComm);
  BatchNormCommSent += curComm;
  s.batch_norm_comm_sent += curComm;
#endif

#ifdef VERIFY_LAYERWISE
  if (party == SERVER) {
    funcReconstruct2PCCons(nullptr, inArr, size);
    funcReconstruct2PCCons(nullptr, multArrVec, size);
    funcReconstruct2PCCons(nullptr, outputArr, size);
  } else {
    signedIntType *VinArr = new signedIntType[size];
    funcReconstruct2PCCons(VinArr, inArr, size);
    signedIntType *VmultArr = new signedIntType[size];
    funcReconstruct2PCCons(VmultArr, multArrVec, size);
    signedIntType *VoutputArr = new signedIntType[size];
    funcReconstruct2PCCons(VoutputArr, outputArr, size);

    std::vector<uint64_t> VinVec(size);
    std::vector<uint64_t> VmultVec(size);
    std::vector<uint64_t> VoutputVec(size);

    for (int i = 0; i < size; i++) {
      VinVec[i] = getRingElt(VinArr[i]);
      VmultVec[i] = getRingElt(VmultArr[i]);
    }

    ElemWiseSecretSharedVectorMult_pt(size, VinVec, VmultVec, VoutputVec);

    bool pass = true;
    for (int i = 0; i < size; i++) {
      if (VoutputArr[i] != getSignedVal(VoutputVec[i])) {
        pass = false;
      }
    }
    if (pass == true)
      std::cout << GREEN << "ElemWiseSecretSharedVectorMult Output Matches"
                << RESET << std::endl;
    else
      std::cout << RED << "ElemWiseSecretSharedVectorMult Output Mismatch"
                << RESET << std::endl;

    delete[] VinArr;
    delete[] VmultArr;
    delete[] VoutputArr;
  }
#endif
}

void Floor(int32_t s1, intType *inArr, intType *outArr, int32_t sf) {
  Floor(*sci::CurrentSession(), s1, inArr, outArr, sf);
}

void Floor(sci::Session &s, int32_t s1, intType *inArr, intType *outArr,
           int32_t sf) {
  (void)s;
  assert(false);
}
