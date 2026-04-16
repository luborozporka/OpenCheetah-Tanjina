// Author: Wen-jie Lu
// Adapter for the SCI's implementation using Cheetah's linear protocols.
#if USE_CHEETAH

#include <gemini/cheetah/tensor.h>

#include "cheetah/cheetah-api.h"
#include "defines_uniform.h"
#include "globals.h"
#include "library_fixed_uniform.h"
#include "session.h"
#include "energy_consumption.hpp"
#include "csv_writer.hpp" // Added by Tanjina for writing the measurement values into a csv file

#define VERIFY_LAYERWISE
#define LOG _LAYERWISE
//#undef VERIFY_LAYERWISE // undefine this to turn OFF the verifcation
//#undef LOG_LAYERWISE // undefine this to turn OFF the log

#ifndef SCI_OT
extern int64_t getSignedVal(uint64_t x);
extern uint64_t getRingElt(int64_t x);
#else
extern uint64_t prime_mod;
extern uint64_t moduloMask;
extern uint64_t moduloMidPt;

static inline int64_t getSignedVal(uint64_t x) {
  assert(x < prime_mod);
  int64_t sx = x;
  if (x >= moduloMidPt) sx = x - prime_mod;
  return sx;
}

static inline uint64_t getRingElt(int64_t x) {
  return ((uint64_t)x) & moduloMask;
}
#endif

extern uint64_t SecretAdd(uint64_t x, uint64_t y);

#ifdef LOG_LAYERWISE
#include <vector>

typedef std::vector<uint64_t> uint64_1D;
typedef std::vector<std::vector<uint64_t>> uint64_2D;
typedef std::vector<std::vector<std::vector<uint64_t>>> uint64_3D;
typedef std::vector<std::vector<std::vector<std::vector<uint64_t>>>> uint64_4D;

extern void funcReconstruct2PCCons(signedIntType *y, const intType *x, int len);

// Helper functions for computing the ground truth
// See `cleartext_library_fixed_uniform.h`
extern void Conv2DWrapper_pt(uint64_t N, uint64_t H, uint64_t W, uint64_t CI,
                             uint64_t FH, uint64_t FW, uint64_t CO,
                             uint64_t zPadHLeft, uint64_t zPadHRight,
                             uint64_t zPadWLeft, uint64_t zPadWRight,
                             uint64_t strideH, uint64_t strideW,
                             uint64_4D &inputArr, uint64_4D &filterArr,
                             uint64_4D &outArr);

extern void MatMul2DEigen_pt(int64_t i, int64_t j, int64_t k, uint64_2D &A,
                             uint64_2D &B, uint64_2D &C, int64_t consSF);

extern void ElemWiseActModelVectorMult_pt(uint64_t s1, uint64_1D &arr1,
                                          uint64_1D &arr2, uint64_1D &outArr);
#endif

void MatMul2D(sci::Session &s, int32_t d0, int32_t d1, int32_t d2,
              const intType *mat_A, const intType *mat_B, intType *mat_C,
              bool is_A_weight_matrix) {
  // Alias session-owned resources
  auto *cheetah_linear = s.cheetah_linear();
  const int party = s.party_value();

#ifdef LOG_LAYERWISE
  INIT_ALL_IO_DATA_SENT;
  INIT_TIMER;

  // Add by Eloise
  std::cout << "*******************" << std::endl;
  auto cur_start = CURRENT_TIME;
  std::cout << "Current time of start for current matmul = " << cur_start
            << std::endl;
  s.matmul_start_time = cur_start;
#endif
/** 
  * Code block for power measurement in MatMul layer starts
  * Added by - Tanjina
**/
#ifdef LOG_LAYERWISE
  // matmul layer counter
  s.matmul_layer_count++;

  std::cout << "STARTING ENERGY MEASUREMENT" << std::endl;
  // Pass the the Power usage file path to the Energy measurement library 
  EnergyMeasurement measurement(power_usage_path);

#endif

  static int ctr = 1;
  printf(
      "Matmul #%d called d0=%ld, d1=%ld, d2=%ld, is_A_weight_matrix=%s\n",
      ctr++, d0, d1, d2, is_A_weight_matrix ? "true" : "false");
  ctr++;

  using namespace gemini;
  CheetahLinear::FCMeta meta;

  TensorShape mat_A_shape({d0, d1});
  TensorShape mat_B_shape({d1, d2});

  TensorShape input_shape = is_A_weight_matrix ? mat_B_shape : mat_A_shape;
  TensorShape weight_shape = is_A_weight_matrix ? mat_A_shape : mat_B_shape;
  meta.input_shape = TensorShape({input_shape.dim_size(1)});
  // Transpose
  meta.weight_shape =
      TensorShape({weight_shape.dim_size(1), weight_shape.dim_size(0)});
  meta.is_shared_input = kIsSharedInput;

  auto weight_mat = is_A_weight_matrix ? mat_A : mat_B;
  auto input_mat = is_A_weight_matrix ? mat_B : mat_A;

  Tensor<intType> weight_matrix;
  if (cheetah_linear->party() == SERVER) {
    // Transpose the weight matrix and convert the uint64_t to ring element
    weight_matrix.Reshape(meta.weight_shape);
    const size_t nrows = weight_shape.dim_size(0);
    const size_t ncols = weight_shape.dim_size(1);
    for (long r = 0; r < nrows; ++r) {
      for (long c = 0; c < ncols; ++c) {
        Arr2DIdxRowM(weight_matrix.data(), ncols, nrows, c, r) =
            getRingElt(Arr2DIdxRowM(weight_mat, nrows, ncols, r, c));
      }
    }
  }

  for (long r = 0; r < input_shape.rows(); ++r) {
    // row-major
    const intType *input_row = input_mat + r * input_shape.cols();

    Tensor<intType> input_vector;
    if (meta.is_shared_input) {
      input_vector = Tensor<intType>::Wrap(const_cast<intType *>(input_row),
                                           meta.input_shape);
    } else {
      input_vector.Reshape(meta.input_shape);
      std::transform(input_row, input_row + meta.input_shape.num_elements(),
                     input_vector.data(),
                     [](uint64_t v) { return getRingElt(v); });
    }

    Tensor<uint64_t> out_vec;
    cheetah_linear->fc(input_vector, weight_matrix, meta, out_vec);
    std::copy_n(out_vec.data(), out_vec.shape().num_elements(),
                mat_C + r * input_shape.cols());
  }

  if (cheetah_linear->party() == SERVER) {
    cheetah_linear->safe_erase(weight_matrix.data(),
                               meta.weight_shape.num_elements());
  }
#ifdef LOG_LAYERWISE
  auto temp = TIMER_TILL_NOW;
  s.matmul_time_ms += temp;
  std::cout << "Time in sec for current matmul = " << (temp / 1000.0)
            << std::endl;

  // Add by Eloise
  auto cur_end = CURRENT_TIME;
  std::cout << "Current time of end for current matmul = " << cur_end
            << std::endl;
  s.matmul_end_time = cur_end;

  uint64_t curComm;
  FIND_ALL_IO_TILL_NOW(curComm);
  s.matmul_comm_sent += curComm;
#endif
/** 
  * Code block for power measurement in MatMul layer ends
  * Added by - Tanjina
**/
#ifdef LOG_LAYERWISE
  std::vector<std::pair<uint64_t, int64_t>> power_readings = measurement.stop();
  s.matmul_execution_time = (s.matmul_end_time - s.matmul_start_time);
  
  for(int i = 0; i < power_readings.size(); ++i){
    uint64_t avgPower = power_readings[i].first;
    int64_t timestampPower = power_readings[i].second;
    double avgPowerUsage = avgPower / 1000000.0;

    s.matmul_total_power_uw += avgPower;
    std::cout << "Tanjina-Power usage values from the power_reading for MatMul #" << s.matmul_layer_count << " : " << avgPowerUsage << " watts " << "Timestamp of the current power reading: " << timestampPower << " Execution time: " << s.matmul_execution_time << " seconds" << std::endl;
  }

#endif

#ifdef VERIFY_LAYERWISE
  int s1 = d0;
  int s2 = d1;
  int s3 = d2;
  auto A = mat_A;
  auto B = mat_B;
  auto C = mat_C;
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

    MatMul2DEigen_pt(s1, s2, s3, VAvec, VBvec, VCvec, 0);

    bool pass = true;
    for (int i = 0; i < s1; i++) {
      for (int j = 0; j < s3; j++) {
        int64_t gnd = getSignedVal(VCvec[i][j]);
        int64_t cmp = Arr2DIdxRowM(VC, s1, s3, i, j);
        if (gnd != cmp) {
          if (pass) {
            std::cout << gnd << " => " << cmp << "\n";
          }
		  if (pass && std::abs(gnd - cmp) > 1) {
            pass = false;
		  }
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

void MatMul2D(int32_t d0, int32_t d1, int32_t d2, const intType *mat_A,
              const intType *mat_B, intType *mat_C, bool is_A_weight_matrix) {
  MatMul2D(*sci::CurrentSession(), d0, d1, d2, mat_A, mat_B, mat_C,
           is_A_weight_matrix);
}

void Conv2DWrapper(sci::Session &s, signedIntType N, signedIntType H,
                   signedIntType W, signedIntType CI, signedIntType FH,
                   signedIntType FW, signedIntType CO, signedIntType zPadHLeft,
                   signedIntType zPadHRight, signedIntType zPadWLeft,
                   signedIntType zPadWRight, signedIntType strideH,
                   signedIntType strideW, intType *inputArr, intType *filterArr,
                   intType *outArr) {
  // Alias session-owned resources
  auto *cheetah_linear = s.cheetah_linear();
  const int party = s.party_value();

#ifdef LOG_LAYERWISE
  sleep(1); // Added by Tanjina to adjust the first power reading timestamp
  INIT_ALL_IO_DATA_SENT;
  INIT_TIMER;
  
  // Add by Eloise
  std::cout << "*******************" << std::endl;
  auto cur_start = CURRENT_TIME;
  std::cout << "Current time of start for current conv = " << cur_start
            << std::endl;
  s.conv_start_time = cur_start;
#endif

/** 
* Code block for power measurement in Conv layer starts
* Added by - Tanjina
**/
#ifdef LOG_LAYERWISE
  // conv layer counter
  s.conv_layer_count++;

  std::cout << "STARTING ENERGY MEASUREMENT" << std::endl;
  // Pass the the Power usage file path to the Energy measurement library 
  EnergyMeasurement measurement(power_usage_path);

#endif

  if (zPadWLeft < zPadWRight) {
    std::swap(zPadWLeft, zPadWRight);
  }
  if (zPadHLeft < zPadHRight) {
    std::swap(zPadHLeft, zPadHRight);
  }
  static int ctr = 1;
  signedIntType newH = (((H + (zPadHLeft + zPadHRight) - FH) / strideH) + 1);
  signedIntType newW = (((W + (zPadWLeft + zPadWRight) - FW) / strideW) + 1);

  gemini::CheetahLinear::ConvMeta meta;
  meta.ishape = gemini::TensorShape({CI, H, W});
  meta.fshape = gemini::TensorShape({CI, FH, FW});
  meta.n_filters = CO;

  std::vector<gemini::Tensor<intType>> filters(CO);
  for (auto &f : filters) {
    f.Reshape(meta.fshape);
  }

  for (int i = 0; i < FH; i++) {
    for (int j = 0; j < FW; j++) {
      for (int k = 0; k < CI; k++) {
        for (int p = 0; p < CO; p++) {
          filters.at(p)(k, i, j) =
              getRingElt(Arr4DIdxRowM(filterArr, FH, FW, CI, CO, i, j, k, p));
        }
      }
    }
  }

  const int npads = zPadHLeft + zPadHRight + zPadWLeft + zPadWRight;
  meta.padding = npads == 0 ? gemini::Padding::VALID : gemini::Padding::SAME;
  meta.stride = strideH;
  meta.is_shared_input = kIsSharedInput;

  // printf(
  //     "HomConv #%d called N=%ld, H=%ld, W=%ld, CI=%ld, FH=%ld, FW=%ld, "
  //     "CO=%ld, S=%ld, Padding %s (%d %d %d %d)\n",
  //     ctr++, N, meta.ishape.height(), meta.ishape.width(),
  //     meta.ishape.channels(), meta.fshape.height(), meta.fshape.width(),
  //     meta.n_filters, meta.stride,
  //     (meta.padding == gemini::Padding::VALID ? "VALID" : "SAME"), zPadHLeft,
  //     zPadHRight, zPadWLeft, zPadWRight);

  printf(
      "HomConv #%d called N=%ld, H=%ld, W=%ld, CI=%ld, FH=%ld, FW=%ld, "
      "CO=%ld, zPadHLeft=%ld, zPadHRight=%ld, zPadWLeft=%ld, zPadWRight=%ld, "
      "strideH=%ld, strideW=%ld\n",
      ctr++, N, H, W, CI, FH, FW, CO, zPadHLeft, zPadHRight, zPadWLeft, zPadWRight,
      strideH, strideW);

#ifdef LOG_LAYERWISE
  const int64_t io_counter = cheetah_linear->io_counter();
#endif

  for (int i = 0; i < N; ++i) {
    gemini::Tensor<intType> image(meta.ishape);
    for (int j = 0; j < H; j++) {
      for (int k = 0; k < W; k++) {
        for (int p = 0; p < CI; p++) {
          image(p, j, k) =
              getRingElt(Arr4DIdxRowM(inputArr, N, H, W, CI, i, j, k, p));
        }
      }
    }

    gemini::Tensor<intType> out_tensor;
    cheetah_linear->conv2d(image, filters, meta, out_tensor);

    for (int j = 0; j < newH; j++) {
      for (int k = 0; k < newW; k++) {
        for (int p = 0; p < CO; p++) {
          Arr4DIdxRowM(outArr, N, newH, newW, CO, i, j, k, p) =
              out_tensor(p, j, k);
        }
      }
    }
  }

#ifdef LOG_LAYERWISE
  auto temp = TIMER_TILL_NOW;
  s.conv_time_ms += temp;
  const int64_t nbytes_sent = cheetah_linear->io_counter() - io_counter;
  std::cout << "Time in sec for current conv = [" << (temp / 1000.0)
            << "] sent [" << (nbytes_sent / 1024. / 1024.) << "] MB"
            << std::endl;

  uint64_t curComm;
  FIND_ALL_IO_TILL_NOW(curComm);
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
#endif  // SCI_HE

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
                     VoutputVec);

    bool pass = true;
    int err_cnt = 0;
    int pos_one = 0, neg_one = 0;
    for (int i = 0; i < N; i++) {
      for (int j = 0; j < newH; j++) {
        for (int k = 0; k < newW; k++) {
          for (int p = 0; p < CO; p++) {
            int64_t gnd = Arr4DIdxRowM(VoutputArr, N, newH, newW, CO, i, j, k, p);
            int64_t cmp = getSignedVal(VoutputVec[i][j][k][p]);
            int64_t diff = gnd - cmp;

            if (diff != 0) {

              if (diff > 0 && pos_one < 2) {
                std::cout << "expect " << gnd << " but got " << cmp << "\n";
              } 

			  if (diff < 0 && neg_one < 2) {
                std::cout << "expect " << gnd << " but got " << cmp << "\n";
              }

              pos_one += (diff > 0);
              neg_one += (diff < 0);
			  if (pass && std::abs(diff) > 1) {
				pass = false;
			  }
              ++err_cnt;
            }
          }
        }
      }
    }

    if (pass == true) {
      std::cout << GREEN << "Convolution Output Matches" << RESET << std::endl;
    } else {
      std::cout << RED << "Convolution Output Mismatch" << RESET << std::endl;
      printf("Error count %d (%d +1, %d -1). %f\%\n", err_cnt, pos_one,
             neg_one, static_cast<double>(err_cnt) * 100. / (N * newH * newW * CO));
    }

    delete[] VinputArr;
    delete[] VfilterArr;
    delete[] VoutputArr;
  }
#endif  // VERIFY_LAYERWISE

// Add by Eloise
#ifdef LOG_LAYERWISE
  auto cur_end = CURRENT_TIME;
  std::cout << "Current time of end for current conv = " << cur_end
            << std::endl;
  s.conv_end_time = cur_end;
  // sleep(1); // Added by Tanjina 
#endif
/** 
  * Code block for power measurement in Conv layer ends
  * Added by - Tanjina
**/
#ifdef LOG_LAYERWISE
  std::vector<std::pair<uint64_t, int64_t>> power_readings = measurement.stop();
  s.conv_execution_time = (s.conv_end_time - s.conv_start_time);

  for(int i = 0; i < power_readings.size(); ++i){
    uint64_t avgPower = power_readings[i].first; // Note-Tanjina: Keep in microwatts, need to do the conversion later
    int64_t timestampPower = power_readings[i].second;
    // double avgPowerUsage = avgPower / 1000000.0;

    s.conv_total_power_uw += power_readings[i].first;
    std::cout << "Tanjina-Power usage values from the power_reading for HomConv #" << s.conv_layer_count << " : " << avgPower << " microwatts " << "Timestamp of the current power reading: " << timestampPower << " Conv layer start Timestamp: " << s.conv_start_time << " Conv layer end Timestamp: " << s.conv_end_time << " Execution time: " << s.conv_execution_time << " milliseconds" << std::endl;
    // std::cout <<  "Tanjina-NN architecture info: " << "Conv_N = " << N << " Conv_H = " << H << " Conv_W = " << W << " Conv_CI = " << CI << " Conv_FH = " << FH << " Conv_FW = " << FW << " Conv_CO = " << CO << " Conv_ zPadHLeft = " << zPadHLeft << " Conv_zPadHRight = " << zPadHRight << " Conv_zPadWLeft = " << zPadWLeft  << " Conv_zPadWRight = " << zPadWRight << " Conv_strideH = " << strideH << " Conv_strideW = " << strideW << std::endl;
  
    std::vector<csv_column_type> conv_data;
    conv_data.push_back(i);
    conv_data.push_back("Conv");
    conv_data.push_back(s.conv_layer_count);
    conv_data.push_back(timestampPower);
    conv_data.push_back(avgPower);
    conv_data.push_back(s.conv_start_time);
    conv_data.push_back(s.conv_end_time);
    conv_data.push_back(s.conv_execution_time);
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

void BatchNorm(sci::Session &s, int32_t B, int32_t H, int32_t W, int32_t C,
               const intType *inputArr, const intType *scales,
               const intType *bias, intType *outArr) {
  // Alias session-owned resources
  auto *cheetah_linear = s.cheetah_linear();

#ifdef LOG_LAYERWISE
  INIT_ALL_IO_DATA_SENT;
  INIT_TIMER;

  // Add by Eloise
  std::cout << "*******************" << std::endl;
  auto cur_start = CURRENT_TIME;
  std::cout << "Current time of start for current BN1 = " << cur_start
            << std::endl;
  s.batch_norm_start_time = cur_start;
#endif
/** 
  * Code block for power measurement in BatchNorm layer starts
  * Added by - Tanjina
**/
#ifdef LOG_LAYERWISE
  // BN layer counter
  s.batch_norm_layer_count++;

  std::cout << "STARTING ENERGY MEASUREMENT" << std::endl;
  // Pass the the Power usage file path to the Energy measurement library 
  EnergyMeasurement measurement(power_usage_path);
           
#endif

  static int batchNormCtr = 1;

  gemini::CheetahLinear::BNMeta meta;
  meta.target_base_mod = prime_mod;
  meta.is_shared_input = kIsSharedInput;
  meta.ishape = gemini::TensorShape({C, H, W});

  std::cout << "HomBN1 #" << batchNormCtr << " on shape " << meta.ishape
            << std::endl;
  batchNormCtr++;

  gemini::Tensor<intType> scale_vec;
  scale_vec.Reshape(gemini::TensorShape({C}));
  if (cheetah_linear->party() == SERVER) {
    std::transform(scales, scales + C, scale_vec.data(), getRingElt);
  }

  gemini::Tensor<intType> in_tensor(meta.ishape);
  gemini::Tensor<intType> out_tensor;
  for (int b = 0; b < B; ++b) {
    for (int32_t h = 0; h < H; ++h) {
      for (int32_t w = 0; w < W; ++w) {
        for (int32_t c = 0; c < C; ++c) {
          in_tensor(c, h, w) =
              getRingElt(Arr4DIdxRowM(inputArr, B, H, W, C, b, h, w, c));
        }
      }
    }

    cheetah_linear->bn_direct(in_tensor, scale_vec, meta, out_tensor);

    for (int32_t h = 0; h < H; ++h) {
      for (int32_t w = 0; w < W; ++w) {
        for (int32_t c = 0; c < C; ++c) {
          Arr4DIdxRowM(outArr, B, H, W, C, b, h, w, c) =
              SecretAdd(out_tensor(c, h, w), bias[c]);
        }
      }
    }
  }

  if (cheetah_linear->party() == SERVER) {
    cheetah_linear->safe_erase(scale_vec.data(), scale_vec.NumElements());
  }

#ifdef LOG_LAYERWISE
  auto temp = TIMER_TILL_NOW;
  s.batch_norm_time_ms += temp;
  uint64_t curComm;
  FIND_ALL_IO_TILL_NOW(curComm);
  s.batch_norm_comm_sent += curComm;
  std::cout << "Time in sec for current BN1 = [" << (temp / 1000.0) << "] sent ["
            << (curComm / 1024. / 1024.) << "] MB" << std::endl;

  // Add by Eloise
  auto cur_end = CURRENT_TIME;
  std::cout << "Current time of end for current BN1 = " << cur_end
            << std::endl;
  s.batch_norm_end_time = cur_end;
#endif
/** 
  * Code block for power measurement in BatchNorm layer ends
  * Added by - Tanjina
**/
#ifdef LOG_LAYERWISE
  std::vector<std::pair<uint64_t, int64_t>> power_readings = measurement.stop();
  s.batch_norm_execution_time = (s.batch_norm_end_time - s.batch_norm_start_time);

  for(int i = 0; i < power_readings.size(); ++i){
    uint64_t avgPower = power_readings[i].first;
    int64_t timestampPower = power_readings[i].second;
    double avgPowerUsage = avgPower / 1000000.0;

    s.batch_norm_total_power_uw += avgPower;
    std::cout << "Tanjina-Power usage values from the power_reading for BN1 #" << s.batch_norm_layer_count << " : " << avgPowerUsage << " watts " << "Timestamp of the current power reading: " << timestampPower << " Execution time: " << s.batch_norm_execution_time << " seconds" << std::endl;

    // std::cout << "Tanjina-NN architecture info: " << "BN1_B = " << B << " BN1_C = " << C << " BN1_H = " << H << " BN1_W = " << W << std::endl;

    // std::vector<csv_column_type> batchnorm1_data;
    // batchnorm1_data.push_back(i);
    // batchnorm1_data.push_back("BatchNorm 1");
    // batchnorm1_data.push_back(BatchNorm_layer_count);
    // batchnorm1_data.push_back(timestampPower);
    // batchnorm1_data.push_back(avgPowerUsage);
    // batchnorm1_data.push_back(BatchNormExecutionTime);
    // batchnorm1_data.push_back(C);
    // batchnorm1_data.push_back(H);
    // batchnorm1_data.push_back(W);

    // writeBatchNorm1CSV.insertDataRow(batchnorm1_data);
  
  }       
#endif

}

void BatchNorm(int32_t B, int32_t H, int32_t W, int32_t C,
               const intType *inputArr, const intType *scales,
               const intType *bias, intType *outArr) {
  BatchNorm(*sci::CurrentSession(), B, H, W, C, inputArr, scales, bias, outArr);
}

void ElemWiseActModelVectorMult(sci::Session &s, int32_t size, intType *inArr,
                                intType *multArrVec, intType *outputArr) {
  // Alias session-owned resources
  auto *cheetah_linear = s.cheetah_linear();
  const int party = s.party_value();

#ifdef LOG_LAYERWISE
  INIT_ALL_IO_DATA_SENT;
  INIT_TIMER;
  // Add by Eloise >> Tanjina-Note: I moved them here. I could still found them in the log before when it was residing outside of this #ifdef LOG_LAYERWISE block, but to be consistent I placed them here!
  std::cout << "*******************" << std::endl;
  auto cur_start = CURRENT_TIME;
  std::cout << "Current time of start for current BN2 = " << cur_start
            << std::endl; 
  s.batch_norm_start_time = cur_start;
#endif

/** 
  * Code block for power measurement in BatchNorm layer starts
  * Added by - Tanjina
**/
#ifdef LOG_LAYERWISE 
  // BN layer counter
  s.batch_norm_layer_count++;

  std::cout << "STARTING ENERGY MEASUREMENT" << std::endl;
  // Pass the the Power usage file path to the Energy measurement library 
  EnergyMeasurement measurement(power_usage_path);
         
#endif

  static int batchNormCtr = 1;
  // Add by Eloise // Tanjina - Need to check if I found this in the log because it is outside the #ifdef LOG_LAYERWISE block! >> Tanjina-Note: Move it to #ifdef LOG_LAYERWISE block???
  // std::cout << "*******************" << std::endl;
  // auto cur_start = CURRENT_TIME;
  // std::cout << "Current time of start for current BN2 = " << cur_start
  //           << std::endl;
  printf("HomBN2 #%d via element-wise mult on %d points\n", batchNormCtr++,
         size);

  gemini::CheetahLinear::BNMeta meta;
  meta.target_base_mod = prime_mod;
  meta.is_shared_input = kIsSharedInput;
  meta.vec_shape = gemini::TensorShape({size});

  gemini::Tensor<intType> in_vec;
  gemini::Tensor<intType> scale_vec;
  scale_vec.Reshape(meta.vec_shape);
  if (cheetah_linear->party() == SERVER) {
    std::transform(multArrVec, multArrVec + size, scale_vec.data(), getRingElt);
  }

  if (meta.is_shared_input) {
    in_vec = gemini::Tensor<intType>::Wrap(inArr, meta.vec_shape);
  } else {
    in_vec.Reshape(meta.vec_shape);
    std::transform(inArr, inArr + size, in_vec.data(), getRingElt);
  }
  gemini::Tensor<intType> out_vec;
  cheetah_linear->bn(in_vec, scale_vec, meta, out_vec);
  std::copy_n(out_vec.data(), out_vec.shape().num_elements(), outputArr);

  if (cheetah_linear->party() == SERVER) {
    cheetah_linear->safe_erase(scale_vec.data(), scale_vec.NumElements());
  }

#ifdef LOG_LAYERWISE
  auto temp = TIMER_TILL_NOW;
  s.batch_norm_time_ms += temp;
  uint64_t curComm;
  FIND_ALL_IO_TILL_NOW(curComm);
  s.batch_norm_comm_sent += curComm;
#endif

#ifdef VERIFY_LAYERWISE
  for (int i = 0; i < size; i++) {
    assert(outputArr[i] < prime_mod);
  }

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
      int64_t gnd = getSignedVal(VoutputVec[i]);
      int64_t cmp = VoutputArr[i];
      if (gnd != cmp) {
        if (pass) {
          std::cout << RED << gnd << " ==> " << cmp << RESET << std::endl;
        }
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

// Add by Eloise
#ifdef LOG_LAYERWISE
  auto cur_end = CURRENT_TIME;
  std::cout << "Current time of end for current BN2 = " << cur_end
            << std::endl;
  s.batch_norm_end_time = cur_end;
#endif
/** 
  * Code block for power measurement in Batch Norm 2 layer ends
  * Added by - Tanjina
**/
#ifdef LOG_LAYERWISE
  std::vector<std::pair<uint64_t, int64_t>> power_readings = measurement.stop();
  s.batch_norm_execution_time = (s.batch_norm_end_time - s.batch_norm_start_time);
  
  for(int i = 0; i < power_readings.size(); ++i){
    uint64_t avgPower = power_readings[i].first;
    int64_t timestampPower = power_readings[i].second;
    double avgPowerUsage = avgPower / 1000000.0;

    s.batch_norm_total_power_uw += avgPower;
    std::cout << "Tanjina-Power usage values from the power_reading for BN2 #" << s.batch_norm_layer_count << " : " << avgPowerUsage << " watts " << "Timestamp of the current power reading: " << timestampPower << " Execution time: " << s.batch_norm_execution_time << " seconds"  << std::endl;
  }
#endif  

}

void ElemWiseActModelVectorMult(int32_t size, intType *inArr,
                                intType *multArrVec, intType *outputArr) {
  ElemWiseActModelVectorMult(*sci::CurrentSession(), size, inArr, multArrVec,
                             outputArr);
}
#endif
