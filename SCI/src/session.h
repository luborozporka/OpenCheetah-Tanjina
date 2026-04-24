/*
Session class for the Cheetah SNNI framework.

Encapsulates the per-connection cryptographic, network, and profiling state.
One Session per client connection in cheetah-server.
*/

#ifndef SCI_SESSION_H__
#define SCI_SESSION_H__

#define MAX_THREADS 4

#include "csv_writer.hpp"
#include "defines.h"
#include "defines_uniform.h"
#include "NonLinear/argmax.h"
#include "NonLinear/maxpool.h"
#include "NonLinear/relu-interface.h"
#include "OT/kkot.h"
#include <atomic>
#include <chrono>
#include <cstdint>
#include <string>

#ifdef SCI_OT
#include "BuildingBlocks/aux-protocols.h"
#include "BuildingBlocks/truncation.h"
#include "LinearOT/linear-ot.h"
#include "LinearOT/linear-uniform.h"
#include "Math/math-functions.h"
#endif

#ifdef SCI_HE
#include "LinearHE/conv-field.h"
#include "LinearHE/elemwise-prod-field.h"
#include "LinearHE/fc-field.h"
#endif

#if USE_CHEETAH
#include "cheetah/cheetah-api.h"
#endif

namespace sci {

class Session;

// Process-wide singleton accessor
// Returns the session owned by StartComputation/EndComputation, or nullptr before / after that window
Session* CurrentSession();

// Publishes/retracts the singleton
// Must only be called from StartComputation / EndComputation in library_fixed_uniform.cpp
void SetCurrentSession(Session* s);

class Session {
 public:
  Session();

  Session(const Session&) = delete;
  Session& operator=(const Session&) = delete;
  Session(Session&&) = delete;
  Session& operator=(Session&&) = delete;

  // Releases every resource held by the session in the reverse order of construction
  ~Session();

  // Allocates all per-connection cryptographic, OT, and linear-layer resources
  void setup(int party, int port, const std::string& address,
             int num_threads, int bitlength, int kScale);

  // Performs the base-OT handshake on the primary IKNP instance. Blocks
  // on network IO. Must be called exactly once per session, after setup()
  void start_base_ot();

  // Releases every resource allocated by setup()
  void teardown();

  // Accessors for the per-session crypto/network handles
  NetIO* io() { return io_; }
  OTPack<NetIO>* otpack() { return otpack_; }

  IKNP<NetIO>* iknpOT() { return iknp_ot_; }
  IKNP<NetIO>* iknpOTRoleReversed() { return iknp_ot_role_reversed_; }
  KKOT<NetIO>* kkot() { return kkot_; }
  PRG128* prg128() { return prg128_; }

  NetIO** ioArr() { return io_arr_; }
  OTPack<NetIO>** otpackArr() { return otpack_arr_; }
  IKNP<NetIO>** otInstanceArr() { return ot_instance_arr_; }
  KKOT<NetIO>** kkotInstanceArr() { return kkot_instance_arr_; }
  PRG128** prgInstanceArr() { return prg_instance_arr_; }

  ReLUProtocol<NetIO, intType>* relu() { return relu_; }
  MaxPoolProtocol<NetIO, intType>* maxpool() { return maxpool_; }
  ArgMaxProtocol<NetIO, intType>* argmax() { return argmax_; }
  ReLUProtocol<NetIO, intType>** reluArr() { return relu_arr_; }
  MaxPoolProtocol<NetIO, intType>** maxpoolArr() { return maxpool_arr_; }

#ifdef SCI_OT
  LinearOT* mult() { return mult_; }
  AuxProtocols* aux() { return aux_; }
  Truncation* truncation() { return truncation_; }
  XTProtocol* xt() { return xt_; }
  MathFunctions* math() { return math_; }
  MatMulUniform<NetIO, intType, IKNP<NetIO>>* multUniform() {
    return mult_uniform_;
  }
  LinearOT** multArr() { return mult_arr_; }
  AuxProtocols** auxArr() { return aux_arr_; }
  Truncation** truncationArr() { return truncation_arr_; }
  XTProtocol** xtArr() { return xt_arr_; }
  MathFunctions** mathArr() { return math_arr_; }
  MatMulUniform<NetIO, intType, IKNP<NetIO>>** multUniformArr() {
    return mult_uniform_arr_;
  }
#endif

#ifdef SCI_HE
  FCField* he_fc() { return he_fc_; }
  ElemWiseProdField* he_prod() { return he_prod_; }
#endif

#if USE_CHEETAH
  gemini::CheetahLinear* cheetah_linear() { return cheetah_linear_; }
#elif defined(SCI_HE)
  ConvField* he_conv() { return he_conv_; }
#endif

  // Per-session profiling counters
  std::chrono::time_point<std::chrono::high_resolution_clock>& start_time() {
    return start_time_;
  }
  uint64_t& comm_threads(int i) { return comm_threads_[i]; }
  uint64_t& num_rounds() { return num_rounds_; }

#ifdef LOG_LAYERWISE
  // Layerwise execution time accumulators
  uint64_t conv_time_ms = 0;
  uint64_t mat_add_time_ms = 0;
  uint64_t batch_norm_time_ms = 0;
  uint64_t truncation_time_ms = 0;
  uint64_t relu_time_ms = 0;
  uint64_t maxpool_time_ms = 0;
  uint64_t avgpool_time_ms = 0;
  uint64_t matmul_time_ms = 0;
  uint64_t mat_add_broadcast_time_ms = 0;
  uint64_t mul_cir_time_ms = 0;
  uint64_t scalar_mul_time_ms = 0;
  uint64_t sigmoid_time_ms = 0;
  uint64_t tanh_time_ms = 0;
  uint64_t sqrt_time_ms = 0;
  uint64_t normalise_l2_time_ms = 0;
  uint64_t argmax_time_ms = 0;

  // Layerwise communication accumulators (bytes sent by this party).
  uint64_t conv_comm_sent = 0;
  uint64_t mat_add_comm_sent = 0;
  uint64_t batch_norm_comm_sent = 0;
  uint64_t truncation_comm_sent = 0;
  uint64_t relu_comm_sent = 0;
  uint64_t maxpool_comm_sent = 0;
  uint64_t avgpool_comm_sent = 0;
  uint64_t matmul_comm_sent = 0;
  uint64_t mat_add_broadcast_comm_sent = 0;
  uint64_t mul_cir_comm_sent = 0;
  uint64_t scalar_mul_comm_sent = 0;
  uint64_t sigmoid_comm_sent = 0;
  uint64_t tanh_comm_sent = 0;
  uint64_t sqrt_comm_sent = 0;
  uint64_t normalise_l2_comm_sent = 0;
  uint64_t argmax_comm_sent = 0;

  // Added by Tanjina - per-layer cumulative power consumption
  uint64_t conv_total_power_uw = 0;
  uint64_t relu_total_power_uw = 0;
  uint64_t maxpool_total_power_uw = 0;
  uint64_t batch_norm_total_power_uw = 0;
  uint64_t matmul_total_power_uw = 0;
  uint64_t avgpool_total_power_uw = 0;
  uint64_t argmax_total_power_uw = 0;

  // Per-layer call counters, used to compute average power per layer
  int conv_layer_count = 0;
  int relu_layer_count = 0;
  int maxpool_layer_count = 0;
  int batch_norm_layer_count = 0;
  int truncation_layer_count = 0;
  int matmul_layer_count = 0;
  int avgpool_layer_count = 0;
  int argmax_layer_count = 0;

  // Per-layer execution-time windows
  uint64_t conv_start_time = 0;
  uint64_t conv_end_time = 0;
  uint64_t conv_execution_time = 0;

  uint64_t relu_start_time = 0;
  uint64_t relu_end_time = 0;
  double relu_execution_time = 0.0;

  uint64_t maxpool_start_time = 0;
  uint64_t maxpool_end_time = 0;
  double maxpool_execution_time = 0.0;

  uint64_t batch_norm_start_time = 0;
  uint64_t batch_norm_end_time = 0;
  double batch_norm_execution_time = 0.0;

  uint64_t matmul_start_time = 0;
  uint64_t matmul_end_time = 0;
  double matmul_execution_time = 0.0;

  uint64_t avgpool_start_time = 0;
  uint64_t avgpool_end_time = 0;
  double avgpool_execution_time = 0.0;

  uint64_t argmax_start_time = 0;
  uint64_t argmax_end_time = 0;
  double argmax_execution_time = 0.0;
#endif  // LOG_LAYERWISE

  // Process-level configuration snapshots taken at setup() time
  int party_value() const { return party_; }
  int num_threads_value() const { return num_threads_; }
  int bitlength_value() const { return bitlength_; }
  int k_scale_value() const { return k_scale_; }

  // Whole-session metrics written by EndComputation()
  uint64_t wall_time_ms = 0;
  uint64_t total_energy_uj = 0;
  double avg_power_w = 0.0;
  // Host idle baseline at the startup
  double idle_power_w = 0.0;
  // "pid<PID>-port<PORT>"
  std::string session_tag;

 private:
  // Primary (thread 0) channel and building blocks.
  NetIO* io_ = nullptr;
  OTPack<NetIO>* otpack_ = nullptr;
  IKNP<NetIO>* iknp_ot_ = nullptr;
  IKNP<NetIO>* iknp_ot_role_reversed_ = nullptr;
  KKOT<NetIO>* kkot_ = nullptr;
  PRG128* prg128_ = nullptr;

  // Per-worker-thread fan-out. Size fixed at MAX_THREADS
  NetIO* io_arr_[MAX_THREADS] = {};
  OTPack<NetIO>* otpack_arr_[MAX_THREADS] = {};
  IKNP<NetIO>* ot_instance_arr_[MAX_THREADS] = {};
  KKOT<NetIO>* kkot_instance_arr_[MAX_THREADS] = {};
  PRG128* prg_instance_arr_[MAX_THREADS] = {};

  // Non-linear protocols
  ReLUProtocol<NetIO, intType>* relu_ = nullptr;
  MaxPoolProtocol<NetIO, intType>* maxpool_ = nullptr;
  ArgMaxProtocol<NetIO, intType>* argmax_ = nullptr;
  ReLUProtocol<NetIO, intType>* relu_arr_[MAX_THREADS] = {};
  MaxPoolProtocol<NetIO, intType>* maxpool_arr_[MAX_THREADS] = {};

#ifdef SCI_OT
  LinearOT* mult_ = nullptr;
  AuxProtocols* aux_ = nullptr;
  Truncation* truncation_ = nullptr;
  XTProtocol* xt_ = nullptr;
  MathFunctions* math_ = nullptr;
  MatMulUniform<NetIO, intType, IKNP<NetIO>>* mult_uniform_ = nullptr;
  LinearOT* mult_arr_[MAX_THREADS] = {};
  AuxProtocols* aux_arr_[MAX_THREADS] = {};
  Truncation* truncation_arr_[MAX_THREADS] = {};
  XTProtocol* xt_arr_[MAX_THREADS] = {};
  MathFunctions* math_arr_[MAX_THREADS] = {};
  MatMulUniform<NetIO, intType, IKNP<NetIO>>* mult_uniform_arr_[MAX_THREADS] =
      {};
#endif

#ifdef SCI_HE
  FCField* he_fc_ = nullptr;
  ElemWiseProdField* he_prod_ = nullptr;
#endif

#if USE_CHEETAH
  gemini::CheetahLinear* cheetah_linear_ = nullptr;
#elif defined(SCI_HE)
  ConvField* he_conv_ = nullptr;
#endif

  // Captured process configuration; immutable after setup()
  int party_ = 0;
  int num_threads_ = 0;
  int bitlength_ = 0;
  int k_scale_ = 0;

  // Per-session timing scaffolding that StartComputation/EndComputation
  // currently read and write through file-scope globals
  std::chrono::time_point<std::chrono::high_resolution_clock> start_time_{};
  uint64_t comm_threads_[MAX_THREADS] = {};
  uint64_t num_rounds_ = 0;

  bool initialised_ = false;
};

extern thread_local int port_override;
extern thread_local int num_threads_override;

// Host idle-power baseline in watts
extern std::atomic<double> host_idle_power_w;

}  // namespace sci

#if USE_CHEETAH
// Per-session toggle
extern thread_local bool kIsSharedInput;
#endif

#endif  // SCI_SESSION_H__
