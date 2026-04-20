/*
Implementation of sci::Session.

The body intentionally mirrors StartComputation() / EndComputation() in library_fixed_uniform.cpp
*/

#include "session.h"

#include <cassert>
#include <cstdio>

#if USE_CHEETAH
thread_local bool kIsSharedInput = false;
#endif

namespace sci {

namespace {
// thread_local so each worker thread in cheetah-server has its own session
thread_local Session* g_current_session = nullptr;
}  // namespace

thread_local int port_override = 0;
thread_local int num_threads_override = 0;

Session* CurrentSession() { return g_current_session; }

void SetCurrentSession(Session* s) { g_current_session = s; }

Session::Session() = default;

Session::~Session() { teardown(); }

void Session::setup(int party, int port, const std::string& address,
                    int num_threads, int bitlength, int k_scale) {
  assert(!initialised_ && "sci::Session::setup() called twice");
  assert(num_threads <= MAX_THREADS);
  assert(bitlength < 64 && bitlength > 0);

  party_ = party;
  num_threads_ = num_threads;
  bitlength_ = bitlength;
  k_scale_ = k_scale;

  for (int i = 0; i < num_threads_; i++) {
    io_arr_[i] = new NetIO(party == ALICE ? nullptr : address.c_str(),
                           port + i, /*quit=*/true);
    ot_instance_arr_[i] = new IKNP<NetIO>(io_arr_[i]);
    prg_instance_arr_[i] = new PRG128();
    kkot_instance_arr_[i] = new KKOT<NetIO>(io_arr_[i]);
#ifdef SCI_OT
    mult_uniform_arr_[i] =
        new MatMulUniform<NetIO, intType, IKNP<NetIO>>(
            party, bitlength, io_arr_[i], ot_instance_arr_[i], nullptr);
#endif
    if (i & 1) {
      otpack_arr_[i] = new OTPack<NetIO>(io_arr_[i], 3 - party);
    } else {
      otpack_arr_[i] = new OTPack<NetIO>(io_arr_[i], party);
    }
  }

  io_ = io_arr_[0];
  otpack_ = otpack_arr_[0];
  iknp_ot_ = new IKNP<NetIO>(io_);
  iknp_ot_role_reversed_ = new IKNP<NetIO>(io_);
  kkot_ = new KKOT<NetIO>(io_);
  prg128_ = new PRG128();

#ifdef SCI_OT
  mult_ = new LinearOT(party, io_, otpack_);
  truncation_ = new Truncation(party, io_, otpack_);
  mult_uniform_ = new MatMulUniform<NetIO, intType, IKNP<NetIO>>(
      party, bitlength, io_, iknp_ot_, iknp_ot_role_reversed_);
  relu_ = new ReLURingProtocol<NetIO, intType>(party, RING, io_, bitlength,
                                               MILL_PARAM, otpack_);
  maxpool_ = new MaxPoolProtocol<NetIO, intType>(
      party, RING, io_, bitlength, MILL_PARAM, 0, otpack_, relu_);
  argmax_ = new ArgMaxProtocol<NetIO, intType>(party, RING, io_, bitlength,
                                               MILL_PARAM, 0, otpack_, relu_);
  math_ = new MathFunctions(party, io_, otpack_);
#endif

#if USE_CHEETAH
  cheetah_linear_ =
      new gemini::CheetahLinear(party, io_, prime_mod, num_threads);
#elif defined(SCI_HE)
  he_conv_ = new ConvField(party, io_);
#endif

#ifdef SCI_HE
  relu_ = new ReLUFieldProtocol<NetIO, intType>(
      party, FIELD, io_, bitlength, MILL_PARAM, prime_mod, otpack_);
  maxpool_ = new MaxPoolProtocol<NetIO, intType>(
      party, FIELD, io_, bitlength, MILL_PARAM, prime_mod, otpack_, relu_);
  argmax_ = new ArgMaxProtocol<NetIO, intType>(
      party, FIELD, io_, bitlength, MILL_PARAM, prime_mod, otpack_, relu_);
  he_fc_ = new FCField(party, io_);
  he_prod_ = new ElemWiseProdField(party, io_);
#endif

#if defined MULTITHREADED_NONLIN && defined SCI_OT
  for (int i = 0; i < num_threads_; i++) {
    if (i & 1) {
      relu_arr_[i] = new ReLURingProtocol<NetIO, intType>(
          3 - party, RING, io_arr_[i], bitlength, MILL_PARAM, otpack_arr_[i]);
      maxpool_arr_[i] = new MaxPoolProtocol<NetIO, intType>(
          3 - party, RING, io_arr_[i], bitlength, MILL_PARAM, 0,
          otpack_arr_[i], relu_arr_[i]);
      mult_arr_[i] = new LinearOT(3 - party, io_arr_[i], otpack_arr_[i]);
      truncation_arr_[i] =
          new Truncation(3 - party, io_arr_[i], otpack_arr_[i]);
    } else {
      relu_arr_[i] = new ReLURingProtocol<NetIO, intType>(
          party, RING, io_arr_[i], bitlength, MILL_PARAM, otpack_arr_[i]);
      maxpool_arr_[i] = new MaxPoolProtocol<NetIO, intType>(
          party, RING, io_arr_[i], bitlength, MILL_PARAM, 0, otpack_arr_[i],
          relu_arr_[i]);
      mult_arr_[i] = new LinearOT(party, io_arr_[i], otpack_arr_[i]);
      truncation_arr_[i] = new Truncation(party, io_arr_[i], otpack_arr_[i]);
    }
  }
#endif

#ifdef SCI_HE
  for (int i = 0; i < num_threads_; i++) {
    if (i & 1) {
      relu_arr_[i] = new ReLUFieldProtocol<NetIO, intType>(
          3 - party, FIELD, io_arr_[i], bitlength, MILL_PARAM, prime_mod,
          otpack_arr_[i]);
      maxpool_arr_[i] = new MaxPoolProtocol<NetIO, intType>(
          3 - party, FIELD, io_arr_[i], bitlength, MILL_PARAM, prime_mod,
          otpack_arr_[i], relu_arr_[i]);
    } else {
      relu_arr_[i] = new ReLUFieldProtocol<NetIO, intType>(
          party, FIELD, io_arr_[i], bitlength, MILL_PARAM, prime_mod,
          otpack_arr_[i]);
      maxpool_arr_[i] = new MaxPoolProtocol<NetIO, intType>(
          party, FIELD, io_arr_[i], bitlength, MILL_PARAM, prime_mod,
          otpack_arr_[i], relu_arr_[i]);
    }
  }
#endif

#ifdef SCI_OT
  for (int i = 0; i < num_threads_; i++) {
    if (i & 1) {
      aux_arr_[i] = new AuxProtocols(3 - party, io_arr_[i], otpack_arr_[i]);
      truncation_arr_[i] =
          new Truncation(3 - party, io_arr_[i], otpack_arr_[i], aux_arr_[i]);
      xt_arr_[i] =
          new XTProtocol(3 - party, io_arr_[i], otpack_arr_[i], aux_arr_[i]);
      math_arr_[i] = new MathFunctions(3 - party, io_arr_[i], otpack_arr_[i]);
    } else {
      aux_arr_[i] = new AuxProtocols(party, io_arr_[i], otpack_arr_[i]);
      truncation_arr_[i] =
          new Truncation(party, io_arr_[i], otpack_arr_[i], aux_arr_[i]);
      xt_arr_[i] =
          new XTProtocol(party, io_arr_[i], otpack_arr_[i], aux_arr_[i]);
      math_arr_[i] = new MathFunctions(party, io_arr_[i], otpack_arr_[i]);
    }
  }
  aux_ = aux_arr_[0];
  truncation_ = truncation_arr_[0];
  xt_ = xt_arr_[0];
  mult_ = mult_arr_[0];
  math_ = math_arr_[0];
#endif

  initialised_ = true;
}

void Session::start_base_ot() {
  assert(initialised_ && "sci::Session::start_base_ot() before setup()");
  if (party_ == ALICE) {
    iknp_ot_->setup_send();
    iknp_ot_role_reversed_->setup_recv();
  } else if (party_ == BOB) {
    iknp_ot_->setup_recv();
    iknp_ot_role_reversed_->setup_send();
  }
}

void Session::teardown() {
  if (!initialised_) {
    return;
  }

#if USE_CHEETAH
  delete cheetah_linear_;
  cheetah_linear_ = nullptr;
#elif defined(SCI_HE)
  delete he_conv_;
  he_conv_ = nullptr;
#endif

#ifdef SCI_HE
  delete he_fc_;
  he_fc_ = nullptr;
  delete he_prod_;
  he_prod_ = nullptr;
#endif

  delete argmax_;
  argmax_ = nullptr;
  delete maxpool_;
  maxpool_ = nullptr;
  delete relu_;
  relu_ = nullptr;

#ifdef SCI_OT
  delete math_;
  math_ = nullptr;
  delete mult_uniform_;
  mult_uniform_ = nullptr;
  delete truncation_;
  truncation_ = nullptr;
  delete mult_;
  mult_ = nullptr;
  for (int i = 0; i < num_threads_; i++) {
    delete math_arr_[i];
    math_arr_[i] = nullptr;
    delete xt_arr_[i];
    xt_arr_[i] = nullptr;
    delete truncation_arr_[i];
    truncation_arr_[i] = nullptr;
    delete aux_arr_[i];
    aux_arr_[i] = nullptr;
    delete mult_uniform_arr_[i];
    mult_uniform_arr_[i] = nullptr;
    delete mult_arr_[i];
    mult_arr_[i] = nullptr;
    delete maxpool_arr_[i];
    maxpool_arr_[i] = nullptr;
    delete relu_arr_[i];
    relu_arr_[i] = nullptr;
  }
  aux_ = nullptr;
  xt_ = nullptr;
#endif

  delete prg128_;
  prg128_ = nullptr;
  delete kkot_;
  kkot_ = nullptr;
  delete iknp_ot_role_reversed_;
  iknp_ot_role_reversed_ = nullptr;
  delete iknp_ot_;
  iknp_ot_ = nullptr;

  for (int i = 0; i < num_threads_; i++) {
    delete otpack_arr_[i];
    otpack_arr_[i] = nullptr;
    delete kkot_instance_arr_[i];
    kkot_instance_arr_[i] = nullptr;
    delete prg_instance_arr_[i];
    prg_instance_arr_[i] = nullptr;
    delete ot_instance_arr_[i];
    ot_instance_arr_[i] = nullptr;
    delete io_arr_[i];
    io_arr_[i] = nullptr;
  }
  otpack_ = nullptr;
  io_ = nullptr;

  num_threads_ = 0;
  initialised_ = false;
}

}  // namespace sci
