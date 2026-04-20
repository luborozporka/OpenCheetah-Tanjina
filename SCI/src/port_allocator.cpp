#include "port_allocator.h"

#include <cassert>

namespace sci {

PortRangeAllocator::PortRangeAllocator(int base_port,
                                       int num_threads_per_session,
                                       int max_active)
    : base_port_(base_port),
      num_threads_per_session_(num_threads_per_session),
      max_active_(max_active) {
  assert(base_port_ > 0);
  assert(num_threads_per_session_ > 0);
  assert(max_active_ > 0);
}

PortRange PortRangeAllocator::allocate() {
  std::lock_guard<std::mutex> lock(mu_);
  int slot;
  if (!free_slots_.empty()) {
    slot = free_slots_.back();
    free_slots_.pop_back();
  } else if (next_slot_ < max_active_) {
    slot = next_slot_++;
  } else {
    return {-1, 0};
  }
  return {base_port_ + slot * num_threads_per_session_,
          num_threads_per_session_};
}

void PortRangeAllocator::release(PortRange range) {
  if (range.base < 0) return;
  std::lock_guard<std::mutex> lock(mu_);
  const int slot = (range.base - base_port_) / num_threads_per_session_;
  assert(slot >= 0 && slot < next_slot_);
  free_slots_.push_back(slot);
}

}  // namespace sci
