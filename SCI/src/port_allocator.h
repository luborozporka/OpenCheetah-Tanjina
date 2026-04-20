// Thread-safe port-range allocator for cheetah-server

#ifndef SCI_PORT_ALLOCATOR_H__
#define SCI_PORT_ALLOCATOR_H__

#include <mutex>
#include <vector>

namespace sci {

struct PortRange {
  int base;
  int num_threads;
};

class PortRangeAllocator {
 public:
  PortRangeAllocator(int base_port, int num_threads_per_session, int max_active);

  PortRangeAllocator(const PortRangeAllocator&) = delete;
  PortRangeAllocator& operator=(const PortRangeAllocator&) = delete;

  PortRange allocate();

  void release(PortRange range);

  int base_port() const { return base_port_; }
  int num_threads_per_session() const { return num_threads_per_session_; }
  int max_active() const { return max_active_; }

 private:
  const int base_port_;
  const int num_threads_per_session_;
  const int max_active_;

  std::mutex mu_;
  int next_slot_ = 0;
  std::vector<int> free_slots_;
};

}  // namespace sci

#endif  // SCI_PORT_ALLOCATOR_H__
