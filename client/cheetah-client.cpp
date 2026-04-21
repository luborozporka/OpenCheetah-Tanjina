// Multi-client cheetah-client

// Handshakes with cheetah-server on a control port, receives a private
// data-port range, then runs the chosen network as client against those ports

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>

#include <unistd.h>

#include <boost/asio.hpp>

#include "defines.h"
#include "energy_consumption.hpp"
#include "session.h"
#include "utils/ArgMapping/ArgMapping.h"

int party = CLIENT;
int port = 32000; // fallback; per-thread port_override takes priority
std::string address = "127.0.0.1";
int num_threads = 4;
int32_t bitlength = 32;
int32_t kScale = 12;

// defined in networks/main_sqnet.cpp / main_resnet50.cpp
void run_sqnet(std::istream& in, std::ostream& out);
void run_resnet50(std::istream& in, std::ostream& out);

namespace asio = boost::asio;
using asio::ip::tcp;

namespace {

enum class NetworkId : uint8_t { kSqnet = 1, kResnet50 = 2 };

constexpr uint32_t kStatusOk = 0;

struct ClientConfig {
  std::string net_name = "sqnet";
  std::string server_ip = "127.0.0.1";
  int control_port = 31000;
  std::string image_path;
  int bitlength = 32;
  int kscale = 12;
  int idle_ms = 2000;
  NetworkId net = NetworkId::kSqnet;
};

std::string hostname_str() {
  char buf[256] = {0};
  if (gethostname(buf, sizeof(buf) - 1) != 0) return "unknown";
  return std::string(buf);
}

void persist_idle_row(const std::string& role, double idle_w, size_t samples,
                      int64_t effective_ms, int requested_ms) {
  const std::string path = "Output/idle.csv";
  const bool write_header = !std::ifstream(path.c_str()).good();
  std::ofstream f(path, std::ios::out | std::ios::app);
  if (!f.is_open()) {
    std::cerr << "[idle] cannot open " << path << " for append\n";
    return;
  }
  if (write_header) {
    f << "epoch_s,host,role,idle_power_w,samples,effective_duration_ms,"
         "requested_duration_ms\n";
  }
  const int64_t epoch_s =
      std::chrono::duration_cast<std::chrono::seconds>(
          std::chrono::system_clock::now().time_since_epoch())
          .count();
  f << epoch_s << "," << hostname_str() << "," << role << "," << idle_w << ","
    << samples << "," << effective_ms << "," << requested_ms << "\n";
}

bool do_handshake(tcp::socket& sock, NetworkId net, uint32_t* base_port,
                  uint32_t* num_threads_out) {
  const uint8_t net_id = static_cast<uint8_t>(net);
  asio::write(sock, asio::buffer(&net_id, sizeof(net_id)));

  uint32_t status = 0;
  asio::read(sock, asio::buffer(&status, sizeof(status)));
  if (status != kStatusOk) {
    std::cerr << "[cheetah-client] server rejected handshake, status="
              << status << "\n";
    return false;
  }
  asio::read(sock, asio::buffer(base_port, sizeof(*base_port)));
  asio::read(sock, asio::buffer(num_threads_out, sizeof(*num_threads_out)));
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  ClientConfig cfg;

  ArgMapping amap;
  amap.arg("net", cfg.net_name, "Network: sqnet | resnet50");
  amap.arg("ip", cfg.server_ip, "Server IP address");
  amap.arg("control_port", cfg.control_port, "Server's control port");
  amap.arg("image", cfg.image_path, "Path to input image file (required)");
  amap.arg("bitlength", cfg.bitlength, "Ring bitlength (must match server; sqnet=32, resnet50=41)");
  amap.arg("kscale", cfg.kscale, "Fixed-point scale (must match server)");
  amap.arg("idle_ms", cfg.idle_ms, "Idle-baseline sampling window in ms (0 = skip)");
  amap.parse(argc, argv);

  if (cfg.net_name == "sqnet") cfg.net = NetworkId::kSqnet;
  else if (cfg.net_name == "resnet50") cfg.net = NetworkId::kResnet50;
  else {
    std::cerr << "[cheetah-client] unknown network: " << cfg.net_name << "\n";
    return 1;
  }

  if (cfg.image_path.empty()) {
    std::cerr << "[cheetah-client] image=<file> is required\n";
    return 1;
  }

  std::ifstream image(cfg.image_path);
  if (!image.is_open()) {
    std::cerr << "[cheetah-client] cannot open image: " << cfg.image_path << "\n";
    return 1;
  }

  // Host idle baseline before any crypto work
  if (cfg.idle_ms > 0) {
    size_t samples = 0;
    int64_t effective_ms = 0;
    std::cerr << "[cheetah-client] sampling idle power for " << cfg.idle_ms
              << "ms from " << power_usage_path << "\n";
    const double idle_w = measure_idle_power_w(power_usage_path, cfg.idle_ms,
                                               &samples, &effective_ms);
    sci::host_idle_power_w.store(idle_w);
    persist_idle_row("CLIENT", idle_w, samples, effective_ms, cfg.idle_ms);
    std::cerr << "[cheetah-client] idle_power=" << idle_w << "W over "
              << samples << " samples / " << effective_ms << "ms\n";
  }

  asio::io_context io;
  tcp::socket sock(io);
  uint32_t base_port = 0;
  uint32_t nt = 0;
  try {
    tcp::endpoint ep(asio::ip::make_address(cfg.server_ip), cfg.control_port);
    sock.connect(ep);
    std::cerr << "[cheetah-client] connected to " << cfg.server_ip << ":"
              << cfg.control_port << "\n";

    if (!do_handshake(sock, cfg.net, &base_port, &nt)) return 1;
    std::cerr << "[cheetah-client] assigned ports " << base_port << ".."
              << (base_port + nt - 1) << " (num_threads=" << nt << ")\n";
  } catch (const std::exception& e) {
    std::cerr << "[cheetah-client] control channel error: " << e.what() << "\n";
    return 1;
  }

  {
    boost::system::error_code ec;
    sock.shutdown(tcp::socket::shutdown_both, ec);
    sock.close(ec);
  }

  party = CLIENT;
  address = cfg.server_ip;
  bitlength = cfg.bitlength;
  kScale = cfg.kscale;
  sci::port_override = static_cast<int>(base_port);
  sci::num_threads_override = static_cast<int>(nt);
  num_threads = static_cast<int>(nt);

  if (cfg.net == NetworkId::kSqnet) run_sqnet(image, std::cout);
  else run_resnet50(image, std::cout);

  sci::port_override = 0;
  sci::num_threads_override = 0;
  return 0;
}
