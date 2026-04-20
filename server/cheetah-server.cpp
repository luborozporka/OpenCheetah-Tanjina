// Multi-client cheetah-server

// Accepts TCP connections on a control port, negotiates a private data-port
// range per client, and runs the chosen network in a dedicated worker thread

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <utility>

#include <boost/asio.hpp>

#include "defines.h"
#include "port_allocator.h"
#include "session.h"
#include "utils/ArgMapping/ArgMapping.h"

int party = SERVER;
int port = 32000; // fallback; per-thread port_override takes priority
std::string address = "0.0.0.0";
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
constexpr uint32_t kStatusErrWrongNet = 1;
constexpr uint32_t kStatusErrExhausted = 2;

struct ServerConfig {
  std::string net_name = "sqnet";
  std::string model_path;
  int control_port = 31000;
  int data_port_base = 32000;
  int max_clients = 2;
  int num_threads_per_session = 4;
  int bitlength = 32;
  int kscale = 12;
  NetworkId net = NetworkId::kSqnet;  // resolved from net_name after parsing
};

void write_u32(tcp::socket& sock, uint32_t v) {
  asio::write(sock, asio::buffer(&v, sizeof(v)));
}

uint8_t read_u8(tcp::socket& sock) {
  uint8_t v = 0;
  asio::read(sock, asio::buffer(&v, sizeof(v)));
  return v;
}

std::string peer_str(const tcp::socket& sock) {
  boost::system::error_code ec;
  auto ep = sock.remote_endpoint(ec);
  if (ec) return "?";
  return ep.address().to_string() + ":" + std::to_string(ep.port());
}

void handle_client(tcp::socket sock, sci::PortRangeAllocator* alloc,
                   NetworkId expected_net, std::string model_path) {
  const std::string peer = peer_str(sock);
  std::cerr << "[cheetah-server] accepted " << peer << "\n";

  sci::PortRange range{-1, 0};
  try {
    const uint8_t requested = read_u8(sock);
    if (static_cast<NetworkId>(requested) != expected_net) {
      std::cerr << "[cheetah-server] " << peer << ": wrong network id "
                << int(requested) << "\n";
      write_u32(sock, kStatusErrWrongNet);
      return;
    }

    range = alloc->allocate();
    if (range.base < 0) {
      std::cerr << "[cheetah-server] " << peer << ": capacity exhausted\n";
      write_u32(sock, kStatusErrExhausted);
      return;
    }

    write_u32(sock, kStatusOk);
    write_u32(sock, static_cast<uint32_t>(range.base));
    write_u32(sock, static_cast<uint32_t>(range.num_threads));
    std::cerr << "[cheetah-server] " << peer << ": assigned ports "
              << range.base << ".." << (range.base + range.num_threads - 1)
              << "\n";

    // publish per-thread overrides so StartComputation() picks them up for
    // this worker's thread_local g_main_session
    sci::port_override = range.base;
    sci::num_threads_override = range.num_threads;

    std::ifstream model(model_path);
    if (!model.is_open()) {
      std::cerr << "[cheetah-server] " << peer
                << ": failed to open model " << model_path << "\n";
    } else {
      std::ostringstream discard;
      if (expected_net == NetworkId::kSqnet) run_sqnet(model, discard);
      else run_resnet50(model, discard);
    }

    sci::port_override = 0;
    sci::num_threads_override = 0;
  } catch (const std::exception& e) {
    std::cerr << "[cheetah-server] " << peer << " error: " << e.what() << "\n";
  }

  if (range.base >= 0) alloc->release(range);

  boost::system::error_code ec;
  sock.shutdown(tcp::socket::shutdown_both, ec);
  sock.close(ec);
  std::cerr << "[cheetah-server] closed " << peer << "\n";
}

}  // namespace

int main(int argc, char** argv) {
  ServerConfig cfg;

  ArgMapping amap;
  amap.arg("net", cfg.net_name, "Network: sqnet | resnet50");
  amap.arg("model", cfg.model_path, "Path to model weights file (required)");
  amap.arg("control_port", cfg.control_port, "Control port");
  amap.arg("data_port_base", cfg.data_port_base, "Base for per-session data ports");
  amap.arg("max_clients", cfg.max_clients, "Max concurrent clients");
  amap.arg("num_threads", cfg.num_threads_per_session, "Threads per session (must be <= MAX_THREADS)");
  amap.arg("bitlength", cfg.bitlength, "Ring bitlength (sqnet=32, resnet50=41)");
  amap.arg("kscale", cfg.kscale, "Fixed-point scale");
  amap.parse(argc, argv);

  if (cfg.net_name == "sqnet") cfg.net = NetworkId::kSqnet;
  else if (cfg.net_name == "resnet50") cfg.net = NetworkId::kResnet50;
  else {
    std::cerr << "[cheetah-server] unknown network: " << cfg.net_name << "\n";
    return 1;
  }

  if (cfg.model_path.empty()) {
    std::cerr << "[cheetah-server] model=<file> is required\n";
    return 1;
  }

  {
    std::ifstream probe(cfg.model_path);
    if (!probe.is_open()) {
      std::cerr << "[cheetah-server] cannot open model: " << cfg.model_path
                << "\n";
      return 1;
    }
  }

  party = SERVER;
  num_threads = cfg.num_threads_per_session;
  bitlength = cfg.bitlength;
  kScale = cfg.kscale;

  sci::PortRangeAllocator alloc(cfg.data_port_base,
                                cfg.num_threads_per_session, cfg.max_clients);

  asio::io_context io;
  tcp::acceptor acceptor(io, tcp::endpoint(tcp::v4(), cfg.control_port));

  std::cerr << "[cheetah-server] net=" << cfg.net_name
            << " control_port=" << cfg.control_port
            << " data_port_base=" << cfg.data_port_base
            << " max_clients=" << cfg.max_clients
            << " num_threads=" << cfg.num_threads_per_session
            << " bitlength=" << cfg.bitlength
            << " kscale=" << cfg.kscale
            << " model=" << cfg.model_path << "\n";
  std::cerr << "[cheetah-server] listening on " << cfg.control_port << "\n";

  while (true) {
    tcp::socket sock(io);
    boost::system::error_code ec;
    acceptor.accept(sock, ec);
    if (ec) {
      std::cerr << "[cheetah-server] accept error: " << ec.message() << "\n";
      continue;
    }
    std::thread(handle_client, std::move(sock), &alloc, cfg.net,
                cfg.model_path)
        .detach();
  }

  return 0;
}
