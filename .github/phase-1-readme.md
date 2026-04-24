## Multi-client SNNI server

End of the first thesis phase: the Cheetah framework can now serve multiple concurrent clients from a single server. Per-session energy readings are also incorporated

### New binaries
- `cheetah-server` - generic multi-client server, `net={sqnet,resnet50}` for NN used, `max_clients` limits concurrent clients
- `cheetah-client` - generic client, `net=…` selects the NN

### Architecture
- Global crypto/profiling state removed, `sci::Session` is used instead
- Thread-per-session concurrency (blocking Boost.Asio accept/control handshake, detached worker thread)
- `sci::PortRangeAllocator` hands each session its own port range for Cheetah's internal multi-threaded networking
- `sci::CurrentSession()` + `thread_local`

### Instrumentation
- Tanjina's per-layer counters (`LOG_LAYERWISE`) preserved, migrated from globals into `sci::Session` members
- Per-session CSV in `Output/session-<pid>-<port>.csv` records `wall_time_ms, total_comm_bytes, total_energy_uj, avg_power_w, idle_power_w`
- Per-session Conv power CSVs are written as `Output/conv-<pid>-<port>.csv` to avoid interleaving layer measurements from concurrent clients
- Baseline power usage is sampled once at startup via `hwmon` and is written to `Output/idle.csv`

### Networks supported
- `sqnet`, `resnet50`

### How to run
```bash
# Server
./build/bin/cheetah-server net=sqnet model=<path-to-weights> \
  control_port=32000 num_threads=4 max_clients=2 idle_ms=2000

# Client
./build/bin/cheetah-client net=sqnet ip=<server-ip> \
  control_port=32000 image=<path-to-image> idle_ms=2000
```