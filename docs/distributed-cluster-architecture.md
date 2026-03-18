# Distributed Cluster Architecture (Planned)

This document defines the intended architecture for a local-network, multi-node runtime/controller system built around this repo.

The goal is not to turn `llama.cpp` or `llama-server` into the whole product. The goal is to reuse local runtime execution where it is already strong, and build discovery, scheduling, orchestration, and persistent node control around it in repo-owned code.

## Product goals

- One Rust egui app for Windows and macOS first.
- Local-network only clustering:
- LAN
- Wi-Fi
- Thunderbolt networking
- No external relay or cloud coordinator.
- Any node can open the controller UI.
- The cluster continues running if the main window closes.
- Prefer single-node placement when a model fits.
- Prefer 2-node split before 3-node split.
- Allow multiple models to stay loaded across available nodes for concurrent pipelines.
- Allow multiple model instances to run across the cluster and be controllable from any node UI.
- Support two retention modes per model instance:
- `keep_loaded`
- `load_on_demand`
- `load_on_demand` must use a post-request grace period before unloading.
- Support same-host multi-GPU and cross-node multi-device execution.

## Non-goals

- Do not depend on upstream `ggml-rpc` as a product-ready surface.
- Do not modify `third_party/llama.cpp/` directly in repo.
- Do not require internet discovery, central accounts, or hosted control planes.
- Do not start with fine-grained tensor parallelism across heterogeneous nodes.

## Architecture summary

The system has three layers:

1. Controller/UI layer
- Rust egui app
- cluster view
- model inventory
- placement controls
- pipeline controls
- session logs and node health

2. Node agent layer
- persistent background process on each machine
- survives UI window close
- owns discovery, resource reporting, worker lifecycle, and local runtime inventory
- exposes a control API to the UI and peer coordinators

3. Runtime worker layer
- local inference workers launched and supervised by the node agent
- reuses existing runtime capabilities where possible
- same-host multi-GPU split stays local to a node
- cross-node execution uses repo-owned transport and scheduling

## Key requirement: hybrid local-plus-remote placement

The system must explicitly support a single request that uses:
- multiple local GPUs on one node
- plus one or more external nodes

Example:
- Windows node:
- internal `CUDA0`
- Thunderbolt eGPU `CUDA1`
- macOS node:
- `Metal0`

Valid placement:
- node A early layers, internally split across `CUDA0 + CUDA1`
- node B late layers on `Metal0`

This hybrid case is a primary design target, not an edge case.

## Process model

Recommended process layout:

- `controller`:
- launched by opening the egui app window
- connects to the local node agent
- can become the active cluster coordinator if elected

- `node agent`:
- background tray/menu-bar runner
- starts automatically or on first app launch
- keeps cluster state alive even if the main window closes
- supervises local workers

- `worker`:
- per-model or per-placement runtime worker
- launched by the node agent
- may be:
- single-node full-model worker
- same-host multi-GPU worker
- distributed layer-block worker

The tray/menu-bar runner is the important lifecycle boundary. The UI must not be the thing keeping the cluster alive.

## Local vs remote execution

### Same-host multi-GPU

This should stay on native local runtime handling.

Use existing local device selection and split support:
- single device where possible
- `split_mode=layer` or `split_mode=row` only when needed
- `tensor_split` for same-host device balancing

This covers cases like:
- Windows laptop internal `CUDA0`
- Thunderbolt eGPU `CUDA1`

Those should be treated as one node with multiple local devices.

### Hybrid local-plus-remote execution

This is the most important non-trivial execution mode:

- one node may use multiple local devices for its assigned share
- the overall request may still span multiple nodes

That means:
- the cluster scheduler must not think only in terms of raw devices
- it must schedule across node-level execution groups
- each node then decides how to realize its own assigned block locally

This avoids trying to globally coordinate every raw GPU in the cluster as if all devices were peers.

### Cross-node execution

Cross-node execution should not depend on raw upstream `ggml-rpc` semantics as the public product API.

Instead:
- reuse local backend execution on each node
- move orchestration, discovery, scheduling, and transport ownership into repo-owned code

For large-model split across nodes, use:
- contiguous layer-block assignment
- one node owns an early block
- next node owns the next block
- last node owns final norm / logits / sampling

For multi-model concurrency, use:
- one model per node when possible
- only split across nodes when a model cannot fit a single node

## Control plane

The control plane should be separate from tensor transport.

Recommended:
- discovery: `mDNS` / Bonjour
- control RPC: protobuf RPC over persistent TCP
- local-only trust model with explicit pairing

The control plane is responsible for:
- node discovery
- node pairing and trust
- resource reporting
- model inventory
- placement proposals
- worker start/stop/load/unload
- state streaming to all controllers
- logs, errors, health, and lease ownership

## Execution groups

The scheduler should operate on `execution groups`, not just raw devices.

An execution group is a node-local compute target that can be one of:
- one device
- multiple local devices under one local runtime split
- CPU-only fallback group

Examples:

- Windows node:
- execution group `win/cuda-single-0`
- execution group `win/cuda-single-1`
- execution group `win/cuda-local-split-0-1`

- macOS node:
- execution group `mac/metal-single-0`

The cluster planner places work across execution groups.
The local node planner decides how each group is realized.

### Why not raw HTTP/JSON for everything

HTTP/JSON is acceptable for simple local admin tools, but it is the wrong choice for the hot path and a poor fit for stateful cluster control.

The system needs:
- long-lived connections
- streaming state updates
- low overhead
- binary compatibility across Windows and macOS

So the control plane should be structured and binary.

## Data plane

The data plane is for hot-path tensor or activation movement.

Recommended:
- persistent binary TCP streams
- one connection per neighbor or worker pair
- fixed-size headers plus raw payloads
- preallocated buffers

Do not use:
- JSON
- REST
- per-token reconnects
- file handoff

### Why TCP is still the right baseline

For mixed Windows/macOS local clusters:
- Thunderbolt networking appears as a network interface
- LAN and Wi-Fi are also network interfaces
- TCP gives one compatible transport story across all of them

The node agent should benchmark link quality and prefer:
1. Thunderbolt
2. wired LAN
3. Wi-Fi

## Discovery and trust

Discovery should be local-subnet only.

Recommended flow:

1. Node agent starts.
2. Agent advertises itself via mDNS.
3. Peer agents discover each other on the local subnet.
4. First-time peer pairing requires explicit approval or a visible pairing code.
5. After pairing, nodes use stored local keys/cert fingerprints.

Requirements:
- do not expose open unauthenticated execution on the subnet
- reject unpaired peers
- default to private interfaces only

This is not internet-grade multi-tenant security. It is a local paired-node trust model.

## Coordinator model

There should be exactly one active coordinator per cluster view.

Any UI can control the cluster, but only one coordinator should own placement decisions at a time.

Recommended:
- lease-based coordinator election
- local node agent can host the coordinator role
- UI windows connect to the local agent, not directly to every worker

This avoids:
- conflicting placements
- duplicate model loads
- controller split-brain

Any controller UI on any node must be able to:
- see all loaded model instances in the cluster
- see which node and execution group each instance is using
- change retention policy for an instance
- request load, unload, pin, or repin actions through the active coordinator

## Node resource model

Each node agent reports:

- node ID
- machine name
- OS
- runtime version
- supported backends
- device list
- total/free VRAM per device
- total/free RAM
- storage path and cache stats
- loaded models
- current worker allocations
- measured link stats to peers

Each node should also report candidate local execution groups:
- single-device groups
- multi-device local split groups
- estimated memory and preferred context limits for each group
- relative performance hints for each group

Each device entry should include:
- backend kind (`CUDA`, `Metal`, `Vulkan`, `CPU`)
- device name
- total memory
- free memory
- whether it is local-only or externally reachable through this node

## Model inventory

The cluster needs a shared model catalog view.

Each model entry should track:
- logical model ID
- local file path on each node
- quantization
- context constraints
- memory estimate by context
- warm/cold status on each node
- current placement

Important:
- model identity must be logical, not just path-based
- multiple nodes may hold the same model
- scheduler must know whether loading is already warm on a node

The system must also track `model instances`, not just logical model definitions.

A model instance is:
- one loaded or loadable runtime instance
- attached to one node or execution group
- governed by a retention policy

Minimum retention policies:
- `keep_loaded`
- `load_on_demand`

`load_on_demand` behavior:
- do not unload immediately when a request completes
- enter a grace window first
- default grace window: `30 seconds`

That grace window is required to avoid reload thrash when another request for the same model arrives immediately after the previous one.

## Placement policy

The default scheduler policy should be:

1. Prefer one node.
2. If not possible, prefer same-host multi-GPU on one node.
3. If not possible, prefer hybrid placement where a strong node uses its local multi-GPU group and one external node takes the remainder.
4. If not possible, prefer 2-node split across simpler groups.
5. Use 3-node split only as last resort.

Additional rules:
- prefer keeping an already-warm model where it is if the fit is still acceptable
- avoid moving a hot model unless there is a clear gain
- keep multimodel pipelines on distinct nodes when possible to maximize concurrency
- keep contiguous layer blocks together
- keep each MoE block whole on a single node
- prefer hybrid placements that minimize cross-node boundaries
- if a compatible `load_on_demand` instance is still inside its grace window, prefer reusing it over a cold load elsewhere

## Scheduler responsibilities

The scheduler should handle two separate problems:

1. Local planning
- on a given node, choose whether assigned work should use:
- one local GPU
- multiple local GPUs
- CPU fallback

2. Cluster planning
- split the global request across nodes or execution groups
- prefer single-node and hybrid placements before broader distributed splits

3. Fit/placement
- where can a model run
- how many nodes are required
- whether it should stay warm

4. Execution planning
- which worker(s) to bind to a request
- whether to run single-node or distributed
- whether a concurrent pipeline should reuse a loaded model or load another one

The scheduler should not start with "optimal" global solving.

Start with a deterministic heuristic policy:
- prefer single node
- then same-host split
- then hybrid local-plus-remote split
- then 2-node split
- then 3-node split
- score by warmness, memory fit, and link quality

## Execution modes

The system should support three execution modes.

### Mode 1: single-node model

- one node loads and serves the full model
- best for latency
- preferred whenever possible
- may be either `keep_loaded` or `load_on_demand`

### Mode 2: same-host split

- one node uses multiple local devices
- handled by local runtime/device split controls
- preferred over cross-node split

### Mode 3: hybrid local-plus-remote split

- one node uses multiple local devices for its local block
- one or more external nodes own later blocks
- cluster planner assigns blocks across nodes
- local planner decides how each node realizes its own block

This mode is the main target for setups like:
- Windows internal GPU + eGPU
- plus a macOS `Metal` node

### Mode 4: cross-node layer split

- one request flows through multiple nodes
- each node owns a contiguous layer block
- each node stores KV cache for its own block
- activations cross node boundaries

This mode is for models that do not fit on one node or one machine.

## Why not tensor parallel first

Tensor parallel across heterogeneous nodes is a poor first target because it requires:
- tight synchronization
- heavier communication
- all-reduce style coordination
- more backend coupling

Layer-block split is simpler and a better fit for:
- 1-3 local nodes
- mixed `CUDA` / `Metal`
- hybrid local-plus-remote execution groups
- one-request-at-a-time workloads

## Interaction with existing runtime code

The intended reuse boundary is:

- keep local execution in the existing runtime stack
- keep same-host GPU split in the existing runtime stack
- build cluster discovery/orchestration outside it

Do not try to make upstream `llama-server` become the cluster manager.

Instead:
- treat the current runtime as the local execution engine
- add repo-owned worker and transport layers around it

### Upstream `ggml-rpc`

Use it as reference material and an implementation starting point only.

Do not use it as the public product boundary because:
- it is upstream-marked proof-of-concept
- it is manual-host registration
- it has no discovery
- it has no scheduler
- it has no model inventory
- it is not the cluster UX you want

## Proposed repo ownership split

Recommended ownership:

- `ENGINE` repo:
- native node worker runtime
- transport layer
- agent APIs
- runtime-facing placement hooks

- Rust egui app repo:
- controller UI
- tray/menu-bar runner shell
- local agent client
- cluster dashboards
- pipeline editor / model controls

If desired, the tray runner can still host or spawn the local native agent, but the separation of concerns should remain.

## Proposed component boundaries

### Native node agent API

Responsibilities:
- start/stop local workers
- enumerate local devices and model files
- report node resources
- accept placement instructions
- stream state updates

### Native worker API

Responsibilities:
- load/unload a model
- report warm state
- serve inference
- participate in distributed layer-block execution

### Rust controller API client

Responsibilities:
- discover the local agent
- observe cluster state
- submit placement or execution requests
- display logs, resources, and running models

## Output/state model for controllers

All controllers should be able to see:
- nodes connected
- devices per node
- models loaded where
- which models are warm
- which requests are running
- which node is active coordinator
- why a placement was chosen

This needs a cluster-state stream, not polling-only status snapshots.

## Suggested MVP phases

### Phase 1: local agent + discovery

- tray/menu-bar runner
- node identity
- mDNS discovery
- paired local trust
- resource reporting
- cluster state view in egui

### Phase 2: single-node model scheduling

- model inventory
- load/unload on chosen node
- single-node inference from any controller
- warm model retention

### Phase 3: same-host multi-GPU scheduling

- expose same-host multi-device capability as one node placement option
- prefer local same-host split before remote split

### Phase 4: hybrid local-plus-remote placement

- add execution groups
- add local planner vs cluster planner separation
- support one node using local multi-GPU while the request also spans an external node

### Phase 5: multi-model cluster scheduling

- preload multiple models across nodes
- support concurrent pipelines using distinct loaded models
- retain warm models based on memory pressure and recent use

### Phase 6: 2-node layer split

- contiguous layer-block execution
- activation transport
- per-node KV ownership
- placement rules and health handling

### Phase 7: 3-node fallback

- same as phase 5, but only when required for fit
- explicit penalty in the scheduler

## Failure model

The cluster must assume nodes can disappear.

Required behavior:
- detect lost nodes quickly
- mark affected placements invalid
- do not silently continue with stale placements
- surface why a request failed
- preserve model inventory state separately from live worker state

For retention handling:
- preserve desired-state metadata for `keep_loaded` instances even if the host node disappears
- allow `load_on_demand` instances to unload cleanly after the grace window expires

## Recommended first implementation choice

Start with:
- local tray/menu-bar agent
- local discovery
- single-node and same-host multi-GPU scheduling
- hybrid local-plus-remote placement
- cluster-wide multi-model preloading
- cluster-visible model instances and retention state

Only then add:
- cross-node layer split

That gets useful product value early without committing to the hardest distributed path first.

## Final recommendation

Build:
- your own cluster control layer
- your own node agent
- your own scheduler
- your own controller UI

Reuse:
- current local runtime execution
- same-host device split
- upstream `ggml-rpc` ideas and code where helpful

Do not:
- make the product depend directly on raw upstream `ggml-rpc`
- push discovery and orchestration into upstream `llama-server`
- start with 3-node distributed inference before the simpler cluster cases work
