# Distributed Cluster Implementation Plan

This document is the single implementation dossier for the GPU-first multi-node runtime.

It replaces the previous draft stub and is intended to be the source of truth for the next engineering pass.

The purpose of this plan is not to restate broad product goals. The purpose is to define the exact implementation rules needed to make the cluster runtime behave correctly for large-model GPU execution across:

- multiple GPUs on one machine
- multiple machines on one local network
- hybrid placements that use both of the above at the same time

It also defines what is deliberately not in scope for the first production-grade cluster runtime, so the system can be shipped in a correct and defensible form instead of pretending unsupported workloads are already split-capable.

## 1. Core Product Decision

The product has two execution modes.

### 1.1 Direct mode

Direct mode is the existing embedded/local mode:

- an app can point at an exact GGUF file
- it can use local multi-GPU behavior on the local host
- it does not require the scheduler app
- it does not require named model instances
- it should remain compatible with existing `bridge` callers

Direct mode remains important. It is the simplest path and must not be broken by cluster work.

### 1.2 Managed mode

Managed mode is the cluster-aware mode:

- the tray host owns node discovery, trust, placement, retention, runtime inventory, and public HTTP APIs
- callers target named managed models or named managed instances
- the tray host chooses a placement and launches or reuses a runtime instance
- the public API routes by model name, not by raw file path

Managed mode is the place where multi-node orchestration exists.

The rest of this document is about managed mode.

## 2. Non-Negotiable Runtime Rules

These are product rules, not best-effort preferences.

### 2.1 GPU first by default

Default behavior must be:

- CPU disabled
- integrated/shared-memory GPUs hidden and disabled on non-macOS systems
- only discrete GPU VRAM is considered valid placement memory
- on Apple silicon, Metal is the GPU backend of interest and CPU should not be presented by default

CPU must still be supported as an explicit opt-in for debugging or edge cases, but the default product behavior must not silently fall back to CPU.

### 2.2 Single-GPU vs multi-device KV policy

Default KV policy is:

- single-GPU runtime: unified KV is allowed and preferred
- multi-GPU runtime on one node: KV must be sharded by owned layer block and stay on the owning GPU
- multi-node runtime: KV must be sharded by owned layer block and stay on the owning device/node

This means:

- `kv_unified = 1` is acceptable only for single-device runtime instances
- `kv_unified = 0` is required for multi-device runtime instances
- host-RAM KV is not the default path
- forcing `no_kv_offload = 1` for multi-device runs is incorrect for this product

### 2.3 No model-sized host RAM mirror

This project does not require literal zero host RAM during load. That is not a realistic cross-platform requirement.

What it does require is:

- no full-model mirror in host RAM
- no repeated full load retries across multiple candidate groups
- no host-RAM KV by default
- bounded host staging buffers only

In practice:

- a few GiB of bounded staging and bookkeeping is acceptable
- tens of GiB of extra host RAM from staging, retries, or host-side shadow copies is not acceptable

### 2.4 One active request per runtime instance at first

The first production-grade scheduler should enforce:

- one active request at a time per runtime instance
- queued requests for that same instance wait in a per-instance queue
- different runtime instances may still run independently

This keeps the initial concurrency model simple and correct.

Pipeline parallelism across devices and nodes can be added later, but it must not be used as an excuse to ship an unstable loader or unstable routing model.

## 3. Workload Capability Matrix

Not every model family needs the same split behavior in the first correct version.

### 3.1 Split-capable families in v1

The following families are expected to support multi-device and multi-node split execution:

- text generation / chat
- vision-language models
- embeddings
- rerank
- other llama-style transformer families that already fit the generic split runtime model

These workloads use the split runtime path.

### 3.2 Single-device-only families in v1

The following families should be explicitly supported in managed mode, but single-device only:

- Voxtral realtime
- Whisper-style transcription
- Sortformer diarization
- other patched audio/realtime families whose runtime path is not yet split-aware

This means:

- they can be scheduled to any node
- they can be scheduled to any allowed single device on that node
- they support `keep_loaded` or `load_on_demand`
- they support managed HTTP APIs
- they do not claim multi-device or multi-node split support

This is still useful because these workloads are small enough to fit one device, and the cluster product still needs to be able to route them to the right machine and keep them warm there.

### 3.3 Why this split matters

This is the clean boundary:

- model-agnostic cluster placement remains true
- supported split-capable families get true split execution
- non-split-capable families still become schedulable managed workloads instead of blocking the cluster product

This prevents a fake promise that "every patched model family already splits across nodes", while still letting the cluster runtime serve every family.

## 4. Runtime Architecture

### 4.1 Process model

The final process model for managed mode is:

1. `ENGINE Node`
- always-on tray or menu-bar host process
- owns discovery, trust, scheduler, runtime inventory, queues, public server, telemetry snapshotting

2. Controller UI
- loaded or opened by the host
- closing the window must not kill the host
- explicit quit must terminate the host

3. Managed runtime
- cluster-aware runtime used by the host to load and serve managed instances
- supports local multi-GPU and cross-node split for the split-capable families

4. Direct runtime / bridge
- continues to exist for direct local callers
- not the authority for managed mode

### 4.2 Authority boundaries

The host owns:

- node discovery
- trust/pairing
- model catalog
- named managed model registry
- instance retention policy
- placement selection
- per-instance queueing
- public HTTP surface
- telemetry aggregation

The runtime owns:

- final device allocation
- model loading
- request execution
- per-instance active state
- split execution across local GPUs and remote nodes

The UI owns:

- presentation
- user commands
- no hard authority over runtime state

## 5. Placement Model

### 5.1 Abstractions

The scheduler works with three levels:

1. Device
- one concrete accelerator
- example: `NVIDIA GeForce RTX 3090`
- example: `NVIDIA GeForce RTX 4090 Laptop GPU`
- example: `Apple M1`

2. Execution group
- one or more devices inside one node
- examples:
  - `3090`
  - `4090`
  - `3090 + 4090`
  - `Apple M1`

3. Placement target
- one or more execution groups across nodes
- examples:
  - `DESKTOP-111OLPQ: 3090`
  - `DESKTOP-111OLPQ: 3090 + 4090`
  - `DESKTOP-111OLPQ: 3090 + 4090 + Users-MacBook-Air Apple M1`

The UI must expose placement targets, not raw backend ids.

### 5.2 Placement policy

Automatic placement must be math-first, not trial-and-error-first.

Default automatic placement order:

1. one discrete GPU if the runtime fits fully there
2. same-host multi-GPU split
3. multi-node split

The default cluster product should prefer the smallest valid placement that fits according to deterministic fit math.

If the user pins a placement, the scheduler must obey it or fail clearly.

The scheduler must not discover fit primarily by repeatedly attempting real loads on different targets.

Correct default behavior is:

1. evaluate all valid single-GPU candidates mathematically
2. if at least one single-GPU candidate fits, pick the best one and stop
3. otherwise evaluate same-host multi-GPU split candidates mathematically
4. if at least one same-host split candidate fits, pick the best one and stop
5. otherwise evaluate multi-node split candidates mathematically
6. if at least one multi-node candidate fits, pick the best one and stop
7. only if no candidate fits mathematically should the system consider a limited probe-load fallback

The probe-load fallback must be:

- disabled as the normal default hammer path
- bounded to a very small number of candidates
- never a blind retry across every candidate in the cluster
- clearly exposed as a fallback or advanced option

The intended practical behavior is:

- if math says a single device fits, use that device
- if math says a single device does not fit but same-host split does, use same-host split
- if math says same-host split does not fit but multi-node does, use multi-node
- if math says nothing fits, fail clearly by default
- only later or by explicit advanced policy, probe a very small number of near-fit candidates

### 5.3 Mathematical fit model

Fit must be estimated before a real load attempt.

At minimum, the estimate per candidate must include:

- model weights bytes assigned to each device/node
- mmproj bytes assigned to each device/node if applicable
- KV bytes assigned to each device/node for the requested context
- backend scratch and compute buffers
- runtime metadata overhead
- explicit safety margin / headroom

The scheduler should reason per device, not only at cluster total.

For each device in a candidate:

`required_bytes(device) = weights_bytes(device) + mmproj_bytes(device) + kv_bytes(device) + scratch_bytes(device) + runtime_overhead_bytes(device) + safety_margin_bytes(device)`

The candidate is valid only if every device in that candidate satisfies:

`required_bytes(device) <= usable_vram_bytes(device)`

Where:

- `usable_vram_bytes(device)` is real available VRAM for that device
- not shared-memory estimates
- not system RAM
- not "total cluster memory"

The safety margin must be explicit and non-trivial.

A practical first implementation should keep a configurable headroom such as:

- percentage headroom, for example `10%`
- plus a fixed backend/runtime margin

If the math says a candidate does not fit, the default scheduler must reject it without trying a full load.

### 5.4 What "fit" means

A placement fits only if:

- weights fit on the target devices/nodes using the target split
- KV for the requested context fits on the target devices/nodes under the runtime KV policy
- runtime scratch/buffer headroom fits
- the placement does not require forbidden CPU or forbidden integrated GPU fallback

Fit calculation must not be based only on raw free memory.

It must include:

- model weights estimate
- mmproj estimate if applicable
- KV estimate for the requested `n_ctx`
- backend buffer overhead
- safety headroom

For multi-device candidates:

- weights are split by the chosen tensor/layer ownership plan
- KV is calculated only for the layers owned by that device
- context length is full, but layer count is the owned subset

For single-device candidates:

- unified KV may be used
- the fit estimate should still include full requested context and backend overhead

### 5.5 No destructive auto-fit retries

The scheduler must not discover fit by repeatedly attempting full runtime loads across different candidates.

Correct flow:

1. estimate candidate fit
2. choose candidate
3. perform a single real load on that chosen candidate

If deeper verification is needed later, add a lightweight probe path, not repeated full loads.

That probe path must obey all of the following:

- only run when mathematical fit is inconclusive or the user explicitly enabled advanced fallback
- only probe the top few ambiguous candidates, not every possible target
- never run continuously in the background
- never be the default behavior when math already says a candidate is valid
- never become a hidden retry storm that hammers RAM, VRAM, or remote nodes

## 6. Low-RAM Managed Load Policy

The earlier custom streaming-loader rewrite is no longer the default plan.

The working direction for the first production pass is:

- keep the existing runtime loader path
- configure it with a strict low-RAM policy for split-capable managed runtimes
- avoid `mmap` and aggressive prefetch-style behavior on the split path
- avoid destructive multi-candidate retry loads

This is the current source-of-truth load strategy unless later proof shows it is insufficient.

### 6.1 What fixed the bad load behavior

The successful direction is not a new bespoke tensor-streaming loader.

The successful direction is:

- one mathematically chosen candidate
- one real load
- low-RAM runtime flags for split-capable managed instances

The important rules are:

- `cache_ram_mib = 0`
- `use_mmap = 0` on split-capable managed loads
- `use_direct_io = 0` by default
- `use_mlock = 0`
- `no_host = 1`
- `no_extra_bufts = 1`
- `kv_unified = 0` for multi-device runtimes
- no CPU fallback by default
- no integrated/shared-memory GPU fallback by default

If later testing proves a different combination is better, the policy may be refined, but the product should not go back to assuming a bespoke custom loader rewrite is the first step.

### 6.2 What "GPU-first load" means in this plan

The practical target is:

- no model-sized host RAM mirror
- no repeated load attempts across multiple candidates
- no host-RAM KV by default for multi-device runtime
- no silent CPU fallback
- bounded physical RAM growth during load

It does not mean:

- literally zero host bytes touched during load

Cross-platform GPU loading still needs some host-side buffers and bookkeeping. The important requirement is that physical RAM remains bounded and does not scale like a large fraction of model size.

### 6.3 Default split-capable managed load policy

For split-capable managed runtimes, the default load policy is:

- select one placement candidate mathematically
- create one runtime instance on that candidate
- load once with the low-RAM flags above

The scheduler must not:

- brute-force full loads across candidates
- probe every possible same-host and multi-node target
- use load attempts as the normal placement mechanism

This means the normal path is:

1. build candidate set
2. filter by GPU-first rules
3. estimate fit mathematically
4. choose best candidate
5. perform one real load

### 6.4 `mmap`, prefetch, and direct IO policy

For the split-capable managed path:

- `mmap` should be off by default
- aggressive prefetch-like behavior should not be part of the default split load path
- `direct_io` should remain an advanced override, not the general default

The reason is simple:

- the validated 35B split run stayed within acceptable physical RAM when `mmap` was disabled and the low-RAM flags were enforced
- the previous blow-up behavior was tied to the older load behavior and retry pattern

So the default plan is not "invent a new loader first". The default plan is "use the current loader in the low-RAM configuration that actually behaved correctly under the validated split run".

### 6.5 Acceptable host RAM target

The target remains:

- a few GiB of bounded physical RAM overhead at most
- not tens of GiB of extra physical RAM

Commit reservation or private reservation may still look larger in tooling than physical working set. The product requirement is focused on preventing destructive physical RAM blow-ups and preventing model-sized host memory mirrors.

### 6.6 Validation rule

No load policy is accepted just because it sounds plausible.

It must be validated with a real large split run, for example:

- `Qwen3.5-35B`
- Windows `RTX 3090`
- Windows `RTX 4090 Laptop GPU`
- Mac `Apple M1`

The validation must check:

- model loads successfully
- inference runs successfully
- physical RAM remains bounded during load
- KV stays sharded for multi-device runtime
- no silent CPU fallback occurs

This validation is more important than theoretical loader design purity.

## 7. KV Cache Design

### 7.1 Single-device runtime

For a single-GPU runtime:

- unified KV is acceptable
- KV lives on that device

This saves memory and keeps the single-device path simple.

### 7.2 Multi-device same-host runtime

For multi-GPU runtime on one machine:

- KV is sharded by owned layer block
- KV for layers owned by GPU A stays on GPU A
- KV for layers owned by GPU B stays on GPU B

This avoids host-RAM KV traffic.

### 7.3 Multi-node runtime

For multi-node runtime:

- KV is sharded by owned layer block
- each node stores full context length for its owned layers only

This is the correct rule:

- context length is not divided by node count
- layer ownership is divided by node count

So every participating node stores KV for the full token history, but only for the subset of layers it owns.

### 7.4 No host-RAM KV by default

Active requests in GPU-first mode should not use host-RAM KV as the default.

Host-RAM KV is acceptable only later as a cold-session spill tier, for example:

- suspended sessions
- inactive chats
- overflow parking

It is not the default for active decode.

## 8. Single-Device Audio and Realtime Families

### 8.1 Product rule

For v1 managed mode:

- Voxtral
- Whisper-style transcription
- Sortformer diarization
- similar patched audio/realtime families

are single-device-only managed workloads.

### 8.2 What that still supports

They still support:

- any node can host them
- any allowed single GPU device on that node can host them
- `keep_loaded`
- `load_on_demand`
- managed HTTP APIs
- instance placement from the scheduler UI

### 8.3 Why this is acceptable

These families are small enough to fit one device, and the important cluster capability for them is:

- location control
- on-demand loading
- staying warm where needed
- being callable through the same managed control plane and managed HTTP layer

This is still a real cluster feature.

### 8.4 Implementation rule

The scheduler must understand a model family capability:

- `split_capable = true`
- `split_capable = false`

If `split_capable = false`:

- only single-device candidates are valid
- multi-device or multi-node split candidates must not be offered

## 9. Per-Instance Queueing

### 9.1 First correct model

Each managed runtime instance gets:

- one active request slot
- one FIFO queue for pending requests

This applies independently per instance.

So:

- instance `qwen-35b-chat` can be busy
- instance `voxtral-mini-realtime` can still run separately

### 9.2 Queue semantics

If a request arrives for a busy instance:

- enqueue it
- expose queue depth and estimated wait in telemetry
- run it when the current request finishes

No request should trigger:

- instance destruction
- placement re-evaluation mid-flight
- hidden fallback to CPU

### 9.3 Why this first model matters

It gives:

- deterministic behavior
- easier debugging
- correct memory accounting

before adding:

- intra-instance parallel slots
- pipeline parallel scheduling across nodes

## 10. Pipeline Parallelism

### 10.1 Scope

Pipeline parallelism is a later phase, not the first blocker.

The current first objective is:

- correct GPU-first placement
- correct low-RAM managed load policy
- stable per-instance queueing
- correct split execution for supported families

### 10.2 Future direction

When pipeline parallelism is added, it should be:

- per runtime instance
- aware of node/device stage boundaries
- compatible with same-host and cross-node stage ownership

The important rule is:

- do not add pipeline parallelism on top of a load path that still explodes host RAM

## 11. Managed HTTP API Surface

Managed mode needs a public server so external tools and web UIs can use it.

### 11.1 Required endpoints

Initial required public endpoints:

- `GET /v1/models`
- `POST /v1/responses`
- `POST /v1/chat/completions`
- `POST /v1/embeddings`
- `POST /v1/rerank`
- `POST /v1/audio/transcriptions`

Optional later:

- streaming responses
- realtime/websocket APIs
- diarization-specific endpoint

### 11.2 Endpoint routing rules

Managed HTTP requests target a managed model or instance name.

The server resolves:

- model family
- split capability
- configured or automatic placement
- retention mode
- existing warm instance reuse

and then routes into the managed runtime.

The managed HTTP layer should also accept explicit scheduling overrides for advanced callers:

- `x_engine_allowed_nodes`
- `x_engine_preferred_owner`
- `x_engine_execution_group`
- `x_engine_retention`
- `x_engine_n_parallel`

These overrides are not the normal user path, but they are important for testing, external web UIs, and power users who need to pin a runtime to one exact GPU target or preserve additional warm request slots.

### 11.3 Open Responses alignment

The public design should be broadly aligned with Open Responses style request/response structure for:

- text
- vision
- multimodal request bodies

while still keeping dedicated typed endpoints for:

- embeddings
- transcription
- diarization extensions

### 11.4 No hot-path HTTP between nodes

Public HTTP is only for external clients.

Node-to-node runtime transport remains internal, binary, and persistent.

## 12. Telemetry and Benchmarking

### 12.1 Default behavior

Telemetry must not continuously hammer the system or link by default.

Default behavior:

- snapshot on app open
- manual refresh
- low-frequency lightweight updates if explicitly enabled

### 12.2 Link benchmarks

Link benchmarks should be:

- startup baseline once
- manual full benchmark on demand

They should not run continuously in the background.

### 12.3 Telemetry content

The telemetry model should show:

- nodes
- visible devices
- real VRAM totals and free VRAM
- managed instances
- per-instance residency
- per-device occupancy by instance
- last benchmark result
- active requests
- queue depth
- request timings and tokens/s

The UI should not foreground host RAM unless the user enables advanced/system views.

## 13. UI Product Rules

### 13.1 Defaults

Default UI must be simple:

- show managed models
- show nodes
- show real GPU VRAM
- show valid placement targets
- hide CPUs by default
- hide shared-memory/integrated GPUs by default on non-macOS

### 13.2 Device naming

Use full, human-readable device names:

- `NVIDIA GeForce RTX 3090`
- `NVIDIA GeForce RTX 4090 Laptop GPU`
- `Apple M1`

Do not make the user map:

- `Vulkan0`
- `Vulkan1`
- `Metal0`

to real hardware in their head.

### 13.3 Placement UI

The placement UI must expose:

- automatic best fit
- current node only
- all reachable GPU nodes
- explicit named multi-device targets
- explicit hybrid multi-node targets

This is a placement-target UI, not a raw backend-id UI.

### 13.4 Tray behavior

Rules:

- close window should not always quit
- explicit `Hide to tray` should minimize to tray
- explicit `Quit ENGINE Node` must terminate the host
- tray `Open controller` must reopen reliably

## 14. Model Catalog Rules

### 14.1 Storage layout

Managed models should live under app data model roots, grouped by family:

- `chat`
- `vision`
- `embeddings`
- `realtime`
- `diarization`

Each model directory should be repo-style and human recognizable, for example:

- `Qwen__Qwen3.5-35B-A3B`
- `OpenAI__gpt-oss-20b`

### 14.2 Catalog metadata

Each catalog entry should describe:

- model family
- model path
- optional mmproj path
- optional diarization companion path
- split capability
- preferred runtime type
- default context
- default retention

This lets the scheduler expose only valid placements for each model family.

## 15. File-Level Implementation Plan

This section maps the plan onto repo-owned files.

### 15.1 `bridge/llama_server_cluster.cpp`

This is the most important native file to change.

It must:

- enforce the low-RAM managed load policy for split-capable runtimes
- enforce:
  - CPU default off
  - integrated/shared-memory GPU default off
  - `kv_unified = 0` for multi-device
- treat `cluster:auto` as:
  - select one best candidate
  - one real load only
  - no destructive multi-candidate retries
- distinguish:
  - split-capable families
  - single-device-only families

New responsibilities here:

- low-RAM loader flag wiring
- model-family-aware runtime policy
- candidate selection handoff from scheduler to one real load
- per-instance queue hooks if kept native-side

### 15.2 `bridge/llama_server_cluster.h`

Must define:

- explicit model family or runtime capability metadata
- placement policy flags
- queueing and scheduler-facing structs
- runtime metrics for managed HTTP and UI

### 15.3 `clusterui/src/agent.rs`

Must own:

- GPU-first filtering defaults
- scheduler policy
- per-instance request queueing policy
- placement candidate building from valid visible devices/groups only
- on-demand telemetry and benchmark triggers

It must not:

- fake fit by repeatedly full-loading different candidates
- background-spawn or babysit subprocesses aggressively

### 15.4 `clusterui/src/controller_ui.rs`

Must own:

- clean placement target UI
- node/device occupancy widgets
- model family-specific controls
- CPU advanced view only when enabled
- explicit quit vs hide behavior

### 15.5 `clusterui/src/main.rs`

Must own:

- tray lifecycle
- explicit quit handling
- hide-to-tray handling
- low-noise refresh defaults

### 15.6 `clusterui/src/public_server.rs`

Must expose:

- `responses`
- `chat/completions`
- `embeddings`
- `rerank`
- `audio/transcriptions`

all routed through the managed scheduler/runtime.

### 15.7 `clusterui/src/protocol.rs`

Must carry:

- placement candidates
- queue state
- occupancy/telemetry snapshots
- benchmark state
- split capability metadata

### 15.8 Native patch or overlay build path

Do not edit repo `third_party`.

All native runtime work must continue to flow through the approved patch or overlay build path, consistent with `AGENTS.md`.

## 16. Acceptance Criteria

This implementation is not done until all of the following are true.

### 16.1 GPU-first behavior

- CPU is off by default
- integrated/shared-memory GPUs are off by default on non-macOS
- default placements use only valid GPU targets

### 16.2 Split execution

Verified split execution exists for supported families on:

- one Windows GPU
- two Windows GPUs
- two Windows GPUs plus Mac GPU

### 16.3 Single-device managed audio families

Verified managed placement exists for:

- Voxtral
- Whisper-style transcription
- Sortformer diarization

on any chosen single allowed device/node, with:

- `keep_loaded`
- `load_on_demand`
- managed HTTP access

### 16.4 Load path

Large split loads:

- do not spike host RAM by tens of GiB
- do not create model-sized host RAM mirrors as the normal path
- do not fall back to CPU silently

### 16.5 Queueing

- one active request per runtime instance
- subsequent requests queue
- separate instances can still run independently

### 16.6 Managed HTTP

External tools can successfully call:

- `/v1/models`
- `/v1/responses`
- `/v1/chat/completions`
- `/v1/embeddings`
- `/v1/audio/transcriptions`

against named managed models.

For advanced validation and external UI integration, these endpoints must also be able to:

- pin a managed model to one exact owner node
- pin a managed model to one exact execution group
- request `keep_loaded` vs `load_on_demand`
- request `n_parallel` warm slots for the runtime instance

### 16.7 UI

The UI clearly shows:

- nodes
- valid devices
- valid placements
- model occupancy per device
- retention mode
- load state
- queue state

without making the user understand raw backend ids.

## 17. Explicitly Deferred Work

This document deliberately defers:

- true split-capable Voxtral/Whisper/Sortformer runtime
- cross-node pipeline parallelism
- aggressive continuous benchmarking
- CPU-first product behavior
- claiming every model family splits today

These can be added later, but they must not block the correct GPU-first cluster foundation.

## 18. Immediate Next Engineering Sequence

The next implementation sequence should be:

1. Lock in the low-RAM managed load policy as the default split-capable load path:
   - one mathematically chosen candidate
   - one real load
   - `mmap` off by default on split-capable managed loads
   - no destructive retry loops
2. Enforce GPU-first runtime policy end to end:
   - no CPU fallback by default
   - no integrated/shared-memory placement by default
   - no host-RAM KV for multi-device runtime
3. Prove the load policy on large split runs while watching physical RAM and VRAM usage.
4. Mark audio/realtime families as managed single-device-only and route them correctly.
5. Finalize per-instance queueing semantics.
6. Keep managed HTTP on top of the same runtime.
7. Finish UI simplification once the runtime rules are correct.

This order matters.

The load policy and placement semantics are the real blockers.
UI polish only becomes worth the time once the runtime obeys these memory and placement rules.
