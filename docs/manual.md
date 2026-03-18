Engine is the desktop control app for the ENGINE runtime. You use it to connect machines together, place models on the right node, create named runtimes, and use those runtimes either through the local `/v1/*` HTTP API or through Engine's cluster-aware in-process paths.

This guide is about how to actually use the app. It assumes you want to get from "I opened Engine" to "my model is running on the machine I want, and I know how to use it."

## 1. What Engine does

Think of Engine as the control surface for one machine or several machines.

On a single machine, you can use it to:

- install the runtime
- import or download model files
- create named runtimes
- run a local HTTP API

Across more than one machine, you can also use it to:

- pair nodes together
- see which GPUs and models each node has
- run a model fully on another machine
- split a model across more than one GPU
- use another node as an RPC worker
- copy selected model files between nodes when needed

Any connected node can be the machine you click from, the machine that actually owns the runtime, or a worker that contributes extra GPU memory.

## 2. First-time setup

If Engine starts and the runtime is missing, the app should take you to `Settings`. Install the runtime there before doing anything else.

Do this on every machine you want to use:

1. Open Engine.
2. Go to `Settings`.
3. Check the `Runtime` section.
4. If the runtime is missing, choose the backend you want and press `Install / Repair`.

Backend choice in simple terms:

- `Vulkan` is the general default on Windows.
- `CUDA` is for supported NVIDIA setups on Windows.
- `Metal` is the normal path on macOS.

You do not copy the runtime between machines. Each machine installs its own local runtime. The thing that moves between machines is the model file, not the runtime itself.

## 3. What the left sidebar is telling you

The left side of the window is your live cluster summary. It is there so you can answer practical questions quickly without opening five different pages.

The small summary cards at the top show:

- `Nodes`: how many machines are currently visible
- `Trusted`: how many paired nodes are saved
- `Loaded`: how many runtimes are loaded right now
- `Serving`: how many active requests are being handled
- `Queued`: how many requests are waiting
- `Free VRAM`: total visible GPU memory still free across visible nodes

Below that you may see a link card for paired nodes. This is the speed test between machines. The main number shows throughput and latency together, for example `10.68 Gbps | 0.7 ms`.

Below that you get one block per visible node. These blocks show:

- the node name
- its control address
- whether it is exposing `RPC`
- whether it is exposing the local `HTTP` API
- how many loaded runtimes it has
- how much GPU memory is free on that node
- each visible GPU and how much memory that GPU has free
- the Engine process RAM on that machine

Read the sidebar like this:

- if a node is missing completely, it is not currently connected
- if `RPC` is missing on a node, that node cannot be used as a worker by other machines
- if `HTTP` is missing on a node, that node is not exposing its local `/v1/*` server
- if a GPU has very little free memory, it is probably already full or nearly full

## 4. Connecting two machines

If you want to use more than one machine, both machines should have Engine open.

Typical example:

- Windows desktop with NVIDIA GPUs
- MacBook on the same network or connected over Thunderbolt

### Step 1: make sure both machines are running

On each machine:

1. Open Engine.
2. Make sure the runtime is installed in `Settings`.
3. Wait for the app to finish its first load.

### Step 2: pair the machines

Go to `Nodes`.

This page is for discovery and trust. It is not where you launch models.

To connect a new machine:

1. On one machine, press `Connect and look for nodes`.
2. Wait for the other machine to appear in `Discovered nodes`.
3. Press `Request pairing`.
4. On the other machine, look at `Incoming requests`.
5. Press `Accept pairing`.

After that, the machine should move into `Saved paired nodes`.

Once nodes are paired, Engine remembers that relationship. You do not need to pair them every time. On later app launches, paired nodes reconnect automatically when they can see each other.

### Step 3: decide whether a machine should offer GPUs to other machines

Still on `Nodes`, look at `Multi-node RPC`.

`Enable multi-node RPC worker on this node` is a local setting for this machine only.

If it is on:

- other machines may use this machine as a worker for split launches

If it is off:

- this machine will not contribute GPUs to other machines

Important:

- turning RPC off here does not disconnect the other machine
- turning RPC off here does not stop this machine from using another node that still has RPC on

Example:

- Windows RPC off
- Mac RPC on

Result:

- Mac cannot use Windows as a worker
- Windows can still use Mac as a worker

If you turn RPC off after it was already running, the embedded RPC host may still exist until a full app restart. Use `Restart Engine` in that section when the app tells you restart is required.

### Step 4: prefer Thunderbolt or USB4 when you have it

If the machines have a direct Thunderbolt or USB4 link, Engine prefers that path for large transfers and cross-node traffic. That matters when you:

- copy model files between nodes
- run split launches across machines

The link widget in the sidebar helps you confirm that the fast path is actually being used.

## 5. Getting models into Engine

Go to `Models`.

This page is where model files enter the app. There are three main ways to do that.

### Download from Hugging Face

Use `Download from Hugging Face` when you know the repo you want.

Typical flow:

1. Paste a repo such as `owner/name` or the full Hugging Face URL.
2. Press `Load repo`.
3. Wait for the file list to appear.
4. Tick the exact files you want.
5. Choose the `Folder name`.
6. Press `Download selected`.

The files are downloaded into Engine's models directory for that machine.

### Import local files

Use `Import local files` when you already downloaded the files yourself and just want Engine to manage them.

Typical flow:

1. Press `Pick files`.
2. Choose the model file or files from disk.
3. Choose a `Model folder name`.
4. Press `Import selected files`.

Engine copies those files into its own models directory. It does not run them from the original random folder you picked.

### Supported audio models

Use `Supported audio models` when you want quick shortcuts for the audio model families that Engine supports directly, such as Whisper-style transcription or diarization-related models.

### Available models

The lower part of `Models` is the important inventory view.

`Available models` is a merged view. It shows model folders from:

- this machine
- connected paired nodes

When you click a folder, Engine shows:

- the files in that folder
- where that folder exists
- where each individual file exists

This matters because a folder may exist on several nodes, but not every file inside it may exist on every node.

## 6. Copying models between machines

Engine now has explicit file transfer buttons. It does not blindly copy an entire model repo unless you specifically selected all of it.

The transfer logic is practical:

- it works inside Engine's models folders only
- it copies the selected model file, not every quant in the folder
- for vision models, it also cares about the selected `MMProj`
- if a destination folder already exists, Engine only sends the missing files

You will see buttons such as:

- `Retrieve to this machine`
- `Upload to <node>`
- `Upload to node`

Use these when:

- the model exists on another node and you want a local copy here
- the model exists locally and you want to send it to another node

Transfers run in the background. Watch the bottom status area for progress. It shows which file is moving, from which node to which node, how many files are done, and the transfer speed.

## 7. Creating a runtime

Go to `Instances` and press `Create new instance`.

This is the most important page in the whole app. If you understand this page, you understand how Engine works.

### Step 1: choose the model type

`Model type` tells Engine what kind of runtime you are creating.

The options matter because they affect routing and API behavior:

- `Text`: normal chat or completion style text models
- `Vision`: text plus image models that also need an `MMProj`
- `Embeddings`: vector embedding models
- `Rerank`: reranker models
- `Whisper`: transcription models
- `Realtime audio`: reserved for the native realtime path
- `Diarization`: speaker-separation or speaker-labeling helper runtimes

Pick the runtime type that matches what the model is actually for. Do not create a text runtime for an embeddings model or vice versa.

### Step 2: choose the model folder

`Model folder` shows the available folders from the merged `Available models` inventory.

Pick the folder that contains the exact file you want.

### Step 3: choose the primary model file

`Primary model file` is the real GGUF or BIN file that will be loaded.

If the file list shows several quants, this is where you choose which exact one you want to run.

Engine also shows which nodes already have that selected file.

### Step 4: choose MMProj for vision models

If `Model type` is `Vision`, you must also choose an `MMProj file`.

That file is separate from the main model. Vision mode needs both:

- the main model
- the matching `MMProj`

Engine will warn you if the folder does not contain an `MMProj`.

### Step 5: understand Whisper and diarization

Whisper and diarization are separate instances now.

If you want transcription with diarization through `/v1/audio/transcriptions`, you usually need:

- one `Whisper` instance
- one `Diarization` instance

Put both on the same owner node when you want them to work together cleanly.

## 8. Launch setup: what every control means

Below the file pickers you will see `Launch setup`.

This is where you decide where the runtime lives and how it uses hardware.

### Presets

At the top you can save a preset.

Use presets when you have a setup you want to reuse later, for example:

- a specific Qwen vision model on the Mac
- a split layout across two GPUs
- a transcription instance with load-on-demand enabled

### Instance name

`Instance name` is the name you will use later in the API.

This matters a lot.

If your instance is called `Henry`, then the local HTTP API will use `Henry` in the `model` field. The API is instance-based, not file-name-based.

### Retention

`Retention` controls what happens after the runtime goes idle.

Options:

- `keep loaded`: keep the runtime in memory after use
- `load on demand`: unload it when it is idle and reload it when needed again

Use `keep loaded` when:

- load time is expensive
- you want the runtime ready immediately

Use `load on demand` when:

- VRAM is tight
- you only use that instance occasionally
- the model is small and cheap to reload

### Grace

`Grace` only matters when retention is `load on demand`.

It is the number of seconds Engine waits after the last request before unloading the runtime.

Examples:

- `0` means unload immediately when the request finishes
- `30` means wait 30 seconds before unloading

For small helper runtimes such as Whisper or diarization, immediate unload is often a sensible choice if you want to avoid wasting VRAM.

### Max predict

`Max predict` is the default generation limit for text-style requests.

If you are not sure what to do, leave it at a reasonable default and tune it only when a workflow needs more output.

## 9. Primary device: the most important rule

The `Primary device` row is the owner of the runtime.

This one rule explains most of Engine:

- the primary device decides which node owns the runtime
- the owner node is the node that actually starts and owns the runtime
- the owner node must physically have the selected model file
- if the runtime is `Vision`, the owner node must also have the selected `MMProj`

If the selected file is missing on the chosen owner, Engine shows an availability badge and a transfer button such as:

- `Retrieve to this machine`
- `Upload to <node>`
- `Copy selected files to <node>`

Use that button before pressing `Load now`.

### What happens when the primary device is on another machine

If you set the primary device to a GPU on the Mac while clicking from Windows:

- the runtime is still created and controlled from your current cluster setup
- but the actual owner is the Mac
- API requests and cluster calls will route to the Mac for that instance

That is normal and intended.

### What happens when you add more devices

After choosing the primary device, you can press `+ Add device`.

Added devices can be:

- more GPUs on the same machine
- GPUs on another machine that is exposing RPC

The primary device still stays the owner. The extra devices are helpers.

That means:

- if the primary device is on the Mac, the Mac owns the runtime
- if you then add a Windows GPU, the Mac stays the owner and uses the Windows GPU as a worker

It does not flip ownership back to the machine you clicked from.

## 10. Layer allocation and split loading

Each device row has a layer control.

This is how you decide how much of the model sits on each GPU.

### Single-device case

If there is only one selected device, `-1` means "put all GPU-offloadable layers on this device."

That is the easiest mode and often the fastest one.

If the model fits on one GPU, prefer that over splitting.

### Multi-device case

When more than one device is selected, each row gets its own layer count.

You are manually telling Engine how to divide the model.

Read the row as:

- selected device
- how many layers you want on it
- how many layers the model reports in total

If Engine can read layer count from the model file, the total is shown directly from the file metadata.

If Engine cannot read that metadata, the total may show as unknown. In that case you can still type values manually.

### How to think about layer splitting

Practical advice:

- start with the biggest or fastest GPU as primary
- keep the owner on the machine you want to truly own the runtime
- only add more devices when one GPU is not enough
- same-host multi-GPU split is usually simpler than cross-node split
- cross-node split is useful when one machine alone cannot fit the model

### Model insights and per-device estimate

Below the device rows, Engine shows:

- model format and architecture if known
- layer count if known
- trained context length if known
- approximate GPU requirement
- per-device estimated memory use
- whether the layout looks ready now, needs eviction, or looks insufficient

These estimates are advisory. Engine does not block launch just because the estimate looks bad. Manual setups can still work beyond the estimator.

### The `allocated X / Y layers` badge

This is the quick summary of how much of the model you have assigned so far.

Read it as:

- `X`: how many layers are currently assigned
- `Y`: total layer count if known

If you are on a single device and using `-1`, Engine shows that as full offload.

## 11. Runtime controls below the device rows

These are the low-level runtime settings.

You usually do not need to obsess over them on day one, but you should know what they mean.

### n_ctx

`n_ctx` is the context window you want the runtime to reserve for use.

In plain language, this is how much conversation or prompt history the runtime can hold at once.

Higher values need more memory.

### n_batch

`n_batch` is a workload batch size setting.

If you do not know exactly why you are changing it, leave the default alone. Bigger values can improve throughput, but they also raise memory pressure.

### n_ubatch

`n_ubatch` is the micro-batch setting.

Again, this is usually an advanced tuning control. Leave the default unless you already know why you are tuning it.

### GPU layers

This is the final total Engine derived from your manual allocation.

With one GPU and `-1`, it means full offload.

### Parallel slots

This controls how many requests the runtime can keep active or ready at the same time.

Simple rule:

- lower value: less VRAM pressure
- higher value: more simultaneous work, but more memory use

If you are trying to squeeze a model onto limited VRAM, use fewer parallel slots.

## 12. Starting the runtime

At the bottom of `Launch setup` you will usually care about two buttons:

- `Refresh cluster state`
- `Load now`

Use `Refresh cluster state` when:

- a node just connected
- you just copied a model
- you want device or model availability to update immediately

Press `Load now` when the setup looks correct.

After that, switch to `Show loaded runtimes` if you want to inspect, load, unload, or remove existing runtimes.

## 13. Managing loaded runtimes

The `Show loaded runtimes` view lists the named runtimes already known to the cluster.

For each runtime you can inspect:

- instance name
- owner node
- loaded state
- retention mode
- active request count
- queued request count
- slot count
- model path
- execution group
- remote workers if any

For the selected runtime you get actions:

- `Load`
- `Unload`
- `Toggle retention`
- `Remove`

Use these when you want to keep an instance definition but temporarily unload it, or when you want to remove it completely.

## 14. Turning on the HTTP API

Go to `Server`.

This is the local `/v1/*` server. It is optional. Runtimes can exist without it.

### What the API serves

The server exposes named loaded cluster instances over HTTP.

Important rule:

- the API uses the instance name in the `model` field

That means if your instance is named `Henry`, clients should call:

- `model: "Henry"`

### Basic setup

1. Turn on `Enable HTTP endpoints for named cluster instances`.
2. Choose `Bind host`.
3. Choose the port.
4. Optionally add an API key.
5. Optionally set IP and CORS limits.
6. Press `Apply server settings`.

### Bind host

`Bind host` decides where the API listens.

Common choices:

- `127.0.0.1`: only this machine
- private LAN IP: other machines on your network can reach it
- link-local / Thunderbolt address: useful for direct node-to-node access

Engine deliberately rejects wildcard and public internet binds here.

### CORS

Turn on `Allow CORS for browser and external web UIs` only if you actually need browser-based access.

### API key

You can:

- paste your own key
- generate a new key
- clear the key

Clients can then use:

- `Authorization: Bearer <key>`
- or `x-api-key`

### Allowed client IPs / CIDRs

Use this if you want to limit which machines are allowed to call the API.

Leave it empty if you want to allow any client that can already reach the chosen bind address.

### What endpoints are available

The server status panel lists the main routes:

- `/v1/models`
- `/v1/responses`
- `/v1/chat/completions`
- `/v1/embeddings`
- `/v1/rerank`
- `/v1/audio/transcriptions`

Use the endpoint that matches the type of instance you created.

Examples:

- text instance -> `/v1/chat/completions` or `/v1/responses`
- embeddings instance -> `/v1/embeddings`
- rerank instance -> `/v1/rerank`
- whisper instance -> `/v1/audio/transcriptions`

For transcription with diarization, create both:

- a `Whisper` instance
- a `Diarization` instance

Put them on the same owner node.

## 15. Settings that matter most

Go to `Settings` when you need to control the local machine itself.

The most important sections are:

### Appearance

Choose:

- `System`
- `Dark`
- `Light`

### Runtime

This is where you:

- see the runtime directory
- choose runtime backend
- install or repair the runtime

If Engine cannot find a usable runtime, come here first.

### Device visibility defaults

These settings control whether CPU devices and certain shared-memory GPUs are shown.

CPU devices are optional. They are not recommended for clustered tensor-model work because they are much slower than GPU-backed placement, and some CPU-heavy cluster layouts are not a path you should treat as well-traveled.

## 16. A simple real-world workflow

Here is a normal example from start to finish.

### Example: run a vision model on a Mac, controlled from Windows

1. Install Engine runtime on both Windows and Mac.
2. Pair the machines in `Nodes`.
3. In `Models`, make sure the folder and selected files are available somewhere.
4. Go to `Instances` -> `Create new instance`.
5. Set `Model type` to `Vision`.
6. Pick the model folder.
7. Pick the main model file.
8. Pick the `MMProj`.
9. In `Launch setup`, set the `Primary device` to the Mac GPU.
10. If the Mac does not yet have the selected model file or `MMProj`, use the transfer button shown there.
11. Optionally add another GPU if you want a split launch.
12. Adjust layer allocation if needed.
13. Press `Load now`.
14. If you want HTTP access, go to `Server`, enable it, and call the instance name through the local `/v1/*` API.

The important idea is that the Mac owns the runtime because the primary device is on the Mac, even though you set it up from Windows.

## 17. When something looks wrong

Use this quick checklist.

### I do not see the other machine

- check both apps are open
- check the machines are on the same network or direct link
- use `Nodes` -> `Connect and look for nodes`
- check the node was paired and not forgotten

### I see the node, but not its GPUs as worker options

- check `Enable multi-node RPC worker on this node` on that machine
- if you recently turned RPC off or on, restart the app if the UI says restart is required

### I see the model folder, but launch says files need transfer

- the selected owner node is missing the selected model file, the selected `MMProj`, or both
- use the transfer button in `Instances`
- or retrieve/upload files from `Models`

### My split layout looks wrong

- check which GPU is primary
- remember the primary device is the owner node
- check how many layers each row has
- check the per-device estimate below the allocator

### The API is on, but requests fail

- make sure the instance is actually loaded
- make sure you used the instance name in the `model` field
- make sure the endpoint matches the instance type
- check bind host, API key, and allowed client IPs

## 18. Final practical advice

If a model fits on one good GPU, use one GPU.

If it does not fit on one GPU, split only as much as you need.

If the owner node is remote, make sure that owner really has the selected model file before loading.

If something changed and the UI still looks stale, press `Refresh`.

And if you are not sure which machine should own a runtime, decide that first. In Engine, the primary device is the answer.
