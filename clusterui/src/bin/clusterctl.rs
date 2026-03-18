#[path = "../agent.rs"]
mod agent;
#[path = "../catalog.rs"]
mod catalog;
#[path = "../cluster_api.rs"]
mod cluster_api;
#[path = "../model_metadata.rs"]
mod model_metadata;
#[path = "../model_store.rs"]
mod model_store;
#[path = "../protocol.rs"]
mod protocol;
#[path = "../public_server.rs"]
mod public_server;
#[path = "../settings.rs"]
mod settings;

use agent::{default_local_agent_addr, AgentClient};
use cluster_api::{CreateInstanceParams, InstanceModelKind, RetentionMode};
use std::collections::BTreeSet;
use std::env;

fn main() {
    if let Err(err) = run() {
        eprintln!("{err}");
        std::process::exit(1);
    }
}

#[derive(Clone, Default)]
struct StartupArgs {
    dump_state: bool,
    dump_telemetry: bool,
    run_link_benchmark: bool,
    list_placement_candidates: bool,
    add_peer: Option<String>,
    remove_peer: Option<String>,
    configure_firewall: bool,
    bind_addr: String,
}

impl StartupArgs {
    fn from_env() -> Self {
        let mut args_out = Self {
            bind_addr: default_local_agent_addr(),
            ..Self::default()
        };

        let mut args = env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--dump-state" => args_out.dump_state = true,
                "--dump-telemetry" => args_out.dump_telemetry = true,
                "--run-link-benchmark" => args_out.run_link_benchmark = true,
                "--list-placement-candidates" => args_out.list_placement_candidates = true,
                "--add-peer" => {
                    if let Some(value) = args.next() {
                        args_out.add_peer = Some(value);
                    }
                }
                "--remove-peer" => {
                    if let Some(value) = args.next() {
                        args_out.remove_peer = Some(value);
                    }
                }
                "--configure-firewall" => args_out.configure_firewall = true,
                "--bind" => {
                    if let Some(value) = args.next() {
                        args_out.bind_addr = value;
                    }
                }
                _ => {}
            }
        }

        args_out
    }
}

fn run() -> Result<(), String> {
    let args = StartupArgs::from_env();

    if args.dump_state {
        return dump_agent_state(&args.bind_addr);
    }

    if args.dump_telemetry {
        return dump_cluster_telemetry(&args.bind_addr);
    }

    if args.run_link_benchmark {
        return run_link_benchmark(&args.bind_addr);
    }

    if args.list_placement_candidates {
        return list_placement_candidates(&args.bind_addr);
    }

    if let Some(control_addr) = args.add_peer {
        return add_agent_peer(&args.bind_addr, &control_addr);
    }

    if let Some(control_addr) = args.remove_peer {
        return remove_agent_peer(&args.bind_addr, &control_addr);
    }

    if args.configure_firewall {
        return configure_firewall(&args.bind_addr);
    }

    Err("no action specified".to_string())
}

fn env_optional(name: &str) -> Option<String> {
    env::var(name)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

fn env_i32(name: &str, default: i32) -> i32 {
    env::var(name)
        .ok()
        .and_then(|value| value.parse::<i32>().ok())
        .unwrap_or(default)
}

fn env_flag(name: &str) -> bool {
    env::var(name)
        .ok()
        .map(|value| {
            matches!(
                value.as_str(),
                "1" | "true" | "TRUE" | "yes" | "YES" | "on" | "ON"
            )
        })
        .unwrap_or(false)
}

fn env_csv(name: &str) -> Option<Vec<String>> {
    let raw = env_optional(name)?;
    let values = raw
        .split(',')
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
        .collect::<Vec<_>>();
    if values.is_empty() {
        None
    } else {
        Some(values)
    }
}

fn list_placement_candidates(control_addr: &str) -> Result<(), String> {
    let model_path = env_optional("CLUSTER_MODEL_PATH")
        .ok_or_else(|| "missing CLUSTER_MODEL_PATH for --list-placement-candidates".to_string())?;
    let client = AgentClient::new(control_addr.to_string());
    let params = CreateInstanceParams {
        name: env_optional("CLUSTER_INSTANCE_NAME").unwrap_or_else(|| "clusterctl".to_string()),
        managed_model_id: None,
        model_path,
        mmproj_path: env_optional("CLUSTER_MMPROJ_PATH"),
        diarization_model_path: env_optional("CLUSTER_DIARIZATION_MODEL_PATH"),
        execution_group_id: env_optional("CLUSTER_GROUP_ID")
            .unwrap_or_else(|| "cluster:auto".to_string()),
        rpc_servers: env_optional("CLUSTER_RPC_SERVERS"),
        manual_device_allocations: Vec::new(),
        manual_devices_csv: None,
        manual_tensor_split: None,
        preferred_owner_control_addr: env_optional("CLUSTER_OWNER"),
        retention_mode: match env_optional("CLUSTER_RETENTION_MODE").as_deref() {
            Some("load_on_demand") => RetentionMode::LoadOnDemand,
            _ => RetentionMode::KeepLoaded,
        },
        load_on_demand_grace_seconds: env_i32(
            "CLUSTER_LOAD_ON_DEMAND_GRACE_SECONDS",
            InstanceModelKind::Text.default_load_on_demand_grace_seconds(),
        ),
        embedding: env_flag("CLUSTER_EMBEDDING"),
        reranking: env_flag("CLUSTER_RERANKING"),
        model_kind: InstanceModelKind::Text,
        single_device_only: env_flag("CLUSTER_SINGLE_DEVICE_ONLY"),
        allow_cpu: env_flag("CLUSTER_ALLOW_CPU"),
        allow_integrated_gpu: env_flag("CLUSTER_ALLOW_INTEGRATED_GPU"),
        n_ctx: env_i32("CLUSTER_N_CTX", 2048),
        n_batch: env_i32("CLUSTER_N_BATCH", 512),
        n_ubatch: env_i32("CLUSTER_N_UBATCH", 512),
        n_parallel: env_i32("CLUSTER_N_PARALLEL", 1),
        n_threads: env_i32("CLUSTER_N_THREADS", 8),
        n_threads_batch: env_i32("CLUSTER_N_THREADS_BATCH", 8),
        n_gpu_layers: env_i32("CLUSTER_N_GPU_LAYERS", -1),
    };
    let plans = client
        .list_placement_candidates(params, env_csv("CLUSTER_ALLOWED_ADDRS"))
        .map_err(|e| e.to_string())?;
    println!("placement_candidates={}", plans.len());
    for plan in &plans {
        println!(
            "plan owner={} group={} rpc={} strategy={:?} devices={} remote_nodes={} required={} free={} ready={} eviction={} reuse={}",
            plan.owner_control_addr,
            plan.execution_group_id,
            plan.rpc_servers,
            plan.strategy,
            plan.device_count,
            plan.remote_node_count,
            plan.estimated_required_bytes,
            plan.estimated_group_free_bytes,
            plan.ready_now,
            plan.requires_eviction,
            plan.reusable_instance_id
                .map(|value| value.to_string())
                .unwrap_or_else(|| "-".to_string())
        );
        println!("label={}", plan.display_label);
    }
    Ok(())
}

fn run_link_benchmark(control_addr: &str) -> Result<(), String> {
    let client = AgentClient::new(control_addr.to_string());
    client
        .run_link_benchmarks(true)
        .map_err(|e| e.to_string())?;
    println!("manual link benchmark completed");
    let snapshot = client.get_local_telemetry().map_err(|e| e.to_string())?;
    println!("link_metrics={}", snapshot.link_metrics.len());
    for link in &snapshot.link_metrics {
        println!(
            "link peer={} transport={} probe_kind={} latency_ms={:.2} goodput_mbps={:.2} payload={} rounds={} duration_ms={:.0} error={}",
            link.peer_control_addr,
            link.transport,
            link.probe_kind,
            link.latency_ms,
            link.goodput_mbps,
            link.payload_bytes,
            link.rounds,
            link.duration_ms,
            link.error.as_deref().unwrap_or("")
        );
    }
    Ok(())
}

fn dump_agent_state(control_addr: &str) -> Result<(), String> {
    let client = AgentClient::new(control_addr.to_string());
    let snapshot = client.get_snapshot().map_err(|e| e.to_string())?;
    println!(
        "node={} id={} os={} arch={} addr={} advertised={}",
        snapshot.node.display_name,
        snapshot.node.node_id,
        snapshot.node.os_name,
        snapshot.node.arch,
        snapshot.control_addr,
        snapshot.advertised_control_addr.as_deref().unwrap_or("")
    );
    println!(
        "rpc running={} endpoint={} advertised={} firewall_status={} firewall_action_required={}",
        snapshot.rpc_running,
        snapshot.rpc_endpoint.as_deref().unwrap_or(""),
        snapshot.advertised_rpc_endpoint.as_deref().unwrap_or(""),
        snapshot.firewall_status.as_deref().unwrap_or(""),
        snapshot.firewall_action_required
    );
    println!("devices={}", snapshot.devices.len());
    for device in &snapshot.devices {
        println!(
            "device index={} backend={} name={} free={} total={}",
            device.bridge_device_index,
            device.backend,
            device.name,
            device.memory_free,
            device.memory_total
        );
    }
    println!("execution_groups={}", snapshot.execution_groups.len());
    for group in &snapshot.execution_groups {
        println!(
            "group id={} split={} devices={} free={} total={}",
            group.id,
            group.uses_local_split,
            group.devices_csv,
            group.memory_free,
            group.memory_total
        );
    }
    println!("instances={}", snapshot.instances.len());
    for instance in &snapshot.instances {
        println!(
            "instance id={} name={} state={} retention={:?} group={} active={} queued={} slots={}",
            instance.instance_id,
            instance.name,
            instance.state,
            instance.retention_mode,
            instance.execution_group_id,
            instance.active_request_count,
            instance.queued_request_count,
            instance.n_parallel
        );
    }
    println!("link_metrics={}", snapshot.link_metrics.len());
    for link in &snapshot.link_metrics {
        println!(
                "link peer={} transport={} probe_kind={} latency_ms={:.2} goodput_mbps={:.2} payload={} rounds={} duration_ms={:.0} error={}",
                link.peer_control_addr,
                link.transport,
                link.probe_kind,
                link.latency_ms,
                link.goodput_mbps,
                link.payload_bytes,
                link.rounds,
                link.duration_ms,
                link.error.as_deref().unwrap_or("")
            );
    }

    let peers = client.list_peers().map_err(|e| e.to_string())?;
    println!("peers={}", peers.len());
    for peer in peers {
        println!(
            "peer node={} display={} addr={} advertised={} rpc_running={} rpc_endpoint={} advertised_rpc={} trusted={}",
            peer.node_id,
            peer.display_name,
            peer.control_addr,
            peer.advertised_control_addr.as_deref().unwrap_or(""),
            peer.rpc_running,
            peer.rpc_endpoint.as_deref().unwrap_or(""),
            peer.advertised_rpc_endpoint.as_deref().unwrap_or(""),
            peer.trusted
        );
    }
    Ok(())
}

fn dump_cluster_telemetry(control_addr: &str) -> Result<(), String> {
    let client = AgentClient::new(control_addr.to_string());
    let mut snapshots = Vec::new();
    let mut seen = BTreeSet::new();

    let mut local = client.get_local_telemetry().map_err(|e| e.to_string())?;
    local.control_addr = control_addr.to_string();
    seen.insert(local.control_addr.clone());
    if let Some(advertised) = &local.advertised_control_addr {
        seen.insert(advertised.clone());
    }
    snapshots.push(local);

    for peer in client.list_peers().map_err(|e| e.to_string())? {
        let connect_addr = peer
            .advertised_control_addr
            .clone()
            .unwrap_or_else(|| peer.control_addr.clone());
        if seen.contains(&connect_addr) || seen.contains(&peer.control_addr) {
            continue;
        }
        let Ok(mut snapshot) = AgentClient::new(connect_addr.clone()).get_local_telemetry() else {
            continue;
        };
        snapshot.control_addr = connect_addr.clone();
        if snapshot.advertised_control_addr.is_none() {
            snapshot.advertised_control_addr = peer.advertised_control_addr.clone();
        }
        seen.insert(snapshot.control_addr.clone());
        if let Some(advertised) = &snapshot.advertised_control_addr {
            seen.insert(advertised.clone());
        }
        snapshots.push(snapshot);
    }

    snapshots.sort_by(|lhs, rhs| {
        lhs.node
            .display_name
            .cmp(&rhs.node.display_name)
            .then(lhs.control_addr.cmp(&rhs.control_addr))
    });
    println!("telemetry_nodes={}", snapshots.len());
    for snapshot in snapshots {
        println!(
            "node={} addr={} advertised={} rpc={} public_api={} proc_mem={} proc_virt={} cpu={:.1} avail_ram={} total_ram={}",
            snapshot.node.display_name,
            snapshot.control_addr,
            snapshot.advertised_control_addr.as_deref().unwrap_or(""),
            snapshot.rpc_running,
            snapshot.public_api_running,
            snapshot.process_memory_bytes,
            snapshot.process_virtual_memory_bytes,
            snapshot.process_cpu_percent,
            snapshot.system_memory_available_bytes,
            snapshot.system_memory_total_bytes
        );
        println!("devices={}", snapshot.devices.len());
        for device in &snapshot.devices {
            println!(
                "device index={} backend={} name={} free={} total={}",
                device.bridge_device_index,
                device.backend,
                device.name,
                device.memory_free,
                device.memory_total
            );
        }
        println!("instances={}", snapshot.instances.len());
        for instance in &snapshot.instances {
            println!(
                "instance id={} name={} state={} active={} queued={} slots={} group={} retention={:?}",
                instance.instance_id,
                instance.name,
                instance.state,
                instance.active_request_count,
                instance.queued_request_count,
                instance.n_parallel,
                instance.execution_group_id,
                instance.retention_mode
            );
        }
        println!("link_metrics={}", snapshot.link_metrics.len());
        for link in &snapshot.link_metrics {
            println!(
                "link peer={} transport={} probe_kind={} latency_ms={:.2} goodput_mbps={:.2} payload={} rounds={} duration_ms={:.0} error={}",
                link.peer_control_addr,
                link.transport,
                link.probe_kind,
                link.latency_ms,
                link.goodput_mbps,
                link.payload_bytes,
                link.rounds,
                link.duration_ms,
                link.error.as_deref().unwrap_or("")
            );
        }
    }
    Ok(())
}

fn add_agent_peer(agent_addr: &str, peer_addr: &str) -> Result<(), String> {
    let client = AgentClient::new(agent_addr.to_string());
    client
        .add_peer(peer_addr.to_string())
        .map_err(|e| e.to_string())
}

fn remove_agent_peer(agent_addr: &str, peer_addr: &str) -> Result<(), String> {
    let client = AgentClient::new(agent_addr.to_string());
    client
        .remove_peer(peer_addr.to_string())
        .map_err(|e| e.to_string())
}

fn configure_firewall(agent_addr: &str) -> Result<(), String> {
    let client = AgentClient::new(agent_addr.to_string());
    client.configure_firewall().map_err(|e| e.to_string())
}
