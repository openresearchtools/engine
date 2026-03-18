use crate::agent::{default_local_agent_addr, ensure_local_agent, run_agent, AgentClient};
use crate::cluster_api::{
    default_runtime_dir, ChatRequest, ClusterApi, CreateInstanceParams, RetentionMode,
    TextGenerationResult,
};
use crate::protocol::{PlacementPlan, ScheduledInstance};
use std::env;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

#[derive(Clone)]
pub struct StartupArgs {
    pub agent_mode: bool,
    pub dump_state: bool,
    pub add_peer: Option<String>,
    pub remove_peer: Option<String>,
    pub runtime_dir: PathBuf,
    pub bind_addr: String,
}

impl StartupArgs {
    pub fn from_env() -> Self {
        let mut agent_mode = false;
        let mut dump_state = false;
        let mut add_peer = None;
        let mut remove_peer = None;
        let mut runtime_dir = default_runtime_dir().unwrap_or_else(|_| PathBuf::new());
        let default_bind_addr = default_local_agent_addr();
        let mut bind_addr = default_bind_addr.clone();
        let mut bind_was_explicit = false;

        let mut args = env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--agent" => agent_mode = true,
                "--dump-state" => dump_state = true,
                "--runtime-dir" => {
                    if let Some(value) = args.next() {
                        runtime_dir = PathBuf::from(value);
                    }
                }
                "--bind" => {
                    if let Some(value) = args.next() {
                        bind_addr = value;
                        bind_was_explicit = true;
                    }
                }
                "--add-peer" => {
                    if let Some(value) = args.next() {
                        add_peer = Some(value);
                    }
                }
                "--remove-peer" => {
                    if let Some(value) = args.next() {
                        remove_peer = Some(value);
                    }
                }
                _ => {}
            }
        }

        // Standalone host/agent mode must accept remote node connections by default.
        // The embedded local controller path still uses loopback when it starts an agent
        // internally, so existing local-only behavior stays intact there.
        if agent_mode && !bind_was_explicit && bind_addr == default_bind_addr {
            if let Some((_, port)) = bind_addr.rsplit_once(':') {
                bind_addr = format!("0.0.0.0:{port}");
            }
        }

        Self {
            agent_mode,
            dump_state,
            add_peer,
            remove_peer,
            runtime_dir,
            bind_addr,
        }
    }
}

pub struct NodeHost {
    runtime_dir: PathBuf,
    control_addr: String,
    local_client: Option<AgentClient>,
    local_api: Mutex<Option<ClusterApi>>,
}

impl NodeHost {
    pub fn new(runtime_dir: PathBuf, control_addr: String) -> Self {
        Self {
            runtime_dir,
            control_addr,
            local_client: None,
            local_api: Mutex::new(None),
        }
    }

    pub fn runtime_dir(&self) -> &Path {
        &self.runtime_dir
    }

    pub fn control_addr(&self) -> &str {
        &self.control_addr
    }

    pub fn local_client(&self) -> Option<AgentClient> {
        self.local_client.clone()
    }

    pub fn set_local_client(&mut self, client: AgentClient) {
        self.local_client = Some(client);
    }

    pub fn clear_local_connection(&mut self) {
        self.local_client = None;
        if let Ok(mut guard) = self.local_api.lock() {
            *guard = None;
        }
    }

    pub fn connect_local(&mut self) -> Result<String, String> {
        match ensure_local_agent(&self.runtime_dir, &self.control_addr) {
            Ok(client) => {
                self.local_client = Some(client);
                Ok(format!(
                    "Connected to local host {} using runtime {}",
                    self.control_addr,
                    self.runtime_dir.display()
                ))
            }
            Err(err) => {
                self.clear_local_connection();
                Err(format!("local host start/connect failed: {err}"))
            }
        }
    }

    pub fn create_instance(&self, params: &CreateInstanceParams) -> Result<i64, String> {
        self.with_local_api(|api| api.create_instance(params).map_err(|err| err.to_string()))
    }

    pub fn plan_instance(
        &self,
        params: &CreateInstanceParams,
        allowed_control_addrs: Option<Vec<String>>,
    ) -> Result<PlacementPlan, String> {
        self.local_client
            .as_ref()
            .ok_or_else(|| "local host is not connected".to_string())?
            .plan_instance(params.clone(), allowed_control_addrs)
            .map_err(|err| err.to_string())
    }

    pub fn list_placement_candidates(
        &self,
        params: &CreateInstanceParams,
        allowed_control_addrs: Option<Vec<String>>,
    ) -> Result<Vec<PlacementPlan>, String> {
        self.local_client
            .as_ref()
            .ok_or_else(|| "local host is not connected".to_string())?
            .list_placement_candidates(params.clone(), allowed_control_addrs)
            .map_err(|err| err.to_string())
    }

    pub fn schedule_instance(
        &self,
        params: &CreateInstanceParams,
        allowed_control_addrs: Option<Vec<String>>,
        load_immediately: bool,
    ) -> Result<ScheduledInstance, String> {
        self.local_client
            .as_ref()
            .ok_or_else(|| "local host is not connected".to_string())?
            .schedule_instance(params.clone(), allowed_control_addrs, load_immediately)
            .map_err(|err| err.to_string())
    }

    pub fn load_instance(&self, instance_id: i64) -> Result<(), String> {
        self.with_local_api(|api| {
            api.load_instance(instance_id)
                .map_err(|err| err.to_string())
        })
    }

    pub fn unload_instance(&self, instance_id: i64) -> Result<(), String> {
        self.with_local_api(|api| {
            api.unload_instance(instance_id)
                .map_err(|err| err.to_string())
        })
    }

    pub fn remove_instance(&self, instance_id: i64) -> Result<(), String> {
        self.with_local_api(|api| {
            api.remove_instance(instance_id)
                .map_err(|err| err.to_string())
        })
    }

    pub fn set_retention_mode(
        &self,
        instance_id: i64,
        retention_mode: RetentionMode,
    ) -> Result<(), String> {
        self.with_local_api(|api| {
            api.set_retention_mode(instance_id, retention_mode)
                .map_err(|err| err.to_string())
        })
    }

    pub fn chat_complete(&self, request: &ChatRequest) -> Result<TextGenerationResult, String> {
        self.with_local_api(|api| api.chat_complete(request).map_err(|err| err.to_string()))
    }

    fn with_local_api<T>(
        &self,
        func: impl FnOnce(&ClusterApi) -> Result<T, String>,
    ) -> Result<T, String> {
        let mut guard = self
            .local_api
            .lock()
            .map_err(|_| "local runtime lock is poisoned".to_string())?;
        if guard.is_none() {
            *guard = Some(ClusterApi::load(&self.runtime_dir).map_err(|err| err.to_string())?);
        }
        let Some(api) = guard.as_ref() else {
            return Err("local runtime is not connected".to_string());
        };
        func(api)
    }
}

pub fn run_host_services(runtime_dir: PathBuf, bind_addr: String) -> Result<(), String> {
    run_agent(runtime_dir, bind_addr).map_err(|err| err.to_string())
}

pub fn dump_agent_state(control_addr: &str) -> Result<(), String> {
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
        "rpc running={} endpoint={} advertised={} public_api_running={} public_api_addr={} advertised_public_api={} models_dir={} firewall_status={} firewall_action_required={}",
        snapshot.rpc_running,
        snapshot.rpc_endpoint.as_deref().unwrap_or(""),
        snapshot
            .advertised_rpc_endpoint
            .as_deref()
            .unwrap_or(""),
        snapshot.public_api_running,
        snapshot.public_api_addr.as_deref().unwrap_or(""),
        snapshot
            .advertised_public_api_addr
            .as_deref()
            .unwrap_or(""),
        snapshot.models_dir,
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
            "link peer={} transport={} latency_ms={:.2} goodput_mbps={:.2} payload={} rounds={} error={}",
            link.peer_control_addr,
            link.transport,
            link.latency_ms,
            link.goodput_mbps,
            link.payload_bytes,
            link.rounds,
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

pub fn add_agent_peer(agent_addr: &str, peer_addr: &str) -> Result<(), String> {
    let client = AgentClient::new(agent_addr.to_string());
    client
        .add_peer(peer_addr.to_string())
        .map_err(|e| e.to_string())
}

pub fn remove_agent_peer(agent_addr: &str, peer_addr: &str) -> Result<(), String> {
    let client = AgentClient::new(agent_addr.to_string());
    client
        .remove_peer(peer_addr.to_string())
        .map_err(|e| e.to_string())
}
