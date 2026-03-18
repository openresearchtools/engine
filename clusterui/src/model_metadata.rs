use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{self, BufReader, Read};
use std::path::Path;
use std::sync::{Mutex, OnceLock};

const GGUF_MAGIC: &[u8; 4] = b"GGUF";
const GGUF_STRING_MAX_BYTES: u64 = 1024 * 1024;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ModelFileMetadata {
    pub format: String,
    pub architecture: Option<String>,
    pub block_count: Option<u32>,
    pub embedding_length: Option<u32>,
    pub head_count: Option<u32>,
    pub head_count_kv: Option<u32>,
    pub trained_context_length: Option<u32>,
    pub uses_recurrent_state: bool,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct RuntimeVramEstimate {
    pub required_gpu_bytes: u64,
    pub model_gpu_bytes: u64,
    pub mmproj_gpu_bytes: u64,
    pub kv_cache_bytes: u64,
    pub workspace_bytes: u64,
    pub overhead_bytes: u64,
    pub detected_layer_count: Option<u32>,
    pub layers_on_gpu: Option<u32>,
}

#[derive(Clone, Copy)]
enum GgufValueType {
    Uint8 = 0,
    Int8 = 1,
    Uint16 = 2,
    Int16 = 3,
    Uint32 = 4,
    Int32 = 5,
    Float32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    Uint64 = 10,
    Int64 = 11,
    Float64 = 12,
}

impl GgufValueType {
    fn from_raw(value: u32) -> Option<Self> {
        match value {
            0 => Some(Self::Uint8),
            1 => Some(Self::Int8),
            2 => Some(Self::Uint16),
            3 => Some(Self::Int16),
            4 => Some(Self::Uint32),
            5 => Some(Self::Int32),
            6 => Some(Self::Float32),
            7 => Some(Self::Bool),
            8 => Some(Self::String),
            9 => Some(Self::Array),
            10 => Some(Self::Uint64),
            11 => Some(Self::Int64),
            12 => Some(Self::Float64),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
enum MetadataValue {
    Integer(i64),
    Unsigned(u64),
    Text(String),
    Bool(bool),
}

pub fn inspect_model_file(path: &Path) -> Option<ModelFileMetadata> {
    let cache = MODEL_METADATA_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let cache_key = path.display().to_string();
    if let Ok(guard) = cache.lock() {
        if let Some(cached) = guard.get(&cache_key) {
            return cached.clone();
        }
    }

    let loaded = inspect_model_file_inner(path).ok();
    if let Ok(mut guard) = cache.lock() {
        guard.insert(cache_key, loaded.clone());
    }
    loaded
}

pub fn expanded_model_dependency_paths(path: &Path) -> Vec<std::path::PathBuf> {
    let mut paths = vec![path.to_path_buf()];
    let Some(stem) = path.file_stem().and_then(|value| value.to_str()) else {
        return paths;
    };
    let Some((base, index, total)) = parse_gguf_shard_suffix(&stem.to_ascii_lowercase()) else {
        return paths;
    };
    if index != 1 || total <= 1 {
        return paths;
    }
    let Some(parent) = path.parent() else {
        return paths;
    };
    let ext = path
        .extension()
        .and_then(|value| value.to_str())
        .unwrap_or("gguf");
    paths.clear();
    for shard_index in 1..=total {
        paths.push(parent.join(format!("{base}-{shard_index:05}-of-{total:05}.{ext}")));
    }
    paths
}

pub fn expanded_model_dependency_relative_paths<'a, I>(
    selected_relative_path: &str,
    available_relative_paths: I,
) -> Vec<String>
where
    I: IntoIterator<Item = &'a str>,
{
    let path = Path::new(selected_relative_path);
    let Some(stem) = path.file_stem().and_then(|value| value.to_str()) else {
        return vec![selected_relative_path.to_string()];
    };
    let Some((base, index, total)) = parse_gguf_shard_suffix(&stem.to_ascii_lowercase()) else {
        return vec![selected_relative_path.to_string()];
    };
    if index != 1 || total <= 1 {
        return vec![selected_relative_path.to_string()];
    }

    let parent = path.parent();
    let ext = path
        .extension()
        .and_then(|value| value.to_str())
        .unwrap_or("gguf");
    let available = available_relative_paths
        .into_iter()
        .map(|value| value.replace('\\', "/"))
        .collect::<std::collections::HashSet<_>>();
    let mut expanded = Vec::new();
    for shard_index in 1..=total {
        let file_name = format!("{base}-{shard_index:05}-of-{total:05}.{ext}");
        let relative = match parent {
            Some(parent) if !parent.as_os_str().is_empty() => {
                format!(
                    "{}/{}",
                    parent.to_string_lossy().replace('\\', "/"),
                    file_name
                )
            }
            _ => file_name,
        };
        if available.contains(&relative) {
            expanded.push(relative);
        }
    }
    if expanded.is_empty() {
        vec![selected_relative_path.to_string()]
    } else {
        expanded
    }
}

pub fn estimate_runtime_vram(
    model_bytes: u64,
    mmproj_bytes: u64,
    metadata: Option<&ModelFileMetadata>,
    n_ctx: i32,
    n_batch: i32,
    n_parallel: i32,
    n_gpu_layers: i32,
) -> RuntimeVramEstimate {
    let layer_count = metadata.and_then(|value| value.block_count);
    let total_layers = layer_count.unwrap_or(0);
    let effective_ctx = n_ctx.max(0) as u64;
    let effective_batch = n_batch.max(1) as u64;
    let effective_parallel = n_parallel.max(1) as u64;

    let layers_on_gpu = if total_layers == 0 {
        None
    } else if n_gpu_layers < 0 {
        Some(total_layers)
    } else {
        Some((n_gpu_layers.max(0) as u32).min(total_layers))
    };

    let model_gpu_bytes = match (layer_count, layers_on_gpu) {
        (Some(total), Some(on_gpu)) if total > 0 && on_gpu < total => {
            let fraction = on_gpu as f64 / total as f64;
            (model_bytes as f64 * fraction).round() as u64
        }
        (Some(_), Some(0)) => 0,
        _ => model_bytes,
    };

    let kv_cache_bytes = metadata
        .and_then(|value| value.embedding_length)
        .zip(layers_on_gpu)
        .map(|(embedding_length, layers)| {
            if effective_ctx == 0 || layers == 0 {
                return 0;
            }
            let head_ratio = match (
                metadata.and_then(|value| value.head_count),
                metadata.and_then(|value| value.head_count_kv),
            ) {
                (Some(heads), Some(kv_heads)) if heads > 0 => kv_heads as f64 / heads as f64,
                _ => 1.0,
            };
            let kv_width = 2.0 * embedding_length as f64 * head_ratio;
            (effective_ctx as f64 * effective_parallel as f64 * layers as f64 * kv_width * 2.0)
                .round() as u64
        })
        .unwrap_or(0);

    let workspace_bytes = metadata
        .and_then(|value| value.embedding_length)
        .map(|embedding_length| {
            let active_tokens = effective_batch.min(effective_ctx.max(1));
            (embedding_length as u64)
                .saturating_mul(active_tokens)
                .saturating_mul(8)
        })
        .unwrap_or(0);

    let subtotal = model_gpu_bytes
        .saturating_add(mmproj_bytes)
        .saturating_add(kv_cache_bytes)
        .saturating_add(workspace_bytes);
    let overhead_bytes = ((subtotal as f64) * 0.08).round() as u64;

    RuntimeVramEstimate {
        required_gpu_bytes: subtotal.saturating_add(overhead_bytes),
        model_gpu_bytes,
        mmproj_gpu_bytes: mmproj_bytes,
        kv_cache_bytes,
        workspace_bytes,
        overhead_bytes,
        detected_layer_count: layer_count,
        layers_on_gpu,
    }
}

fn inspect_model_file_inner(path: &Path) -> io::Result<ModelFileMetadata> {
    let extension = path
        .extension()
        .and_then(|value| value.to_str())
        .map(|value| value.to_ascii_lowercase())
        .unwrap_or_default();
    if extension != "gguf" {
        return Ok(ModelFileMetadata {
            format: if extension.is_empty() {
                "unknown".to_string()
            } else {
                extension
            },
            ..ModelFileMetadata::default()
        });
    }

    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    let mut magic = [0u8; 4];
    reader.read_exact(&mut magic)?;
    if &magic != GGUF_MAGIC {
        return Ok(ModelFileMetadata {
            format: "gguf".to_string(),
            ..ModelFileMetadata::default()
        });
    }

    let version = read_u32(&mut reader)?;
    if version < 2 {
        return Ok(ModelFileMetadata {
            format: format!("gguf-v{version}"),
            ..ModelFileMetadata::default()
        });
    }

    let _tensor_count = read_u64(&mut reader)?;
    let kv_count = read_u64(&mut reader)?;
    let mut values = HashMap::new();
    for _ in 0..kv_count {
        let key = read_gguf_string(&mut reader)?;
        let value_type = read_u32(&mut reader)?;
        let parsed = read_metadata_value(&mut reader, value_type)?;
        if let Some(value) = parsed {
            values.insert(key, value);
        }
    }

    let architecture = value_text(&values, "general.architecture");
    let block_count = architecture
        .as_deref()
        .and_then(|arch| value_u32(&values, &format!("{arch}.block_count")))
        .or_else(|| value_u32(&values, "llama.block_count"));
    let embedding_length = architecture
        .as_deref()
        .and_then(|arch| value_u32(&values, &format!("{arch}.embedding_length")))
        .or_else(|| value_u32(&values, "llama.embedding_length"));
    let head_count = architecture
        .as_deref()
        .and_then(|arch| {
            value_u32(&values, &format!("{arch}.head_count"))
                .or_else(|| value_u32(&values, &format!("{arch}.attention.head_count")))
        })
        .or_else(|| value_u32(&values, "llama.head_count"))
        .or_else(|| value_u32(&values, "llama.attention.head_count"));
    let head_count_kv = architecture
        .as_deref()
        .and_then(|arch| {
            value_u32(&values, &format!("{arch}.head_count_kv"))
                .or_else(|| value_u32(&values, &format!("{arch}.attention.head_count_kv")))
        })
        .or_else(|| value_u32(&values, "llama.head_count_kv"))
        .or_else(|| value_u32(&values, "llama.attention.head_count_kv"));
    let trained_context_length = architecture
        .as_deref()
        .and_then(|arch| {
            value_u32(&values, &format!("{arch}.context_length"))
                .or_else(|| value_u32(&values, &format!("{arch}.context_length_train")))
        })
        .or_else(|| value_u32(&values, "llama.context_length"));
    let uses_recurrent_state = architecture.as_deref().is_some_and(|arch| {
        value_u32(&values, &format!("{arch}.ssm.state_size")).is_some()
            || value_u32(&values, &format!("{arch}.ssm.inner_size")).is_some()
            || value_u32(&values, &format!("{arch}.ssm.group_count")).is_some()
    });

    Ok(ModelFileMetadata {
        format: format!("gguf-v{version}"),
        architecture,
        block_count,
        embedding_length,
        head_count,
        head_count_kv,
        trained_context_length,
        uses_recurrent_state,
    })
}

fn read_metadata_value<R: Read>(
    reader: &mut R,
    raw_type: u32,
) -> io::Result<Option<MetadataValue>> {
    let Some(value_type) = GgufValueType::from_raw(raw_type) else {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("unknown GGUF value type {raw_type}"),
        ));
    };
    match value_type {
        GgufValueType::Uint8 => Ok(Some(MetadataValue::Unsigned(read_u8(reader)? as u64))),
        GgufValueType::Int8 => Ok(Some(MetadataValue::Integer(read_i8(reader)? as i64))),
        GgufValueType::Uint16 => Ok(Some(MetadataValue::Unsigned(read_u16(reader)? as u64))),
        GgufValueType::Int16 => Ok(Some(MetadataValue::Integer(read_i16(reader)? as i64))),
        GgufValueType::Uint32 => Ok(Some(MetadataValue::Unsigned(read_u32(reader)? as u64))),
        GgufValueType::Int32 => Ok(Some(MetadataValue::Integer(read_i32(reader)? as i64))),
        GgufValueType::Uint64 => Ok(Some(MetadataValue::Unsigned(read_u64(reader)?))),
        GgufValueType::Int64 => Ok(Some(MetadataValue::Integer(read_i64(reader)?))),
        GgufValueType::Bool => Ok(Some(MetadataValue::Bool(read_bool(reader)?))),
        GgufValueType::String => Ok(Some(MetadataValue::Text(read_gguf_string(reader)?))),
        GgufValueType::Float32 => {
            skip_bytes(reader, 4)?;
            Ok(None)
        }
        GgufValueType::Float64 => {
            skip_bytes(reader, 8)?;
            Ok(None)
        }
        GgufValueType::Array => {
            let nested_type = read_u32(reader)?;
            let item_count = read_u64(reader)?;
            for _ in 0..item_count {
                let _ = read_metadata_value(reader, nested_type)?;
            }
            Ok(None)
        }
    }
}

fn value_text(values: &HashMap<String, MetadataValue>, key: &str) -> Option<String> {
    match values.get(key)? {
        MetadataValue::Text(value) => Some(value.clone()),
        _ => None,
    }
}

fn value_u32(values: &HashMap<String, MetadataValue>, key: &str) -> Option<u32> {
    match values.get(key)? {
        MetadataValue::Integer(value) if *value >= 0 => u32::try_from(*value).ok(),
        MetadataValue::Unsigned(value) => u32::try_from(*value).ok(),
        MetadataValue::Bool(value) => Some(u32::from(*value)),
        _ => None,
    }
}

fn read_gguf_string<R: Read>(reader: &mut R) -> io::Result<String> {
    let length = read_u64(reader)?;
    if length > GGUF_STRING_MAX_BYTES {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("GGUF string too large: {length}"),
        ));
    }
    let mut bytes = vec![0u8; length as usize];
    reader.read_exact(&mut bytes)?;
    Ok(String::from_utf8_lossy(&bytes).into_owned())
}

fn skip_bytes<R: Read>(reader: &mut R, len: usize) -> io::Result<()> {
    let mut remaining = len;
    let mut scratch = [0u8; 256];
    while remaining > 0 {
        let to_read = remaining.min(scratch.len());
        reader.read_exact(&mut scratch[..to_read])?;
        remaining -= to_read;
    }
    Ok(())
}

fn read_bool<R: Read>(reader: &mut R) -> io::Result<bool> {
    Ok(read_u8(reader)? != 0)
}

fn read_u8<R: Read>(reader: &mut R) -> io::Result<u8> {
    let mut buf = [0u8; 1];
    reader.read_exact(&mut buf)?;
    Ok(buf[0])
}

fn read_i8<R: Read>(reader: &mut R) -> io::Result<i8> {
    Ok(read_u8(reader)? as i8)
}

fn read_u16<R: Read>(reader: &mut R) -> io::Result<u16> {
    let mut buf = [0u8; 2];
    reader.read_exact(&mut buf)?;
    Ok(u16::from_le_bytes(buf))
}

fn read_i16<R: Read>(reader: &mut R) -> io::Result<i16> {
    let mut buf = [0u8; 2];
    reader.read_exact(&mut buf)?;
    Ok(i16::from_le_bytes(buf))
}

fn read_u32<R: Read>(reader: &mut R) -> io::Result<u32> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_i32<R: Read>(reader: &mut R) -> io::Result<i32> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    Ok(i32::from_le_bytes(buf))
}

fn read_u64<R: Read>(reader: &mut R) -> io::Result<u64> {
    let mut buf = [0u8; 8];
    reader.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

fn read_i64<R: Read>(reader: &mut R) -> io::Result<i64> {
    let mut buf = [0u8; 8];
    reader.read_exact(&mut buf)?;
    Ok(i64::from_le_bytes(buf))
}

fn parse_gguf_shard_suffix(value: &str) -> Option<(String, i32, i32)> {
    let of_pos = value.rfind("-of-")?;
    let total_text = value.get(of_pos + 4..)?;
    if total_text.len() != 5 || !total_text.chars().all(|ch| ch.is_ascii_digit()) {
        return None;
    }
    let before = value.get(..of_pos)?;
    let dash_pos = before.rfind('-')?;
    let index_text = before.get(dash_pos + 1..)?;
    if index_text.len() != 5 || !index_text.chars().all(|ch| ch.is_ascii_digit()) {
        return None;
    }
    let base = before.get(..dash_pos)?.to_string();
    let index = index_text.parse::<i32>().ok()?;
    let total = total_text.parse::<i32>().ok()?;
    Some((base, index, total))
}

static MODEL_METADATA_CACHE: OnceLock<Mutex<HashMap<String, Option<ModelFileMetadata>>>> =
    OnceLock::new();
