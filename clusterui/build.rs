use std::env;
use std::fs::File;
use std::io::BufWriter;
use std::path::{Path, PathBuf};

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("missing manifest dir"));
    let icon_png = manifest_dir.join("assets").join("engine.png");
    println!("cargo:rerun-if-changed={}", icon_png.display());

    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if target_os != "windows" {
        return;
    }

    if let Err(err) = embed_windows_icon(&icon_png) {
        panic!("failed to embed Windows icon resource: {err}");
    }
}

fn embed_windows_icon(icon_png: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let out_dir = PathBuf::from(env::var("OUT_DIR")?);
    let icon_ico = out_dir.join("engine.ico");
    write_multi_size_ico(icon_png, &icon_ico)?;
    let version = env::var("CARGO_PKG_VERSION").unwrap_or_else(|_| "0.1.0".to_string());

    let mut resource = winresource::WindowsResource::new();
    resource.set_icon(icon_ico.to_string_lossy().as_ref());
    resource.set("FileDescription", "Engine");
    resource.set("ProductName", "Openresearchtools-Engine");
    resource.set("InternalName", "Engine");
    resource.set("OriginalFilename", "Engine.exe");
    resource.set("ProductVersion", &version);
    resource.set("FileVersion", &version);
    resource.compile()?;
    Ok(())
}

fn write_multi_size_ico(
    source_png: &Path,
    out_ico: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let source = image::open(source_png)?.to_rgba8();
    let mut icon_dir = ico::IconDir::new(ico::ResourceType::Icon);

    for size in [16_u32, 24, 32, 40, 48, 64, 96, 128, 256] {
        let resized =
            image::imageops::resize(&source, size, size, image::imageops::FilterType::Lanczos3);
        let icon_image = ico::IconImage::from_rgba_data(size, size, resized.into_raw());
        icon_dir.add_entry(ico::IconDirEntry::encode(&icon_image)?);
    }

    let writer = BufWriter::new(File::create(out_ico)?);
    icon_dir.write(writer)?;
    Ok(())
}
