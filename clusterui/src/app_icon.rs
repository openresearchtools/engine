use anyhow::{Context, Result};
use eframe::egui;
use image::{imageops::FilterType, ImageFormat};

const ENGINE_ICON_PNG: &[u8] =
    include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/assets/engine.png"));

pub(crate) fn build_engine_window_icon() -> egui::IconData {
    icon_rgba(64)
        .map(|(rgba, width, height)| egui::IconData {
            rgba,
            width,
            height,
        })
        .unwrap_or_else(|_| fallback_window_icon())
}

#[cfg(any(target_os = "windows", target_os = "macos"))]
pub(crate) fn build_tray_icon_rgba() -> Result<(Vec<u8>, u32, u32)> {
    icon_rgba(32)
}

fn icon_rgba(size: u32) -> Result<(Vec<u8>, u32, u32)> {
    let decoded = image::load_from_memory_with_format(ENGINE_ICON_PNG, ImageFormat::Png)
        .context("failed to decode bundled engine icon PNG")?;
    let resized = decoded
        .resize_exact(size, size, FilterType::Lanczos3)
        .to_rgba8();
    let (width, height) = resized.dimensions();
    Ok((resized.into_raw(), width, height))
}

fn fallback_window_icon() -> egui::IconData {
    const WIDTH: u32 = 32;
    const HEIGHT: u32 = 32;
    let mut rgba = Vec::with_capacity((WIDTH * HEIGHT * 4) as usize);
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            let border = x <= 2 || y <= 2 || x >= WIDTH - 3 || y >= HEIGHT - 3;
            let diagonal = (x as i32 - y as i32).abs() <= 2
                || ((WIDTH - 1 - x) as i32 - y as i32).abs() <= 2;
            let (r, g, b) = if diagonal {
                (56, 189, 248)
            } else if border {
                (15, 23, 42)
            } else {
                (30, 41, 59)
            };
            rgba.extend_from_slice(&[r, g, b, 255]);
        }
    }
    egui::IconData {
        rgba,
        width: WIDTH,
        height: HEIGHT,
    }
}
