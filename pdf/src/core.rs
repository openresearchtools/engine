use anyhow::{Context, Result, anyhow};
use once_cell::sync::Lazy;
use pdfium_render::prelude::*;
use regex::Regex;
use std::cmp::Ordering;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

static MULTI_BLANK_LINES_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"\n{3,}").expect("valid regex"));

#[derive(Debug, Clone)]
pub struct ExtractedDocument {
    pub text: String,
    pub pages: usize,
    pub chars: usize,
}

pub struct PdfiumTextExtractor {
    pdfium: Pdfium,
}

#[derive(Debug, Clone)]
struct MarkdownChunk {
    top: f32,
    left: f32,
    font_size: f32,
    text: String,
}

#[derive(Debug, Clone)]
struct MarkdownLine {
    text: String,
    font_size: f32,
}

#[derive(Debug, Clone)]
struct MarkdownHeadingGroup {
    level: usize,
    lines: Vec<String>,
}

#[derive(Debug, Clone)]
struct PageMarkdownData {
    normalized_text: String,
    lines: Vec<MarkdownLine>,
}

#[derive(Debug, Default)]
struct ReadabilityStats {
    content_lines: usize,
    suspicious_lines: usize,
    suspicious_pages: Vec<usize>,
}

impl PdfiumTextExtractor {
    pub fn new(pdfium_lib: Option<&Path>) -> Result<Self> {
        let pdfium = bind_pdfium(pdfium_lib)?;
        Ok(Self { pdfium })
    }

    pub fn extract_pdf_to_text(
        &self,
        pdf_path: &Path,
        password: Option<&str>,
    ) -> Result<ExtractedDocument> {
        let document = self
            .pdfium
            .load_pdf_from_file(pdf_path, password)
            .map_err(|err| anyhow!("PDF load failed: {err:?}"))?;

        let pages = document.pages().len() as usize;
        let mut page_blocks = Vec::with_capacity(pages.max(1));
        let mut readability = ReadabilityStats::default();

        for (page_idx, page) in document.pages().iter().enumerate() {
            let raw = extract_page_text_native(&page)?;
            let normalized = normalize_text_basic(&raw);
            update_readability_stats(&mut readability, &normalized, page_idx + 1);
            page_blocks.push(normalized);
        }

        ensure_machine_readable_text(pdf_path, pages, &readability)?;

        let text = join_pages_with_markers(&page_blocks);
        let chars = text.chars().count();

        Ok(ExtractedDocument { text, pages, chars })
    }

    pub fn extract_pdf_to_markdown(
        &self,
        pdf_path: &Path,
        password: Option<&str>,
    ) -> Result<ExtractedDocument> {
        let document = self
            .pdfium
            .load_pdf_from_file(pdf_path, password)
            .map_err(|err| anyhow!("PDF load failed: {err:?}"))?;

        let pages = document.pages().len() as usize;
        let mut page_blocks = Vec::with_capacity(pages.max(1));
        let mut page_data = Vec::with_capacity(pages.max(1));
        let mut readability = ReadabilityStats::default();

        for (page_idx, page) in document.pages().iter().enumerate() {
            let raw = extract_page_text_native(&page)?;
            let normalized = normalize_text_basic(&raw);
            update_readability_stats(&mut readability, &normalized, page_idx + 1);
            let lines = collect_page_lines_with_font_size(&page).unwrap_or_default();
            page_data.push(PageMarkdownData {
                normalized_text: normalized,
                lines,
            });
        }

        ensure_machine_readable_text(pdf_path, pages, &readability)?;

        let (heading_one, heading_two) = infer_global_heading_buckets(&page_data);
        for page in &page_data {
            let markdown = render_page_markdown_from_lines(
                &page.normalized_text,
                &page.lines,
                heading_one,
                heading_two,
            );
            page_blocks.push(markdown);
        }

        let text = join_pages_with_markers(&page_blocks);
        let chars = text.chars().count();

        Ok(ExtractedDocument { text, pages, chars })
    }
}

pub fn write_utf8_bom(path: &Path, content: &str, overwrite: bool) -> Result<()> {
    if path.exists() && !overwrite {
        return Err(anyhow!(
            "Output exists (use --overwrite to replace): {}",
            clean_display_path(path)
        ));
    }

    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).with_context(|| {
            format!(
                "Unable to create output directory {}",
                clean_display_path(parent)
            )
        })?;
    }

    let mut bytes = Vec::with_capacity(content.len() + 3);
    bytes.extend_from_slice(&[0xEF, 0xBB, 0xBF]); // UTF-8 BOM for reliable encoding detection.
    bytes.extend_from_slice(content.as_bytes());

    fs::write(path, bytes)
        .with_context(|| format!("Unable to write output file {}", clean_display_path(path)))
}

pub fn is_pdf(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| ext.eq_ignore_ascii_case("pdf"))
}

pub fn clean_display_path(path: &Path) -> String {
    let raw = path.to_string_lossy();
    raw.strip_prefix(r"\\?\")
        .unwrap_or(raw.as_ref())
        .to_string()
}

pub fn absolute_from_cwd(path: &Path) -> Result<PathBuf> {
    if path.is_absolute() {
        Ok(path.to_path_buf())
    } else {
        Ok(std::env::current_dir()?.join(path))
    }
}

fn bind_pdfium(pdfium_lib: Option<&Path>) -> Result<Pdfium> {
    if let Some(path) = pdfium_lib {
        let lib_path = resolve_pdfium_library_path(path)?;
        let bindings = Pdfium::bind_to_library(&lib_path)
            .map_err(|err| anyhow!("Failed to bind PDFium at {}: {err:?}", lib_path.display()))?;
        return Ok(Pdfium::new(bindings));
    }

    for env_key in ["PDFIUM_LIB", "PDFIUM_DLL"] {
        if let Ok(env_pdfium) = std::env::var(env_key) {
            let lib_path = resolve_pdfium_library_path(Path::new(&env_pdfium))?;
            if let Ok(bindings) = Pdfium::bind_to_library(&lib_path) {
                return Ok(Pdfium::new(bindings));
            }
        }
    }

    for candidate in default_pdfium_candidates() {
        let lib_path = resolve_pdfium_library_path(&candidate)?;
        if let Ok(bindings) = Pdfium::bind_to_library(&lib_path) {
            return Ok(Pdfium::new(bindings));
        }
    }

    let local_lib = Pdfium::pdfium_platform_library_name_at_path("./");
    if let Ok(bindings) = Pdfium::bind_to_library(&local_lib) {
        return Ok(Pdfium::new(bindings));
    }

    let bindings = Pdfium::bind_to_system_library().map_err(|err| {
        anyhow!(
            "Unable to bind PDFium. Provide --pdfium-lib PATH to a PDFium library file (or containing folder). Error: {err:?}"
        )
    })?;

    Ok(Pdfium::new(bindings))
}

fn default_pdfium_candidates() -> Vec<PathBuf> {
    let mut paths = Vec::new();

    if let Ok(cwd) = std::env::current_dir() {
        paths.push(cwd.join("vendor").join("pdfium"));
        paths.push(cwd.clone());
        paths.push(cwd.join("third_party").join("pdfium").join("bin"));
        paths.push(cwd.join("third_party").join("pdfium"));
    }

    if let Ok(exe) = std::env::current_exe() {
        if let Some(exe_dir) = exe.parent() {
            paths.push(exe_dir.join("vendor").join("pdfium"));
            paths.push(exe_dir.to_path_buf());
            paths.push(exe_dir.join("third_party").join("pdfium").join("bin"));
            paths.push(exe_dir.join("third_party").join("pdfium"));
        }
    }

    paths
}

fn resolve_pdfium_library_path(path: &Path) -> Result<PathBuf> {
    let abs = absolute_from_cwd(path)?;
    if abs.is_dir() {
        Ok(Pdfium::pdfium_platform_library_name_at_path(&abs))
    } else {
        Ok(abs)
    }
}

fn extract_page_text_native(page: &PdfPage<'_>) -> Result<String> {
    page.text()
        .map_err(|err| anyhow!("FPDFText_LoadPage failed: {err:?}"))
        .map(|text| text.all())
}

fn collect_page_lines_with_font_size(page: &PdfPage<'_>) -> Result<Vec<MarkdownLine>> {
    let mut chunks = Vec::new();

    for object in page.objects().iter() {
        let Some(text_object) = object.as_text_object() else {
            continue;
        };

        let bounds = match object.bounds() {
            Ok(bounds) => bounds,
            Err(_) => continue,
        };

        let scaled = text_object.scaled_font_size().value;
        let unscaled = text_object.unscaled_font_size().value;
        let font_size = if scaled > 0.0 { scaled } else { unscaled };
        if font_size <= 0.0 {
            continue;
        }

        let raw = text_object.text();
        if raw.trim().is_empty() {
            continue;
        }

        for line in raw
            .replace("\r\n", "\n")
            .replace('\r', "\n")
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty())
        {
            chunks.push(MarkdownChunk {
                top: bounds.top().value,
                left: bounds.left().value,
                font_size,
                text: line.to_string(),
            });
        }
    }

    if chunks.is_empty() {
        return Ok(Vec::new());
    }

    chunks.sort_by(|a, b| {
        b.top
            .total_cmp(&a.top)
            .then_with(|| a.left.total_cmp(&b.left))
            .then(Ordering::Equal)
    });

    let mut rows: Vec<Vec<MarkdownChunk>> = Vec::new();
    for chunk in chunks {
        let same_row = rows.last().is_some_and(|row| {
            let reference_top = row.first().map(|item| item.top).unwrap_or(chunk.top);
            let row_max_font = row
                .iter()
                .map(|item| item.font_size)
                .fold(chunk.font_size, f32::max);
            let tolerance = (row_max_font * 0.20).clamp(0.8, 2.0);
            (reference_top - chunk.top).abs() <= tolerance
        });

        if same_row {
            if let Some(row) = rows.last_mut() {
                row.push(chunk);
            }
        } else {
            rows.push(vec![chunk]);
        }
    }

    let mut lines = Vec::new();
    for mut row in rows {
        row.sort_by(|a, b| a.left.total_cmp(&b.left).then(Ordering::Equal));
        let mut font_buckets: HashMap<i32, usize> = HashMap::new();
        for chunk in &row {
            let bucket = (chunk.font_size * 10.0).round() as i32;
            let weight = chunk
                .text
                .chars()
                .filter(|ch| !ch.is_whitespace())
                .count()
                .max(1);
            *font_buckets.entry(bucket).or_default() += weight;
        }
        let row_font = font_buckets
            .into_iter()
            .max_by_key(|(_, weight)| *weight)
            .map(|(bucket, _)| bucket as f32 / 10.0)
            .unwrap_or(0.0);

        let mut rendered = String::new();
        for chunk in row {
            let text = chunk.text.trim();
            if text.is_empty() {
                continue;
            }

            if !rendered.is_empty() {
                rendered.push(' ');
            }

            rendered.push_str(text);
        }

        let normalized = normalize_text_basic(&rendered);
        if normalized.is_empty() {
            continue;
        }

        lines.push(MarkdownLine {
            text: normalized,
            font_size: row_font,
        });
    }

    Ok(lines)
}

fn infer_global_heading_buckets(pages: &[PageMarkdownData]) -> (Option<f32>, Option<f32>) {
    let mut all_lines = Vec::new();
    for page in pages {
        all_lines.extend(page.lines.iter().cloned());
    }

    if all_lines.is_empty() {
        return (None, None);
    }

    let body_font = infer_body_font_size(&all_lines);
    let min_heading_size = (body_font + 2.2).max(body_font * 1.22);

    let mut counts: HashMap<i32, usize> = HashMap::new();
    for line in all_lines {
        let bucket = (line.font_size * 10.0).round() as i32;
        let bucket_size = bucket as f32 / 10.0;
        if bucket_size >= min_heading_size {
            *counts.entry(bucket).or_default() += 1;
        }
    }

    let mut buckets = counts
        .into_iter()
        .filter(|(_, count)| *count >= 2)
        .map(|(bucket, _)| bucket as f32 / 10.0)
        .collect::<Vec<_>>();

    if buckets.is_empty() {
        return (None, None);
    }

    buckets.sort_by(|a, b| b.total_cmp(a));
    buckets.dedup_by(|a, b| same_font_bucket(*a, *b));

    (buckets.first().copied(), buckets.get(1).copied())
}

fn render_page_markdown_from_lines(
    page_text: &str,
    lines: &[MarkdownLine],
    heading_one_bucket: Option<f32>,
    heading_two_bucket: Option<f32>,
) -> String {
    if lines.is_empty() || heading_one_bucket.is_none() {
        return page_text.to_string();
    }

    let groups = detect_heading_groups_by_size(lines, heading_one_bucket, heading_two_bucket);
    if groups.is_empty() {
        return page_text.to_string();
    }

    apply_heading_groups_to_text(page_text, &groups)
}

fn detect_heading_groups_by_size(
    lines: &[MarkdownLine],
    heading_one_bucket: Option<f32>,
    heading_two_bucket: Option<f32>,
) -> Vec<MarkdownHeadingGroup> {
    if heading_one_bucket.is_none() {
        return Vec::new();
    }

    let mut groups = Vec::new();
    let mut current: Option<MarkdownHeadingGroup> = None;

    for line in lines {
        let bucket = (line.font_size * 10.0).round() / 10.0;
        let level = if heading_one_bucket.is_some_and(|top| same_font_bucket(bucket, top)) {
            Some(1)
        } else if heading_two_bucket.is_some_and(|second| same_font_bucket(bucket, second)) {
            Some(2)
        } else {
            None
        };

        let Some(level) = level else {
            if let Some(group) = current.take() {
                groups.push(group);
            }
            continue;
        };

        let should_continue = current.as_ref().is_some_and(|group| group.level == level);

        if should_continue {
            if let Some(group) = current.as_mut() {
                group.lines.push(line.text.trim().to_string());
            }
        } else {
            if let Some(group) = current.take() {
                groups.push(group);
            }
            current = Some(MarkdownHeadingGroup {
                level,
                lines: vec![line.text.trim().to_string()],
            });
        }
    }

    if let Some(group) = current.take() {
        groups.push(group);
    }

    groups
}

fn apply_heading_groups_to_text(page_text: &str, groups: &[MarkdownHeadingGroup]) -> String {
    let lines: Vec<String> = page_text.lines().map(str::to_string).collect();
    if lines.is_empty() {
        return String::new();
    }

    let mut cursor = 0usize;
    let mut out = String::new();

    for group in groups {
        if group.lines.is_empty() {
            continue;
        }

        let Some((start, end)) = find_heading_match(&lines, cursor, &group.lines) else {
            continue;
        };

        for line in &lines[cursor..start] {
            out.push_str(line);
            out.push('\n');
        }

        let joined = lines[start..end]
            .iter()
            .map(|line| line.trim())
            .filter(|line| !line.is_empty())
            .collect::<Vec<_>>()
            .join(" ");
        let heading_text = cleanup_heading_text(&joined);
        if !heading_text.is_empty() {
            if !out.is_empty() && !out.ends_with("\n\n") {
                out.push('\n');
            }
            out.push_str(&"#".repeat(group.level));
            out.push(' ');
            out.push_str(&heading_text);
            out.push_str("\n\n");
        }

        cursor = end;
    }

    for line in &lines[cursor..] {
        out.push_str(line);
        out.push('\n');
    }

    MULTI_BLANK_LINES_RE
        .replace_all(out.trim_end(), "\n\n")
        .to_string()
}

fn find_heading_match(
    lines: &[String],
    start_at: usize,
    heading_lines: &[String],
) -> Option<(usize, usize)> {
    for candidate in start_at..lines.len() {
        let mut line_cursor = candidate;
        let mut heading_cursor = 0usize;

        while heading_cursor < heading_lines.len() {
            while line_cursor < lines.len() && lines[line_cursor].trim().is_empty() {
                line_cursor += 1;
            }

            if line_cursor >= lines.len() {
                break;
            }

            if !lines_equivalent_for_match(&lines[line_cursor], &heading_lines[heading_cursor]) {
                break;
            }

            heading_cursor += 1;
            line_cursor += 1;
        }

        if heading_cursor == heading_lines.len() {
            return Some((candidate, line_cursor));
        }
    }

    None
}

fn lines_equivalent_for_match(a: &str, b: &str) -> bool {
    let left = canonicalize_for_match(a);
    let right = canonicalize_for_match(b);

    if left.is_empty() || right.is_empty() {
        return false;
    }

    if left == right {
        return true;
    }

    let min_len = left.len().min(right.len());
    min_len >= 10 && (left.contains(&right) || right.contains(&left))
}

fn canonicalize_for_match(input: &str) -> String {
    input
        .chars()
        .filter(|ch| ch.is_alphanumeric())
        .flat_map(|ch| ch.to_lowercase())
        .collect()
}

fn cleanup_heading_text(input: &str) -> String {
    let words: Vec<String> = input.split_whitespace().map(str::to_string).collect();
    if words.is_empty() {
        return String::new();
    }

    let mut merged = Vec::new();
    let mut idx = 0usize;

    while idx < words.len() {
        if idx + 1 < words.len() {
            let first = &words[idx];
            let second = &words[idx + 1];
            let should_merge = first.chars().count() == 1
                && idx == 0
                && first.chars().all(|ch| ch.is_alphabetic())
                && !first.eq_ignore_ascii_case("a")
                && !first.eq_ignore_ascii_case("i")
                && second.chars().next().is_some_and(|ch| ch.is_lowercase());

            if should_merge {
                merged.push(format!("{first}{second}"));
                idx += 2;
                continue;
            }
        }

        merged.push(words[idx].clone());
        idx += 1;
    }

    merged.join(" ")
}

fn same_font_bucket(a: f32, b: f32) -> bool {
    (a - b).abs() <= 0.11
}

fn infer_body_font_size(lines: &[MarkdownLine]) -> f32 {
    let mut buckets: HashMap<i32, usize> = HashMap::new();
    for line in lines {
        let trimmed = line.text.trim();
        if trimmed.is_empty() {
            continue;
        }

        let bucket = (line.font_size * 10.0).round() as i32;
        let weight = trimmed.chars().count().max(1);
        *buckets.entry(bucket).or_default() += weight;
    }

    buckets
        .into_iter()
        .max_by_key(|(_, weight)| *weight)
        .map(|(bucket, _)| bucket as f32 / 10.0)
        .unwrap_or(11.0)
}

fn normalize_text_basic(input: &str) -> String {
    let text = normalize_newlines(input);
    let text = strip_invisible_text_controls(&text).replace('\u{00A0}', " ");
    let text = fix_common_mojibake(&text);
    let text = strip_invisible_text_controls(&text);
    let text = text
        .lines()
        .map(str::trim_end)
        .collect::<Vec<_>>()
        .join("\n");
    let text = MULTI_BLANK_LINES_RE.replace_all(&text, "\n\n").to_string();
    text.trim().to_string()
}

fn join_pages_with_markers(page_blocks: &[String]) -> String {
    let mut out = String::new();

    for (idx, page_text) in page_blocks.iter().enumerate() {
        out.push_str(&format!("<-page{}->\n", idx + 1));
        out.push_str(page_text.trim_end());
        if idx + 1 < page_blocks.len() {
            out.push_str("\n\n");
        }
    }

    out
}

fn ensure_machine_readable_text(
    pdf_path: &Path,
    page_count: usize,
    stats: &ReadabilityStats,
) -> Result<()> {
    if !looks_non_machine_readable(page_count, stats) {
        return Ok(());
    }

    let sample_pages = stats
        .suspicious_pages
        .iter()
        .take(6)
        .map(|page| page.to_string())
        .collect::<Vec<_>>()
        .join(", ");

    Err(anyhow!(
        "Text extraction quality check failed: content appears non-machine-readable \
         (likely broken/missing Unicode font mapping). \
         Suspicious pages: [{}]. Use a different PDF source or OCR fallback.",
        sample_pages
    )
    .context(format!("Machine-readability gate rejected {}", clean_display_path(pdf_path))))
}

fn looks_non_machine_readable(page_count: usize, stats: &ReadabilityStats) -> bool {
    if stats.content_lines == 0 {
        return true;
    }

    let suspicious_pages = stats.suspicious_pages.len();
    let suspicious_page_ratio = suspicious_pages as f32 / page_count.max(1) as f32;

    (suspicious_page_ratio >= 0.20 && stats.suspicious_lines >= 20)
        || (suspicious_page_ratio >= 0.10 && stats.suspicious_lines >= 35)
        || (suspicious_pages >= 8 && stats.suspicious_lines >= 60)
}

fn update_readability_stats(stats: &mut ReadabilityStats, page_text: &str, page_number: usize) {
    let mut page_content_lines = 0usize;
    let mut page_suspicious_lines = 0usize;

    for line in page_text.lines().map(str::trim) {
        if line.is_empty() {
            continue;
        }

        page_content_lines += 1;
        if line_looks_garbled(line) {
            page_suspicious_lines += 1;
        }
    }

    stats.content_lines += page_content_lines;
    stats.suspicious_lines += page_suspicious_lines;

    let page_suspicious = (page_content_lines >= 8
        && page_suspicious_lines >= 5
        && page_suspicious_lines as f32 / page_content_lines as f32 >= 0.45)
        || page_suspicious_lines >= 12;

    if page_suspicious {
        stats.suspicious_pages.push(page_number);
    }
}

fn line_looks_garbled(line: &str) -> bool {
    let trimmed = line.trim();
    let char_count = trimmed.chars().filter(|ch| !ch.is_whitespace()).count();
    if char_count < 28 {
        return false;
    }

    let tokens: Vec<&str> = trimmed.split_whitespace().collect();
    if tokens.len() < 5 {
        return false;
    }

    let alpha = trimmed
        .chars()
        .filter(|ch| !ch.is_whitespace() && ch.is_alphabetic())
        .count();
    let ascii_punct = trimmed
        .chars()
        .filter(|ch| !ch.is_whitespace() && ch.is_ascii_punctuation())
        .count();

    let alpha_ratio = alpha as f32 / char_count as f32;
    let punct_ratio = ascii_punct as f32 / char_count as f32;

    let wordlike_tokens = tokens.iter().filter(|token| token_looks_wordlike(token)).count();
    let wordlike_ratio = wordlike_tokens as f32 / tokens.len() as f32;

    let long_punct_run = has_long_ascii_punctuation_run(trimmed, 5);

    (punct_ratio >= 0.30 && alpha_ratio <= 0.55 && wordlike_ratio <= 0.35)
        || (punct_ratio >= 0.45 && alpha_ratio <= 0.70)
        || (long_punct_run && wordlike_ratio <= 0.45 && alpha_ratio <= 0.60)
}

fn token_looks_wordlike(token: &str) -> bool {
    let normalized = token.trim_matches(|ch: char| {
        ch.is_ascii_punctuation() || matches!(ch, '\u{2018}' | '\u{2019}' | '\u{201C}' | '\u{201D}')
    });

    if normalized.is_empty() {
        return false;
    }

    let mut letter_count = 0usize;
    let mut vowel_count = 0usize;

    for ch in normalized.chars() {
        if ch.is_alphabetic() {
            letter_count += 1;
            if matches!(ch.to_ascii_lowercase(), 'a' | 'e' | 'i' | 'o' | 'u' | 'y') {
                vowel_count += 1;
            }
        }
    }

    letter_count >= 3 && vowel_count >= 1
}

fn has_long_ascii_punctuation_run(input: &str, min_len: usize) -> bool {
    let mut run = 0usize;
    for ch in input.chars() {
        if ch.is_ascii_punctuation() {
            run += 1;
            if run >= min_len {
                return true;
            }
        } else if !ch.is_whitespace() {
            run = 0;
        }
    }
    false
}

fn normalize_newlines(input: &str) -> String {
    input.replace("\r\n", "\n").replace('\r', "\n")
}

fn strip_invisible_text_controls(input: &str) -> String {
    input
        .chars()
        .filter(|ch| !is_invisible_text_control(*ch))
        .collect()
}

fn is_invisible_text_control(ch: char) -> bool {
    if matches!(ch, '\n' | '\t') {
        return false;
    }

    let code = ch as u32;
    ((0x00..=0x1F).contains(&code) || (0x7F..=0x9F).contains(&code))
        || matches!(
            ch,
            '\u{FEFF}' | '\u{200B}' | '\u{200C}' | '\u{200D}' | '\u{2060}'
        )
}

fn fix_symbol_mojibake(input: &str) -> String {
    let mut text = input.to_string();

    for (bad, good) in [
        ("Î±", "α"),
        ("Î²", "β"),
        ("Î³", "γ"),
        ("Î´", "δ"),
        ("Îµ", "ε"),
        ("Î¸", "θ"),
        ("Î»", "λ"),
        ("Î¼", "μ"),
        ("Î”", "Δ"),
        ("Î£", "Σ"),
        ("Î©", "Ω"),
        ("Ï€", "π"),
        ("Ïƒ", "σ"),
        ("Ï‰", "ω"),
        ("Ï†", "φ"),
        ("Ïˆ", "ψ"),
        ("Â±", "±"),
        ("Â·", "·"),
        ("Âµ", "μ"),
        ("âˆ’", "−"),
        ("â‰¤", "≤"),
        ("â‰¥", "≥"),
        ("â‰ˆ", "≈"),
        ("\u{00E2}\u{2030}\u{00A0}", "≠"),
        ("âˆš", "√"),
        ("âˆ‘", "∑"),
        ("âˆ", "∏"),
        ("âˆ«", "∫"),
        ("âˆ‚", "∂"),
        ("âˆž", "∞"),
        ("â†’", "→"),
        ("â‡’", "⇒"),
    ] {
        text = text.replace(bad, good);
    }

    text
}

fn fix_common_mojibake(input: &str) -> String {
    let mut text = input.to_string();
    for (bad, good) in [
        ("Ã¯Â»Â¿", ""),
        ("Ã¢â‚¬â„¢", "\u{2019}"),
        ("Ã¢â‚¬Ëœ", "\u{2018}"),
        ("Ã¢â‚¬Å“", "\u{201C}"),
        ("Ã¢â‚¬\u{9d}", "\u{201D}"),
        ("Ã¢â‚¬â€œ", "\u{2013}"),
        ("Ã¢â‚¬â€", "\u{2014}"),
        ("Ã¢â‚¬Â¦", "\u{2026}"),
        ("Ã¢â‚¬Â¢", "\u{2022}"),
        ("Ã‚ ", " "),
        ("Ã‚", ""),
        ("â€™", "\u{2019}"),
        ("â€˜", "\u{2018}"),
        ("â€œ", "\u{201C}"),
        ("â€", "\u{201D}"),
        ("â€“", "\u{2013}"),
        ("â€”", "\u{2014}"),
        ("â€¦", "\u{2026}"),
        ("â€¢", "\u{2022}"),
        ("Â±", "\u{00B1}"),
        ("Â·", "\u{00B7}"),
        ("âˆ’", "\u{2212}"),
        ("â‰¤", "\u{2264}"),
        ("â‰¥", "\u{2265}"),
        ("â‰ˆ", "\u{2248}"),
        ("âˆš", "\u{221A}"),
        ("â†’", "\u{2192}"),
        ("â‡’", "\u{21D2}"),
    ] {
        text = text.replace(bad, good);
    }
    fix_symbol_mojibake(&text)
}
