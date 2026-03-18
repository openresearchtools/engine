#[allow(dead_code)]
mod audio_assembler;
#[allow(dead_code)]
mod audio_orchestrator;
mod llama_bridge;

use std::env;

fn usage() -> &'static str {
    "example-cli usage:
  example-cli <bridge_subcommand> [bridge args...]
  example-cli bridge <bridge_subcommand> [bridge args...]
  example-cli pdf [pdf args...]
  example-cli pdfvlm [pdf_to_markdown args...]
  example-cli help

bridge subcommands:
  list-devices
  vlm
  audio
  audio-session
  chat
  embed
  rerank

examples:
  example-cli list-devices
  example-cli bridge audio --audio-file <audio-file> --mode speech --custom default --whisper-model <whisper.bin>
  example-cli bridge audio-session --audio-file <audio-file> --diarization-model-path <sortformer.gguf> --whisper-model <whisper.bin>
  example-cli bridge audio-session --stdin-pcm-s16le --transcription-realtime-model <voxtral.gguf>
  example-cli chat --model <gguf> --markdown <file.md>
  example-cli vlm --model <gguf> --mmproj <gguf> --image <image.png>
  example-cli pdf extract --input <file-or-dir> --output <path>
  example-cli pdfvlm --pdf <file.pdf> --model <gguf> --mmproj <gguf>"
}

fn run_pdf_subcommand(args: &[String]) -> Result<(), String> {
    let mut argv = Vec::with_capacity(args.len() + 1);
    argv.push("pdf".to_string());
    argv.extend_from_slice(args);
    pdf::run_pdf_cli_from_args(&argv).map_err(|e| e.to_string())
}

fn run_pdfvlm_subcommand(args: &[String]) -> Result<(), String> {
    let mut argv = Vec::with_capacity(args.len() + 1);
    argv.push("pdf_to_markdown".to_string());
    argv.extend_from_slice(args);
    pdfvlm::run_pdf_to_markdown_cli_from_args(&argv).map_err(|e| e.to_string())
}

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        eprintln!("{}", usage());
        std::process::exit(1);
    }

    let result = match args[1].as_str() {
        "help" | "--help" | "-h" => {
            println!("{}", usage());
            return;
        }
        "pdf" | "convert" => run_pdf_subcommand(&args[2..]),
        "pdfvlm" | "pdf-vlm" | "vlmpdf" | "pdf-to-markdown" => run_pdfvlm_subcommand(&args[2..]),
        "bridge" => {
            if args.len() < 3 {
                Err("missing bridge subcommand".to_string())
            } else {
                llama_bridge::run_bridge_cli_subcommand(args[2].as_str(), &args[3..])
            }
        }
        _ => llama_bridge::run_bridge_cli_subcommand(args[1].as_str(), &args[2..]),
    };

    if let Err(e) = result {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}
