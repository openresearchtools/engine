#pragma once

// In-process entrypoint that reuses whisper.cpp CLI logic without spawning a child process.
int whisper_cli_inproc_main(int argc, char ** argv);
