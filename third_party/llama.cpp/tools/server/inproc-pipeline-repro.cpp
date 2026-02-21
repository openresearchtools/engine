#include <filesystem>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include "tools/pyannote/pyannote-entrypoints.h"
#include "tools/whisper/whisper-cli-entrypoint.h"

static int run_inproc_main(int (*entrypoint)(int, char **), const std::vector<std::string> & args) {
    if (entrypoint == nullptr || args.empty()) {
        return 2;
    }

    std::vector<std::string> argv_storage = args;
    std::vector<char *> argv;
    argv.reserve(argv_storage.size());
    for (auto & s : argv_storage) {
        argv.push_back(s.data());
    }

    return entrypoint((int) argv.size(), argv.data());
}

int main(int argc, char ** argv) {
    if (argc < 6) {
        std::cerr << "Usage: " << argv[0]
                  << " <audio_path> <whisper_model> <segmentation_gguf> <work_dir> <device>\n";
        return 2;
    }

    const std::string audio_path = std::filesystem::absolute(argv[1]).string();
    const std::string whisper_model = std::filesystem::absolute(argv[2]).string();
    const std::string segmentation_gguf = std::filesystem::absolute(argv[3]).string();
    const std::string work_dir = std::filesystem::absolute(argv[4]).string();
    const std::string device = argv[5];
    const bool run_threaded = argc >= 7 && std::string(argv[6]) == "--threaded";

    const std::string whisper_out_prefix = (std::filesystem::path(work_dir) / "whisper_repro").string();
    const std::string diar_out_dir = (std::filesystem::path(work_dir) / "diarization").string();

    std::filesystem::create_directories(work_dir);
    std::filesystem::create_directories(diar_out_dir);

    auto run_pipeline = [&]() -> int {
    const std::vector<std::string> whisper_args = {
        "whisper-cli",
        "-m", whisper_model,
        "-f", audio_path,
        "-ojf",
        "-of", whisper_out_prefix,
        "-np",
        "-l", "en",
        "-dev", "0",
        "-fa",
    };

    std::cout << "[repro] calling whisper inproc\n";
    const int whisper_rc = run_inproc_main(whisper_cli_inproc_main, whisper_args);
    std::cout << "[repro] whisper rc=" << whisper_rc << "\n";
    if (whisper_rc != 0) {
        return whisper_rc;
    }

    const std::vector<std::string> diar_args = {
        "llama-pyannote-diarize",
        "--audio", audio_path,
        "--segmentation-gguf", segmentation_gguf,
        "--output-dir", diar_out_dir,
        "--device", device,
        "--offline",
        "--num-speakers", "2",
        "--min-duration-off", "0.0",
    };

    std::cout << "[repro] calling diarization inproc\n";
    const int diar_rc = run_inproc_main(llama_pyannote_diarize_main, diar_args);
    std::cout << "[repro] diarization rc=" << diar_rc << "\n";
    if (diar_rc != 0) {
        return diar_rc;
    }

    // Stress string allocation/deallocation after diarization return.
    std::cout << "[repro] post-diarization string stress\n";
    for (int i = 0; i < 200000; ++i) {
        std::string s = work_dir + "/" + std::to_string(i);
        if (s.empty()) {
            std::cerr << "[repro] impossible branch\n";
            return 1;
        }
    }

    std::cout << "[repro] success\n";
    return 0;
    };

    if (!run_threaded) {
        return run_pipeline();
    }

    int rc = 1;
    std::thread th([&]() {
        rc = run_pipeline();
    });
    th.join();
    std::cout << "[repro] threaded mode done rc=" << rc << "\n";
    return rc;
}
