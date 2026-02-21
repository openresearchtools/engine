#include "whisper-cli-entrypoint.h"

#include <fstream>

// Minimal dependency from whisper.cpp examples/common.h used by cli.cpp.
bool is_file_exist(const char * filename) {
    if (filename == nullptr || filename[0] == '\0') {
        return false;
    }
    std::ifstream ifs(filename, std::ios::binary);
    return ifs.good();
}

// Reuse upstream whisper-cli implementation but expose it as an in-process function.
#define main whisper_cli_inproc_main
#include "../../../whisper.cpp/examples/cli/cli.cpp"
#undef main
