# Minimal upstream patch

Add this option near other build options in `llama.cpp/CMakeLists.txt`:

```cmake
option(LLAMA_BUILD_MARKDOWN_BRIDGE "llama: build markdown in-process bridge library" OFF)
```

Add this near other `add_subdirectory(...)` build sections:

```cmake
if (LLAMA_BUILD_COMMON AND LLAMA_BUILD_TOOLS AND LLAMA_BUILD_MARKDOWN_BRIDGE)
    add_subdirectory(MARKDOWN/bridge)
endif()
```

Then drop this folder into `llama.cpp/MARKDOWN/bridge`.

