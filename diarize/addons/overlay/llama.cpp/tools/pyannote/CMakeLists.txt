set(TARGET_INSPECT llama-pyannote-inspect)
set(TARGET_ALIGN   llama-pyannote-align)
set(TARGET_DIARIZE llama-pyannote-diarize)

add_executable(${TARGET_INSPECT} pyannote-inspect.cpp)
target_link_libraries(${TARGET_INSPECT} PRIVATE ggml)
target_include_directories(${TARGET_INSPECT} PRIVATE ../../ggml/include)
target_compile_features(${TARGET_INSPECT} PRIVATE cxx_std_17)

add_executable(${TARGET_ALIGN} pyannote-align.cpp)
target_compile_features(${TARGET_ALIGN} PRIVATE cxx_std_17)
target_include_directories(${TARGET_ALIGN} PRIVATE ../../vendor ../../ggml/include)
target_link_libraries(${TARGET_ALIGN} PRIVATE ggml)

add_executable(${TARGET_DIARIZE} pyannote-diarize.cpp)
target_compile_features(${TARGET_DIARIZE} PRIVATE cxx_std_17)
target_include_directories(${TARGET_DIARIZE} PRIVATE ../../vendor ../../ggml/include)
target_link_libraries(${TARGET_DIARIZE} PRIVATE ggml)

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET_INSPECT} ${TARGET_ALIGN} ${TARGET_DIARIZE} RUNTIME)
endif()
