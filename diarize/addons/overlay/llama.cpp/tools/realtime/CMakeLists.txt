set(TARGET_LIB   llama-realtime-core)
set(TARGET_SMOKE llama-realtime-smoke)

add_library(${TARGET_LIB} STATIC
    audio-ring.cpp
    backend-factory.cpp
    stream-manager.cpp
    stream-session.cpp
    voxtral/voxtral-backend.cpp
    voxtral/voxtral-runtime.cpp
    sortformer/sortformer-audio.cpp
    sortformer/sortformer-backend.cpp
    sortformer/sortformer-gguf.cpp
    sortformer/sortformer-encoder.cpp
    sortformer/sortformer-frontend.cpp
    sortformer/sortformer-layer0.cpp
    sortformer/sortformer-model.cpp
    sortformer/sortformer-postprocess.cpp
    sortformer/sortformer-postnet.cpp
    sortformer/sortformer-preencode.cpp
    sortformer/sortformer-schema.cpp
    sortformer/sortformer-state.cpp
    sortformer/sortformer-streaming.cpp
)
target_compile_features(${TARGET_LIB} PRIVATE cxx_std_17)
target_include_directories(${TARGET_LIB} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/voxtral
    ${CMAKE_CURRENT_SOURCE_DIR}/sortformer
    ../../ggml/include
)
target_link_libraries(${TARGET_LIB} PUBLIC ggml)
if (TARGET ggml-cpu)
    # Voxtral runtime uses the CPU backend API directly for fallback contexts.
    target_link_libraries(${TARGET_LIB} PUBLIC ggml-cpu)
endif()

add_executable(${TARGET_SMOKE} realtime-smoke.cpp)
target_compile_features(${TARGET_SMOKE} PRIVATE cxx_std_17)
target_link_libraries(${TARGET_SMOKE} PRIVATE ${TARGET_LIB})

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET_SMOKE} RUNTIME)
endif()
