find_package(Threads REQUIRED)

if (NOT TARGET server-context)
    message(FATAL_ERROR "LLAMA markdown bridge requires LLAMA_BUILD_SERVER=ON")
endif()

if (NOT TARGET llama-realtime-core)
    message(FATAL_ERROR "LLAMA markdown bridge requires llama-realtime-core")
endif()

option(LLAMA_SERVER_BRIDGE_ENABLE_FFMPEG "Enable FFmpeg-based in-memory audio conversion for raw audio requests" OFF)
set(LLAMA_SERVER_BRIDGE_FFMPEG_ROOT "" CACHE PATH "Path to FFmpeg root containing include/ and lib/ for shared LGPL builds")
option(LLAMA_SERVER_AUDIO_BUILD "Build native live audio capture bridge library" ON)
option(LLAMA_SERVER_AUDIO_ENABLE_WEBRTC "Enable WebRTC AudioProcessing in the native live audio bridge" OFF)
set(LLAMA_SERVER_AUDIO_MINIAUDIO_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/vendor/miniaudio" CACHE PATH "Path containing miniaudio.h for the native live audio bridge")
set(LLAMA_SERVER_AUDIO_WEBRTC_ROOT "" CACHE PATH "Path to the webrtc-audio-processing source root")
set(LLAMA_SERVER_AUDIO_WEBRTC_LIBRARY "" CACHE FILEPATH "Deprecated single-path WebRTC AudioProcessing static library")
set(LLAMA_SERVER_AUDIO_WEBRTC_LIBRARIES "" CACHE STRING "Semicolon-separated paths to the built webrtc-audio-processing static libraries")

set(TARGET llama-server-bridge)
set(TARGET_MULTI_NODE multi-node-server)
set(TARGET_MULTI_NODE_CORE multi-node-server-core)
set(TARGET_AUDIO llama-server-audio)
set(TARGET_RT_SMOKE llama-server-bridge-realtime-smoke)
set(MULTI_NODE_NATIVE_DIR "${CMAKE_SOURCE_DIR}/MARKDOWN/multi_node_server/native")

add_library(${TARGET} SHARED
    llama_server_bridge.cpp
    llama_server_bridge.h
    llama_server_cluster.cpp
    llama_server_cluster.h
)

target_compile_features(${TARGET} PRIVATE cxx_std_17)

target_include_directories(${TARGET} PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/common
    ${CMAKE_SOURCE_DIR}/tools/server
)

target_compile_definitions(${TARGET} PRIVATE
    LLAMA_SERVER_BRIDGE_BUILD
    LLAMA_SERVER_CLUSTER_BUILD
)

target_link_libraries(${TARGET}
    PRIVATE
        server-context
        llama-realtime-core
        common
        mtmd
        ${CMAKE_THREAD_LIBS_INIT}
)

if (LLAMA_SERVER_BRIDGE_ENABLE_FFMPEG)
    if (NOT LLAMA_SERVER_BRIDGE_FFMPEG_ROOT)
        message(FATAL_ERROR "LLAMA_SERVER_BRIDGE_ENABLE_FFMPEG=ON requires LLAMA_SERVER_BRIDGE_FFMPEG_ROOT")
    endif()

    target_compile_definitions(${TARGET} PRIVATE LLAMA_SERVER_BRIDGE_USE_FFMPEG)
    target_include_directories(${TARGET} PRIVATE "${LLAMA_SERVER_BRIDGE_FFMPEG_ROOT}/include")
    target_link_directories(${TARGET} PRIVATE "${LLAMA_SERVER_BRIDGE_FFMPEG_ROOT}/lib")
    target_link_libraries(${TARGET}
        PRIVATE
            avformat
            avcodec
            avutil
            swresample
            swscale
    )

    if (APPLE)
        set_target_properties(${TARGET} PROPERTIES
            BUILD_RPATH "@loader_path/vendor/ffmpeg/lib"
            INSTALL_RPATH "@loader_path/vendor/ffmpeg/lib")
    elseif(UNIX)
        set_target_properties(${TARGET} PROPERTIES
            BUILD_RPATH "\$ORIGIN/vendor/ffmpeg/lib"
            INSTALL_RPATH "\$ORIGIN/vendor/ffmpeg/lib")
    endif()
endif()

install(TARGETS ${TARGET}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

if (NOT EXISTS "${MULTI_NODE_NATIVE_DIR}/server-context.cpp")
    message(FATAL_ERROR "multi-node-server native source tree is missing at ${MULTI_NODE_NATIVE_DIR}")
endif()

set(MULTI_NODE_SERVER_ASSET_DIR "${CMAKE_CURRENT_BINARY_DIR}/multi-node-server-assets")
set(MULTI_NODE_SERVER_ASSET_SRCS)
foreach(asset index.html.gz loading.html)
    set(input "${MULTI_NODE_NATIVE_DIR}/public/${asset}")
    set(output "${MULTI_NODE_SERVER_ASSET_DIR}/${asset}.hpp")
    list(APPEND MULTI_NODE_SERVER_ASSET_SRCS "${output}")
    add_custom_command(
        DEPENDS "${input}"
        OUTPUT "${output}"
        COMMAND "${CMAKE_COMMAND}" "-DINPUT=${input}" "-DOUTPUT=${output}" -P "${CMAKE_SOURCE_DIR}/scripts/xxd.cmake"
    )
    set_source_files_properties("${output}" PROPERTIES GENERATED TRUE)
endforeach()

add_library(${TARGET_MULTI_NODE_CORE} STATIC
    ${MULTI_NODE_SERVER_ASSET_SRCS}
    ${MULTI_NODE_NATIVE_DIR}/server-task.cpp
    ${MULTI_NODE_NATIVE_DIR}/server-task.h
    ${MULTI_NODE_NATIVE_DIR}/server-queue.cpp
    ${MULTI_NODE_NATIVE_DIR}/server-queue.h
    ${MULTI_NODE_NATIVE_DIR}/server-common.cpp
    ${MULTI_NODE_NATIVE_DIR}/server-common.h
    ${MULTI_NODE_NATIVE_DIR}/server-context.cpp
    ${MULTI_NODE_NATIVE_DIR}/server-context.h
    ${MULTI_NODE_NATIVE_DIR}/server-http.cpp
    ${MULTI_NODE_NATIVE_DIR}/server-http.h
    ${MULTI_NODE_NATIVE_DIR}/server-models.cpp
    ${MULTI_NODE_NATIVE_DIR}/server-models.h
)

if (BUILD_SHARED_LIBS)
    set_target_properties(${TARGET_MULTI_NODE_CORE} PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

target_compile_features(${TARGET_MULTI_NODE_CORE} PRIVATE cxx_std_17)
target_include_directories(${TARGET_MULTI_NODE_CORE} PRIVATE
    ${MULTI_NODE_NATIVE_DIR}
    ${MULTI_NODE_SERVER_ASSET_DIR}
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/common
    ${CMAKE_SOURCE_DIR}/tools/mtmd
)
target_link_libraries(${TARGET_MULTI_NODE_CORE}
    PUBLIC
        common
        mtmd
        cpp-httplib
        ${CMAKE_THREAD_LIBS_INIT}
)

add_library(${TARGET_MULTI_NODE} SHARED
    llama_server_multi_node_server.cpp
    llama_server_multi_node_server.h
)

target_compile_features(${TARGET_MULTI_NODE} PRIVATE cxx_std_17)

target_include_directories(${TARGET_MULTI_NODE} PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/common
    ${CMAKE_SOURCE_DIR}/tools/server
)

target_compile_definitions(${TARGET_MULTI_NODE} PRIVATE
    LLAMA_SERVER_CLUSTER_BUILD
)

target_link_libraries(${TARGET_MULTI_NODE}
    PRIVATE
        ${TARGET_MULTI_NODE_CORE}
        ${TARGET}
        server-context
        llama-realtime-core
        common
        mtmd
        cpp-httplib
        ${CMAKE_THREAD_LIBS_INIT}
)

install(TARGETS ${TARGET_MULTI_NODE}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

if (LLAMA_SERVER_AUDIO_BUILD)
    if (NOT EXISTS "${LLAMA_SERVER_AUDIO_MINIAUDIO_INCLUDE_DIR}/miniaudio.h")
        message(FATAL_ERROR "LLAMA_SERVER_AUDIO_BUILD=ON requires miniaudio.h under LLAMA_SERVER_AUDIO_MINIAUDIO_INCLUDE_DIR")
    endif()

    add_library(${TARGET_AUDIO} SHARED
        llama_server_audio_capture.cpp
        llama_server_audio_capture.h
    )

    target_compile_features(${TARGET_AUDIO} PRIVATE cxx_std_17)
    target_compile_definitions(${TARGET_AUDIO} PRIVATE LLAMA_SERVER_AUDIO_BUILD)
    target_include_directories(${TARGET_AUDIO} PRIVATE
        ${CMAKE_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/ggml/include
        ${CMAKE_SOURCE_DIR}/common
        ${CMAKE_SOURCE_DIR}/tools/server
        ${LLAMA_SERVER_AUDIO_MINIAUDIO_INCLUDE_DIR}
    )

    target_link_libraries(${TARGET_AUDIO}
        PRIVATE
            ${TARGET}
            ${CMAKE_THREAD_LIBS_INIT}
    )

    if (UNIX)
        target_link_libraries(${TARGET_AUDIO} PRIVATE ${CMAKE_DL_LIBS})
    endif()

    if (LLAMA_SERVER_AUDIO_ENABLE_WEBRTC)
        set(_llama_server_audio_webrtc_libs "${LLAMA_SERVER_AUDIO_WEBRTC_LIBRARIES}")
        if (NOT _llama_server_audio_webrtc_libs AND LLAMA_SERVER_AUDIO_WEBRTC_LIBRARY)
            set(_llama_server_audio_webrtc_libs "${LLAMA_SERVER_AUDIO_WEBRTC_LIBRARY}")
        endif()

        if (NOT LLAMA_SERVER_AUDIO_WEBRTC_ROOT)
            message(FATAL_ERROR "LLAMA_SERVER_AUDIO_ENABLE_WEBRTC=ON requires LLAMA_SERVER_AUDIO_WEBRTC_ROOT")
        endif()
        if (NOT _llama_server_audio_webrtc_libs)
            message(FATAL_ERROR "LLAMA_SERVER_AUDIO_ENABLE_WEBRTC=ON requires LLAMA_SERVER_AUDIO_WEBRTC_LIBRARIES")
        endif()

        target_compile_definitions(${TARGET_AUDIO} PRIVATE LLAMA_SERVER_AUDIO_HAVE_WEBRTC)
        target_compile_definitions(${TARGET_AUDIO} PRIVATE
            WEBRTC_AUDIO_PROCESSING_ONLY_BUILD
            NDEBUG
        )

        if (WIN32)
            target_compile_definitions(${TARGET_AUDIO} PRIVATE
                WEBRTC_WIN
                _WIN32
                _WINSOCKAPI_
                NOMINMAX
                _USE_MATH_DEFINES
            )
        elseif(APPLE)
            target_compile_definitions(${TARGET_AUDIO} PRIVATE
                WEBRTC_MAC
                WEBRTC_THREAD_RR
                WEBRTC_CLOCK_TYPE_REALTIME
                WEBRTC_POSIX
            )
        elseif(UNIX)
            target_compile_definitions(${TARGET_AUDIO} PRIVATE
                WEBRTC_LINUX
                WEBRTC_THREAD_RR
                WEBRTC_POSIX
            )
        endif()

        target_include_directories(${TARGET_AUDIO} PRIVATE "${LLAMA_SERVER_AUDIO_WEBRTC_ROOT}")
        target_link_libraries(${TARGET_AUDIO} PRIVATE ${_llama_server_audio_webrtc_libs})

        if (WIN32)
            target_link_libraries(${TARGET_AUDIO} PRIVATE winmm)
        elseif(UNIX AND NOT APPLE)
            target_link_libraries(${TARGET_AUDIO} PRIVATE rt)
        endif()
    endif()

    install(TARGETS ${TARGET_AUDIO}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    )
endif()

add_executable(${TARGET_RT_SMOKE}
    llama_server_bridge_realtime_smoke.cpp
)

target_compile_features(${TARGET_RT_SMOKE} PRIVATE cxx_std_17)

target_include_directories(${TARGET_RT_SMOKE} PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/common
    ${CMAKE_SOURCE_DIR}/tools/server
)

target_link_libraries(${TARGET_RT_SMOKE}
    PRIVATE
        ${TARGET}
        llama-realtime-core
)

if (LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET_RT_SMOKE} RUNTIME)
endif()
