find_package(Threads REQUIRED)

if (NOT TARGET server-context)
    message(FATAL_ERROR "LLAMA markdown bridge requires LLAMA_BUILD_SERVER=ON")
endif()

if (NOT TARGET llama-realtime-core)
    message(FATAL_ERROR "LLAMA markdown bridge requires llama-realtime-core")
endif()

option(LLAMA_SERVER_BRIDGE_ENABLE_FFMPEG "Enable FFmpeg-based in-memory audio conversion for raw audio requests" OFF)
set(LLAMA_SERVER_BRIDGE_FFMPEG_ROOT "" CACHE PATH "Path to FFmpeg root containing include/ and lib/ for shared LGPL builds")

set(TARGET llama-server-bridge)
set(TARGET_RT_SMOKE llama-server-bridge-realtime-smoke)

add_library(${TARGET} SHARED
    llama_server_bridge.cpp
    llama_server_bridge.h
)

target_compile_features(${TARGET} PRIVATE cxx_std_17)

target_include_directories(${TARGET} PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/common
    ${CMAKE_SOURCE_DIR}/tools/server
)

target_compile_definitions(${TARGET} PRIVATE LLAMA_SERVER_BRIDGE_BUILD)

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
