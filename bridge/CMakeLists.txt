find_package(Threads REQUIRED)

if (NOT TARGET server-context)
    message(FATAL_ERROR "LLAMA markdown bridge requires LLAMA_BUILD_SERVER=ON")
endif()

option(LLAMA_SERVER_BRIDGE_ENABLE_FFMPEG "Enable FFmpeg-based in-memory audio conversion for raw audio requests" OFF)
set(LLAMA_SERVER_BRIDGE_FFMPEG_ROOT "" CACHE PATH "Path to FFmpeg root containing include/ and lib/ for shared LGPL builds")

set(TARGET llama-server-bridge)

add_library(${TARGET} SHARED
    llama_server_bridge.cpp
    llama_server_bridge.h
)

target_compile_features(${TARGET} PRIVATE cxx_std_17)

target_include_directories(${TARGET} PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/common
    ${CMAKE_SOURCE_DIR}/tools/server
)

target_compile_definitions(${TARGET} PRIVATE LLAMA_SERVER_BRIDGE_BUILD)

target_link_libraries(${TARGET}
    PRIVATE
        server-context
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
