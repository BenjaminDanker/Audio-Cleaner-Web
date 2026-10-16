cmake_minimum_required(VERSION 3.28...3.30)

include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/common/bootstrap.cmake" NO_POLICY_SCOPE)

project(${_name} VERSION ${_version})

option(ACW_BUILD_PLUGIN "Build the OBS Studio module (needs libobs)" ON)
option(ACW_BUILD_TESTS "Build the acw_core unit tests and the live streaming test tool (no OBS needed)" OFF)
option(ENABLE_FRONTEND_API "Use obs-frontend-api (needed to send captions to the stream)" ON)

if(OS_MACOS AND NOT ACW_BUILD_PLUGIN)
  # The template's macOS rules insist on the Xcode generator (they set up the plugin bundle). The core library and
  # its tests need none of that: they build with any generator, and only get the shared compiler options.
  include(compiler_common)
else()
  include(compilerconfig)
endif()

if(ACW_BUILD_PLUGIN)
  include(defaults)
  include(helpers)
endif()

# OBS-independent core: ring buffers, reframer, resampler, protocol, WebSocket client, output stage.
add_subdirectory(core)

if(ACW_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

if(NOT ACW_BUILD_PLUGIN)
  return()
endif()

add_library(${CMAKE_PROJECT_NAME} MODULE)

find_package(libobs REQUIRED)
target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE OBS::libobs)

if(ENABLE_FRONTEND_API)
  find_package(obs-frontend-api REQUIRED)
  target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE OBS::obs-frontend-api)
  target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE ACW_HAVE_FRONTEND_API)
endif()

target_sources(
  ${CMAKE_PROJECT_NAME}
  PRIVATE
    src/plugin-main.c
    src/audio-cleaner-filter.cpp
    src/caption-output.cpp
    src/caption-output.h
    src/update-notice.cpp
    src/update-notice.h
)
target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE acw_core)
set_target_properties(${CMAKE_PROJECT_NAME} PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)

if(OS_WINDOWS)
  target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE shell32) # ShellExecute: opens the release page in the browser
endif()

if(OS_LINUX)
  # Keep the statically linked third-party code private to the module so it can
  # never bind to (or override) another copy loaded by OBS.
  target_link_options(${CMAKE_PROJECT_NAME} PRIVATE -Wl,--exclude-libs,ALL -Wl,-Bsymbolic)
endif()

set_target_properties_plugin(${CMAKE_PROJECT_NAME} PROPERTIES OUTPUT_NAME ${_name})

# Licence texts travel with every binary package (GPL and BSD notices must be passed on).
target_add_resource(${CMAKE_PROJECT_NAME} "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE")
target_add_resource(${CMAKE_PROJECT_NAME} "${CMAKE_CURRENT_SOURCE_DIR}/THIRD_PARTY_LICENSES.md")

if(ACW_BUILD_TESTS AND UNIX)
  # Headless libobs host that loads the module and drives the filter (needs a running server).
  add_executable(obs_smoke EXCLUDE_FROM_ALL tests/obs_smoke.c)
  target_link_libraries(obs_smoke PRIVATE OBS::libobs m)
endif()
