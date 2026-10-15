# Third-party code for acw_core. Everything is fetched as a release tarball and
# pinned by SHA-256 (no git history, no moving tags).
#
#   nlohmann/json   3.11.3   MIT          header only
#   IXWebSocket     11.4.6   BSD-3-Clause WebSocket client (we compile its sources ourselves)
#   mbedTLS         3.6.7    Apache-2.0 OR GPL-2.0-or-later   (TLS backend on Windows / fallback)
#   libopus         1.5.2    BSD-3-Clause 20 ms audio packets, ~60 kbps instead of ~770 kbps PCM
#   OpenSSL         system   Apache-2.0   (TLS backend on Linux when its headers are installed)
#   Security.framework (macOS SecureTransport, via IXWebSocket)

include_guard(GLOBAL)
include(FetchContent)

# Some of the dependencies still declare cmake_minimum_required(< 3.5).
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
set(FETCHCONTENT_QUIET ON)

# Third-party sources are not ours to fix: never turn their warnings into errors.
set(_acw_saved_werror "${CMAKE_COMPILE_WARNING_AS_ERROR}")
set(CMAKE_COMPILE_WARNING_AS_ERROR OFF)

# ---------------------------------------------------------------- nlohmann/json
FetchContent_Declare(
  acw_json
  URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
  URL_HASH SHA256=d6c65aca6b1ed68e7a182f4757257b107ae403032760ed6ef121c9d55e81757d
  SOURCE_SUBDIR acw_no_cmake # header only: do not add_subdirectory
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(acw_json)
add_library(acw_json INTERFACE)
target_include_directories(acw_json SYSTEM INTERFACE "${acw_json_SOURCE_DIR}/include")

# ----------------------------------------------------------------- TLS backend
set(
  ACW_TLS_BACKEND
  "auto"
  CACHE STRING
  "TLS backend for the WebSocket client: auto, openssl, mbedtls, securetransport"
)
set_property(CACHE ACW_TLS_BACKEND PROPERTY STRINGS auto openssl mbedtls securetransport)

set(_acw_tls "${ACW_TLS_BACKEND}")
if(_acw_tls STREQUAL "auto")
  if(APPLE)
    set(_acw_tls securetransport) # system trust store, no extra dependency
  elseif(WIN32)
    set(_acw_tls mbedtls) # reads the Windows root store (IXWebSocket), static, no DLL
  else()
    find_package(OpenSSL QUIET)
    if(OPENSSL_FOUND)
      set(_acw_tls openssl) # system OpenSSL when its headers are installed
    else()
      set(_acw_tls mbedtls) # otherwise build mbedTLS and use the distribution CA bundle
    endif()
  endif()
endif()
message(STATUS "acw_core: TLS backend = ${_acw_tls}")

if(_acw_tls STREQUAL "mbedtls")
  set(ENABLE_PROGRAMS OFF CACHE BOOL "" FORCE)
  set(ENABLE_TESTING OFF CACHE BOOL "" FORCE)
  set(GEN_FILES OFF CACHE BOOL "" FORCE)
  set(MBEDTLS_FATAL_WARNINGS OFF CACHE BOOL "" FORCE)
  set(USE_SHARED_MBEDTLS_LIBRARY OFF CACHE BOOL "" FORCE)
  set(USE_STATIC_MBEDTLS_LIBRARY ON CACHE BOOL "" FORCE)
  set(DISABLE_PACKAGE_CONFIG_AND_INSTALL ON CACHE BOOL "" FORCE)
  set(INSTALL_MBEDTLS_HEADERS OFF CACHE BOOL "" FORCE)
  set(_acw_saved_shared "${BUILD_SHARED_LIBS}")
  set(_acw_saved_pic "${CMAKE_POSITION_INDEPENDENT_CODE}")
  set(BUILD_SHARED_LIBS OFF)
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)
  FetchContent_Declare(
    acw_mbedtls
    URL https://github.com/Mbed-TLS/mbedtls/releases/download/mbedtls-3.6.7/mbedtls-3.6.7.tar.bz2
    URL_HASH SHA256=a7e8bcbec0e6f761b4af24f25677626b35f762f68eef79c08677a363212d11f6
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    # Linked statically and privately: never part of `cmake --install` / the packages (needs CMake >= 3.28).
    EXCLUDE_FROM_ALL
  )
  FetchContent_MakeAvailable(acw_mbedtls)
  set(BUILD_SHARED_LIBS "${_acw_saved_shared}")
  set(CMAKE_POSITION_INDEPENDENT_CODE "${_acw_saved_pic}")
elseif(_acw_tls STREQUAL "openssl")
  find_package(OpenSSL REQUIRED)
endif()

# ----------------------------------------------------------------- IXWebSocket
FetchContent_Declare(
  acw_ixwebsocket_src
  URL https://github.com/machinezone/IXWebSocket/archive/refs/tags/v11.4.6.tar.gz
  URL_HASH SHA256=c024334f8e45980836c67008979a884d6dcc5ef067dd2eb1fa7241f4c17ddc32
  SOURCE_SUBDIR acw_no_cmake # we build the sources ourselves
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(acw_ixwebsocket_src)

set(_ix "${acw_ixwebsocket_src_SOURCE_DIR}/ixwebsocket")
set(
  _ix_sources
  IXBench IXCancellationRequest IXConnectionState IXDNSLookup IXExponentialBackoff IXGetFreePort IXGzipCodec
  IXHttp IXHttpClient IXHttpServer IXNetSystem IXSelectInterrupt IXSelectInterruptFactory IXSelectInterruptPipe
  IXSelectInterruptEvent IXSetThreadName IXSocket IXSocketConnect IXSocketFactory IXSocketServer
  IXSocketTLSOptions IXStrCaseCompare IXUdpSocket IXUrlParser IXUuid IXUserAgent IXWebSocket
  IXWebSocketCloseConstants IXWebSocketHandshake IXWebSocketHttpHeaders IXWebSocketPerMessageDeflate
  IXWebSocketPerMessageDeflateCodec IXWebSocketPerMessageDeflateOptions IXWebSocketProxyServer
  IXWebSocketServer IXWebSocketTransport
)
set(_ix_files)
foreach(_s IN LISTS _ix_sources)
  list(APPEND _ix_files "${_ix}/${_s}.cpp")
endforeach()

add_library(acw_ixwebsocket STATIC ${_ix_files})
target_include_directories(acw_ixwebsocket SYSTEM PUBLIC "${acw_ixwebsocket_src_SOURCE_DIR}")
set_target_properties(
  acw_ixwebsocket
  PROPERTIES POSITION_INDEPENDENT_CODE ON CXX_STANDARD 17 CXX_VISIBILITY_PRESET hidden
)
if(MSVC)
  target_compile_options(acw_ixwebsocket PRIVATE /W0)
else()
  target_compile_options(acw_ixwebsocket PRIVATE -w)
endif()
# No permessage-deflate (IXWEBSOCKET_USE_ZLIB stays undefined): audio does not compress.

if(_acw_tls STREQUAL "openssl")
  target_sources(acw_ixwebsocket PRIVATE "${_ix}/IXSocketOpenSSL.cpp")
  target_compile_definitions(acw_ixwebsocket PUBLIC IXWEBSOCKET_USE_TLS IXWEBSOCKET_USE_OPEN_SSL)
  target_link_libraries(acw_ixwebsocket PUBLIC OpenSSL::SSL OpenSSL::Crypto)
elseif(_acw_tls STREQUAL "mbedtls")
  target_sources(acw_ixwebsocket PRIVATE "${_ix}/IXSocketMbedTLS.cpp")
  target_compile_definitions(
    acw_ixwebsocket
    PUBLIC IXWEBSOCKET_USE_TLS IXWEBSOCKET_USE_MBED_TLS IXWEBSOCKET_USE_MBED_TLS_MIN_VERSION_3 ACW_TLS_MBEDTLS
  )
  target_link_libraries(acw_ixwebsocket PUBLIC mbedtls mbedx509 mbedcrypto)
elseif(_acw_tls STREQUAL "securetransport")
  target_sources(acw_ixwebsocket PRIVATE "${_ix}/IXSocketAppleSSL.cpp")
  target_compile_definitions(acw_ixwebsocket PUBLIC IXWEBSOCKET_USE_TLS IXWEBSOCKET_USE_SECURE_TRANSPORT)
  target_link_libraries(acw_ixwebsocket PUBLIC "-framework Foundation" "-framework Security")
else()
  message(FATAL_ERROR "acw_core: unknown ACW_TLS_BACKEND '${_acw_tls}'")
endif()
target_compile_definitions(acw_ixwebsocket PUBLIC ACW_TLS_BACKEND_NAME="${_acw_tls}")

if(WIN32)
  target_link_libraries(acw_ixwebsocket PUBLIC wsock32 ws2_32 shlwapi Crypt32)
  target_compile_definitions(acw_ixwebsocket PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()
set(THREADS_PREFER_PTHREAD_FLAG TRUE)
find_package(Threads REQUIRED)
target_link_libraries(acw_ixwebsocket PUBLIC Threads::Threads)

# ----------------------------------------------------------------------- libopus
option(ACW_USE_OPUS "Build libopus and offer the Opus codec (default on the wire)" ON)
if(ACW_USE_OPUS)
  set(OPUS_BUILD_PROGRAMS OFF CACHE BOOL "" FORCE)
  set(OPUS_BUILD_TESTING OFF CACHE BOOL "" FORCE)
  set(OPUS_BUILD_SHARED_LIBRARY OFF CACHE BOOL "" FORCE)
  set(OPUS_INSTALL_PKG_CONFIG_MODULE OFF CACHE BOOL "" FORCE)
  set(OPUS_INSTALL_CMAKE_CONFIG_MODULE OFF CACHE BOOL "" FORCE)
  set(OPUS_DRED OFF CACHE BOOL "" FORCE)
  set(OPUS_OSCE OFF CACHE BOOL "" FORCE)
  if(APPLE)
    # Universal (arm64;x86_64) builds cannot probe SIMD per architecture; encoding 50 packets/s does not need it.
    set(OPUS_DISABLE_INTRINSICS ON CACHE BOOL "" FORCE)
  endif()
  set(_acw_saved_shared_opus "${BUILD_SHARED_LIBS}")
  set(_acw_saved_pic_opus "${CMAKE_POSITION_INDEPENDENT_CODE}")
  set(BUILD_SHARED_LIBS OFF)
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)
  FetchContent_Declare(
    acw_opus
    URL https://github.com/xiph/opus/releases/download/v1.5.2/opus-1.5.2.tar.gz
    URL_HASH SHA256=65c1d2f78b9f2fb20082c38cbe47c951ad5839345876e46941612ee87f9a7ce1
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    # Linked statically and privately: its install rules (libopus.a, include/opus) must not end up in the
    # .deb / zip, where they would clash with the system's libopus-dev (needs CMake >= 3.28).
    EXCLUDE_FROM_ALL
  )
  FetchContent_MakeAvailable(acw_opus)
  set(BUILD_SHARED_LIBS "${_acw_saved_shared_opus}")
  set(CMAKE_POSITION_INDEPENDENT_CODE "${_acw_saved_pic_opus}")
endif()

# Keep third-party warnings out of our build output.
foreach(_t IN ITEMS opus mbedtls mbedx509 mbedcrypto everest p256m)
  if(TARGET ${_t})
    if(MSVC)
      target_compile_options(${_t} PRIVATE /W0)
    else()
      target_compile_options(${_t} PRIVATE -w)
    endif()
  endif()
endforeach()

set(CMAKE_COMPILE_WARNING_AS_ERROR "${_acw_saved_werror}")
