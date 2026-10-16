# The version of this CI build, in the same way as the desktop app (desktop/packaging/version.py):
#
#   a branch build   MAJOR.MINOR from buildspec.json, the CI run number as the last part (0.1.57), so every build
#                    has its own, always increasing version (dpkg and the installers compare them)
#   a pushed tag     obs-plugin-vX.Y.Z builds exactly X.Y.Z; X.Y must match buildspec.json
#   anywhere else    the version in buildspec.json
#
# Usage:  cmake -P cmake/ci-version.cmake
# It prints the version and, when GITHUB_ENV is set, appends ACW_VERSION=<version> there: the later steps of the job
# see it as an environment variable, and CMake (cmake/common/bootstrap.cmake) builds the module with it.
cmake_minimum_required(VERSION 3.28)

file(READ "${CMAKE_CURRENT_LIST_DIR}/../buildspec.json" _spec)
string(JSON _base GET "${_spec}" version)
set(_num "(0|[1-9][0-9]*)")
if(NOT _base MATCHES "^${_num}\\.${_num}\\.${_num}$")
  message(FATAL_ERROR "buildspec.json: version '${_base}' is not MAJOR.MINOR.PATCH")
endif()
set(_major "${CMAKE_MATCH_1}")
set(_minor "${CMAKE_MATCH_2}")
set(_base_patch "${CMAKE_MATCH_3}")

if("$ENV{GITHUB_REF_TYPE}" STREQUAL "tag")
  set(_tag "$ENV{GITHUB_REF_NAME}")
  if(NOT _tag MATCHES "^obs-plugin-v${_num}\\.${_num}\\.${_num}$")
    message(FATAL_ERROR "tag '${_tag}' is not obs-plugin-vX.Y.Z")
  endif()
  set(_tag_major "${CMAKE_MATCH_1}")
  set(_tag_minor "${CMAKE_MATCH_2}")
  set(_patch "${CMAKE_MATCH_3}")
  if(NOT _tag_major EQUAL _major OR NOT _tag_minor EQUAL _minor)
    message(
      FATAL_ERROR
      "tag ${_tag} does not match the plugin's version ${_major}.${_minor} (obs-plugin/buildspec.json)"
    )
  endif()
  set(_version "${_tag_major}.${_tag_minor}.${_patch}")
elseif(NOT "$ENV{GITHUB_RUN_NUMBER}" STREQUAL "")
  if(NOT "$ENV{GITHUB_RUN_NUMBER}" MATCHES "^[1-9][0-9]*$")
    message(FATAL_ERROR "GITHUB_RUN_NUMBER '$ENV{GITHUB_RUN_NUMBER}' is not a number")
  endif()
  set(_patch "$ENV{GITHUB_RUN_NUMBER}")
  set(_version "${_major}.${_minor}.${_patch}")
else()
  set(_patch "${_base_patch}")
  set(_version "${_base}")
endif()

# The Windows resource file stores each part in 16 bits.
if(_patch GREATER 65535 OR _major GREATER 65535 OR _minor GREATER 65535)
  message(FATAL_ERROR "version ${_version}: a part is larger than 65535 (the Windows file version cannot hold it)")
endif()

execute_process(COMMAND "${CMAKE_COMMAND}" -E echo "OBS plugin version ${_version}")
if(NOT "$ENV{GITHUB_ENV}" STREQUAL "")
  file(APPEND "$ENV{GITHUB_ENV}" "ACW_VERSION=${_version}\n")
endif()
