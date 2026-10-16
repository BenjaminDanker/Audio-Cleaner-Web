# Plugin bootstrap module

include_guard(GLOBAL)

# Map fallback configurations for optimized build configurations
# gersemi: off
set(
  CMAKE_MAP_IMPORTED_CONFIG_RELWITHDEBINFO
    RelWithDebInfo
    Release
    MinSizeRel
    None
    ""
)
set(
  CMAKE_MAP_IMPORTED_CONFIG_MINSIZEREL
    MinSizeRel
    Release
    RelWithDebInfo
    None
    ""
)
set(
  CMAKE_MAP_IMPORTED_CONFIG_RELEASE
    Release
    RelWithDebInfo
    MinSizeRel
    None
    ""
)
# gersemi: on

# Prohibit in-source builds
if("${CMAKE_CURRENT_BINARY_DIR}" STREQUAL "${CMAKE_CURRENT_SOURCE_DIR}")
  message(
    FATAL_ERROR
    "In-source builds are not supported. "
    "Specify a build directory via 'cmake -S <SOURCE DIRECTORY> -B <BUILD_DIRECTORY>' instead."
  )
  file(REMOVE_RECURSE "${CMAKE_CURRENT_SOURCE_DIR}/CMakeCache.txt" "${CMAKE_CURRENT_SOURCE_DIR}/CMakeFiles")
endif()

# Add common module directories to default search path
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/common")

file(READ "${CMAKE_CURRENT_SOURCE_DIR}/buildspec.json" buildspec)

string(JSON _name GET ${buildspec} name)
string(JSON _website GET ${buildspec} website)
string(JSON _author GET ${buildspec} author)
string(JSON _email GET ${buildspec} email)
string(JSON _version GET ${buildspec} version)
string(JSON _bundleId GET ${buildspec} platformConfig macos bundleId)

# CI builds are versioned MAJOR.MINOR.<run number> (cmake/ci-version.cmake): the number is handed in with
# -DACW_VERSION=... or the environment variable of that name, instead of being edited into buildspec.json.
# Everything below (the module's compiled version, resource files, the Info.plist, the package names) follows it.
if(NOT DEFINED ACW_VERSION AND DEFINED ENV{ACW_VERSION})
  set(ACW_VERSION "$ENV{ACW_VERSION}")
endif()
if(ACW_VERSION)
  if(NOT ACW_VERSION MATCHES "^[0-9]+\\.[0-9]+\\.[0-9]+$")
    message(FATAL_ERROR "ACW_VERSION must be MAJOR.MINOR.PATCH, got '${ACW_VERSION}'")
  endif()
  set(_version "${ACW_VERSION}")
endif()

set(PLUGIN_AUTHOR ${_author})
set(PLUGIN_WEBSITE ${_website})
set(PLUGIN_EMAIL ${_email})
set(PLUGIN_VERSION ${_version})
set(MACOS_BUNDLEID ${_bundleId})

string(REPLACE "." ";" _version_canonical "${_version}")
list(GET _version_canonical 0 PLUGIN_VERSION_MAJOR)
list(GET _version_canonical 1 PLUGIN_VERSION_MINOR)
list(GET _version_canonical 2 PLUGIN_VERSION_PATCH)
unset(_version_canonical)

include(buildnumber)
include(osconfig)

# Allow selection of common build types via UI
if(NOT CMAKE_GENERATOR MATCHES "(Xcode|Visual Studio .+)")
  if(NOT CMAKE_BUILD_TYPE)
    set(
      CMAKE_BUILD_TYPE
      "RelWithDebInfo"
      CACHE STRING
      "OBS build type [Release, RelWithDebInfo, Debug, MinSizeRel]"
      FORCE
    )
    set_property(
      CACHE CMAKE_BUILD_TYPE
      PROPERTY STRINGS Release RelWithDebInfo Debug MinSizeRel
    )
  endif()
endif()

# Disable exports automatically going into the CMake package registry
set(CMAKE_EXPORT_PACKAGE_REGISTRY FALSE)
# Enable default inclusion of targets' source and binary directory
set(CMAKE_INCLUDE_CURRENT_DIR TRUE)
