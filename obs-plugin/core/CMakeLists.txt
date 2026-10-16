# acw_core: OBS-independent streaming core (ring buffers, reframer, resampler,
# protocol v1, WebSocket client, fixed-delay output stage).

include(cmake/third_party.cmake)

add_library(
  acw_core
  STATIC
  src/caption_pacer.cpp
  src/opus_codec.cpp
  src/output_stage.cpp
  src/protocol.cpp
  src/reference.cpp
  src/resampler.cpp
  src/session.cpp
  src/update_check.cpp
)
target_include_directories(acw_core PUBLIC include)
target_compile_features(acw_core PUBLIC cxx_std_17)
set_target_properties(acw_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(acw_core PRIVATE acw_ixwebsocket acw_json)
if(ACW_USE_OPUS)
  target_link_libraries(acw_core PRIVATE opus)
  target_compile_definitions(acw_core PRIVATE ACW_HAVE_OPUS)
endif()

if(WIN32)
  target_compile_definitions(acw_core PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
endif()
