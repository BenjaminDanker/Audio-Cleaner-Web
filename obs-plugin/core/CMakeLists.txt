# acw_core: OBS-independent streaming core (ring buffers, reframer, resampler,
# protocol v1, WebSocket client, fixed-delay output stage).

include(cmake/third_party.cmake)

add_library(
  acw_core
  STATIC
  src/caption_pacer.cpp
  src/caption_router.cpp
  src/delayed_sender.cpp
  src/languages.cpp
  src/opus_codec.cpp
  src/output_select.cpp
  src/overlay_text.cpp
  src/output_stage.cpp
  src/protocol.cpp
  src/reference.cpp
  src/resampler.cpp
  src/session.cpp
  src/subtitles.cpp
  src/http_get.cpp
  src/update_check.cpp
  src/update_download.cpp
  src/update_install.cpp
  src/update_verify.cpp
)
target_include_directories(acw_core PUBLIC include)
target_compile_features(acw_core PUBLIC cxx_std_17)
set_target_properties(acw_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(acw_core PRIVATE acw_ixwebsocket acw_json acw_monocypher)
if(ACW_USE_OPUS)
  target_link_libraries(acw_core PRIVATE opus)
  target_compile_definitions(acw_core PRIVATE ACW_HAVE_OPUS)
endif()

if(WIN32)
  target_compile_definitions(acw_core PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
  target_link_libraries(acw_core PUBLIC shell32) # ShellExecute: the elevated start of the update script
endif()
