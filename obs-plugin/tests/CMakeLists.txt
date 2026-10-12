# Tests that need no OBS: unit tests (offline) and the live streaming tool
# (talks to a real server).

add_executable(core_unit_test core_unit_test.cpp)
target_link_libraries(core_unit_test PRIVATE acw_core)
add_test(NAME core_unit_test COMMAND core_unit_test)

add_executable(core_stream_test core_stream_test.cpp)
target_link_libraries(core_stream_test PRIVATE acw_core)
