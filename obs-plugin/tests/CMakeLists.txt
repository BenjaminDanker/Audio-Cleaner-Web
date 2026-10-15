# Tests that need no OBS: unit tests (offline), threaded loopback tests (a fake server on 127.0.0.1) and the
# live streaming tool (talks to a real server).

add_executable(core_unit_test core_unit_test.cpp)
target_link_libraries(core_unit_test PRIVATE acw_core)
add_test(NAME core_unit_test COMMAND core_unit_test)

# The real Session against an IXWebSocket server standing in for the service: handshake, errors, retries, late
# replies, garbage input, shutdown. Also run under the sanitizers in CI.
add_executable(core_loopback_test core_loopback_test.cpp)
target_link_libraries(core_loopback_test PRIVATE acw_core acw_ixwebsocket)
add_test(NAME core_loopback_test COMMAND core_loopback_test)
set_tests_properties(core_loopback_test PROPERTIES TIMEOUT 300)

add_executable(core_stream_test core_stream_test.cpp)
target_link_libraries(core_stream_test PRIVATE acw_core)
