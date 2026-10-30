"""acw.capacity: the client of the host capacity service, against a small fake server on a private abstract socket."""
import json
import socket
import threading
import uuid

import pytest

from acw import capacity


class Fake:
    def __init__(self, reply: bytes):
        self.name = "\0acw-test-" + uuid.uuid4().hex
        self.reply = reply
        self.requests: list[bytes] = []
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.bind(self.name)
        self.sock.listen(4)
        self.thread = threading.Thread(target=self._serve, daemon=True)
        self.thread.start()

    def _serve(self):
        while True:
            try:
                conn, _ = self.sock.accept()
            except OSError:
                return
            with conn:
                data = b""
                while b"\r\n\r\n" not in data:
                    chunk = conn.recv(65536)
                    if not chunk:
                        break
                    data += chunk
                head, _, body = data.partition(b"\r\n\r\n")
                length = 0
                for line in head.split(b"\r\n")[1:]:
                    k, _, v = line.partition(b":")
                    if k.strip().lower() == b"content-length":
                        length = int(v)
                while len(body) < length:
                    body += conn.recv(65536)
                self.requests.append(head + b"\r\n\r\n" + body)
                conn.sendall(self.reply)

    def close(self):
        self.sock.close()


def http(body, status="200 OK") -> bytes:
    payload = body if isinstance(body, bytes) else json.dumps(body).encode()
    return f"HTTP/1.0 {status}\r\nContent-Type: application/json\r\n\r\n".encode() + payload


@pytest.fixture
def serve(monkeypatch):
    started: list[Fake] = []

    def start(reply: bytes) -> Fake:
        fake = Fake(reply)
        started.append(fake)
        monkeypatch.setattr(capacity, "SOCKET", fake.name)
        return fake

    yield start
    for f in started:
        f.close()


@pytest.fixture
def reserve_job():
    return capacity.reserve_job.real  # the conftest stub hides the real function from every other test


def test_a_granted_reply(serve, reserve_job):
    fake = serve(http({"granted": True}))
    assert reserve_job("job-7", 3 * 1024**3) == (True, "")
    request = fake.requests[0]
    head, _, body = request.partition(b"\r\n\r\n")
    assert head.startswith(b"POST /reserve HTTP/1.0\r\n")
    assert json.loads(body) == {"kind": "acw_job", "name": "job-7", "bytes": 3 * 1024**3}


def test_a_refused_reply_carries_the_reason(serve, reserve_job):
    serve(http({"granted": False, "reason": "4.2 GB free, job needs 6.0 GB"}))
    assert reserve_job("j", 6) == (False, "4.2 GB free, job needs 6.0 GB")


def test_a_refusal_without_a_reason_has_an_empty_one(serve, reserve_job):
    serve(http({"granted": False}))
    assert reserve_job("j", 6) == (False, "")


@pytest.mark.parametrize("reply", [
    http({"granted": True}, "500 Internal Server Error"),
    http({"granted": True}, "404 Not Found"),
    http(b"this is not json"),
    http(b""),
    http(b"[1, 2]"),  # JSON, but not an object
    http({"unrelated": 1}),  # an object without "granted"
    b"garbage with no http head at all",
    b"",
])
def test_anything_else_is_no_answer(serve, reserve_job, reply):
    serve(reply)
    assert reserve_job("j", 1) is None


def test_nothing_listening_is_no_answer(monkeypatch, reserve_job):
    monkeypatch.setattr(capacity, "SOCKET", "\0acw-test-" + uuid.uuid4().hex)
    assert reserve_job("j", 1) is None
    assert capacity._post("/reserve", {}) is None


def test_post_returns_the_json_object(serve):
    serve(http({"granted": True, "extra": 1}))
    assert capacity._post("/reserve", {"a": 1}) == {"granted": True, "extra": 1}


def test_the_job_id_is_sent_as_text_and_the_size_as_an_integer(serve, reserve_job):
    fake = serve(http({"granted": True}))
    reserve_job(42, 1.9e9)
    assert json.loads(fake.requests[0].partition(b"\r\n\r\n")[2]) == {"kind": "acw_job", "name": "42", "bytes": 1900000000}


def test_a_server_that_hangs_up_without_a_reply_is_no_answer(serve, reserve_job):
    serve(b"")
    assert reserve_job("j", 1) is None
