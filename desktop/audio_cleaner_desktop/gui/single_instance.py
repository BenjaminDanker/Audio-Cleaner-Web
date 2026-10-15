"""Only one Audio Cleaner window at a time: starting it again (or choosing several files and "Open with")
hands the files to the window that is already open instead of opening a second one. The two talk over a local
socket that only the same user can reach."""
from __future__ import annotations

import getpass
import json
import re

from PySide6.QtCore import QObject, Signal
from PySide6.QtNetwork import QLocalServer, QLocalSocket

_MAX_MESSAGE = 4 << 20  # bytes; a list of paths, nothing more


def server_name() -> str:
    user = re.sub(r"[^A-Za-z0-9_.-]", "_", _user())
    return f"audio-cleaner-{user}"


def _user() -> str:
    try:
        return getpass.getuser()
    except Exception:  # noqa: BLE001 - no user name known: one shared name is still better than none
        return "user"


def forward(name: str, paths: list[str], timeout_ms: int = 1500) -> bool:
    """Send `paths` to the window that is already running. False when there is none."""
    socket = QLocalSocket()
    socket.connectToServer(name)
    if not socket.waitForConnected(timeout_ms):
        return False
    socket.write(json.dumps({"paths": paths}).encode("utf-8") + b"\n")
    socket.waitForBytesWritten(timeout_ms)
    socket.disconnectFromServer()
    if socket.state() != QLocalSocket.LocalSocketState.UnconnectedState:
        socket.waitForDisconnected(timeout_ms)
    return True


class InstanceServer(QObject):
    """Listens for the files other starts hand over. `received(paths)` also fires with an empty list when the
    app was started again without files (the window should simply come to the front)."""

    received = Signal(list)

    def __init__(self, name: str, parent: QObject | None = None):
        super().__init__(parent)
        self._name = name
        self._server = QLocalServer(self)
        self._server.setSocketOptions(QLocalServer.SocketOption.UserAccessOption)
        self._server.newConnection.connect(self._accept)
        self._buffers: dict[QLocalSocket, bytearray] = {}

    def listen(self) -> bool:
        QLocalServer.removeServer(self._name)  # a socket file a crashed run left behind
        return self._server.listen(self._name)

    def close(self) -> None:
        self._server.close()

    def _accept(self) -> None:
        while self._server.hasPendingConnections():
            socket = self._server.nextPendingConnection()
            self._buffers[socket] = bytearray()
            socket.readyRead.connect(lambda s=socket: self._read(s))
            socket.disconnected.connect(lambda s=socket: self._finish(s))

    def _read(self, socket: QLocalSocket) -> None:
        buffer = self._buffers.get(socket)
        if buffer is None:
            return
        buffer += bytes(socket.readAll())
        if len(buffer) > _MAX_MESSAGE:
            socket.abort()
        elif b"\n" in buffer:
            self._finish(socket)

    def _finish(self, socket: QLocalSocket) -> None:
        buffer = self._buffers.pop(socket, None)
        if buffer is None:
            return
        try:
            paths = [str(p) for p in json.loads(bytes(buffer).split(b"\n")[0])["paths"]]
        except (ValueError, KeyError, TypeError):
            paths = []
        socket.deleteLater()
        self.received.emit(paths)
