"""
framing.py - Chapter 5
TCP delivers a stream of bytes, not messages. This adds message boundaries:
each message = [4-byte header length][JSON header][4-byte payload length][payload bytes].
"""
import json
import socket
import struct

LEN = struct.Struct("!I")   # unsigned 32-bit int, network byte order (big-endian): 4 bytes


def recv_exact(sock: socket.socket, n: int) -> bytes:
    """Read exactly n bytes. recv() may return fewer, so loop until we have them all."""
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:                        # b"" means the other side closed the connection
            raise ConnectionError("peer closed the connection")
        buf += chunk
    return bytes(buf)


def send_msg(sock: socket.socket, header: dict, payload: bytes = b"") -> None:
    h = json.dumps(header).encode("utf-8")
    sock.sendall(LEN.pack(len(h)) + h + LEN.pack(len(payload)) + payload)


def recv_msg(sock: socket.socket) -> tuple[dict, bytes]:
    (h_len,) = LEN.unpack(recv_exact(sock, LEN.size))
    header = json.loads(recv_exact(sock, h_len))
    (p_len,) = LEN.unpack(recv_exact(sock, LEN.size))
    return header, recv_exact(sock, p_len)
