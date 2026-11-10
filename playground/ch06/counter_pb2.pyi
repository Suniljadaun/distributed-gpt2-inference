from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class EchoRequest(_message.Message):
    __slots__ = ("payload",)
    PAYLOAD_FIELD_NUMBER: _ClassVar[int]
    payload: bytes
    def __init__(self, payload: _Optional[bytes] = ...) -> None: ...

class EchoReply(_message.Message):
    __slots__ = ("payload",)
    PAYLOAD_FIELD_NUMBER: _ClassVar[int]
    payload: bytes
    def __init__(self, payload: _Optional[bytes] = ...) -> None: ...

class AddRequest(_message.Message):
    __slots__ = ("request_id", "amount", "delay_s")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    AMOUNT_FIELD_NUMBER: _ClassVar[int]
    DELAY_S_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    amount: int
    delay_s: float
    def __init__(self, request_id: _Optional[str] = ..., amount: _Optional[int] = ..., delay_s: _Optional[float] = ...) -> None: ...

class GetRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class CounterReply(_message.Message):
    __slots__ = ("counter",)
    COUNTER_FIELD_NUMBER: _ClassVar[int]
    counter: int
    def __init__(self, counter: _Optional[int] = ...) -> None: ...

class CountUpRequest(_message.Message):
    __slots__ = ("n", "interval_s")
    N_FIELD_NUMBER: _ClassVar[int]
    INTERVAL_S_FIELD_NUMBER: _ClassVar[int]
    n: int
    interval_s: float
    def __init__(self, n: _Optional[int] = ..., interval_s: _Optional[float] = ...) -> None: ...

class Tick(_message.Message):
    __slots__ = ("i",)
    I_FIELD_NUMBER: _ClassVar[int]
    i: int
    def __init__(self, i: _Optional[int] = ...) -> None: ...
