from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class DType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    DTYPE_UNSPECIFIED: _ClassVar[DType]
    FLOAT32: _ClassVar[DType]
    FLOAT16: _ClassVar[DType]
    BFLOAT16: _ClassVar[DType]
    INT64: _ClassVar[DType]
DTYPE_UNSPECIFIED: DType
FLOAT32: DType
FLOAT16: DType
BFLOAT16: DType
INT64: DType

class Tensor(_message.Message):
    __slots__ = ("shape", "dtype", "data")
    SHAPE_FIELD_NUMBER: _ClassVar[int]
    DTYPE_FIELD_NUMBER: _ClassVar[int]
    DATA_FIELD_NUMBER: _ClassVar[int]
    shape: _containers.RepeatedScalarFieldContainer[int]
    dtype: DType
    data: bytes
    def __init__(self, shape: _Optional[_Iterable[int]] = ..., dtype: _Optional[_Union[DType, str]] = ..., data: _Optional[bytes] = ...) -> None: ...

class ForwardRequest(_message.Message):
    __slots__ = ("request_id", "step_id", "input", "start_pos")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    STEP_ID_FIELD_NUMBER: _ClassVar[int]
    INPUT_FIELD_NUMBER: _ClassVar[int]
    START_POS_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    step_id: int
    input: Tensor
    start_pos: int
    def __init__(self, request_id: _Optional[str] = ..., step_id: _Optional[int] = ..., input: _Optional[_Union[Tensor, _Mapping]] = ..., start_pos: _Optional[int] = ...) -> None: ...

class ForwardReply(_message.Message):
    __slots__ = ("request_id", "step_id", "output", "next_token", "compute_ms", "cache_len")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    STEP_ID_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_FIELD_NUMBER: _ClassVar[int]
    NEXT_TOKEN_FIELD_NUMBER: _ClassVar[int]
    COMPUTE_MS_FIELD_NUMBER: _ClassVar[int]
    CACHE_LEN_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    step_id: int
    output: Tensor
    next_token: int
    compute_ms: float
    cache_len: int
    def __init__(self, request_id: _Optional[str] = ..., step_id: _Optional[int] = ..., output: _Optional[_Union[Tensor, _Mapping]] = ..., next_token: _Optional[int] = ..., compute_ms: _Optional[float] = ..., cache_len: _Optional[int] = ...) -> None: ...

class InfoRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class InfoReply(_message.Message):
    __slots__ = ("worker_id", "first_layer", "last_layer", "has_embed", "has_head", "pid", "param_bytes", "cached_requests", "kv_bytes")
    WORKER_ID_FIELD_NUMBER: _ClassVar[int]
    FIRST_LAYER_FIELD_NUMBER: _ClassVar[int]
    LAST_LAYER_FIELD_NUMBER: _ClassVar[int]
    HAS_EMBED_FIELD_NUMBER: _ClassVar[int]
    HAS_HEAD_FIELD_NUMBER: _ClassVar[int]
    PID_FIELD_NUMBER: _ClassVar[int]
    PARAM_BYTES_FIELD_NUMBER: _ClassVar[int]
    CACHED_REQUESTS_FIELD_NUMBER: _ClassVar[int]
    KV_BYTES_FIELD_NUMBER: _ClassVar[int]
    worker_id: str
    first_layer: int
    last_layer: int
    has_embed: bool
    has_head: bool
    pid: int
    param_bytes: int
    cached_requests: int
    kv_bytes: int
    def __init__(self, worker_id: _Optional[str] = ..., first_layer: _Optional[int] = ..., last_layer: _Optional[int] = ..., has_embed: _Optional[bool] = ..., has_head: _Optional[bool] = ..., pid: _Optional[int] = ..., param_bytes: _Optional[int] = ..., cached_requests: _Optional[int] = ..., kv_bytes: _Optional[int] = ...) -> None: ...

class ReleaseRequest(_message.Message):
    __slots__ = ("request_id",)
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    def __init__(self, request_id: _Optional[str] = ...) -> None: ...

class ReleaseReply(_message.Message):
    __slots__ = ("existed", "freed_bytes")
    EXISTED_FIELD_NUMBER: _ClassVar[int]
    FREED_BYTES_FIELD_NUMBER: _ClassVar[int]
    existed: bool
    freed_bytes: int
    def __init__(self, existed: _Optional[bool] = ..., freed_bytes: _Optional[int] = ...) -> None: ...
