import ast

from collections.abc import Mapping, Sequence
from typing import Any, MutableSequence

import attrs
from llama_cpp import Llama


@attrs.define
class Node:
    stmt: ast.stmt = attrs.field()
    value: Any = attrs.field(default=None)
    exc: Exception | None = attrs.field(default=None)
    stdout: str = attrs.field(default="")
    stderr: str = attrs.field(default="")


@attrs.define
class State:
    context: str = attrs.field()
    llm: Llama = attrs.field()
    prefix: str = attrs.field(default="")
    nodes: MutableSequence[Node] = attrs.field(factory=list)
    budget: int = attrs.field(default=10)
    cost: int = attrs.field(default=0)
    globals: dict[str, Any] = attrs.field(factory=lambda: {"__name__": "__main__"})
    locals: Mapping[str, Any] = attrs.field(factory=dict)
