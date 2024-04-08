import ast
import contextlib
import io
from collections.abc import Mapping
from typing import Any, Protocol

import attrs

from agent.state import Node


class Context(Protocol):
    globals: dict[str, Any]
    locals: Mapping[str, Any]

    @classmethod
    def empty(cls):
        tmp = attrs.make_class("Context", ["globals", "locals"])
        return tmp(globals={}, locals={})


def execute(ctx: Context, node: Node):
    stdout = io.StringIO()
    stderr = io.StringIO()
    stmt = node.stmt

    with contextlib.redirect_stdout(stdout):
        with contextlib.redirect_stderr(stderr):
            try:
                if isinstance(stmt, ast.Expr):
                    node.value = eval_expr(ctx, stmt.value)
                else:
                    node.value = exec_stmt(ctx, stmt)
            except Exception as e:
                node.exc = e
            finally:
                node.stdout = stdout.getvalue()
                node.stderr = stderr.getvalue()


def exec_stmt(ctx: Context, stmt: ast.stmt):
    exec(ast.unparse(stmt), ctx.globals, ctx.locals)


def eval_expr(ctx: Context, expr: ast.expr):
    return eval(ast.unparse(expr), ctx.globals, ctx.locals)
