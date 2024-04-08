import ast
import contextlib
import io

from agent.state import Node, State


def execute(state: State, node: Node):
    stdout = io.StringIO()
    stderr = io.StringIO()
    stmt = node.stmt

    with contextlib.redirect_stdout(stdout):
        with contextlib.redirect_stderr(stderr):
            try:
                if isinstance(stmt, ast.Expr):
                    node.value = eval_expr(state, stmt.value)
                else:
                    node.value = exec_stmt(state, stmt)
            except Exception as e:
                node.exc = e
            finally:
                node.stdout = stdout.getvalue()
                node.stderr = stderr.getvalue()


def exec_stmt(state: State, stmt: ast.stmt):
    exec(ast.unparse(stmt), state.globals, state.locals)


def eval_expr(state: State, expr: ast.expr):
    return eval(ast.unparse(expr), state.globals, state.locals)
