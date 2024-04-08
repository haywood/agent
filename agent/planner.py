import ast
import json
import logging
import re
import textwrap

from agent.state import Node, State

_PYTHON_RE = re.compile(r"```(python)?(?P<code>.*?)```", re.DOTALL)


def plan(state: State) -> ast.Module | None:
    prompt = format(state)
    logging.debug(f"Prompt:\n{textwrap.indent(prompt, '  ')}")

    output = state.llm(
        prompt,
        max_tokens=512,
        temperature=0.8,
        stop=[">>>"],
        repeat_penalty=1.0,
    )
    logging.debug(f"Output:\n{textwrap.indent(json.dumps(output, indent=2), '  ')}")
    choices = output.get("choices")
    if choices and (text := choices[0].get("text")):
        if match := _PYTHON_RE.search(text):
            text = match.group("code")
        else:
            text = text.removesuffix("```")
        try:
            return ast.parse(text.strip())
        except Exception as e:
            logging.exception(
                f"Unable to parse model output:\n{textwrap.indent(text, '  ')}"
            )
            return ast.parse("quit()")

    raise ValueError("Model produced no output")


def format(state: State) -> str:
    context = textwrap.indent(state.context.strip(), "# ")
    nodes = "\n".join(format_node(node) for node in state.nodes)
    if nodes and not nodes.endswith("\n"):
        nodes += "\n"

    return f"{state.prefix}{context}\n{nodes}>>> "


def format_node(node: Node) -> str:
    stmt = format_stmt(node.stmt)
    output = ""
    if node.exc:
        output = repr(node.exc)
    elif node.stdout:
        output = node.stdout
    elif node.value is not None:
        output = str(node.value)

    if output:
        return f"{stmt}\n{output}"
    else:
        return stmt


def format_stmt(stmt: ast.stmt) -> str:
    src = ast.unparse(stmt)

    return textwrap.indent(src, ">>> ")