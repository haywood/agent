import ast
import json
import logging
import re
import textwrap
import traceback

from agent.state import Node, State

_PYTHON_RE = re.compile(r"```(python)?(?P<code>.*?)```", re.DOTALL)


def plan(state: State) -> ast.Module | None:
    prompt = format(state)
    logging.debug(f"Prompt:\n{textwrap.indent(prompt, '  ')}")

    output = state.llm(
        prompt,
        max_tokens=512,
        temperature=0.8,
        stop=["```"],
        repeat_penalty=1.0,
    )
    logging.debug(f"Output:\n{textwrap.indent(json.dumps(output, indent=2), '  ')}")
    choices = output.get("choices")
    if choices and (text := choices[0].get("text")):
        try:
            return ast.parse(text.strip(), filename="main")
        except SyntaxError as e:
            logging.exception(
                f"Unable to parse model output:\n{textwrap.indent(text, '  ')}"
            )
            stdout = traceback.format_exc(limit=0)
            node = Node(src=text, stmt=None, stdout=stdout)
            state.nodes.append(node)
            return ast.Module(body=[], type_ignores=[])

    raise ValueError("Model produced no output")


def format(state: State) -> str:
    context = textwrap.indent(state.context.strip(), "# ")
    nodes = "\n".join(format_node(node) for node in state.nodes)
    if nodes and not nodes.endswith("\n"):
        nodes += "\n"

    return f"{state.prefix}<start_of_turn>user\n{context}<end_of_turn>\n{nodes}<start_of_turn>model\n```python\n"


def format_node(node: Node) -> str:
    src = format_src(node.src or ast.unparse(node.stmt))
    output = ""

    if node.exc:
        output = repr(node.exc)
    elif node.stdout:
        output = node.stdout
    elif node.value is not None:
        output = str(node.value)

    if output:
        return f"{src}\n<start_of_turn>user\n{output.strip()}<end_of_turn>"
    else:
        return src


def format_src(src: str) -> str:
    return f"<start_of_turn>model\n```python\n{src}\n```<end_of_turn>"
