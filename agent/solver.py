import ast
import logging

from agent import executor, planner
from agent.state import Node, State

EXAMPLES = [
    (
        "What time is it?",
        """
from datetime import datetime
datetime.now()
""".strip(),
    ),
    (
        "What is the square root of Pi?",
        """
import math
math.sqrt(math.pi)
""",
    ),
    (
        "What is the 10th Fibonacci number?",
        """
def fibonacci(n):
    a, b = (0, 1)
    for _ in range(n-1):
        a, b = (b, a + b)
    return b
fibonacci(10)
""",
    ),
]


def solve(state: State):
    print(planner.format(state))
    pending = []

    while state.cost < state.budget:
        state.cost += 1

        if pending:
            node = pending.pop(0)
            executor.execute(state, node)
            state.nodes.append(node)
            if node.exc:
                # re-plan on exception
                pending.clear()
            print(planner.format_node(node))
        elif plan := planner.plan(state):
            pending.extend(Node(stmt) for stmt in plan.body)
            print(ast.unparse(plan))
        else:
            return

        logging.debug(f"remaining budget: {state.budget - state.cost}/{state.budget}")


def create_state(**kwargs) -> State:
    parts = []
    locals = {}
    for context, src in EXAMPLES:
        tmp = State(
            llm=kwargs.get("llm"),
            context=context,
            locals=locals,
            nodes=[Node(stmt) for stmt in ast.parse(src).body],
        )
        for node in tmp.nodes:
            executor.execute(tmp, node)
        locals = locals | tmp.locals

        parts.append(planner.format(tmp) + "quit()\n\n")

    return State(**kwargs, locals=locals, prefix="".join(parts))
