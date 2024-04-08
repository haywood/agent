import ast
import logging

from agent import examples, executor, planner
from agent.state import Node, State

EXAMPLES = [
    (
        "What time is it?",
        """
# Import datetime to access needed functions.
from datetime import datetime
# Get the current time using datetime.now()
datetime.now()
""".strip(),
    ),
    (
        "What is the square root of Pi?",
        """
# Import math to access sqrt() and pi
import math
# Use math.sqrt() to calculate the square root of math.pi.
math.sqrt(math.pi)
""",
    ),
    (
        "What is the 10th Fibonacci number?",
        """
# Create a function for calculating Fibonacci numbers.
def fibonacci(n):
    \"""Calculate the nth Fibonacci number.\"""
    # a = f(0), b = f(1)
    a, b = (0, 1)
    for _ in range(n-1):
        # Use b to compute the next number, and store the previous number in a
        a, b = (b, a + b)

    # At the end of the loop, b = f(n)
    return b
# Use the function to calcuate f(10)
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
            try:
                executor.execute(state, node)
            except SystemExit:
                return
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
    ex = examples.create_examples()
    parts = []
    for context, nodes in examples.create_examples():
        nodes_str = "\n".join(planner.format_node(node) for node in nodes)
        parts.append(
            f"<start_of_turn>user\n{context}<end_of_turn>\n{nodes_str}\n{planner.format_src('quit()')}\n\n"
        )

    return State(**kwargs, prefix="".join(parts))
