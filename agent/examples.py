import ast
from collections.abc import Sequence

from agent import executor
from agent.state import Node


def create_examples() -> Sequence[tuple[str, Node]]:
    ctx = executor.Context.empty()
    examples = [
        (context, [Node(stmt) for stmt in ast.parse(src).body])
        for context, src in _EXAMPLES
    ]
    for _, nodes in examples:
        for node in nodes:
            executor.execute(ctx, node)

    return examples


_EXAMPLES = [
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
