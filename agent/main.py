import argparse
import logging

from agent import solver
from llama_cpp import Llama

parser = argparse.ArgumentParser()
parser.add_argument(
    "--context",
    default=None,
    required=True,
    type=str,
    help="The agent's initial context.",
)
parser.add_argument(
    "--budget",
    default=10,
    type=int,
    help="The agent's budget.",
)
parser.add_argument(
    "--model",
    default="./models/gemma-1.1-2b-it.gguf",
    help="Path to llama.cpp compatible model weights.",
)
parser.add_argument("--loglevel", default=logging.DEBUG)
args = parser.parse_args()

logging.basicConfig(level=args.loglevel)

state = solver.create_state(
    context=args.context,
    budget=args.budget,
    llm=Llama(
        args.model,
        n_gpu_layers=1,
        n_ctx=8192,
        verbose=False,
    ),
)
solver.solve(state)
