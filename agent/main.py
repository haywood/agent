import argparse
import logging

from agent import solver
from llama_cpp import Llama
from llama_cpp.llama_tokenizer import LlamaHFTokenizer

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
parser.add_argument(
    "--loglevel",
    default=logging.DEBUG,
    type=int,
    choices=[
        logging.DEBUG,
        logging.INFO,
        logging.WARNING,
        logging.ERROR,
        logging.FATAL,
    ],
)
args = parser.parse_args()

logging.basicConfig(level=args.loglevel)

# We should use HF AutoTokenizer instead of llama.cpp's tokenizer because we found that Llama.cpp's tokenizer doesn't give the same result as that from Huggingface. The reason might be in the training, we added new tokens to the tokenizer and Llama.cpp doesn't handle this successfully
llm = Llama.from_pretrained(
    repo_id="meetkai/functionary-small-v2.4-GGUF",
    filename="functionary-small-v2.4.Q4_0.gguf",
    chat_format="functionary-v2",
    tokenizer=LlamaHFTokenizer.from_pretrained("meetkai/functionary-small-v2.4-GGUF"),
    n_gpu_layers=-1,
    # n_gpu_layers=1,
    n_ctx=8192,
    verbose=True,
)

state = solver.create_state(
    context=args.context,
    budget=args.budget,
    llm=llm,
    # llm=Llama(
    #     args.model,
    #     n_gpu_layers=1,
    #     n_ctx=8192,
    #     verbose=False,
    # ),
)
solver.solve(state)
