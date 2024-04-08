import argparse
import code
import contextlib
import inspect
import io
import itertools
import json
import logging
import random
import re

import google.auth
import llama_cpp.llama_chat_format as llama_chat_format

from google.cloud import bigquery, datacatalog_v1 as dc
from llama_cpp import Llama
from llama_cpp.llama_tokenizer import LlamaHFTokenizer

parser = argparse.ArgumentParser()
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
    verbose=False,
)

messages = [
    {
        "role": "system",
        "content": "You are a helpful assistant with knowledge of GCP BigQuery and GoogleSQL.",
    }
]
formatter = llama_chat_format.hf_autotokenizer_to_chat_formatter(
    "meetkai/functionary-small-v2.4-GGUF"
)

credentials, quota_project = google.auth.default()
bq_client = bigquery.Client(
    credentials=credentials, project=quota_project or "rosewater-diary"
)
dc_client = dc.DataCatalogClient(credentials=credentials)


def search_tables(description: str):
    """Search for BigQuery tables according to a natural language description."""

    filter = "|".join(
        part.strip() for part in re.split(r"[\s_]+", description) if part.strip()
    )
    resource_names = [
        re.sub(r"^.*\.googleapis.com/", "", result.linked_resource)
        for result in itertools.islice(
            dc_client.search_catalog(
                request=dc.SearchCatalogRequest(
                    query=f"type=table system=bigquery name:{filter}",
                    scope=dc.SearchCatalogRequest.Scope(
                        include_gcp_public_datasets=True,
                    ),
                    page_size=10,
                ),
            ),
            10,
        )
    ]
    table_ids = [".".join(name.split("/")[1::2]) for name in resource_names]

    return json.dumps(table_ids, indent=2)


def get_table_details(dataset: str, table: str):
    """Get the schema for a BigQuery table."""
    return json.dumps(bq_client.get_table(f"{dataset}.{table}").to_api_repr(), indent=2)


def run_sql(query: str):
    """Run a query using BigQuery."""
    return "\n" + bq_client.query_and_wait(query).to_dataframe().to_string(index=False)


repl = code.InteractiveInterpreter()


def python(src: str):
    """Run Python code in an interactive REPL."""
    logging.debug("Preparing to execute Python source:\n%s", src)
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        with contextlib.redirect_stderr(output):
            result = repl.runsource(src, symbol="exec")
    if result:
        return "Error: incomplete Python input"
    else:
        return output.getvalue()


functions = [search_tables, get_table_details, run_sql, python]
tools = [
    {
        "type": "function",
        "function": {
            "name": fn.__name__,
            "description": inspect.getdoc(fn) or "",
            "parameters": {
                "type": "object",
                "properties": {
                    name: {
                        "type": {
                            "str": "string",
                        }[value.annotation.__name__]
                    }
                    for name, value in inspect.signature(fn).parameters.items()
                },
                "required": [
                    name
                    for name, value in inspect.signature(fn).parameters.items()
                    if value.kind
                    in [
                        inspect.Parameter.POSITIONAL_ONLY,
                        inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    ]
                ],
            },
        },
    }
    for fn in functions
]

print(json.dumps({"tools": tools}, indent=2))


def execute_tool(tool_call):
    if (type := tool_call.get("type")) != "function":
        return f"Error: tool_call `{tool_call.get(id)}` has invalid type: `{type}`"

    function_call = tool_call.get("function")
    name = function_call.get("name")
    if not name:
        return f"Error: tool_call `{tool_call.get(id)}` missing function name"

    if name not in [fn.__name__ for fn in functions]:
        return f"Error: call to unknown function `{name}`"

    args_json = function_call.get("arguments")
    if not args_json:
        return f"Error: missing arguments for call to `{name}`"
    try:
        kwargs = json.loads(args_json)
    except Exception as e:
        return f"Error: failed to parse arguments for call to `{name}`: {e}"

    try:
        fn = next(fn for fn in functions if fn.__name__ == name)
        return fn(**kwargs)
    except Exception as e:
        return f"Error: failed to call function `{name}`: {e}"


def call_assistant():
    result = llm.create_chat_completion(
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )
    choices = result.get("choices")
    if choices and (assistant_message := choices[0].get("message")):
        messages.append(assistant_message)
        if tool_calls := assistant_message.get("tool_calls"):
            logging.debug(formatter([assistant_message]).prompt)
            names = [
                tool_call.get("function", {}).get("name") for tool_call in tool_calls
            ]
            outputs = [execute_tool(tool_call) for tool_call in tool_calls]
            tool_messages = [
                {"role": "tool", "name": name, "content": output}
                for name, output in zip(names, outputs)
            ]
            messages.extend(tool_messages)
            print(formatter(tool_messages).prompt)
            call_assistant()
        else:
            print(formatter([assistant_message]).prompt)


if messages:
    print(formatter(messages).prompt)

while True:
    try:
        user_input = input(">>> ")
    except KeyboardInterrupt:
        print("Exiting due to interrupt...")
        exit(0)
    print()

    messages.append({"role": "user", "content": user_input})
    try:
        call_assistant()
    except KeyboardInterrupt:
        logging.warning(
            f"Assistant received keyboard interrupt. Awaiting further instructions."
        )
