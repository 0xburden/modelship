"""Native service entrypoint. Never use wildcard binds on provider-managed hosts."""
import argparse
import os
import re

import uvicorn

parser = argparse.ArgumentParser()
parser.add_argument("--run-id", required=True)
args = parser.parse_args()
if not re.fullmatch(r"[a-f0-9]{32}", args.run_id):
    raise RuntimeError("invalid run identity")
os.environ["MODEL_RUN_ID"] = args.run_id
module = "text_server" if os.environ["MODEL_TASK"] == "text-generation" else "server"
uvicorn.run(
    module + ":app",
    host="127.0.0.1",
    port=int(os.environ["MODEL_PORT"]),
    workers=1,
    access_log=False,
)
