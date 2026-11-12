"""Native entrypoint: loopback by default; non-loopback requires bearer auth."""
import argparse
import importlib
import os
import re

import uvicorn
from api_auth import BearerAuth, load_access

# Fail closed before importing inference modules or downloading/loading weights.
access = load_access()

parser = argparse.ArgumentParser()
parser.add_argument("--run-id", required=True)
args = parser.parse_args()
if not re.fullmatch(r"[a-f0-9]{32}", args.run_id):
    raise RuntimeError("invalid run identity")
os.environ["MODEL_RUN_ID"] = args.run_id
task = os.environ["MODEL_TASK"]
if task == "text-generation":
    module = "text_server"
elif task in ("text-to-image", "image-to-image"):
    module = "server"
elif task == "text-to-video":
    module = "video_server"
else:
    raise RuntimeError("unsupported native model task")
uvicorn.run(
    BearerAuth(importlib.import_module(module).app, access),
    host=access.listen,
    port=int(os.environ["MODEL_PORT"]),
    workers=1,
    access_log=False,
)
