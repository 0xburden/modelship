#!/usr/bin/env python3
"""Opt-in native GPU API checks. Runs against an existing localhost deployment.

No installation or deployment changes. Saves outputs/summary under --output.
Use one task at a time; image-to-image needs --input from a previous image run.
"""
import argparse
import base64
import concurrent.futures
import json
from pathlib import Path
import struct
import threading
import time
import urllib.error
import urllib.request


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["text-generation", "text-to-image", "image-to-image"], required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--expect-completion-substring", help="optional semantic check for the France completion, e.g. Paris")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True, mode=0o700)
    base = f"http://127.0.0.1:{args.port}"
    checks = []

    def request(path, body=None):
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(base + path, data=data, headers={"Content-Type": "application/json"})
        start = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=600) as response:
                return response.status, json.load(response), time.monotonic() - start
        except urllib.error.HTTPError as exc:
            return exc.code, json.load(exc), time.monotonic() - start

    def check(label, path, body=None, status=200):
        actual, result, elapsed = request(path, body)
        entry = {"check": label, "status": actual, "expected": status, "seconds": round(elapsed, 3)}
        checks.append(entry)
        print(json.dumps(entry), flush=True)
        assert actual == status, (label, actual, result)
        return result

    health = check("health", "/health")
    assert health["ready"] and health["task"] == args.task and len(health["run_id"]) == 32
    model = health["model"]
    if args.task == "text-generation":
        models = check("models", "/v1/models")
        assert model in [entry["id"] for entry in models["data"]]
        completion = check("completion", "/v1/completions", {"model": model, "prompt": "The capital of France is", "max_tokens": 16, "temperature": 0})
        assert completion["choices"][0]["text"].strip()
        if args.expect_completion_substring:
            assert args.expect_completion_substring.lower() in completion["choices"][0]["text"].lower(), completion["choices"][0]["text"]
        chat = check("chat", "/v1/chat/completions", {"model": model, "messages": [{"role": "user", "content": "Explain what a GPU does in one short sentence."}], "max_tokens": 64, "temperature": 0})
        assert chat["choices"][0]["message"]["content"].strip()
        check("unknown-model", "/v1/completions", {"model": "missing/model", "prompt": "x"}, status=404)
        check("stream-rejected", "/v1/completions", {"prompt": "x", "stream": True}, status=422)
        check("invalid-token-budget", "/v1/completions", {"prompt": "x", "max_tokens": 9999}, status=422)
        check("context-overflow", "/v1/completions", {"prompt": "hello " * 5000, "max_tokens": 8}, status=422)
        barrier = threading.Barrier(2)
        def concurrent_request(_):
            barrier.wait()
            return request("/v1/completions", {"prompt": "List twenty different kinds of trees with a brief description of each.", "max_tokens": 128})[0]
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            statuses = sorted(pool.map(concurrent_request, range(2)))
        assert statuses in ([200, 503], [200, 200]), statuses
        checks.append({"check": "concurrent-busy", "statuses": statuses, "observed_busy": 503 in statuses})
        (args.output / "text-responses.json").write_text(json.dumps({"completion": completion, "chat": chat}, indent=2))
    else:
        endpoint = "/v1/images/generations" if args.task == "text-to-image" else "/v1/images/edits"
        if args.task == "image-to-image":
            if args.input is None:
                parser.error("--input is required for image-to-image")
            encoded = base64.b64encode(args.input.read_bytes()).decode()
            sizes = (512,)
        else:
            encoded = None
            sizes = (512, 1024)
        pipeline = health.get("pipeline", "")
        guidance = 1.0 if pipeline == "QwenImage21Pipeline" else 3.0 if pipeline == "MOSSImagePipeline" else 7.0
        for size in sizes:
            body = {"prompt": "A small mountain observatory at sunset, watercolor painting" if encoded is None else "A small mountain observatory at night beneath a starry sky, watercolor painting", "width": size, "height": size, "steps": 20, "guidance_scale": guidance, "seed": 42 if encoded is None else 43}
            if encoded is not None:
                body.update(image_b64=encoded)
                if pipeline != "QwenImage21Pipeline":
                    body["strength"] = 0.65
            result = check(f"{args.task}-{size}", endpoint, body)
            data = base64.b64decode(result["data"][0]["b64_json"], validate=True)
            assert data.startswith(b"\x89PNG\r\n\x1a\n")
            assert struct.unpack(">II", data[16:24]) == (size, size)
            name = f"{args.task}-{size}.png"
            (args.output / name).write_bytes(data)
            checks[-1].update(file=name, bytes=len(data), seed=result["seed"])
        wrong_endpoint = "/v1/images/edits" if encoded is None else "/v1/images/generations"
        check("wrong-task-endpoint", wrong_endpoint, {"prompt": "x"}, status=409)
        check("invalid-dimensions", endpoint, {"prompt": "x", "width": 513}, status=422)
        if encoded is not None:
            check("image-required", endpoint, {"prompt": "x"}, status=422)
            check("image-url-rejected", endpoint, {"prompt": "x", "image_b64": "https://example.com/image.png"}, status=422)
    report = {"model": model, "revision": health["revision"], "task": args.task, "checks": checks}
    path = args.output / (args.task + "-report.json")
    path.write_text(json.dumps(report, indent=2))
    print("LIVE_API_CHECK_PASS", args.task, path, flush=True)


if __name__ == "__main__":
    main()
