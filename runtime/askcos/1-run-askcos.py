"""Run an ASKCOS HTTP deployment over a RetroCast task."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import requests
from tqdm import tqdm

from askcos_runtime import (
    DATA_DIR,
    RAW_DIR,
    ExecutionTimer,
    HttpClient,
    configure_logging,
    load_task,
    logger,
    positive_int,
    request_counts,
    write_effective_config,
    write_run_artifacts,
)

DEFAULT_ASKCOS_URL = "http://localhost:9321/get_buyable_paths"


def call_askcos_api(
    smiles: str,
    askcos_url: str,
    *,
    timeout: int,
    client: HttpClient = requests,
) -> dict[str, Any] | None:
    try:
        response = client.post(
            askcos_url,
            headers={"Content-Type": "application/json"},
            json={"smiles": smiles},
            timeout=timeout,
        )
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException as error:
        logger.error("ASKCOS request failed for %s: %s", smiles, error)
        return None
    except ValueError as error:
        logger.error("ASKCOS returned invalid JSON for %s: %s", smiles, error)
        return None
    if not isinstance(payload, dict):
        logger.error("ASKCOS returned %s instead of an object for %s", type(payload).__name__, smiles)
        return None
    return payload


def run(
    *,
    task_name: str,
    askcos_url: str,
    timeout: int,
    limit: int | None,
    server_release: str | None,
) -> Path:
    task, source_path, stock_name = load_task(task_name)
    save_dir = RAW_DIR / "askcos" / task["name"]
    save_dir.mkdir(parents=True, exist_ok=True)

    effective_config = {
        "endpoint": askcos_url,
        "request": {
            "method": "POST",
            "content_type": "application/json",
            "body": {"smiles": "<target smiles>"},
        },
        "server_release": server_release,
        "task_stock_name": stock_name,
        "timeout_seconds": timeout,
    }
    effective_config_path = write_effective_config(effective_config, save_dir)

    targets = list(task["targets"].values())
    if limit is not None:
        targets = targets[:limit]

    logger.info("Calling ASKCOS at %s for %d targets", askcos_url, len(targets))
    results: dict[str, dict[str, Any] | None] = {}
    timer = ExecutionTimer()
    for target in tqdm(targets, desc="Processing targets"):
        with timer.measure(target["id"]):
            results[target["id"]] = call_askcos_api(
                target["smiles"],
                askcos_url,
                timeout=timeout,
            )

    parameters: dict[str, Any] = {
        "askcos_url": askcos_url,
        "effective_config_path": str(effective_config_path.relative_to(DATA_DIR)),
        "timeout": timeout,
    }
    if stock_name is not None:
        parameters["stock_name"] = stock_name
    if server_release is not None:
        parameters["server_release"] = server_release
    if limit is not None:
        parameters["limit"] = limit

    write_run_artifacts(
        results=results,
        execution_stats=timer.to_dict(),
        save_dir=save_dir,
        task=task,
        task_source=source_path,
        effective_config_path=effective_config_path,
        parameters=parameters,
    )
    successful_requests, failed_requests = request_counts(results)
    logger.info(
        "Saved %d successful ASKCOS responses (%d failed requests) to %s",
        successful_requests,
        failed_requests,
        save_dir,
    )
    return save_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", "--benchmark", dest="task_name", required=True)
    parser.add_argument("--askcos-url", default=DEFAULT_ASKCOS_URL)
    parser.add_argument("--timeout", type=positive_int, default=300)
    parser.add_argument("--limit", type=positive_int)
    parser.add_argument(
        "--server-release",
        help="ASKCOS server image tag, git revision, or deployment identifier used for this run.",
    )
    args = parser.parse_args()
    run(
        task_name=args.task_name,
        askcos_url=args.askcos_url,
        timeout=args.timeout,
        limit=args.limit,
        server_release=args.server_release,
    )


if __name__ == "__main__":
    configure_logging()
    main()
