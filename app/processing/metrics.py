import os
import time
from typing import Any, Dict

try:
    import psutil
except ImportError as exc:
    raise RuntimeError(
        "psutil is required for resource metrics. Install it with `pip install psutil`."
    ) from exc

PROCESS = psutil.Process(os.getpid())


def snapshot() -> Dict[str, Any]:
    cpu_times = PROCESS.cpu_times()
    memory_info = PROCESS.memory_info()

    return {
        "timestamp": time.perf_counter(),
        "cpu_user": cpu_times.user,
        "cpu_system": cpu_times.system,
        "rss": memory_info.rss,
        "vms": memory_info.vms,
    }


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return len(text.split())


def format_bytes(value: int) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if value < 1024.0:
            return f"{value:.1f}{unit}"
        value /= 1024.0
    return f"{value:.1f}TB"


def diff_metrics(before: Dict[str, Any], after: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "duration": after["timestamp"] - before["timestamp"],
        "cpu_user": after["cpu_user"] - before["cpu_user"],
        "cpu_system": after["cpu_system"] - before["cpu_system"],
        "rss_delta": after["rss"] - before["rss"],
        "vms_delta": after["vms"] - before["vms"],
    }


def print_stage_metrics(stage: str, before: Dict[str, Any], after: Dict[str, Any], extra: Dict[str, Any] = None) -> None:
    metrics = diff_metrics(before, after)
    extra = extra or {}

    print(f"[metrics] {stage}:")
    print(f"  duration: {metrics['duration']:.4f}s")
    print(f"  cpu_user: {metrics['cpu_user']:.4f}s")
    print(f"  cpu_system: {metrics['cpu_system']:.4f}s")
    print(f"  rss change: {format_bytes(metrics['rss_delta'])}")
    print(f"  vms change: {format_bytes(metrics['vms_delta'])}")

    if "embedding_tokens" in extra:
        print(f"  embedding token estimate: {extra['embedding_tokens']}")
    if "groq_prompt_tokens" in extra:
        print(f"  groq prompt token estimate: {extra['groq_prompt_tokens']}")
    if "groq_response_tokens" in extra:
        print(f"  groq response token estimate: {extra['groq_response_tokens']}")
    if "stage_tokens" in extra:
        print(f"  stage token estimate: {extra['stage_tokens']}")

    if extra:
        for key, value in extra.items():
            if key not in {"embedding_tokens", "groq_prompt_tokens", "groq_response_tokens", "stage_tokens"}:
                print(f"  {key}: {value}")


def print_total_metrics(before: Dict[str, Any], after: Dict[str, Any]) -> None:
    metrics = diff_metrics(before, after)
    print("[metrics] total search:")
    print(f"  duration: {metrics['duration']:.4f}s")
    print(f"  cpu_user: {metrics['cpu_user']:.4f}s")
    print(f"  cpu_system: {metrics['cpu_system']:.4f}s")
    print(f"  rss change: {format_bytes(metrics['rss_delta'])}")
    print(f"  vms change: {format_bytes(metrics['vms_delta'])}")
