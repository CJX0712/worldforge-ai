"""一键验证：裸 CPython 即可运行的分阶段门禁链。

任一阶段失败立即短路并返回非零退出码。报告落盘为 verify-report.json，
CI 直接当 artifact 上传。
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
REPORT = ROOT / "verify-report.json"

sys.path.insert(0, str(SRC))


def _run(cmd: list[str], label: str) -> dict[str, Any]:
    started = time.perf_counter()
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
    tail = (proc.stdout or "").strip().splitlines()[-6:]
    return {
        "stage": label,
        "returncode": proc.returncode,
        "seconds": round(time.perf_counter() - started, 3),
        "stdout_tail": tail,
        "stderr_tail": (proc.stderr or "").strip().splitlines()[-4:],
    }


def stage_imports() -> dict[str, Any]:
    import importlib

    modules = [
        "worldforge",
        "worldforge.config",
        "worldforge.types",
        "worldforge.errors",
        "worldforge.pipeline",
        "worldforge.core.bm25",
        "worldforge.core.rrf",
        "worldforge.core.text",
        "worldforge.core.tokenizer",
        "worldforge.core.safe_eval",
        "worldforge.core.seeding",
        "worldforge.core.kernels",
        "worldforge.ports",
        "worldforge.adapters.registry",
        "worldforge.retrieval",
        "worldforge.agents",
        "worldforge.serving",
        "worldforge.eval",
        "worldforge.observability",
    ]
    for name in modules:
        importlib.import_module(name)
    return {"stage": "imports", "modules": len(modules), "returncode": 0}


def stage_smoke() -> dict[str, Any]:
    from worldforge.pipeline import build_pipeline

    pipeline = build_pipeline()
    for doc_id, text in (
        ("kb-a", "BM25 由 Robertson 提出，用逆文档频率衡量词项区分度。"),
        ("kb-b", "Transformer 用多头注意力建模任意位置之间的依赖关系。"),
        ("kb-c", "检索增强生成先召回文档再让模型生成答案，可缓解幻觉。"),
    ):
        pipeline.ingest(doc_id, text)
    hits = pipeline.retrieve("BM25 是谁提出的", top_k=3)
    if not hits:
        raise AssertionError("默认后端检索召回为空：默认实现退化成 stub")
    answer = pipeline.answer("BM25 是谁提出的")
    if not answer["answer"]:
        raise AssertionError("答案为空")
    act = pipeline.act("帮我算 12*(3+4)")
    if act["answer"] != "84":
        raise AssertionError(f"工具路由结果错误: {act['answer']}")
    return {"stage": "smoke", "returncode": 0, "top_hit": hits[0].chunk_id, "grounding": answer["grounding"]}


def stage_eval() -> dict[str, Any]:
    from worldforge.eval.runner import evaluate
    from worldforge.pipeline import build_pipeline

    report = evaluate(build_pipeline, enforce=True)
    return {"stage": "eval_gate", "returncode": 0, "gate": report["gate"]}


def stage_e2e() -> dict[str, Any]:
    return _run([sys.executable, str(ROOT / "scripts" / "smoke.py")], "http_e2e")


def main() -> int:
    stages: list[Callable[[], dict[str, Any]]] = [
        lambda: _run([sys.executable, str(ROOT / "scripts" / "scan_emoji.py")], "p0_char_gate"),
        lambda: _run([sys.executable, "-m", "pytest", "tests", "-q", "--no-header", "-p", "no:cacheprovider"], "pytest"),
        stage_imports,
        stage_smoke,
        stage_eval,
        stage_e2e,
    ]
    results: list[dict[str, Any]] = []
    for run in stages:
        try:
            result = run()
        except Exception as exc:  # noqa: BLE001 - 门禁链必须把异常转成阶段失败
            result = {"stage": getattr(run, "__name__", "unknown"), "returncode": 1, "error": str(exc), "traceback": traceback.format_exc()[-800:]}
        results.append(result)
        status = "PASS" if result.get("returncode") == 0 else "FAIL"
        print(f"[{status}] {result.get('stage')}")
        if status == "FAIL":
            for line in result.get("stdout_tail", [])[-4:]:
                print(f"       {line}")
            for line in result.get("stderr_tail", []):
                print(f"       {line}")
            if result.get("error"):
                print(f"       {result['error']}")
            REPORT.write_text(json.dumps({"passed": False, "stages": results}, ensure_ascii=False, indent=2), encoding="utf-8")
            return 1
    REPORT.write_text(json.dumps({"passed": True, "stages": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nAll stages passed. Report -> verify-report.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
__author__ = "晨星"
