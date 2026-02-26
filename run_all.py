"""Run every benchmark test and write results to a CSV file.

Auto-discovers ``tests/test_*.py`` modules, executes each one's
``run()`` function, captures stdout output, and persists a
timestamped CSV report under ``results/``.
"""

import csv
import importlib
import io
import os
import sys
import time
from contextlib import redirect_stdout
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path so `core` imports work regardless
# of how the script is invoked.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Auto-activate the venv when executed with the system Python.
# ---------------------------------------------------------------------------
VENV_PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"
if VENV_PYTHON.exists() and Path(sys.executable).resolve() != VENV_PYTHON.resolve():
    os.execv(str(VENV_PYTHON), [str(VENV_PYTHON), *sys.argv])


def discover_tests() -> list[str]:
    """Return sorted dotted module names for every ``tests/test_*.py`` file.

    Returns
    -------
    list[str]
        e.g. ``['tests.test_01_semantic_search', ...]``
    """
    tests_dir = PROJECT_ROOT / "tests"
    modules = sorted(
        f"tests.{p.stem}" for p in tests_dir.glob("test_*.py") if p.is_file()
    )
    return modules


def run_single_test(module_name: str) -> dict:
    """Import *module_name*, call its ``run()`` function, and capture results.

    Parameters
    ----------
    module_name : str
        Fully-qualified dotted module name (e.g. ``tests.test_01_semantic_search``).

    Returns
    -------
    dict
        Keys: ``test_module``, ``test_label``, ``status``, ``duration_s``,
        ``output``, ``error``, ``timestamp``.
    """
    # Derive a human-friendly label from the module name
    label = module_name.replace("tests.", "").replace("_", " ").title()

    record = {
        "test_module": module_name,
        "test_label": label,
        "status": "PASS",
        "duration_s": 0.0,
        "output": "",
        "error": "",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }

    buf = io.StringIO()
    start = time.perf_counter()

    try:
        mod = importlib.import_module(module_name)
        if not hasattr(mod, "run"):
            raise AttributeError(f"{module_name} has no run() function")

        with redirect_stdout(buf):
            mod.run()

        record["output"] = buf.getvalue().strip()

    except Exception as exc:  # noqa: BLE001
        record["status"] = "FAIL"
        record["error"] = f"{type(exc).__name__}: {exc}"

    record["duration_s"] = round(time.perf_counter() - start, 3)
    return record


def write_csv(records: list[dict], csv_path: Path) -> None:
    """Persist *records* as a CSV file at *csv_path*.

    Parameters
    ----------
    records : list[dict]
        One dict per test, as returned by :func:`run_single_test`.
    csv_path : Path
        Destination file path (parent dirs created automatically).
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "test_module",
        "test_label",
        "status",
        "duration_s",
        "output",
        "error",
        "timestamp",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    """Discover, execute, and report all benchmark tests."""
    modules = discover_tests()
    total = len(modules)

    print(f"Discovered {total} test(s).\n")

    results: list[dict] = []

    for idx, mod_name in enumerate(modules, 1):
        short = mod_name.replace("tests.", "")
        print(f"[{idx}/{total}] Running {short} ... ", end="", flush=True)

        record = run_single_test(mod_name)
        results.append(record)

        status_icon = "✓" if record["status"] == "PASS" else "✗"
        print(f"{status_icon}  ({record['duration_s']:.1f}s)")

    # ----- summary -----
    passed = sum(1 for r in results if r["status"] == "PASS")
    failed = total - passed
    total_time = sum(r["duration_s"] for r in results)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = PROJECT_ROOT / "results" / f"benchmark_{timestamp}.csv"
    write_csv(results, csv_path)

    print("\n" + "=" * 60)
    print(f"  DONE — {passed}/{total} passed, {failed} failed  ({total_time:.1f}s)")
    print(f"  CSV  → {csv_path.relative_to(PROJECT_ROOT)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
