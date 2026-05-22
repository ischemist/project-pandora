from __future__ import annotations

from _pytest.reports import TestReport


def _duration(seconds: float) -> str:
    if seconds >= 60:
        minutes, remainder = divmod(seconds, 60)
        return f"{int(minutes)}m{remainder:04.1f}s"
    return f"{seconds:.2f}s"


def pytest_report_teststatus(report: TestReport) -> tuple[str, str, str] | None:
    if report.when != "call":
        return None

    duration = _duration(report.duration)
    if report.passed:
        return "passed", f"PASSED {duration}", f"PASSED {duration}"
    if report.skipped:
        return "skipped", f"SKIPPED {duration}", f"SKIPPED {duration}"
    if report.failed:
        return "failed", f"FAILED {duration}", f"FAILED {duration}"
    return None
