"""Behavioral lifecycle contract for the reusable documentation gate."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
GATE_SCRIPT = REPO_ROOT / "scripts" / "check_docs.sh"
INJECTOR_SCRIPT = REPO_ROOT / "scripts" / "inject_docs_accessibility.py"
HOOK_ID = "jaxstro-docs-disclosure-labels"


@dataclass(frozen=True)
class GateRun:
    root: Path
    completed: subprocess.CompletedProcess[str]
    final_source: str
    server_stopped: bool
    server_alive_after_exit: bool
    audit_args: str


def _write_executable(path: Path, source: str) -> None:
    path.write_text(source, encoding="utf-8")
    path.chmod(0o755)


def _pid_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _stop_lingering_process(pid: int) -> None:
    if not _pid_is_alive(pid):
        return
    os.kill(pid, signal.SIGKILL)
    deadline = time.monotonic() + 2.0
    while _pid_is_alive(pid) and time.monotonic() < deadline:
        time.sleep(0.01)
    assert not _pid_is_alive(pid), f"failed to stop fake MyST process {pid}"


def _run_docs_gate(tmp_path: Path, script_text: str | None = None) -> GateRun:
    root = tmp_path / "repo"
    scripts = root / "scripts"
    docs = root / "docs"
    fake_bin = tmp_path / "bin"
    temp_dir = tmp_path / "tmp"
    for directory in (scripts, docs, fake_bin, temp_dir):
        directory.mkdir(parents=True, exist_ok=True)

    gate = scripts / "check_docs.sh"
    if script_text is None:
        gate.symlink_to(GATE_SCRIPT)
    else:
        _write_executable(gate, script_text)
    (scripts / "inject_docs_accessibility.py").symlink_to(INJECTOR_SCRIPT)

    artifact = docs / "_build" / "html" / "index.html"
    ready = tmp_path / "server-ready"
    stopped = tmp_path / "server-stopped"
    pid_file = tmp_path / "server-pid"
    audit_args = tmp_path / "audit-args"
    injection_started = tmp_path / "injection-started"
    final_verified = tmp_path / "final-verified"

    _write_executable(
        fake_bin / "myst",
        textwrap.dedent(
            f"""\
            #!{sys.executable}
            import os
            import signal
            import sys
            import time
            from pathlib import Path

            artifact = Path(os.environ["FAKE_ARTIFACT"])
            command = sys.argv[1]
            if command == "build":
                artifact.parent.mkdir(parents=True, exist_ok=True)
                artifact.write_text("<html><body>strict build</body></html>", encoding="utf-8")
                raise SystemExit(0)
            if command != "start":
                raise SystemExit(f"unexpected fake myst command: {{command}}")

            artifact.write_text("<html><body>server start rewrite</body></html>", encoding="utf-8")
            Path(os.environ["FAKE_SERVER_PID"]).write_text(str(os.getpid()), encoding="utf-8")
            Path(os.environ["FAKE_SERVER_READY"]).write_text("ready", encoding="utf-8")

            def stop_server(_signum, _frame):
                injection_started = Path(os.environ["FAKE_INJECTION_STARTED"])
                final_verified = Path(os.environ["FAKE_FINAL_VERIFIED"])
                start_deadline = time.monotonic() + 1.0
                while not injection_started.exists() and time.monotonic() < start_deadline:
                    time.sleep(0.01)
                if injection_started.exists():
                    verify_deadline = time.monotonic() + 1.0
                    while not final_verified.exists() and time.monotonic() < verify_deadline:
                        time.sleep(0.01)
                artifact.write_text(
                    "<html><body>server shutdown rewrite</body></html>",
                    encoding="utf-8",
                )
                Path(os.environ["FAKE_SERVER_STOPPED"]).write_text("stopped", encoding="utf-8")
                raise SystemExit(0)

            signal.signal(signal.SIGTERM, stop_server)
            while True:
                time.sleep(0.05)
            """
        ),
    )
    _write_executable(
        fake_bin / "uv",
        textwrap.dedent(
            f"""\
            #!{sys.executable}
            import os
            import subprocess
            import sys
            import time
            from pathlib import Path

            args = sys.argv[1:]
            python_args = args[args.index("python") + 1:]
            target = Path(python_args[0])
            if target.name == "inject_docs_accessibility.py":
                Path(os.environ["FAKE_INJECTION_STARTED"]).write_text(
                    "started", encoding="utf-8"
                )
                completed = subprocess.run(
                    [os.environ["PYTHON_EXECUTABLE"], *python_args], check=False
                )
                if "--check" in python_args and completed.returncode == 0:
                    Path(os.environ["FAKE_FINAL_VERIFIED"]).write_text(
                        "verified", encoding="utf-8"
                    )
                raise SystemExit(completed.returncode)
            if target.name == "check_docs_site.py":
                deadline = time.monotonic() + 5.0
                ready = Path(os.environ["FAKE_SERVER_READY"])
                while not ready.exists() and time.monotonic() < deadline:
                    time.sleep(0.01)
                if not ready.exists():
                    raise SystemExit("fake MyST server did not become ready")
                Path(os.environ["FAKE_AUDIT_ARGS"]).write_text(
                    " ".join(python_args), encoding="utf-8"
                )
            raise SystemExit(0)
            """
        ),
    )

    env = os.environ.copy()
    env.update(
        {
            "BASE_URL": "/fake-base",
            "FAKE_ARTIFACT": str(artifact),
            "FAKE_AUDIT_ARGS": str(audit_args),
            "FAKE_FINAL_VERIFIED": str(final_verified),
            "FAKE_INJECTION_STARTED": str(injection_started),
            "FAKE_SERVER_PID": str(pid_file),
            "FAKE_SERVER_READY": str(ready),
            "FAKE_SERVER_STOPPED": str(stopped),
            "PATH": f"{fake_bin}{os.pathsep}{env['PATH']}",
            "PYTHON_EXECUTABLE": sys.executable,
            "TMPDIR": str(temp_dir),
        }
    )

    completed: subprocess.CompletedProcess[str] | None = None
    try:
        completed = subprocess.run(
            ["bash", str(gate)],
            cwd=root,
            env=env,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    finally:
        if completed is None and pid_file.exists():
            _stop_lingering_process(int(pid_file.read_text(encoding="utf-8")))

    assert completed is not None
    assert pid_file.is_file(), completed.stdout + completed.stderr
    pid = int(pid_file.read_text(encoding="utf-8"))
    if _pid_is_alive(pid):
        deadline = time.monotonic() + 1.5
        while _pid_is_alive(pid) and time.monotonic() < deadline:
            time.sleep(0.01)
    server_alive = _pid_is_alive(pid)
    final_source = artifact.read_text(encoding="utf-8")
    recorded_audit_args = (
        audit_args.read_text(encoding="utf-8") if audit_args.exists() else ""
    )
    if server_alive:
        _stop_lingering_process(pid)

    return GateRun(
        root=root,
        completed=completed,
        final_source=final_source,
        server_stopped=stopped.is_file(),
        server_alive_after_exit=server_alive,
        audit_args=recorded_audit_args,
    )


def _assert_finalized_lifecycle(run: GateRun) -> None:
    assert run.completed.returncode == 0, run.completed.stdout + run.completed.stderr
    assert "ALL DOCS GATES PASSED" in run.completed.stdout
    assert run.server_stopped
    assert not run.server_alive_after_exit
    assert "server shutdown rewrite" in run.final_source
    assert run.final_source.count(f'<script id="{HOOK_ID}">') == 1
    assert "--base-path /fake-base" in run.audit_args

    verified = subprocess.run(
        [
            sys.executable,
            str(INJECTOR_SCRIPT),
            str(run.root / "docs" / "_build" / "html"),
            "--check",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert verified.returncode == 0, verified.stdout + verified.stderr
    assert "1 pages verified current" in verified.stdout


def _mutate_gate_script(mutation: str) -> str:
    source = GATE_SCRIPT.read_text(encoding="utf-8")
    if mutation == "no-op stop":
        start = source.index("stop_docs_server() {")
        end = source.index("\n}\n\ncleanup()", start) + 2
        mutated = f"{source[:start]}stop_docs_server() {{\n  :\n}}{source[end:]}"
    elif mutation == "removed wait":
        mutated = source.replace('    wait "$pid" 2>/dev/null || true\n', "", 1)
    elif mutation == "injection before stop":
        mutated = source.replace("\nstop_docs_server\n\n# MyST", "\n# MyST", 1)
        mutated = mutated.replace(
            'echo "== docs: final static artifact postcondition =="',
            'stop_docs_server\n\necho "== docs: final static artifact postcondition =="',
            1,
        )
    elif mutation == "rewrite after finalization":
        mutated = source.replace(
            'echo "ALL DOCS GATES PASSED"',
            "printf '<html><body>late rewrite</body></html>\\n' > "
            '"$ROOT_DIR/docs/_build/html/index.html"\n\n'
            'echo "ALL DOCS GATES PASSED"',
            1,
        )
    else:
        raise ValueError(f"unknown lifecycle mutation: {mutation}")
    assert mutated != source
    return mutated


def test_docs_gate_finalizes_the_artifact_after_the_server_is_reaped(
    tmp_path: Path,
) -> None:
    _assert_finalized_lifecycle(_run_docs_gate(tmp_path))


@pytest.mark.parametrize(
    "mutation",
    (
        "no-op stop",
        "removed wait",
        "injection before stop",
        "rewrite after finalization",
    ),
)
def test_docs_gate_behavioral_harness_rejects_lifecycle_mutations(
    tmp_path: Path, mutation: str
) -> None:
    run = _run_docs_gate(tmp_path, script_text=_mutate_gate_script(mutation))

    if mutation == "no-op stop":
        assert not run.server_stopped
        assert run.server_alive_after_exit
    elif mutation == "removed wait":
        assert run.completed.returncode == 0
        assert run.server_stopped
        assert f'<script id="{HOOK_ID}">' not in run.final_source
    elif mutation == "injection before stop":
        assert run.completed.returncode != 0
        assert run.server_stopped
        assert f'<script id="{HOOK_ID}">' not in run.final_source
    elif mutation == "rewrite after finalization":
        assert run.completed.returncode == 0
        assert run.final_source == "<html><body>late rewrite</body></html>\n"
