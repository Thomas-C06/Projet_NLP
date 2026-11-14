"""Prepare and open the application without manual terminal configuration."""

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import urlopen
import webbrowser

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
STATE = ROOT / "tmp/application-process.json"
INSTANCE = hashlib.sha256(str(ROOT).encode()).hexdigest()[:16]


@contextmanager
def startup_lock():
    STATE.parent.mkdir(parents=True, exist_ok=True)
    with (STATE.parent / "launcher.lock").open("a+b") as handle:
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt
            for _ in range(360):
                try:
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    time.sleep(0.5)
            else:
                raise RuntimeError("Another launch is still preparing the application.")
            try:
                yield
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)


def run(arguments):
    subprocess.run([str(value) for value in arguments], cwd=ROOT, check=True)


def health(port):
    try:
        with urlopen(f"http://127.0.0.1:{port}/api/health", timeout=2) as response:
            return json.loads(response.read())
    except (URLError, ValueError, OSError):
        return None


def prepare():
    if not PYTHON.exists():
        if shutil.which("uv"):
            run(["uv", "venv", "--python", "3.12", ROOT / ".venv"])
        elif sys.version_info[:2] == (3, 12):
            run([sys.executable, "-m", "venv", ROOT / ".venv"])
        else:
            run(["py", "-3.12", "-m", "venv", ROOT / ".venv"])
    check = subprocess.run([str(PYTHON), "-c", "import importlib.util; assert all(importlib.util.find_spec(name) for name in ('fastapi', 'sentence_transformers', 'httpx', 'langdetect'))"],
                           cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if check.returncode:
        run([PYTHON, "-m", "pip", "install", "-r", "requirements.txt"])
    if not (ROOT / "models/all-MiniLM-L6-v2/config.json").exists():
        run([PYTHON, "-m", "scripts.prepare_model"])
    if not (ROOT / "data/movies.json").exists():
        run([PYTHON, "-m", "scripts.prepare_catalog"])
    if not (ROOT / "frontend/dist/index.html").exists():
        npm = shutil.which("npm") or shutil.which("npm.cmd")
        if not npm:
            raise RuntimeError("Node.js is needed to build the interface.")
        run([npm, "--prefix", "frontend", "ci"])
        run([npm, "--prefix", "frontend", "run", "build"])


def stop():
    if not STATE.exists():
        print("No application process was started by this launcher.")
        return
    state = json.loads(STATE.read_text())
    pid = int(state["pid"])
    status = health(state["port"])
    if not status or status.get("instance_id") != INSTANCE:
        print("The recorded application is not running.")
        return
    if os.name == "nt":
        command = subprocess.run(["powershell", "-NoProfile", "-Command",
            f"(Get-CimInstance Win32_Process -Filter 'ProcessId = {pid}').CommandLine"],
            capture_output=True, text=True, check=True).stdout
        if str(ROOT / "scripts/run_server.py").casefold() not in command.casefold():
            raise RuntimeError("The recorded process does not belong to this project.")
    os.kill(pid, signal.SIGTERM)
    STATE.unlink(missing_ok=True)
    print("Movie Match stopped.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--stop", action="store_true")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    with startup_lock():
        start_or_stop(args)


def start_or_stop(args):
    if args.stop:
        stop()
        return
    status = health(args.port)
    if status and status.get("instance_id") != INSTANCE:
        raise RuntimeError(f"Port {args.port} is already used by another service.")
    if not status:
        prepare()
        STATE.parent.mkdir(parents=True, exist_ok=True)
        log_path = ROOT / "tmp/application.log"
        with log_path.open("a", encoding="utf-8") as log:
            process = subprocess.Popen([str(PYTHON), str(ROOT / "scripts/run_server.py"), "--port", str(args.port)],
                cwd=ROOT, stdout=log, stderr=log, stdin=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        for _ in range(180):
            if process.poll() is not None:
                raise RuntimeError(f"The application did not start. Read {log_path}.")
            status = health(args.port)
            if status and status.get("catalog_size"):
                break
            time.sleep(1)
        else:
            raise RuntimeError(f"The application is still preparing. Read {log_path}.")
    # Windows virtual environments can use a redirector process with another PID.
    # Keep the PID reported by the actual HTTP service for a reliable stop command.
    if status and status.get("server_pid"):
        STATE.write_text(json.dumps({"pid": status["server_pid"], "port": args.port}), encoding="utf-8")
    url = f"http://127.0.0.1:{args.port}"
    print(f"Movie Match is ready: {url}")
    if not args.no_browser:
        webbrowser.open(url)


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, OSError, subprocess.CalledProcessError) as error:
        print(f"Startup failed: {error}", file=sys.stderr)
        raise SystemExit(1)
