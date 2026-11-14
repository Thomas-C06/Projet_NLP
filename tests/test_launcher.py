from argparse import Namespace
import json

import scripts.launch as launch


def test_reusing_service_records_actual_server_pid(tmp_path, monkeypatch):
    state = tmp_path / "application-process.json"
    monkeypatch.setattr(launch, "STATE", state)
    monkeypatch.setattr(launch, "health", lambda port: {
        "instance_id": launch.INSTANCE, "server_pid": 12345, "catalog_size": 150,
    })
    monkeypatch.setattr(launch, "prepare", lambda: (_ for _ in ()).throw(AssertionError("Unexpected setup")))
    launch.start_or_stop(Namespace(stop=False, port=8000, no_browser=True))
    assert json.loads(state.read_text())["pid"] == 12345


def test_stop_does_not_kill_another_project(tmp_path, monkeypatch):
    state = tmp_path / "application-process.json"
    state.write_text(json.dumps({"pid": 12345, "port": 8000}))
    monkeypatch.setattr(launch, "STATE", state)
    monkeypatch.setattr(launch, "health", lambda port: {"instance_id": "another-project"})
    monkeypatch.setattr(launch.os, "kill", lambda *args: (_ for _ in ()).throw(AssertionError("Unrelated process")))
    launch.stop()
    assert state.exists()
