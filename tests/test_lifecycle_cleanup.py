import os
import signal
import socket
import subprocess
import time
from unittest.mock import MagicMock
import pytest
from fastapi.testclient import TestClient
from backend.db.connection import ROOT_DIR


def get_ephemeral_port():
    """Finds an unused port provided by the OS."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.listen(1)
        return s.getsockname()[1]


def is_port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


class TestLifecycleCleanup:
    def test_ml_engine_close_cleans_up_resources(self, mocker):
        from backend.ml.engine import MLEngine

        # Avoid downloading/loading full models during unit test
        mocker.patch("backend.ml.engine.resolve_device_and_dtype", return_value=("cpu", "float32"))
        mocker.patch("backend.ml.engine.registry.get_semantic", return_value=MagicMock())
        mocker.patch("backend.ml.engine.registry.get_nmt", return_value=MagicMock())
        mocker.patch("backend.ml.engine.registry.get_fluency", return_value=MagicMock())
        mocker.patch("backend.ml.engine.hf_hub_download", return_value="/tmp/fake_lid.bin")
        mocker.patch("backend.ml.engine.fasttext.load_model", return_value=MagicMock())

        mock_lang_tool = MagicMock()
        mocker.patch("backend.ml.engine.language_tool_python.LanguageTool", return_value=mock_lang_tool)

        engine = MLEngine()
        assert engine.lang_tool is mock_lang_tool

        engine.close()
        mock_lang_tool.close.assert_called_once()
        assert engine.lang_tool is None

        # Verify idempotency of close()
        engine.close()

    def test_eager_precompute_manager_shutdown(self, isolated_data_manager):
        from backend.ml.precompute import EagerPrecomputeManager

        mock_engine_fn = MagicMock()
        manager = EagerPrecomputeManager(get_engine_fn=mock_engine_fn, data_manager=isolated_data_manager)

        assert manager._is_shutdown is False
        manager.shutdown(wait=True)
        assert manager._is_shutdown is True

    @pytest.mark.anyio
    async def test_fastapi_lifespan_cleanup(self, monkeypatch):
        from backend import server

        mock_ml_engine = MagicMock()
        mock_eager = MagicMock()

        monkeypatch.setattr(server, "ml_engine", mock_ml_engine)
        monkeypatch.setattr(server, "EagerPrecomputeManager", lambda *args, **kwargs: mock_eager)
        monkeypatch.setattr(server, "DataManager", lambda *args, **kwargs: MagicMock())
        monkeypatch.setattr(server, "ScoringEngine", lambda *args, **kwargs: MagicMock())
        monkeypatch.setattr(server, "get_tts_engine", lambda *args, **kwargs: MagicMock())

        async with server.lifespan(server.app):
            # Lifespan startup completed
            pass

        # Lifespan shutdown completed
        mock_eager.shutdown.assert_called_once_with(wait=False)
        mock_ml_engine.close.assert_called_once()


    def test_subprocess_server_sigint_frees_port(self):
        """Spawns the backend server in a separate process group, verifies it listens, sends SIGINT,

        and asserts the port is completely freed and unbound immediately.
        """
        port = get_ephemeral_port()
        assert not is_port_in_use(port)

        cmd = [
            "uv", "run", "--no-sync", "python", "-m", "backend.server",
            "--port", str(port),
            "--host", "127.0.0.1",
        ]

        # Start server with its own process group
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=os.setsid,
            cwd=str(ROOT_DIR),
        )

        try:
            # Poll until server is listening on port
            start_time = time.time()
            listening = False
            while time.time() - start_time < 20.0:
                if is_port_in_use(port):
                    listening = True
                    break
                if proc.poll() is not None:
                    break
                time.sleep(0.3)

            assert listening, f"Server failed to start listening on port {port} within 20s"

            # Send SIGINT to the process group (simulating Ctrl+C)
            os.killpg(os.getpgid(proc.pid), signal.SIGINT)

            # Wait for exit
            retcode = proc.wait(timeout=10)
            assert retcode in (0, -signal.SIGINT, 130)

            # Assert port is freed immediately
            time.sleep(0.5)
            assert not is_port_in_use(port), f"Port {port} remained bound after SIGINT"

            # Verify port can be bound immediately
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("127.0.0.1", port))
                s.listen(1)

        finally:
            if proc.poll() is None:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except Exception:
                    pass
                proc.wait()
