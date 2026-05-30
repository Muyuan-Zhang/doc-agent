"""Browser E2E fixtures — starts a real uvicorn server with mocked infra."""
import asyncio
import threading
import time

import pytest
import uvicorn

from tests.e2e.conftest import make_app

_PORT = 18765


def _run_server(server: uvicorn.Server) -> None:
    asyncio.set_event_loop(asyncio.new_event_loop())
    server.run()


@pytest.fixture(scope="session")
def live_server():
    app = make_app()
    config = uvicorn.Config(
        app,
        host="127.0.0.1",
        port=_PORT,
        log_level="error",
        loop="asyncio",
        lifespan="off",  # make_app() pre-injects mocked state; skip real infra lifespan
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=_run_server, args=(server,), daemon=True)
    thread.start()

    deadline = time.time() + 10
    while not server.started:
        if time.time() > deadline:
            raise RuntimeError("Browser test server failed to start within 10 s")
        time.sleep(0.05)

    yield f"http://127.0.0.1:{_PORT}"

    server.should_exit = True
    thread.join(timeout=5)


@pytest.fixture(scope="session")
def base_url(live_server: str) -> str:
    return live_server
