"""ASGI entrypoint wrapper for OpenEnv validator compatibility."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import app as app  # noqa: F401, E402


def main() -> None:
    import argparse

    import uvicorn
    parser = argparse.ArgumentParser(description="Run the RAG Context Optimizer server")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host IP address")
    parser.add_argument("--port", type=int, default=8000, help="Port number")
    args = parser.parse_args()
    uvicorn.run("server.app:app", host=args.host, port=args.port, reload=False)

if __name__ == "__main__":
    main()
