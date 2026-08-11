import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from server.app import main  # noqa: E402


@patch("uvicorn.run")
def test_server_main_default_args(mock_run):
    with patch.object(sys, 'argv', ['app.py']):
        main()
        mock_run.assert_called_once_with("server.app:app", host="127.0.0.1", port=8000, reload=False)

@patch("uvicorn.run")
def test_server_main_with_args(mock_run):
    with patch.object(sys, 'argv', ['app.py', '--host', '127.0.0.1', '--port', '9000']):
        main()
        mock_run.assert_called_once_with("server.app:app", host="127.0.0.1", port=9000, reload=False)
