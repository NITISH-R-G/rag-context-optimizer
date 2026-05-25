import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from server.app import main  # noqa: E402


@patch("uvicorn.run")
def test_server_main(mock_run):
    main()
    mock_run.assert_called_once_with("server.app:app", host="0.0.0.0", port=7860, reload=False)
