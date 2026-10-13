import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import httpx  # noqa: E402
import streamlit_app  # noqa: E402


@patch("streamlit_app.st.error")
@patch("streamlit_app.httpx.get")
def test_api_get_error(mock_get, mock_error):
    mock_get.side_effect = httpx.RequestError("Connection Refused")
    result = streamlit_app.api_get("/test")
    assert result is None
    mock_error.assert_called_once_with("API Error: Connection Refused")
