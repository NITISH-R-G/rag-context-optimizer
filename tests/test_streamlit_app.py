import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit_app  # noqa: E402


@patch("streamlit_app.st.error")
@patch("streamlit_app.httpx.get")
def test_api_get_error(mock_get, mock_error):
    mock_get.side_effect = Exception("Connection Refused")
    result = streamlit_app.api_get("/test")
    assert result is None
    mock_error.assert_called_once_with("API Error: Connection Refused")


@patch("streamlit_app.httpx.post")
def test_api_post(mock_post):
    class MockResponse:
        def json(self):
            return {"success": True}

        def raise_for_status(self):
            pass

    mock_post.return_value = MockResponse()
    result = streamlit_app.api_post("/test", {"a": 1})
    assert result == {"success": True}
