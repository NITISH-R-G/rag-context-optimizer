import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit_app  # noqa: E402
import httpx  # noqa: E402


@patch("streamlit_app.st")
@patch("streamlit_app.httpx.post")
def test_streamlit_flow(mock_post, mock_st):
    mock_st.session_state = {
        "payload": {
            "observation": {
                "task_name": "t",
                "token_budget": 10,
                "total_tokens_used": 1,
                "step_number": 1,
                "query": "q",
                "available_chunks": [
                    {"chunk_id": "c1", "domain": "d", "tokens": 10, "keywords": ["k"]}
                ],
                "selected_chunks": [],
            },
            "done": False,
        }
    }
    mock_post.return_value.json.return_value = {
        "observation": {
            "task_name": "t",
            "token_budget": 10,
            "total_tokens_used": 1,
            "step_number": 1,
            "query": "q",
            "available_chunks": [
                {"chunk_id": "c1", "domain": "d", "tokens": 10, "keywords": ["k"]}
            ],
            "selected_chunks": [],
        }
    }

    mock_c1 = MagicMock()
    mock_c2 = MagicMock()
    mock_container = MagicMock()
    mock_container.columns.return_value = [mock_c1, mock_c2]
    mock_st.columns.return_value[0].container.return_value = mock_container

    streamlit_app.start_episode("task", "query", 100, 10)
    streamlit_app.do_step({"action": "test"})
    streamlit_app.render_chunks(mock_st.session_state["payload"]["observation"])

    with patch("streamlit_app.httpx.get") as mock_get:
        mock_get.side_effect = httpx.RequestError("error", request=MagicMock())
        streamlit_app.api_get("/test")
