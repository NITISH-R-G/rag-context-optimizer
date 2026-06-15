import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit_app  # noqa: E402


@pytest.fixture(autouse=True)
def _mock_streamlit_secrets():
    with patch("streamlit_app.st.secrets", new={"API_URL": "http://test"}):
        yield


def test_api_get_success():
    with patch("streamlit_app.httpx.get") as mock_get:
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"status": "ok"}
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp

        result = streamlit_app.api_get("/test")
        assert result == {"status": "ok"}
        mock_get.assert_called_once()


@patch("streamlit_app.st.error")
@patch("streamlit_app.httpx.get")
def test_api_get_error(mock_get, mock_error):
    mock_get.side_effect = Exception("Connection Refused")
    result = streamlit_app.api_get("/test")
    assert result is None
    mock_error.assert_called_once_with("API Error: Connection Refused")


def test_api_post_success():
    with patch("streamlit_app.httpx.post") as mock_post:
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"status": "ok"}
        mock_resp.raise_for_status.return_value = None
        mock_post.return_value = mock_resp

        result = streamlit_app.api_post("/test", {"payload": 1})
        assert result == {"status": "ok"}
        mock_post.assert_called_once()


@patch("streamlit_app.api_post")
@patch("streamlit_app.st.session_state", new={})
def test_start_episode(mock_api_post):
    mock_api_post.return_value = {"started": True}
    streamlit_app.start_episode("test_task", "test query", 100, 10)
    mock_api_post.assert_called_once()
    assert streamlit_app.st.session_state["payload"] == {"started": True}


@patch("streamlit_app.api_post")
@patch("streamlit_app.st.session_state", new={})
def test_do_step(mock_api_post):
    mock_api_post.return_value = {"stepped": True}
    streamlit_app.do_step({"action": "test"})
    mock_api_post.assert_called_once()
    assert streamlit_app.st.session_state["payload"] == {"stepped": True}


@patch("streamlit_app.st")
def test_render_sidebar_empty_query(mock_st):
    task_map = {"task1": {"token_budget": 100, "max_steps": 10}}
    mock_st.sidebar.selectbox.return_value = "task1"
    mock_st.session_state = {"custom_query": ""}
    mock_st.sidebar.text_area.return_value = "  "  # Empty after strip
    mock_st.sidebar.number_input.side_effect = [100, 10]

    col_mock = MagicMock()
    # Click start button
    col_mock.button.side_effect = [True, False]
    mock_st.sidebar.columns.return_value = [col_mock, col_mock]

    streamlit_app.render_sidebar(task_map)
    mock_st.sidebar.error.assert_called_once()


@patch("streamlit_app.st")
@patch("streamlit_app.start_episode")
def test_render_sidebar_start(mock_start, mock_st):
    task_map = {"task1": {"token_budget": 100, "max_steps": 10}}
    mock_st.sidebar.selectbox.return_value = "task1"
    mock_st.session_state = {"custom_query": ""}
    mock_st.sidebar.text_area.return_value = "valid query"
    mock_st.sidebar.number_input.side_effect = [100, 10]

    col_mock1 = MagicMock()
    col_mock1.button.return_value = True  # Start button
    col_mock2 = MagicMock()
    col_mock2.button.return_value = False  # Refresh button
    mock_st.sidebar.columns.return_value = [col_mock1, col_mock2]

    streamlit_app.render_sidebar(task_map)
    mock_start.assert_called_once_with("task1", "valid query", 100, 10)
    mock_st.rerun.assert_called_once()


@patch("streamlit_app.st")
def test_render_sidebar_refresh(mock_st):
    mock_st.sidebar.selectbox.return_value = None
    mock_st.session_state = {"custom_query": ""}
    mock_st.sidebar.text_area.return_value = "valid query"
    mock_st.sidebar.number_input.side_effect = [100, 10]

    col_mock1 = MagicMock()
    col_mock1.button.return_value = False  # Start button
    col_mock2 = MagicMock()
    col_mock2.button.return_value = True  # Refresh button
    mock_st.sidebar.columns.return_value = [col_mock1, col_mock2]

    streamlit_app.render_sidebar({})
    mock_st.rerun.assert_called_once()


@patch("streamlit_app.st")
def test_render_metrics(mock_st):
    cols = [MagicMock(), MagicMock(), MagicMock(), MagicMock()]
    mock_st.columns.return_value = cols
    observation = {
        "task_name": "t1",
        "token_budget": 100,
        "total_tokens_used": 50,
        "step_number": 2,
    }
    streamlit_app.render_metrics(observation)
    cols[0].metric.assert_called_once()
    cols[3].metric.assert_called_once()


@patch("streamlit_app.st")
def test_render_query_and_feedback(mock_st):
    payload = {"info": {"grader_breakdown": {"score": 1.0}}, "reward": 0.5}
    observation = {"query": "q", "last_action_feedback": "warn"}
    streamlit_app.render_query_and_feedback(payload, observation)
    mock_st.warning.assert_called_once_with("warn")
    mock_st.success.assert_called_once()
    mock_st.json.assert_called_once()


@patch("streamlit_app.api_post")
@patch("streamlit_app.do_step")
@patch("streamlit_app.st")
def test_render_actions_auto_step(mock_st, mock_do_step, mock_api_post):
    mock_api_post.return_value = {"action": "test"}
    cols = [MagicMock(), MagicMock(), MagicMock()]
    cols[0].button.return_value = True
    cols[1].button.return_value = False
    mock_st.columns.return_value = cols
    mock_st.button.return_value = False

    streamlit_app.render_actions()
    mock_api_post.assert_called_once_with("/optimize-step")
    mock_do_step.assert_called_once()
    mock_st.rerun.assert_called_once()


@patch("streamlit_app.api_post")
@patch("streamlit_app.do_step")
@patch("streamlit_app.st")
def test_render_actions_auto_run(mock_st, mock_do_step, mock_api_post):
    mock_api_post.return_value = {"action_type": "submit_answer"}
    mock_st.session_state = {"payload": {"done": False}}
    cols = [MagicMock(), MagicMock(), MagicMock()]
    cols[0].button.return_value = False
    cols[1].button.return_value = True
    mock_st.columns.return_value = cols
    mock_st.button.return_value = False

    streamlit_app.render_actions()
    mock_api_post.assert_called_once()
    mock_st.rerun.assert_called_once()


@patch("streamlit_app.do_step")
@patch("streamlit_app.st")
def test_render_actions_manual_submit(mock_st, mock_do_step):
    cols = [MagicMock(), MagicMock(), MagicMock()]
    cols[0].button.return_value = False
    cols[1].button.return_value = False
    cols[2].text_input.return_value = "my answer"
    mock_st.columns.return_value = cols
    mock_st.button.return_value = True

    streamlit_app.render_actions()
    mock_do_step.assert_called_once_with(
        {"action_type": "submit_answer", "answer": "my answer"}
    )
    mock_st.rerun.assert_called_once()


@patch("streamlit_app.do_step")
@patch("streamlit_app.st")
def test_render_chunks(mock_st, mock_do_step):
    observation = {
        "available_chunks": [
            {"chunk_id": "c1", "domain": "d", "tokens": 10, "keywords": ["k"]},
            {"chunk_id": "c2", "domain": "d", "tokens": 10, "keywords": ["k"]},
        ],
        "selected_chunks": ["c1"],
    }
    col1 = MagicMock()
    col2 = MagicMock()
    mock_st.columns.return_value = [col1, col2]

    # Container setup
    cont1 = MagicMock()
    c1_1 = MagicMock()
    c1_2 = MagicMock()
    cont1.columns.return_value = [c1_1, c1_2]
    col1.container.return_value = cont1

    cont2 = MagicMock()
    c2_1 = MagicMock()
    c2_2 = MagicMock()
    cont2.columns.return_value = [c2_1, c2_2]
    col2.container.return_value = cont2

    # Click deselect on c1
    c1_1.button.return_value = True
    c1_2.button.return_value = False
    # Click select on c2
    c2_1.button.return_value = True
    c2_2.button.return_value = False

    streamlit_app.render_chunks(observation)

    assert mock_do_step.call_count == 2
    mock_st.rerun.call_count == 2


@patch("streamlit_app.do_step")
@patch("streamlit_app.st")
def test_render_chunks_compress(mock_st, mock_do_step):
    observation = {
        "available_chunks": [
            {"chunk_id": "c1", "domain": "d", "tokens": 10, "keywords": ["k"]}
        ],
        "selected_chunks": ["c1"],
    }
    col1 = MagicMock()
    col2 = MagicMock()
    mock_st.columns.return_value = [col1, col2]

    cont1 = MagicMock()
    c1_1 = MagicMock()
    c1_2 = MagicMock()
    cont1.columns.return_value = [c1_1, c1_2]
    col1.container.return_value = cont1

    c1_1.button.return_value = False
    c1_2.button.return_value = True

    streamlit_app.render_chunks(observation)
    mock_do_step.assert_called_once_with(
        {"action_type": "compress_chunk", "chunk_id": "c1", "compression_ratio": 0.5}
    )


@patch("streamlit_app.api_get")
@patch("streamlit_app.render_sidebar")
@patch("streamlit_app.render_metrics")
@patch("streamlit_app.render_query_and_feedback")
@patch("streamlit_app.render_actions")
@patch("streamlit_app.render_chunks")
@patch("streamlit_app.st")
def test_main_no_payload(
    mock_st, mock_rc, mock_ra, mock_rq, mock_rm, mock_rs, mock_api_get
):
    class StopExecution(Exception):
        pass

    mock_api_get.return_value = [{"name": "t1"}]
    mock_st.session_state = {}
    mock_st.stop.side_effect = StopExecution()
    with pytest.raises(StopExecution):
        streamlit_app.main()


@patch("streamlit_app.api_get")
@patch("streamlit_app.render_sidebar")
@patch("streamlit_app.render_metrics")
@patch("streamlit_app.render_query_and_feedback")
@patch("streamlit_app.render_actions")
@patch("streamlit_app.render_chunks")
@patch("streamlit_app.st")
def test_main_with_payload(
    mock_st, mock_rc, mock_ra, mock_rq, mock_rm, mock_rs, mock_api_get
):
    mock_api_get.side_effect = [[{"name": "t1"}], {"state": "test"}]
    mock_st.session_state = {"payload": {"observation": {}}}
    streamlit_app.main()
    mock_rc.assert_called_once()
    mock_st.json.call_count == 2
