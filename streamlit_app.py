from __future__ import annotations

import requests
import streamlit as st


API_URL = st.secrets.get("API_URL", "http://localhost:7860") if hasattr(st, "secrets") else "http://localhost:7860"

st.set_page_config(page_title="rag-context-optimizer", page_icon="R", layout="wide")
st.title("RAG Context Optimizer")
st.caption("Use any prompt, keep the token budget tight, and let the optimizer pick the best evidence per token.")


def api_get(path: str):
    response = requests.get(f"{API_URL}{path}", timeout=20)
    response.raise_for_status()
    return response.json()


def api_post(path: str, payload: dict | None = None):
    response = requests.post(f"{API_URL}{path}", json=payload or {}, timeout=20)
    response.raise_for_status()
    return response.json()


def start_episode(task_name: str, query: str, token_budget: int, max_steps: int):
    st.session_state["payload"] = api_post(
        "/reset",
        {
            "task_name": task_name,
            "custom_query": query,
            "token_budget": token_budget,
            "max_steps": max_steps,
        },
    )


def do_step(payload: dict):
    st.session_state["payload"] = api_post("/step", payload)


tasks = api_get("/tasks")
task_map = {task["name"]: task for task in tasks}

selected_task = st.sidebar.selectbox("Task preset", list(task_map))
task_meta = task_map[selected_task]

default_query = st.session_state.get("custom_query", "")
custom_query = st.sidebar.text_area(
    "Custom prompt",
    value=default_query,
    height=180,
    placeholder="Enter any prompt you want to optimize for minimal token usage.",
)
token_budget = st.sidebar.number_input(
    "Token budget",
    min_value=50,
    value=int(task_meta["token_budget"]),
    step=10,
)
max_steps = st.sidebar.number_input(
    "Max steps",
    min_value=1,
    value=int(task_meta["max_steps"]),
    step=1,
)

st.session_state["custom_query"] = custom_query

sidebar_cols = st.sidebar.columns(2)
if sidebar_cols[0].button("Start / Reset", use_container_width=True):
    if not custom_query.strip():
        st.sidebar.error("Enter a custom prompt first.")
    else:
        start_episode(selected_task, custom_query.strip(), int(token_budget), int(max_steps))
        st.rerun()

if sidebar_cols[1].button("Refresh", use_container_width=True):
    st.rerun()

if "payload" not in st.session_state:
    st.info("Add your prompt in the sidebar and press Start / Reset.")
    st.stop()

payload = st.session_state["payload"]
observation = payload["observation"]

col1, col2, col3, col4 = st.columns(4)
col1.metric("Task", observation["task_name"])
col2.metric("Budget", observation["token_budget"])
col3.metric("Used", observation["total_tokens_used"])
col4.metric("Step", observation["step_number"])

st.subheader("Active Query")
st.info(observation["query"])

feedback = observation.get("last_action_feedback")
if feedback:
    st.warning(feedback)
if payload.get("info", {}).get("grader_breakdown"):
    st.success(f"Final score: {payload.get('reward', 0):.4f}")
    st.json(payload["info"]["grader_breakdown"])

action_cols = st.columns(3)
if action_cols[0].button("Auto Optimize Step", use_container_width=True):
    suggestion = api_post("/optimize-step")
    do_step(suggestion)
    st.rerun()
if action_cols[1].button("Auto Run", use_container_width=True):
    for _ in range(20):
        suggestion = api_post("/optimize-step")
        do_step(suggestion)
        if suggestion["action_type"] == "submit_answer" or st.session_state["payload"]["done"]:
            break
    st.rerun()

manual_answer = action_cols[2].text_input("Manual answer", value="")
if st.button("Submit Manual Answer", type="primary", use_container_width=True):
    do_step(
        {
            "action_type": "submit_answer",
            "answer": manual_answer.strip() or "Concise answer synthesized from the selected evidence.",
        }
    )
    st.rerun()

st.subheader("Available Chunks")
chunk_columns = st.columns(2)
for index, chunk in enumerate(observation["available_chunks"]):
    selected = chunk["chunk_id"] in set(observation["selected_chunks"])
    container = chunk_columns[index % 2].container(border=True)
    container.markdown(f"**{chunk['chunk_id']}**")
    container.caption(f"{chunk['domain']} | {chunk['tokens']} tokens")
    container.write(", ".join(chunk["keywords"]))
    c1, c2 = container.columns(2)
    if selected:
      if c1.button("Deselect", key=f"deselect-{chunk['chunk_id']}", use_container_width=True):
          do_step({"action_type": "deselect_chunk", "chunk_id": chunk["chunk_id"]})
          st.rerun()
    else:
      if c1.button("Select", key=f"select-{chunk['chunk_id']}", use_container_width=True):
          do_step({"action_type": "select_chunk", "chunk_id": chunk["chunk_id"]})
          st.rerun()
    if c2.button("Compress 50%", key=f"compress-{chunk['chunk_id']}", use_container_width=True):
        do_step(
            {
                "action_type": "compress_chunk",
                "chunk_id": chunk["chunk_id"],
                "compression_ratio": 0.5,
            }
        )
        st.rerun()

st.subheader("Observation")
st.json(payload)

st.subheader("State")
st.json(api_get("/state"))
