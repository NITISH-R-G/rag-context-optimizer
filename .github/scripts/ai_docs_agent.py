import os
import json
import logging
from typing import Dict, Any

# Configure simple logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def get_repo_context() -> Dict[str, Any]:
    """Loads repository context from generated files."""
    context = {}
    try:
        with open("docs/knowledge_graph.json", "r") as f:
            context['graph'] = json.load(f)
    except Exception as e:
         logging.warning(f"Could not load knowledge graph: {e}")
         context['graph'] = {}

    try:
        with open("docs/architecture.md", "r") as f:
            context['architecture'] = f.read()
    except Exception as e:
         logging.warning(f"Could not load architecture diagrams: {e}")
         context['architecture'] = ""

    try:
        with open("README.md", "r") as f:
            context['readme'] = f.read()
    except Exception as e:
         logging.warning(f"Could not load README: {e}")
         context['readme'] = ""

    return context

def build_prompt(context: Dict[str, Any]) -> str:
    """Builds a prompt for the AI to update documentation."""
    graph = context.get('graph', {})

    prompt = f"""
    You are an expert AI repository maintainer.
    Your task is to review the current repository state and generate completely updated documentation.

    Here is the Knowledge Graph for the repository:
    {json.dumps(graph, indent=2)}

    Here are the auto-generated Architecture Diagrams:
    {context.get('architecture', '')}

    Current README.md:
    {context.get('readme', '')}

    Please output a JSON object containing three keys:
    1. 'readme': A completely rewritten README.md that includes status badges, tech stack, system architecture, API documentation, and setup instructions. It MUST incorporate the architecture diagrams.
    2. 'contributing': A new or updated docs/contributing.md file.
    3. 'onboarding': A new or updated docs/onboarding.md file explaining the codebase to new developers.

    Ensure the output is high-quality Markdown, ready to be committed.
    """
    return prompt

def generate_documentation(api_key: str, context: Dict[str, Any]) -> Dict[str, str]:
    """Uses OpenAI API to generate new documentation."""
    try:
        from openai import OpenAI
    except ImportError:
        logging.error("OpenAI package not installed.")
        return {}

    client = OpenAI(api_key=api_key)
    prompt = build_prompt(context)

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini", # Standard faster model for doc generation
            messages=[
                {"role": "system", "content": "You are a senior technical writer and software architect. Return ONLY valid JSON matching the requested keys."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )
        content = response.choices[0].message.content
        if content:
             return json.loads(content)
        return {}
    except Exception as e:
        logging.error(f"Error calling OpenAI API: {e}")
        return {}

def main():
    api_key = os.getenv("OPENAI_API_KEY")
    context = get_repo_context()

    if not api_key:
         logging.warning("No OPENAI_API_KEY found. AI Documentation generation skipped.")
         # In a real run, this might fail or we could generate a deterministic basic template
         # For this environment, we'll write a placeholder if API key is absent
         docs = {
             "contributing": "# Contributing\n\nWelcome to the project! Please follow standard PR guidelines.",
             "onboarding": "# Developer Onboarding\n\nReview the `docs/architecture.md` file to get started."
         }
    else:
        logging.info("Calling OpenAI API to generate docs...")
        docs = generate_documentation(api_key, context)

    # Write outputs
    if docs.get("readme"):
        with open("README.md", "w") as f:
            f.write(docs["readme"])
        logging.info("Updated README.md")

    if docs.get("contributing"):
        with open("docs/contributing.md", "w") as f:
            f.write(docs["contributing"])
        logging.info("Updated docs/contributing.md")

    if docs.get("onboarding"):
        with open("docs/onboarding.md", "w") as f:
            f.write(docs["onboarding"])
        logging.info("Updated docs/onboarding.md")

if __name__ == "__main__":
    main()
