import os
import logging
import subprocess

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def read_file(filepath):
    """Safely read a text file, returning an empty string if it doesn't exist."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        logging.warning(f"Could not read {filepath}: {e}")
        return ""


def load_reports():
    """Load the output reports from various QA tools."""
    reports = {
        "ruff": read_file("reports/ruff.txt"),
        "mypy": read_file("reports/mypy.txt"),
        "vulture": read_file("reports/vulture.txt"),
        "pylint": read_file("reports/pylint.txt"),
        "radon": read_file("reports/radon.json"),
        "bandit": read_file("reports/bandit.json"),
        "pip_audit": read_file("reports/pip-audit.json"),
        "safety": read_file("reports/safety.json"),
        "coverage": read_file("reports/coverage.txt"),
        "pip_licenses": read_file("reports/pip-licenses.json"),
    }
    return reports


def build_prompt(reports):
    """Build the prompt for the LLM to generate the insights report."""
    prompt = f"""
    You are an expert AI repository maintainer and QA engineer.
    Your task is to review the current repository state based on the automated QA tools reports and generate a comprehensive AI-powered insight report.

    Here are the limited extracts from the reports (truncated to avoid context limits):

    1. Ruff (Linting & Formatting):
    {reports["ruff"][:2000]}

    2. Mypy (Type Checking):
    {reports["mypy"][:2000]}

    3. Vulture (Dead Code):
    {reports["vulture"][:2000]}

    4. Pylint (Duplicate Code / Others):
    {reports["pylint"][:2000]}

    5. Radon (Complexity):
    {reports["radon"][:2000]}

    6. Bandit (Security Scanning):
    {reports["bandit"][:2000]}

    7. Pip Audit (Dependency Security):
    {reports["pip_audit"][:2000]}

    8. Safety (Dependency Security):
    {reports["safety"][:2000]}

    9. Pytest Coverage:
    {reports["coverage"][:2000]}

    10. License Compliance:
    {reports["pip_licenses"][:2000]}

    Please generate a Markdown report with the following sections:
    - **Executive Summary**: A brief overview of the codebase quality.
    - **Key Findings in Plain English**: Explain the most critical issues from the reports.
    - **Prioritized Issues by Severity**: List issues categorized as Critical, High, Medium, Low.
    - **Recommended Fixes**: Specific advice on how to fix the issues.
    - **Refactoring Opportunities**: Suggestions for improving code structure or reducing complexity.
    - **Architectural Concerns**: Highlight any high-level design issues.
    - **Actionable Maintenance Tasks**: A bulleted list of tasks for the maintainers to work on.

    Ensure the output is high-quality Markdown.
    """
    return prompt


def generate_insights(api_key, reports):
    """Call OpenAI API to generate insights based on reports."""
    try:
        from openai import OpenAI
    except ImportError:
        logging.error("OpenAI package not installed.")
        return "No OpenAI package installed. AI insights skipped."

    client = OpenAI(api_key=api_key)
    prompt = build_prompt(reports)

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a senior QA engineer and software architect. Provide a clear, actionable markdown report based on the provided QA data.",
                },
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content
    except Exception as e:
        logging.error(f"Error calling OpenAI API: {e}")
        return f"Error generating insights: {e}"


def post_pr_comment(markdown_report):
    """Post the generated markdown report as a PR comment using the GitHub CLI."""
    # Getting PR number from a generic environment variable we can pass in Actions
    pr_number = os.getenv("PR_NUMBER")

    if pr_number:
        logging.info(f"Posting comment to PR {pr_number}")
        try:
            with open("temp_report.md", "w") as f:
                f.write(markdown_report)

            # Use gh cli to comment
            subprocess.run(
                [
                    "/usr/bin/env",
                    "gh",
                    "pr",
                    "comment",
                    pr_number,
                    "-F",
                    "temp_report.md",
                ],
                check=True,
                shell=False,
            )
            os.remove("temp_report.md")
        except Exception as e:
            logging.error(f"Error posting PR comment: {e}")
    else:
        logging.info("Not a PR or PR_NUMBER not set, skipping PR comment.")


def main():
    api_key = os.getenv("OPENAI_API_KEY")
    reports = load_reports()

    if not api_key:
        logging.warning(
            "No OPENAI_API_KEY found. Generating a basic AI insights template."
        )
        markdown_report = "# AI Quality Insights\n\nNo OpenAI API key provided. Skipping detailed analysis."
    else:
        logging.info("Generating AI insights via OpenAI...")
        markdown_report = generate_insights(api_key, reports)

    os.makedirs("reports", exist_ok=True)
    with open("reports/ai_insights.md", "w") as f:
        f.write(markdown_report)
    logging.info("Wrote AI insights to reports/ai_insights.md")

    # Post to PR if PR_NUMBER is available
    post_pr_comment(markdown_report)

    # Output to GitHub Actions step summary if available
    github_step_summary = os.getenv("GITHUB_STEP_SUMMARY")
    if github_step_summary:
        try:
            with open(github_step_summary, "a") as f:
                f.write(markdown_report)
        except Exception as e:
            logging.warning(f"Could not write to GITHUB_STEP_SUMMARY: {e}")


if __name__ == "__main__":
    main()
