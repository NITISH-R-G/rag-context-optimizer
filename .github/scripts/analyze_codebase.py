"""Module for GitHub Actions automation."""
# pylint: disable=line-too-long,import-outside-toplevel,missing-function-docstring,redefined-outer-name,too-many-nested-blocks,duplicate-code
import ast
import os
import json
import re


def parse_python_file(filepath):
    """Parses a Python file and extracts its AST, catching basic syntax errors."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            source = f.read()
        return ast.parse(source), source
    except Exception as e:  # pylint: disable=broad-exception-caught
        print(f"Error parsing {filepath}: {e}")
        return None, None


def extract_dependencies(tree):
    """Extracts imported modules from AST."""
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.append(node.module.split(".")[0])
    return list(set(imports))


def extract_endpoints(tree):
    """Detects FastAPI endpoints and their methods."""
    endpoints = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            for decorator in node.decorator_list:
                if isinstance(decorator, ast.Call) and isinstance(
                    decorator.func, ast.Attribute
                ):
                    if decorator.func.attr in ["get", "post", "put", "delete", "patch"]:
                        path = ""
                        if decorator.args and isinstance(
                            decorator.args[0], ast.Constant
                        ):
                            path = decorator.args[0].value
                        endpoints.append(
                            {
                                "method": decorator.func.attr.upper(),
                                "path": path,
                                "function": node.name,
                            }
                        )
    return endpoints


def extract_classes_functions(tree):
    """Extracts class and function definitions."""
    entities = {"classes": [], "functions": []}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            entities["classes"].append(node.name)
        elif isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
            entities["functions"].append(node.name)
    return entities


def extract_env_vars(source):
    """Extracts environment variables requested via os.getenv or os.environ."""
    env_vars = set()
    env_vars.update(re.findall(r'os\.getenv\([\'"]([A-Z0-9_]+)[\'"]', source))
    env_vars.update(re.findall(r'os\.environ\.get\([\'"]([A-Z0-9_]+)[\'"]', source))
    return list(env_vars)


def get_project_metadata():
    """Reads pyproject.toml to extract project info and top-level dependencies."""
    metadata = {"dependencies": [], "name": "", "version": ""}
    try:
        with open("pyproject.toml", "r", encoding="utf-8") as f:
            content = f.read()
            name_match = re.search(r'name\s*=\s*"([^"]+)"', content)
            if name_match:
                metadata["name"] = name_match.group(1)

            deps_match = re.search(r"dependencies\s*=\s*\[(.*?)\]", content, re.DOTALL)
            if deps_match:
                deps_str = deps_match.group(1)
                deps = re.findall(r'"([^"=><]+)[^"]*"', deps_str)
                metadata["dependencies"] = deps
    except Exception as e:  # pylint: disable=broad-exception-caught
        print(f"Error reading pyproject.toml: {e}")
    return metadata


def analyze_codebase(root_dir="."):
    """Walks the codebase to extract architectural details."""
    graph = {
        "files": {},
        "frameworks": set(),
        "external_deps": set(),
        "internal_imports": set(),
        "api_endpoints": [],
        "env_vars": set(),
        "metadata": get_project_metadata(),
    }

    graph["external_deps"].update(graph["metadata"]["dependencies"])

    for subdir, _, files in os.walk(root_dir):
        if (
            ".git" in subdir
            or ".github" in subdir
            or "venv" in subdir
            or "env" in subdir
            and "env" != subdir[2:]
        ):
            if "env" not in subdir:  # Allow scanning the actual env package if any
                continue

        for file in files:
            if file.endswith(".py"):
                filepath = os.path.join(subdir, file)
                rel_path = os.path.relpath(filepath, root_dir)

                tree, source = parse_python_file(filepath)
                if not tree:
                    continue

                imports = extract_dependencies(tree)
                endpoints = extract_endpoints(tree)
                entities = extract_classes_functions(tree)
                envs = extract_env_vars(source)

                file_info = {
                    "imports": imports,
                    "endpoints": endpoints,
                    "classes": entities["classes"],
                    "functions": entities["functions"],
                    "env_vars": envs,
                }

                graph["files"][rel_path] = file_info

                # Check for frameworks based on imports
                if "fastapi" in imports:
                    graph["frameworks"].add("FastAPI")
                if "streamlit" in imports:
                    graph["frameworks"].add("Streamlit")
                if "pydantic" in imports:
                    graph["frameworks"].add("Pydantic")

                graph["api_endpoints"].extend(endpoints)
                graph["env_vars"].update(envs)

                # Heuristics for internal vs external imports
                for imp in imports:
                    # Very simple heuristic: if it matches a top-level dir/file it's internal
                    if os.path.exists(imp) or os.path.exists(f"{imp}.py"):
                        graph["internal_imports"].add(imp)
                    else:
                        if imp not in [
                            "os",
                            "sys",
                            "json",
                            "re",
                            "pathlib",
                            "typing",
                            "ast",
                            "dataclasses",
                            "contextlib",
                        ]:
                            graph["external_deps"].add(imp)

    # Convert sets to lists for JSON serialization
    graph["frameworks"] = list(graph["frameworks"])
    graph["external_deps"] = list(graph["external_deps"])
    graph["internal_imports"] = list(graph["internal_imports"])
    graph["env_vars"] = list(graph["env_vars"])

    return graph


if __name__ == "__main__":
    graph = analyze_codebase()
    os.makedirs("docs", exist_ok=True)
    with open("docs/knowledge_graph.json", "w", encoding="utf-8") as f:
        json.dump(graph, f, indent=2)
    print("Knowledge graph written to docs/knowledge_graph.json")
