import ast
from pathlib import Path
import pytest


def load_check_file_size():
    path = Path(__file__).resolve().parents[1] / "video-annotation-app.py"
    source = path.read_text()
    tree = ast.parse(source)
    nodes = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            if isinstance(node.targets[0], ast.Name) and node.targets[0].id in {"WARNING_SIZE_BYTES", "MAX_SIZE_BYTES"}:
                nodes.append(node)
        if isinstance(node, ast.FunctionDef) and node.name == "check_file_size":
            nodes.append(node)
    module = {}
    exec(compile(ast.Module(nodes, []), filename=str(path), mode="exec"), module)
    return module["check_file_size"], module["WARNING_SIZE_BYTES"], module["MAX_SIZE_BYTES"]


def load_get_project_videos(tmpdir):
    path = Path(__file__).resolve().parents[1] / "video-annotation-app.py"
    source = path.read_text()
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "get_project_videos":
            module = {"STORAGE_DIR": Path(tmpdir), "Path": Path}
            exec(compile(ast.Module([node], []), filename=str(path), mode="exec"), module)
            return module["get_project_videos"]
    raise RuntimeError("get_project_videos function not found")


check_file_size, WARNING_SIZE_BYTES, MAX_SIZE_BYTES = load_check_file_size()


def test_check_file_size_ok():
    assert check_file_size(WARNING_SIZE_BYTES - 1) == "ok"


def test_check_file_size_warn():
    assert check_file_size(WARNING_SIZE_BYTES + 1) == "warn"


def test_check_file_size_reject():
    assert check_file_size(MAX_SIZE_BYTES + 1) == "reject"


def test_get_project_videos_extensions(tmp_path):
    get_videos = load_get_project_videos(tmp_path)
    proj_dir = tmp_path / "user_1" / "project_1"
    proj_dir.mkdir(parents=True)
    exts = [".mp4", ".avi", ".mov", ".mkv", ".m4v", ".3gp", ".txt"]
    for ext in exts:
        (proj_dir / f"test{ext}").touch()

    videos = get_videos(1, 1)
    suffixes = sorted(p.suffix for p in videos)
    assert ".m4v" in suffixes
    assert ".3gp" in suffixes
    assert ".txt" not in suffixes
