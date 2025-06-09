import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pathlib import Path

from src.utils import check_file_size, WARNING_SIZE_BYTES, MAX_SIZE_BYTES
from src.database import get_project_videos


def test_check_file_size_ok():
    assert check_file_size(WARNING_SIZE_BYTES - 1) == "ok"


def test_check_file_size_warn():
    assert check_file_size(WARNING_SIZE_BYTES + 1) == "warn"


def test_check_file_size_reject():
    assert check_file_size(MAX_SIZE_BYTES + 1) == "reject"


def test_get_project_videos_extensions(tmp_path, monkeypatch):
    from src import database as db

    monkeypatch.setattr(db, "STORAGE_DIR", Path(tmp_path))
    proj_dir = db.STORAGE_DIR / "user_1" / "project_1"
    proj_dir.mkdir(parents=True)
    exts = [".mp4", ".avi", ".mov", ".mkv", ".m4v", ".3gp", ".txt"]
    for ext in exts:
        (proj_dir / f"test{ext}").touch()

    videos = get_project_videos(1, 1)
    suffixes = sorted(p.suffix for p in videos)
    assert ".m4v" in suffixes
    assert ".3gp" in suffixes
    assert ".txt" not in suffixes
