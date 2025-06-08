import ast
import io
import zipfile
from pathlib import Path


def load_function(func_name):
    path = Path(__file__).resolve().parents[1] / "video-annotation-app.py"
    source = path.read_text()
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            module = {}
            exec(compile(ast.Module([node], []), filename=str(path), mode="exec"), globals(), module)
            return module[func_name]
    raise RuntimeError(f"{func_name} not found")


create_zip_with_video_and_xml = load_function("create_zip_with_video_and_xml")


def test_zip_contains_video_and_xml(tmp_path):
    video_path = tmp_path / "dummy.mp4"
    # create small dummy file to represent video
    video_path.write_bytes(b"dummy video data")

    xml_map = {0: "<annotation></annotation>"}
    zip_bytes = create_zip_with_video_and_xml(str(video_path), xml_map)

    with zipfile.ZipFile(io.BytesIO(zip_bytes), 'r') as zipf:
        names = zipf.namelist()
    assert any(name.endswith('.mp4') for name in names)
    assert any(name.endswith('.xml') for name in names)

