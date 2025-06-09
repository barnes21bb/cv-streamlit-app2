import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.annotation_utils import generate_pascal_voc_xml


def test_generate_pascal_voc_xml_single():
    anns = {0: [{"class": "good-cup", "bbox": [1, 2, 3, 4]}]}
    xml_map = generate_pascal_voc_xml(anns, "video", (10, 10, 3))
    assert 0 in xml_map
    assert "<name>good-cup</name>" in xml_map[0]


def test_generate_pascal_voc_xml_empty():
    assert generate_pascal_voc_xml({}, "video", (10, 10, 3)) == {}
