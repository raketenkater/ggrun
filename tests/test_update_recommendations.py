import importlib.util
from pathlib import Path


def load_updater():
    root = Path(__file__).resolve().parents[1]
    path = root / "tools" / "models" / "update_recommendations.py"
    spec = importlib.util.spec_from_file_location("update_recommendations", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parse_open_weight_next_payload():
    updater = load_updater()
    payload = (
        '<script>self.__next_f.push([1,"x:{\\"models\\":[{'
        '\\"id\\":\\"m1\\",\\"name\\":\\"Open Model\\",'
        '\\"slug\\":\\"open-model\\",\\"isOpenWeights\\":true,'
        '\\"intelligenceIndex\\":42.5,\\"huggingfaceUrl\\":\\"https://huggingface.co/org/model\\"'
        '},{\\"id\\":\\"m2\\",\\"name\\":\\"Closed Model\\",'
        '\\"slug\\":\\"closed-model\\",\\"isOpenWeights\\":false}]}"'
        '])</script>'
    )
    rows = updater.parse_open_weight_page_models(payload)
    assert len(rows) == 1
    assert rows[0]["name"] == "Open Model"
    assert updater.huggingface_repo(rows[0]) == "org/model"


def test_representative_size_prefers_q4_over_q8():
    updater = load_updater()
    quants = [
        {"name": "Q2_K_XL", "size_gb": 10},
        {"name": "Q8_0", "size_gb": 30},
        {"name": "Q4_K_XL", "size_gb": 18},
    ]
    assert updater.representative_size_gb(quants) == 18
