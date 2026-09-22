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


def test_repo_candidates_use_bare_model_name():
    """Artificial Analysis dropped the HF repo link in September 2026.

    With no link, the repo name is rebuilt from the row. display_name()
    prefixes the creator ("Z AI GLM-5.3-Flash"), which names no real repo, so
    every model whose creator is not already part of its name silently lost its
    GGUF match and vanished from the catalog (142 candidates -> 72). The bare
    model name must be tried, and tried before the prefixed spelling.
    """
    updater = load_updater()
    row = {
        "name": "GLM-5.3-Flash",
        "slug": "glm-5-3-flash",
        "isOpenWeights": True,
        "intelligenceIndex": 41.8,
        "creator": {"name": "Z AI", "slug": "zai"},
    }
    assert updater.bare_model_name(row) == "GLM-5.3-Flash"
    # display_name still carries the creator; that is what broke the lookup.
    assert updater.display_name(row) == "Z AI GLM-5.3-Flash"

    repos = updater.direct_repo_candidates(row)
    assert "unsloth/GLM-5.3-Flash-GGUF" in repos, repos
    # resolve_gguf_repo only inspects the first four candidates, so the correct
    # bare spelling has to outrank the creator-prefixed one.
    assert repos.index("unsloth/GLM-5.3-Flash-GGUF") < repos.index(
        "unsloth/Z-AI-GLM-5.3-Flash-GGUF"
    ), repos
    assert "GLM-5.3-Flash GGUF" in updater.search_queries(row)


def test_repo_candidates_keep_creator_named_models():
    """A creator that is already part of the model name must not regress."""
    updater = load_updater()
    row = {
        "name": "DeepSeek V4.1 Flash",
        "slug": "deepseek-v4-1-flash",
        "isOpenWeights": True,
        "intelligenceIndex": 39.5,
        "creator": {"name": "DeepSeek", "slug": "deepseek"},
    }
    repos = updater.direct_repo_candidates(row)
    assert "unsloth/DeepSeek-V4.1-Flash-GGUF" in repos, repos
