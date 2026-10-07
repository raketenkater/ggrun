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


def test_identity_rejects_other_generation_and_owner_digits():
    """Mistral Small 4 was resolved to a Small-24B *Base* 2501 repo whose owner,
    DevQuasar-4, supplied the "4"."""
    updater = load_updater()
    row = {"name": "Mistral Small 4", "slug": "mistral-small-4", "creator": {"name": "Mistral", "slug": "mistral"}}
    assert not updater.candidate_relevant("DevQuasar-4/mistralai.Mistral-Small-24B-Base-2501-GGUF", row)
    assert updater.candidate_relevant("unsloth/Mistral-Small-4-119B-2603-GGUF", row)


def test_identity_rejects_a_different_variant():
    """Devstral 2 was resolved to Devstral Small 2."""
    updater = load_updater()
    row = {"name": "Devstral 2", "slug": "devstral-2", "creator": {"name": "Mistral", "slug": "mistral"}}
    assert not updater.candidate_relevant("bartowski/mistralai_Devstral-Small-2-24B-Instruct-2512-GGUF", row)
    assert updater.candidate_relevant("unsloth/Devstral-2-123B-Instruct-2512-GGUF", row)
    small = {"name": "Devstral Small 2", "slug": "devstral-small-2", "creator": {"name": "Mistral", "slug": "mistral"}}
    assert updater.candidate_relevant("bartowski/mistralai_Devstral-Small-2-24B-Instruct-2512-GGUF", small)


def test_identity_keeps_valid_spellings():
    updater = load_updater()
    cases = [
        ({"name": "GLM-5.3-Flash", "creator": {"name": "Z AI"}}, "unsloth/GLM-5.3-Flash-GGUF", True),
        ({"name": "GLM-5.3-Flash", "creator": {"name": "Z AI"}}, "unsloth/GLM-5.3-GGUF", False),
        ({"name": "DeepSeek V4.1 Flash", "creator": {"name": "DeepSeek"}}, "unsloth/DeepSeek-V4.1-Flash-GGUF", True),
        ({"name": "Qwen3.6 27B", "creator": {"name": "Alibaba"}}, "unsloth/Qwen3.6-27B-GGUF", True),
        ({"name": "Qwen3.6 27B", "creator": {"name": "Alibaba"}}, "unsloth/Qwen3.6-Coder-27B-GGUF", False),
        ({"name": "MiniMax M3", "creator": {"name": "MiniMax"}}, "unsloth/MiniMax-M3-GGUF", True),
    ]
    for row, repo, want in cases:
        assert updater.candidate_relevant(repo, row) is want, (row["name"], repo)


def test_hyphenated_size_is_not_a_version_mismatch():
    updater = load_updater()
    row = {"name": "Qwen3.6 27B", "creator": {"name": "Alibaba"}}
    assert updater.candidate_relevant("unsloth/Qwen3.6-27B-GGUF", row)
    assert not updater.candidate_relevant("unsloth/Qwen3.5-27B-GGUF", row)


def test_lineage_notes_are_not_variant_qualifiers():
    updater = load_updater()
    row = {"name": "INTELLECT-3 (based on GLM-4.5-Air)", "creator": {"name": "Prime Intellect"}}
    assert updater.candidate_relevant("bartowski/PrimeIntellect_INTELLECT-3-GGUF", row)


def test_modified_derivatives_do_not_inherit_the_original_score():
    updater = load_updater()
    row = {"name": "Mistral Small 4 (Reasoning)", "creator": {"name": "Mistral"}}
    assert not updater.candidate_relevant("timteh673/Mistral-Small-4-119B-Uncensored-GGUF", row)
    magistral = {"name": "Magistral Small 1.2", "creator": {"name": "Mistral"}}
    assert not updater.candidate_relevant("mradermacher/Magistral-Small-2509-Heretic-v1.2-i1-GGUF", magistral)


def test_quant_suffix_and_retune_prefixes_are_not_identity():
    updater = load_updater()
    small4 = {"name": "Mistral Small 4 (Reasoning)", "creator": {"name": "Mistral"}}
    assert not updater.candidate_relevant("pipilok/Mistral-Small-Instruct-2409-Q4_0_4_8-GGUF", small4)
    omni = {"name": "Qwen3 Omni 30B A3B Instruct", "creator": {"name": "Alibaba"}}
    assert not updater.candidate_relevant("mradermacher/MANGO-Qwen3-Omni-30B-A3B-Instruct-i1-GGUF", omni)
    assert updater.candidate_relevant("unsloth/Qwen3-Omni-30B-A3B-Instruct-GGUF", omni)
    for row, repo in [
        ({"name": "LFM2.5-2.6B", "creator": {"name": "Liquid AI"}}, "bartowski/LiquidAI_LFM2.5-2.6B-GGUF"),
        ({"name": "Devstral Small 2", "creator": {"name": "Mistral"}}, "bartowski/mistralai_Devstral-Small-2-24B-Instruct-2512-GGUF"),
        ({"name": "Llama 3.1 Instruct 405B", "creator": {"name": "Meta"}}, "ThomasBaruzier/Meta-Llama-3.1-405B-Instruct-GGUF"),
    ]:
        assert updater.candidate_relevant(repo, row), repo


ARCH_SOURCE = '''
static const std::map<llm_arch, const char *> LLM_ARCH_NAMES = {
    { LLM_ARCH_LLAMA,           "llama"        },
    { LLM_ARCH_K2_HORIZON,      "k2-horizon"   },
    { LLM_ARCH_UNKNOWN,         "(unknown)"    },
};
static const std::map<llm_kv, const char *> LLM_KV_NAMES = {
    { LLM_KV_GENERAL_TYPE, "general.type" },
};
'''


def test_parse_arch_names_reads_only_the_arch_table():
    updater = load_updater()
    assert updater.parse_arch_names(ARCH_SOURCE) == {"llama", "k2-horizon", "(unknown)"}
    assert updater.parse_arch_names("no table here") == set()


def test_stamp_runnable_marks_rows_no_upstream_backend_loads():
    updater = load_updater()
    rows = [{"arch": "llama"}, {"arch": "K2-Horizon"}, {"arch": "axk2"}, {"name": "legacy row"}]
    updater.stamp_runnable(rows, {"llama", "k2-horizon"})
    assert [row.get("runnable") for row in rows] == [True, True, False, None]


def test_unreadable_upstream_table_stamps_nothing(monkeypatch):
    updater = load_updater()

    def fetch_text(url, **_):
        if "ik_llama" in url:
            raise OSError("offline")
        return ARCH_SOURCE

    monkeypatch.setattr(updater, "fetch_text", fetch_text)
    # A partial union would hide rows only the missing backend loads.
    assert updater.fetch_upstream_arches() is None
    rows = [{"arch": "axk2"}]
    updater.stamp_runnable(rows, updater.fetch_upstream_arches())
    assert "runnable" not in rows[0]


def _sibling(name, size):
    return {"rfilename": name, "lfs": {"size": size}}


def test_draft_heads_and_projectors_are_not_model_quants(monkeypatch):
    updater = load_updater()
    gb = 1024**3
    siblings = [
        _sibling("Model-MXFP4-00001-of-00002.gguf", 10 * 1024**2),
        _sibling("Model-MXFP4-00002-of-00002.gguf", 500 * gb),
        _sibling("Model-Q6_K-00001-of-00002.gguf", 300 * gb),
        _sibling("Model-Q6_K-00002-of-00002.gguf", 300 * gb),
        _sibling("mtp-Model-Q4_0.gguf", 2 * gb),
        _sibling("mtp-Model-BF16.gguf", 7 * gb),
        _sibling("dflash-Model-BF16.gguf", 5 * gb),
        _sibling("MTP/mtp-Model-shared-Q8_0.gguf", 3 * gb),
        _sibling("mmproj-Model-BF16.gguf", 3 * gb),
    ]
    monkeypatch.setattr(updater, "fetch_hf_model_info", lambda repo: {"siblings": siblings})
    quants = {q["name"]: q["size_bytes"] for q in updater.fetch_hf_quants("owner/Model-GGUF")}
    assert quants == {"MXFP4": 500 * gb + 10 * 1024**2, "Q6_K": 600 * gb}
    assert updater._representative_gguf_file(siblings) == "Model-MXFP4-00001-of-00002.gguf"


def test_a_draft_only_repo_cannot_supply_a_scored_main_model(monkeypatch):
    updater = load_updater()
    siblings = [_sibling("dflash-Model-BF16.gguf", 5 * 1024**3), _sibling("dflash-Model-Q8_0.gguf", 3 * 1024**3)]
    monkeypatch.setattr(updater, "fetch_hf_model_info", lambda repo: {"siblings": siblings})
    assert updater.fetch_hf_quants("owner/Model-DFlash-GGUF") == []
    assert updater._representative_gguf_file(siblings) is None


def test_quant_identity_uses_only_the_final_label(monkeypatch):
    updater = load_updater()
    siblings = [_sibling("Model-BF16-Q4_K_M.gguf", 4 * 1024**3),
                _sibling("Model-BF16-Q8_0.gguf", 8 * 1024**3),
                _sibling("UD-IQ3_XXS/Model-UD-IQ3_XXS.gguf", 3 * 1024**3)]
    monkeypatch.setattr(updater, "fetch_hf_model_info", lambda repo: {"siblings": siblings})
    quants = {q["name"]: q["size_bytes"] for q in updater.fetch_hf_quants("owner/Model-BF16-GGUF")}
    assert quants == {"Q4_K_M": 4 * 1024**3, "Q8_0": 8 * 1024**3, "UD-IQ3_XXS": 3 * 1024**3}


def test_partial_shards_and_unknown_sizes_never_look_like_small_models(monkeypatch):
    updater = load_updater()
    siblings = [_sibling("Model-Q4_K_M-00001-of-00002.gguf", 1024**3),
                _sibling("Model-Q8_0-00001-of-00002.gguf", 1024**3),
                {"rfilename": "Model-Q8_0-00002-of-00002.gguf"},
                _sibling("Model-Q6_K.gguf", 6 * 1024**3)]
    monkeypatch.setattr(updater, "fetch_hf_model_info", lambda repo: {"siblings": siblings})
    assert [q["name"] for q in updater.fetch_hf_quants("owner/Model")] == ["Q6_K"]


def test_distinct_models_with_same_quant_are_not_summed(monkeypatch):
    updater = load_updater()
    siblings = [_sibling("Model-2B-Q4_K_M.gguf", 1024**3),
                _sibling("Model-20B-Q4_K_M.gguf", 10 * 1024**3)]
    monkeypatch.setattr(updater, "fetch_hf_model_info", lambda repo: {"siblings": siblings})
    assert updater.fetch_hf_quants("owner/Model") == []


def test_unknown_size_does_not_hide_an_ambiguous_variant(monkeypatch):
    updater = load_updater()
    siblings = [_sibling("Model-2B-Q4_K_M.gguf", 1024**3),
                _sibling("Model-20B-Q4_K_M.gguf", None)]
    monkeypatch.setattr(updater, "fetch_hf_model_info", lambda repo: {"siblings": siblings})
    assert updater.fetch_hf_quants("owner/Model") == []


def test_search_cannot_promote_a_draft_through_base_model_tags(monkeypatch):
    updater = load_updater()
    row = {"name": "Qwen3.8-Flash-Next", "creator": {"name": "Alibaba"},
           "huggingfaceUrl": "https://huggingface.co/Qwen/Qwen3.8-Flash-Next"}
    draft = "unsloth/Qwen3.8-Flash-Next-MTP-GGUF"
    assert not updater.candidate_relevant(draft, row)
    inspected = []
    monkeypatch.setattr(updater, "search_hf_models", lambda *args: [
        {"id": draft, "tags": ["base_model:Qwen/Qwen3.8-Flash-Next"]}])
    def quants(repo):
        inspected.append(repo)
        return [{"name": "Q4_K_M", "size_gb": 2}] if repo == draft else []
    monkeypatch.setattr(updater, "fetch_hf_quants", quants)
    assert updater.resolve_gguf_repo(row, 20) is None
    assert draft not in inspected


def test_shard_completeness_requires_each_index_once():
    updater = load_updater()
    assert updater.complete_shards(["m-00001-of-00002.gguf", "m-00002-of-00002.gguf"])
    assert not updater.complete_shards(["m-00001-of-00002.gguf"])
    assert not updater.complete_shards(["m-00001-of-00002.gguf", "m-00001-of-00002.gguf"])
    assert not updater.complete_shards(["m-00000-of-00002.gguf", "m-00002-of-00002.gguf"])
    assert not updater.complete_shards(["m-00001-of-00002.gguf", "m-00002-of-00003.gguf"])


def test_gguf_importance_matrix_is_not_a_tiny_bf16_model(monkeypatch):
    updater = load_updater()
    siblings = [_sibling("GLM-5.3-Flash-BF16-imatrix.gguf", 512687648),
                _sibling("GLM-5.3-Flash-BF16-Q4_K_M.gguf", 200828233184)]
    monkeypatch.setattr(updater, "fetch_hf_model_info", lambda repo: {"siblings": siblings})
    assert [(q["name"], q["size_bytes"]) for q in updater.fetch_hf_quants("owner/GLM")] == [
        ("Q4_K_M", 200828233184)]
    assert updater._representative_gguf_file(siblings) == siblings[1]["rfilename"]
