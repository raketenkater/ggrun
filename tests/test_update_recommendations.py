import importlib.util
import math
import json
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


def test_current_table_spans_flight_chunks_and_preserves_quoted_names():
    updater = load_updater()
    row = {"slug": "open-model", "name": 'Open "Reasoning" Model',
           "isOpenWeights": True, "modelCreatorName": "Creator", "intelligenceIndex": 42.5}
    wire = "x:" + json.dumps({
        "models": [{"slug": "navigation-only", "name": "Not a scored model"}],
        "initialModels": [row, {"name": "Closed model", "isOpenWeights": False}],
    })
    cut = wire.index("Reasoning") + 3
    page = "".join('<script>self.__next_f.push(' + json.dumps([1, chunk]) + ')</script>'
                   for chunk in [wire[:cut], wire[cut:]])
    rows = updater.parse_open_weight_page_models(page)
    assert rows == [row]
    assert updater.model_creator_name(rows[0]) == "Creator"
    assert updater.intelligence(rows[0]) == 42.5


def test_plain_json_table_and_malformed_transport():
    updater = load_updater()
    row = {"name": "Open model", "isOpenWeights": True, "intelligenceIndex": 30}
    page = '<script type="application/json">' + json.dumps({"models": [row]}) + '</script>'
    assert updater.parse_open_weight_page_models(page) == [row]
    assert updater.parse_open_weight_page_models('<script>self.__next_f.push(notJSON)</script>') == []


def test_flattened_row_keeps_gguf_experts_and_artifact_parameter_count(monkeypatch):
    updater = load_updater()
    monkeypatch.setattr(updater, "fetch_gguf_arch", lambda repo: {"arch": "glm-dsa", "experts": 256, "exp_used": 8})
    monkeypatch.setattr(updater, "fetch_hf_model_info", lambda repo: {"gguf": {"total": 753864139008, "totalFileSize": 435200000000, "architecture": "glm-dsa"}})
    row = {"name": "GLM-5.3", "isOpenWeights": True, "intelligenceIndex": 44.7}
    cand = updater.candidate_from_row(row, "org/model", [{"name": "Q4_K_M", "size_gb": 435.2, "size_bytes": 435200000000}], "direct")
    assert cand["moe"] is True
    assert cand["total_params_b"] == 753.864
    assert cand["active_params_b"] == 0  # not total * 8 / 256
    assert cand["experts"] == 256


def test_shape_preserves_known_counts_and_dense_or_missing_evidence():
    updater = load_updater()
    shape = updater.candidate_model_shape({"totalParameters": 744, "activeParameters": 40}, {}, {"gguf": {"total": 753864139008}}, [])
    assert shape == {"total_params_b": 744, "active_params_b": 40, "moe": True}
    for experts in (0, 1):
        shape = updater.candidate_model_shape({}, {"experts": experts}, {"gguf": {"total": 27000000000, "totalFileSize": 15000000000}}, [{"size_bytes": 15000000000}])
        assert shape == {"total_params_b": 27, "active_params_b": 0, "moe": False}
    assert updater.candidate_model_shape({}, {}, {}, []) == {"total_params_b": 0, "active_params_b": 0, "moe": False}
    for invalid in (True, "27B", float("nan"), float("inf"), -1):
        assert updater.candidate_model_shape({}, {}, {"gguf": {"total": invalid, "totalFileSize": 1}}, [{"size_bytes": 1}])["total_params_b"] == 0


def test_repo_summary_of_a_companion_cannot_supply_main_parameter_count():
    updater = load_updater()
    arch = {"arch": "nemotron_h_moe", "experts": 128}
    quants = [{"size_bytes": 46000000000}]
    for summary in [
        {"total": 1632922404, "architecture": "dflash", "totalFileSize": 1290795104},
        {"total": 1632922404, "architecture": "nemotron_h_moe", "totalFileSize": 1290795104},
        {"total": 1632922404, "architecture": "dflash", "totalFileSize": 46000000000},
    ]:
        shape = updater.candidate_model_shape({}, arch, {"gguf": summary}, quants)
        assert shape == {"total_params_b": 0, "active_params_b": 0, "moe": True}


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


def test_stamp_runnable_records_mainline_separately():
    updater = load_updater()
    rows = [{"arch": "llama"}, {"arch": "deepseek41"}, {"arch": "axk2"}]
    updater.stamp_upstream(rows, {"llama.cpp": {"llama"}, "ik_llama.cpp": {"llama", "deepseek41"}})
    # An ik-only architecture is runnable, but no mainline build loads it.
    assert [row.get("runnable") for row in rows] == [True, True, False]
    assert [row.get("runnable_mainline") for row in rows] == [True, False, False]


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
    updater.stamp_upstream(rows, updater.fetch_upstream_arch_tables())
    assert "runnable" not in rows[0] and "runnable_mainline" not in rows[0]


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
        _sibling("sparse-indexer/indexer-Model-BF16.gguf", 874342784),
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


def test_quant_labels_for_ternary_and_xl_variants_match_downloader():
    """UD-TQ1_0 is ternary, not Q1_0; IQ4_NL_XL is not IQ4_NL. Both tools must
    agree, or the catalog offers a label the downloader cannot find."""
    import importlib.util
    root = Path(__file__).resolve().parents[1]

    def load(rel):
        spec = importlib.util.spec_from_file_location(Path(rel).stem, root / rel)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    updater = load("tools/models/update_recommendations.py")
    downloader = load("tools/download/download_any_gguf.py")
    assert updater.QUANT_PATTERN.pattern == downloader.QUANT_PATTERN.pattern
    cases = {
        "Qwen3-Coder-Next-UD-TQ1_0.gguf": "UD-TQ1_0",
        "UD-TQ2_0/Kimi-K3-UD-TQ2_0-00002-of-00013.gguf": "UD-TQ2_0",
        "Qwen3.6-35B-A3B-UD-IQ4_NL.gguf": "UD-IQ4_NL",
        "Qwen3.6-35B-A3B-UD-IQ4_NL_XL.gguf": "UD-IQ4_NL_XL",
        "Q2_K/meta-models.Muse-Glimmer-30B.f16.gguf.Q2_K.gguf": "Q2_K",
        "GLM-5.3-Flash-BF16-Q4_0/GLM-5.3-Flash-BF16-Q4_0-00001-of-00005.gguf": "Q4_0",
    }
    for path, want in cases.items():
        assert updater.artifact_quant(path) == want, path
        assert downloader.artifact_quant(path) == want, path


def test_identity_rejects_a_domain_finetune():
    """Ling 3.0 Flash was resolved to the finance retune Ling-3.0-flash-Fin."""
    updater = load_updater()
    base = {"name": "Ling 3.0 Flash", "creator": {"name": "InclusionAI"}}
    fin = {"name": "Ling-3.0-flash-Fin", "creator": {"name": "InclusionAI"}}
    assert not updater.candidate_relevant("bartowski/Ling-3.0-flash-Fin-GGUF", base)
    assert updater.candidate_relevant("AtomicChat/Ling-3.0-flash-GGUF", base)
    assert updater.candidate_relevant("bartowski/Ling-3.0-flash-Fin-GGUF", fin)


def test_companion_files_are_not_model_quants(monkeypatch):
    """unsloth/DeepSeek-V4.1-Flash-1017-GGUF, 2026-10-27: dspark heads of
    7-8 GB as Q8_0 and BF16 beside UD quant folders; an unnamed one sharing
    the model's label must not hide the model's quant either."""
    updater = load_updater()
    gb = 1024**3
    siblings = [
        _sibling("UD-Q2_K_XL/M-UD-Q2_K_XL-00001-of-00002.gguf", gb // 100),
        _sibling("UD-Q2_K_XL/M-UD-Q2_K_XL-00002-of-00002.gguf", 312 * gb),
        _sibling("dspark-M-Q8_0.gguf", int(7.43 * gb)),
        _sibling("dspark/dspark-M-BF16.gguf", int(7.92 * gb)),
        _sibling("Q8_0/M-Q8_0-00001-of-00002.gguf", 400 * gb),
        _sibling("Q8_0/M-Q8_0-00002-of-00002.gguf", 400 * gb),
        _sibling("helper-M-Q8_0.gguf", 8 * gb),
    ]
    monkeypatch.setattr(updater, "fetch_hf_model_info", lambda repo: {"siblings": siblings})
    assert {q["name"] for q in updater.fetch_hf_quants("owner/M-GGUF")} == {"UD-Q2_K_XL", "Q8_0"}


def test_companion_rule_matches_downloader():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("dl", root / "tools/download/download_any_gguf.py")
    downloader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(downloader)
    updater = load_updater()
    for name in ("DRAFT_HEAD_NAMES", "DRAFT_HEAD_WORDS", "BYTES_PER_WEIGHT", "COMPANION_PARAM_RATIO"):
        assert getattr(updater, name) == getattr(downloader, name), name
    # Token-aware: an Eagle-named main model is not a draft head; EAGLE drafts are.
    for path, want in [("RWKV-Eagle-7B-Q4_K_M.gguf", False), ("rwkv/RWKV-Eagle-7B-Q8_0.gguf", False),
                       ("eagle3-M-BF16.gguf", True), ("M-eagle-draft-Q8_0.gguf", True),
                       ("M-EAGLE3-Q8_0.gguf", True), ("eagle/M-Q8_0.gguf", True),
                       ("dspark-M-Q8_0.gguf", True), ("M-medusa-BF16.gguf", True),
                       ("M-Q4_K_M.gguf", False)]:
        for module in (updater, downloader):
            assert module.is_draft_head_gguf(path) is want, (module.__name__, path)


def test_dspark_heads_and_companion_sized_files_are_not_quants(monkeypatch, capsys):
    """2026-10-25: unsloth/DeepSeek-V4.1-Flash-1017-GGUF listed its 8 GB
    dspark head as the Q8_0 of a 508 GB model and failed the refresh."""
    updater = load_updater()
    siblings = [_sibling(f"UD-Q4_K_XL/DS-1017-UD-Q4_K_XL-0000{i}-of-00002.gguf", 250 * 1024**3) for i in (1, 2)]
    siblings += [_sibling("dspark-DS-1017-Q8_0.gguf", 7976914240),
                 _sibling("dspark/dspark-DS-1017-BF16.gguf", 8506281280),
                 _sibling("DS-1017-draft-Q4_K_M.gguf", 4 * 1024**3),
                 _sibling("mmproj-BF16.gguf", 972208288)]
    monkeypatch.setattr(updater, "fetch_hf_model_info", lambda repo: {"siblings": siblings})
    assert [q["name"] for q in updater.fetch_hf_quants("unsloth/DS-1017-GGUF")] == ["UD-Q4_K_XL"]
    # An unlabelled companion is caught by size, with a warning.
    siblings[:] = [_sibling("Model-Q4_K_M.gguf", 300 * 1024**3), _sibling("Model-Q8_0.gguf", 8 * 1024**3),
                   _sibling("Model-IQ1_S.gguf", 60 * 1024**3)]
    assert [q["name"] for q in updater.fetch_hf_quants("owner/Model-GGUF")] == ["IQ1_S", "Q4_K_M"]
    assert "Q8_0" in capsys.readouterr().err
    assert updater.auxiliary_repo("someone/DS-1017-DSpark-GGUF")
    assert not updater.is_draft_head_gguf("RWKV-Eagle-7B-Q4_K_M.gguf")


def test_identity_requires_every_word_of_the_model_name():
    updater = load_updater()
    cases = [
        # 2026-10-25 regressions: a distill, a "+" variant and a later version.
        ({"name": "A.X-K2", "creator": {"name": "SK Telecom"}}, "someone/A.X-K2-Distilled-3.7B-GGUF", False),
        ({"name": "A.X-K2", "creator": {"name": "SK Telecom"}}, "skt/A.X-K2-GGUF", True),
        ({"name": "Command A", "creator": {"name": "Cohere"}}, "bartowski/command-a-plus-05-2026-GGUF", False),
        ({"name": "Command A", "creator": {"name": "Cohere"}}, "someone/Command-A+-GGUF", False),
        ({"name": "Command A", "creator": {"name": "Cohere"}}, "bartowski/CohereForAI_c4ai-command-a-03-2025-GGUF", True),
        ({"name": "LFM2 24B A2B", "creator": {"name": "Liquid AI"}}, "bartowski/LiquidAI_LFM2.5-24B-A2B-GGUF", False),
        ({"name": "LFM2 24B A2B", "creator": {"name": "Liquid AI"}}, "bartowski/LiquidAI_LFM2-24B-A2B-GGUF", True),
        ({"name": "MiniMax-M3", "creator": {"name": "MiniMax"}}, "someone/MiniMax-M3.1-GGUF", False),
        ({"name": "Qwen3.6 27B", "creator": {"name": "Alibaba"}}, "someone/Qwen3.6-27B-Distillation-GGUF", False),
        ({"name": "Llama Nemotron Super 49B v1.5", "creator": {"name": "NVIDIA"}},
         "bartowski/nvidia_Llama-3_3-Nemotron-Super-49B-v1_5-GGUF", True),
    ]
    for row, repo, want in cases:
        assert updater.candidate_relevant(repo, row) is want, (row["name"], repo)


def test_quant_count_cannot_outweigh_identity():
    updater = load_updater()
    row = {"name": "A.X-K2", "creator": {"name": "SK Telecom"}}
    few = [{"name": "Q4_K_M", "size_gb": 300}]
    many = [{"name": n, "size_gb": 300} for n in ("Q4_0", "Q4_1", "Q4_K_S", "IQ4_XS", "IQ4_NL", "MXFP4")] + few
    assert updater.hf_repo_score("o/A.X-K2-GGUF", row, many) - updater.hf_repo_score("o/A.X-K2-GGUF", row, few) <= 30


def test_resolution_rejects_a_repo_far_off_the_previous_size(monkeypatch, capsys):
    updater = load_updater()
    row = {"name": "A.X-K2", "creator": {"name": "SK Telecom"}}
    monkeypatch.setattr(updater, "search_hf_models", lambda *a: [{"id": "skt/A.X-K2-GGUF"}])
    monkeypatch.setattr(updater, "fetch_hf_quants", lambda repo: [{"name": "Q4_K_M", "size_gb": 2.3, "size_bytes": 1}])
    assert updater.resolve_gguf_repo(row, 20, {"size_gb": 344.99}) is None
    assert "off the previous" in capsys.readouterr().err
    assert updater.resolve_gguf_repo(row, 20, {"size_gb": 3.0})[0] == "skt/A.X-K2-GGUF"
    assert updater.resolve_gguf_repo(row, 20)[0] == "skt/A.X-K2-GGUF"


def _previous(name, repo, seen, index=40.0, slug=""):
    return {"name": name, "repo": repo, "family": "Kimi", "aa_slug": slug, "aa_intelligence_index": index,
            "aa_updated_at": seen, "quality": round(index * 1.65), "size_gb": 500, "notes": "row",
            "quants": [{"name": "Q4_K_M", "size_gb": 500, "size_bytes": 500 * 1024**3}]}


def _offline_catalog(updater, monkeypatch, resolved):
    monkeypatch.setattr(updater, "resolve_gguf_repo", lambda row, limit, prev=None: resolved.get(row["name"]))
    monkeypatch.setattr(updater, "candidate_from_row", lambda row, repo, quants, method: {
        "name": updater.display_name(row), "repo": repo, "quants": quants, "aa_slug": updater.row_slug(row),
        "aa_intelligence_index": updater.intelligence(row), "quality": 50, "notes": "fresh"})
    monkeypatch.setattr(updater, "REVIEWED_UNSCORED_MODELS", ())


def test_a_row_missing_from_one_scrape_is_carried_until_it_expires(monkeypatch, capsys):
    """Kimi K3 vanished from one leaderboard scrape and was deleted."""
    from datetime import datetime, timezone
    updater = load_updater()
    quants = [{"name": "Q4_K_M", "size_gb": 10, "size_bytes": 10 * 1024**3}]
    _offline_catalog(updater, monkeypatch, {"GLM-5.3": ("unsloth/GLM-5.3-GGUF", quants, "hf-search")})
    rows = [{"name": "GLM-5.3", "slug": "glm-5-3", "intelligenceIndex": 44, "isOpenWeights": True}]
    previous = [_previous("Kimi K3 (Max)", "unsloth/Kimi-K3-GGUF", "2026-10-19T11:04:37Z", slug="kimi-k3")]
    now = datetime(2026, 10, 25, tzinfo=timezone.utc)
    catalog = updater.build_catalog(rows, "test", 100, 20, previous, now)
    kimi = [c for c in catalog["candidates"] if c["repo"] == "unsloth/Kimi-K3-GGUF"]
    assert kimi and "last seen on the leaderboard 2026-10-19" in kimi[0]["notes"]
    assert "kept 1 previous rows" in catalog["source"]
    err = capsys.readouterr().err
    assert "top-10 model Kimi K3 (Max) is missing" in err and "until 2026-11-02" in err
    # Carried twice, the note is not repeated; past the expiry the row goes.
    again = updater.build_catalog(rows, "test", 100, 20, catalog["candidates"], now)
    assert [c["notes"] for c in again["candidates"] if "Kimi" in c["name"]] == [kimi[0]["notes"]]
    late = updater.build_catalog(rows, "test", 100, 20, previous, datetime(2026, 11, 3, tzinfo=timezone.utc))
    assert [c["repo"] for c in late["candidates"]] == ["unsloth/GLM-5.3-GGUF"]


def test_an_unresolved_row_keeps_its_previous_entry_unless_it_was_a_mismatch(monkeypatch):
    from datetime import datetime, timezone
    updater = load_updater()
    _offline_catalog(updater, monkeypatch, {})
    now = datetime(2026, 10, 25, tzinfo=timezone.utc)
    rows = [{"name": "Kimi K3 (Max)", "slug": "kimi-k3", "intelligenceIndex": 43.6, "isOpenWeights": True,
             "creator": {"name": "Kimi"}},
            {"name": "Command A", "slug": "command-a", "intelligenceIndex": 7, "isOpenWeights": True,
             "creator": {"name": "Cohere"}}]
    previous = [_previous("Kimi K3 (Max)", "unsloth/Kimi-K3-GGUF", "2026-10-19T11:04:37Z", slug="kimi-k3"),
                _previous("Cohere Command A", "bartowski/command-a-plus-05-2026-GGUF", "2026-10-19T11:04:37Z",
                          index=7, slug="command-a")]
    catalog = updater.build_catalog(rows, "test", 100, 20, previous, now)
    assert [c["repo"] for c in catalog["candidates"]] == ["unsloth/Kimi-K3-GGUF"]


def test_one_failing_repo_does_not_fail_the_refresh(monkeypatch, capsys):
    updater = load_updater()
    monkeypatch.setattr(updater, "REVIEWED_UNSCORED_MODELS", ())
    quants = [{"name": "Q4_K_M", "size_gb": 10, "size_bytes": 10 * 1024**3}]

    def resolve(row, limit, prev=None):
        if row["name"] == "Bad":
            raise OSError("connection reset")
        return ("o/Good-GGUF", quants, "hf-search")
    monkeypatch.setattr(updater, "resolve_gguf_repo", resolve)
    monkeypatch.setattr(updater, "candidate_from_row", lambda row, repo, q, m: {
        "name": row["name"], "repo": repo, "quants": q, "aa_intelligence_index": 1})
    rows = [{"name": "Bad", "intelligenceIndex": 50, "isOpenWeights": True},
            {"name": "Good", "intelligenceIndex": 40, "isOpenWeights": True}]
    catalog = updater.build_catalog(rows, "test", 100, 20)
    assert [c["name"] for c in catalog["candidates"]] == ["Good"]
    assert "connection reset" in capsys.readouterr().err


def test_validation_repairs_or_drops_rows_the_go_tests_reject(capsys):
    updater = load_updater()
    gb = 1024**3
    rows = [{"name": "dense-labelled MoE", "repo": "o/a", "experts": 128, "moe": False,
             "quants": [{"name": "Q4_K_M", "size_gb": 60, "size_bytes": 60 * gb}]},
            {"name": "companion", "repo": "o/b", "size_gb": 7.4,
             "quants": [{"name": "Q8_0", "size_gb": 7.4, "size_bytes": 8 * gb},
                        {"name": "UD-Q4_K_XL", "size_gb": 480, "size_bytes": 500 * gb}]},
            {"name": "empty", "repo": "o/c", "quants": []}]
    out = updater.validate_candidates(rows)
    assert [r["name"] for r in out] == ["dense-labelled MoE", "companion"]
    assert out[0]["moe"] is True
    assert [q["name"] for q in out[1]["quants"]] == ["UD-Q4_K_XL"] and out[1]["size_gb"] == 480
    assert "marked MoE" in capsys.readouterr().err


def test_reviewed_unscored_models_rank_one_point_under_their_scored_anchor(monkeypatch):
    updater = load_updater()
    gb = 1024**3
    monkeypatch.setattr(updater, "fetch_hf_quants", lambda repo: [{"name": "UD-Q4_K_XL", "size_gb": 240, "size_bytes": 240 * gb}])
    monkeypatch.setattr(updater, "fetch_gguf_arch", lambda repo: {"arch": "minimax-m3", "experts": 128, "exp_used": 4} if "MiniMax" in repo else {"arch": "llama"})
    monkeypatch.setattr(updater, "fetch_hf_model_info", lambda repo: {})
    scored = [{"name": "MiniMax-M3", "repo": "unsloth/MiniMax-M3-GGUF", "aa_intelligence_index": 29.22, "quality": 48},
              {"name": "Nanbeige4.1-3B", "repo": "o/Nanbeige4.1-3B-GGUF", "aa_intelligence_index": 8.4, "quality": 14}]
    rows = {r["name"]: r for r in updater.reviewed_candidates(scored, [])}
    assert set(rows) == {"MiniMax-M3.1-Flash", "Nanbeige4.3-33B"}
    flash = rows["MiniMax-M3.1-Flash"]
    assert flash["aa_intelligence_index"] == 0 and flash["quality"] == 47
    assert flash["moe"] is True and flash["layers"] == 60 and flash["total_params_b"] == 428
    assert flash["repo"] == "unsloth/MiniMax-M3.1-Flash-GGUF" and flash["notes"].startswith("Unscored")
    dense = rows["Nanbeige4.3-33B"]
    assert dense["quality"] == 13 and dense["moe"] is False and dense["aa_intelligence_index"] == 0
    # No scored anchor, no estimate: the row is not listed at all.
    assert [r["name"] for r in updater.reviewed_candidates(scored[1:], [])] == ["Nanbeige4.3-33B"]
    # An anchor missing from this scrape (2026-10-29) is taken from the previous catalog.
    assert [r["name"] for r in updater.reviewed_candidates(scored[:1], scored[1:])] == ["MiniMax-M3.1-Flash", "Nanbeige4.3-33B"]
    # Once the leaderboard scores it, the scored row wins.
    scored.append({"name": "MiniMax-M3.1-Flash", "repo": "x/y", "aa_intelligence_index": 33, "quality": 54})
    assert [r["name"] for r in updater.reviewed_candidates(scored, [])] == ["Nanbeige4.3-33B"]


def test_reviewed_rows_stay_out_of_the_candidates_released_clients_rank(monkeypatch):
    """Released binaries fetch main's catalog and rank "candidates" by quality
    alone, so a reviewed estimate must live under its own key."""
    updater = load_updater()
    quants = [{"name": "Q4_K_M", "size_gb": 10, "size_bytes": 10 * 1024**3}]
    _offline_catalog(updater, monkeypatch, {"GLM-5.3": ("unsloth/GLM-5.3-GGUF", quants, "hf-search")})
    seen = []

    def reviewed(candidates, previous, scale=None):
        seen.append([c["repo"] for c in previous])
        return [{"name": "Guess", "repo": "o/Guess-GGUF", "quality": 49, "aa_intelligence_index": 0, "quants": quants}]
    monkeypatch.setattr(updater, "reviewed_candidates", reviewed)
    rows = [{"name": "GLM-5.3", "slug": "glm-5-3", "intelligenceIndex": 44, "isOpenWeights": True}]
    catalog = updater.build_catalog(rows, "test", 100, 20)
    assert [c["repo"] for c in catalog["candidates"]] == ["unsloth/GLM-5.3-GGUF"]
    assert [c["repo"] for c in catalog["reviewed_candidates"]] == ["o/Guess-GGUF"]
    assert "listed 1 reviewed unscored rows" in catalog["source"]
    # The next refresh reads both keys back as previous rows.
    assert [c["repo"] for c in updater.catalog_rows(catalog)] == ["unsloth/GLM-5.3-GGUF", "o/Guess-GGUF"]
    updater.build_catalog(rows, "test", 100, 20, updater.catalog_rows(catalog))
    assert seen[-1] == ["unsloth/GLM-5.3-GGUF", "o/Guess-GGUF"]


def test_api_403_and_failures_fall_back_to_the_public_page(monkeypatch, capsys):
    updater = load_updater()
    page_rows = [{"name": "Open", "isOpenWeights": True, "intelligenceIndex": 30}]
    monkeypatch.setattr(updater, "fetch_public_open_weight_rows", lambda: page_rows)
    calls = []

    def fetch(api_key, url, *, paginated):
        calls.append(url)
        if url == updater.API_URL:
            raise updater.APIRequestError(403, url, "Forbidden")
        return [{"name": "Legacy row without licensing"}]
    monkeypatch.setattr(updater, "fetch_models_from_url", fetch)
    assert updater.load_open_weight_rows("key") == (page_rows, "Artificial Analysis open-weight leaderboard page")
    assert calls == [updater.API_URL, updater.LEGACY_API_URL]
    assert "v2 API returned 403; using the legacy endpoint" in capsys.readouterr().err

    def broken(api_key, url, *, paginated):
        raise updater.APIRequestError(500, url, "Server Error")
    monkeypatch.setattr(updater, "fetch_models_from_url", broken)
    assert updater.load_open_weight_rows("key")[0] == page_rows


def _gguf(kvs):
    import struct

    def string(s):
        b = s.encode()
        return struct.pack("<Q", len(b)) + b
    out = b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0) + struct.pack("<Q", len(kvs))
    for key, value in kvs:
        out += string(key)
        if isinstance(value, int):
            out += struct.pack("<I", 4) + struct.pack("<I", value)
        elif isinstance(value, str):
            out += struct.pack("<I", 8) + string(value)
        else:
            out += struct.pack("<I", 9) + struct.pack("<I", 8) + struct.pack("<Q", len(value))
            out += b"".join(string(v) for v in value)
    return out


def test_a_large_tokenizer_does_not_hide_moe_geometry(monkeypatch):
    """Sarvam 105B's 262k-token vocabulary overran the 16 MiB header range;
    the whole header was discarded and the MoE was catalogued as dense."""
    updater = load_updater()
    raw = _gguf([("general.architecture", "deepseek2"), ("deepseek2.block_count", 32),
                 ("deepseek2.expert_count", 128), ("deepseek2.expert_used_count", 8),
                 ("tokenizer.ggml.tokens", [f"tok{i}" for i in range(4000)]),
                 ("deepseek2.attention.head_count_kv", 1), ("deepseek2.attention.key_length", 576),
                 ("deepseek2.attention.value_length", 512)])
    monkeypatch.setattr(updater, "fetch_hf_model_info", lambda repo: {"siblings": [_sibling("m-Q4_K_M.gguf", 60 * 1024**3)]})
    monkeypatch.setattr(updater, "_fetch_gguf_range", lambda url, n: raw[:n])
    monkeypatch.setattr(updater, "GGUF_HEADER_FETCH_STEPS", (1024, len(raw) + 10))
    full = updater.fetch_gguf_arch("o/sarvam-a")
    assert full["experts"] == 128 and full["layers"] == 32 and full["vl"] == 512
    monkeypatch.setattr(updater, "GGUF_HEADER_FETCH_STEPS", (512, 1024))
    assert updater.fetch_gguf_arch("o/sarvam-b") == {"arch": "deepseek2", "experts": 128, "exp_used": 8}
    monkeypatch.setattr(updater, "_fetch_gguf_range", lambda url, n: b"not a gguf")
    assert updater.fetch_gguf_arch("o/sarvam-c") is None


def test_refresh_guard_refuses_a_gutted_or_rescaled_scrape():
    """2026-10-29: 21 of 100 rows scored and scores rescaled (GLM-5.3 44.8 -> 31.7);
    the refresh would have shipped 21 rows instead of 119."""
    updater = load_updater()
    prev = [{"name": f"M{i}", "repo": f"o/M{i}-GGUF", "aa_slug": f"m{i}", "aa_intelligence_index": 40.0 - i}
            for i in range(10)]
    same = [{"name": f"M{i}", "slug": f"m{i}", "intelligenceIndex": 40.0 - i, "isOpenWeights": True} for i in range(10)]
    full = {"candidates": [dict(p) for p in prev]}
    assert updater.refresh_guard(same, prev, full) is None
    # Most rows unscored.
    sparse = same[:3] + [dict(r, intelligenceIndex=0) for r in same[3:]]
    assert "carry an intelligence score" in updater.refresh_guard(sparse, prev, full)
    # Same index version, but the scores moved as a whole.
    moved = {"candidates": [dict(c, aa_intelligence_index=c["aa_intelligence_index"] * 0.7) for c in prev]}
    assert "median factor 0.70" in updater.refresh_guard(same, prev, moved)
    # A refused rescale is reported as is.
    assert updater.refresh_guard(same, prev, {"aa_index_scale": {"refused": "weak fit"}, "candidates": []}) == "weak fit"
    # Scores fine, but the catalog collapsed.
    assert "freshly resolved rows" in updater.refresh_guard(same, prev, {"candidates": full["candidates"][:3]})
    # Carried rows do not count as fresh.
    carried = [dict(c, notes="row; last seen on the leaderboard 2026-10-19") for c in full["candidates"]]
    assert "freshly resolved rows" in updater.refresh_guard(same, prev, {"candidates": carried})


def test_reviewed_row_keeps_its_score_when_the_anchor_leaves_both_catalogs(monkeypatch):
    updater = load_updater()
    gb = 1024**3
    monkeypatch.setattr(updater, "fetch_hf_quants", lambda repo: [{"name": "Q4_K_M", "size_gb": 20, "size_bytes": 20 * gb}])
    monkeypatch.setattr(updater, "fetch_gguf_arch", lambda repo: {"arch": "llama"})
    monkeypatch.setattr(updater, "fetch_hf_model_info", lambda repo: {})
    anchor = [{"name": "Nanbeige4.1-3B", "repo": "o/Nanbeige4.1-3B-GGUF", "aa_intelligence_index": 8.4, "quality": 14}]
    first = [r for r in updater.reviewed_candidates([], anchor) if r["name"] == "Nanbeige4.3-33B"]
    assert first and first[0]["quality"] == 13 and first[0]["anchor_name"] == "Nanbeige4.1-3B"
    # Next refresh: the anchor is gone from both the scrape and the catalog it wrote.
    second = [r for r in updater.reviewed_candidates([], first) if r["name"] == "Nanbeige4.3-33B"]
    assert second and second[0]["quality"] == 13 and "Nanbeige4.1-3B" in second[0]["notes"]


def _prev_rows(n, version_score=lambda i: 4.0 + 3 * i):
    return [{"name": f"M{i}", "repo": f"o/M{i}-GGUF", "aa_slug": f"m{i}", "aa_intelligence_index": version_score(i),
             "quality": round(version_score(i) * 1.65), "aa_updated_at": "2026-10-19T10:55:25Z", "notes": "row",
             "quants": [{"name": "Q4_K_M", "size_gb": 10, "size_bytes": 10 * 1024**3}]} for i in range(n)]


def _v5(x):
    return round(0.5 * x ** 1.1, 6)


def test_index_change_converts_unscored_rows_with_a_power_fit():
    """v4 -> v5.0 (2026-10): most rows lost their score. A straight line had a
    -5.8 intercept and clamped 39 small models; the power fit stays positive."""
    updater = load_updater()
    prev = _prev_rows(14)  # v4 scores 4 .. 43
    rows = [{"name": f"M{i}", "slug": f"m{i}", "isOpenWeights": True,
             "intelligenceIndex": _v5(4.0 + 3 * i) if i >= 4 else None} for i in range(14)]
    rows.append({"name": "New", "slug": "new", "isOpenWeights": True, "intelligenceIndex": 33.0})
    out, scale = updater.apply_index_scale(rows, prev, "v4", "v5.0")
    by = {updater.row_slug(r): r for r in out}
    assert scale["version"] == "v5" and scale["pairs"] == 10 and scale["r2"] > 0.999 and "refused" not in scale
    assert abs(scale["fit"]["a"] - 1.1) < 1e-4
    small = [updater.intelligence(by[f"m{i}"]) for i in range(4)]
    assert all(by[f"m{i}"]["_mapped_from"] == "v4" for i in range(4))
    assert small == sorted(small) and small[0] > 0.1 and abs(small[0] - _v5(4.0)) < 1e-3
    assert "_mapped_from" not in by["m5"] and updater.intelligence(by["new"]) == 33.0


def test_same_index_version_keeps_converted_scores_without_refitting():
    """Second refresh on v5.0: no fit is needed, so the job cannot stall as
    v4-era models leave; a re-evaluated model takes its real score."""
    updater = load_updater()
    prev = _prev_rows(3, lambda i: [2.0, 7.5, 20.0][i])
    prev[0]["aa_index_mapped_from"] = prev[1]["aa_index_mapped_from"] = "v4"
    rows = [{"name": "M0", "slug": "m0", "intelligenceIndex": None},
            {"name": "M1", "slug": "m1", "intelligenceIndex": 9.0},
            {"name": "M2", "slug": "m2", "intelligenceIndex": 20.4}]
    out, scale = updater.apply_index_scale(rows, prev, "v5.0", "v5.0")
    by = {updater.row_slug(r): r for r in out}
    assert "refused" not in scale and "fit" not in scale and scale["version"] == "v5"
    assert updater.intelligence(by["m0"]) == 2.0 and by["m0"]["_mapped_from"] == "v4"
    assert updater.intelligence(by["m1"]) == 9.0 and "_mapped_from" not in by["m1"]


def test_next_index_change_converts_from_the_previous_version():
    """v5.0 -> v6: anchors are the previous version's scores, mapped ones included."""
    updater = load_updater()
    prev = _prev_rows(12)
    prev[0]["aa_index_mapped_from"] = "v4"
    rows = [{"name": f"M{i}", "slug": f"m{i}", "intelligenceIndex": 2 * (4.0 + 3 * i) if i else None}
            for i in range(12)]
    out, scale = updater.apply_index_scale(rows, prev, "v5.0", "v6")
    m0 = next(r for r in out if updater.row_slug(r) == "m0")
    assert scale["pairs"] == 11 and m0["_mapped_from"] == "v5" and abs(updater.intelligence(m0) - 8.0) < 1e-3


def test_index_change_refuses_a_thin_or_weak_fit():
    updater = load_updater()
    prev = _prev_rows(12)
    thin = [{"name": f"M{i}", "slug": f"m{i}", "intelligenceIndex": _v5(4.0 + 3 * i) if i < 5 else None}
            for i in range(12)]
    _, scale = updater.apply_index_scale(thin, prev, "v4", "v5.0")
    assert "only 5 models" in scale["refused"]
    noisy = [{"name": f"M{i}", "slug": f"m{i}", "intelligenceIndex": [9, 31, 2, 28, 3, 30, 4, 27, 13, 1][i]}
             for i in range(10)]
    _, scale = updater.apply_index_scale(noisy, prev, "v4", "v5.0")
    assert "is weak" in scale["refused"] and "fit" not in scale


def test_carried_rows_are_converted_or_dropped_on_a_version_change():
    updater = load_updater()
    scale = {"version": "v5.0", "previous_version": "v4", "fit": {"kind": "power", "a": 1.1, "b": math.log(0.5)}}
    row = _prev_rows(1, lambda i: 20.0)[0]
    mapped = updater.map_carried(dict(row), scale)
    assert abs(mapped["aa_intelligence_index"] - _v5(20.0)) < 1e-3 and mapped["aa_index_mapped_from"] == "v4"
    assert updater.map_carried(dict(row, aa_intelligence_index=0), scale) is None
    # Scored by this scrape already: kept as is.
    assert updater.map_carried(dict(row, _on_scale=True), scale)["aa_intelligence_index"] == 20.0
    assert updater.map_carried(dict(row), dict(scale, version="v4")) == row


def test_reviewed_fallbacks_are_converted_on_a_version_change(monkeypatch):
    updater = load_updater()
    gb = 1024**3
    monkeypatch.setattr(updater, "fetch_hf_quants", lambda repo: [{"name": "Q4_K_M", "size_gb": 20, "size_bytes": 20 * gb}])
    monkeypatch.setattr(updater, "fetch_gguf_arch", lambda repo: {"arch": "llama"})
    monkeypatch.setattr(updater, "fetch_hf_model_info", lambda repo: {})
    scale = {"version": "v5.0", "previous_version": "v4", "fit": {"kind": "power", "a": 1.1, "b": math.log(0.5)}}
    old_anchor = [{"name": "Nanbeige4.1-3B", "repo": "o/N-GGUF", "aa_intelligence_index": 8.4, "quality": 14}]
    row = next(r for r in updater.reviewed_candidates([], old_anchor, scale) if r["name"] == "Nanbeige4.3-33B")
    assert row["quality"] == updater.quality_score(_v5(8.4)) - 1


def test_build_catalog_across_two_refreshes_keeps_one_scale(monkeypatch):
    """Real candidate_from_row; the first refresh's catalog is fed back in."""
    from datetime import datetime, timezone
    updater = load_updater()
    quants = [{"name": "Q4_K_M", "size_gb": 10, "size_bytes": 10 * 1024**3}]
    monkeypatch.setattr(updater, "resolve_gguf_repo",
                        lambda row, limit, prev=None: (f"o/{updater.display_name(row)}-GGUF", quants, "hf-search")
                        if updater.display_name(row) != "M13" else None)
    monkeypatch.setattr(updater, "fetch_gguf_arch", lambda repo: {})
    monkeypatch.setattr(updater, "fetch_hf_model_info", lambda repo: {})
    monkeypatch.setattr(updater, "REVIEWED_UNSCORED_MODELS", ())
    prev = _prev_rows(14)
    rows = [{"name": f"M{i}", "slug": f"m{i}", "isOpenWeights": True,
             "intelligenceIndex": _v5(4.0 + 3 * i) if i >= 4 else None} for i in range(13)]  # M13 left the scrape
    now = datetime(2026, 10, 29, tzinfo=timezone.utc)
    first = updater.build_catalog(rows, "test", 100, 20, prev, now, prev_version="v4", page_version="v5.0")
    by = {c["repo"]: c for c in first["candidates"]}
    assert first["aa_index_version"] == "v5" and len(by) == 14
    assert by["o/M1-GGUF"]["aa_index_mapped_from"] == "v4" and "~intelligence mapped" in by["o/M1-GGUF"]["notes"]
    assert by["o/M13-GGUF"]["aa_index_mapped_from"] == "v4"  # carried and converted
    assert abs(by["o/M13-GGUF"]["aa_intelligence_index"] - _v5(43.0)) < 1e-2
    assert max(c["aa_intelligence_index"] for c in first["candidates"]) < _v5(44.0)  # no v4-scale row on top
    assert updater.refresh_guard(rows, prev, first, "v4") is None
    # Second refresh, same page: no refit, same scores, nothing refused.
    second = updater.build_catalog(rows, "test", 100, 20, updater.catalog_rows(first), now,
                                   prev_version="v5.0", page_version="v5.0")
    assert "fit" not in second["aa_index_scale"] and "refused" not in second["aa_index_scale"]
    again = {c["repo"]: c["aa_intelligence_index"] for c in second["candidates"]}
    assert again["o/M1-GGUF"] == by["o/M1-GGUF"]["aa_intelligence_index"]
    assert updater.refresh_guard(rows, updater.catalog_rows(first), second, "v5.0") is None


def test_version_labels_ignore_a_trailing_zero_and_keep_fit_provenance():
    updater = load_updater()
    prev = _prev_rows(3)
    rows = [{"name": f"M{i}", "slug": f"m{i}", "intelligenceIndex": 4.0 + 3 * i} for i in range(3)]
    fit = {"kind": "power", "a": 1.3, "b": -1.6}
    _, scale = updater.apply_index_scale(rows, prev, "v5.0", "v5", fit)
    assert scale["version"] == "v5" and "pairs" not in scale and scale["fit_from_previous_version"] == fit

