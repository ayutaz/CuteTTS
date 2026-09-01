# Copyright 2026 OPPO and Fudan University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""manifest schema / validator の決定的テスト（実データに依存しない）。"""

from __future__ import annotations

import dataclasses
import json

import pytest

from cutetts.training.manifest import (
    DATASET_IDS,
    VALIDATION_CODES,
    Utterance,
    ValidationIssue,
    load_manifest,
    manifest_checksum,
    split_audio_ref,
    summarize,
    validate,
    write_manifest,
)


def make_utterance(**overrides) -> Utterance:
    """検証を通る最小構成のrecord。テスト側は必要なfieldだけ上書きする。"""
    base = {
        "utterance_id": "gol:game0001:0000001",
        "dataset_id": "gol",
        "audio_ref": "data/raw/gol/audio/game0001.tar::game0001/0000001.wav",
        "text_raw": "今日はいい天気ですね。",
        "speaker_id": "0123456789ABCDEF0123456789ABCDEF",
        # 11文字 / 2.2秒 = 0.20 秒/文字。自然な日本語の範囲に収めて
        # text_audio_mismatch を誘発しないようにする（既定閾値 0.40）。
        "duration": 2.2,
        "sample_rate": 48000,
    }
    base.update(overrides)
    return Utterance(**base)


def codes(issues: list[ValidationIssue]) -> set[str]:
    return {issue.code for issue in issues}


# --- schema ------------------------------------------------------------------


def test_to_json_from_json_roundtrip_is_equal() -> None:
    record = make_utterance(
        text_normalized="今日はいい天気ですね。",
        reading="キョウワイイテンキデスネ",
        voice_cluster_id="vc-000123",
        quality_score=0.96,
        split="train",
        game_id="game0001",
        source_checksum="a" * 64,
    )

    restored = Utterance.from_json(record.to_json())

    assert restored == record


def test_to_json_omits_none_fields_and_from_json_restores_them() -> None:
    record = make_utterance()
    payload = record.to_json()

    optional_fields = [
        field.name
        for field in dataclasses.fields(Utterance)
        if getattr(record, field.name) is None
    ]
    # 少なくともこれらは None のはずで、JSONに現れてはいけない。
    assert set(optional_fields) >= {
        "text_normalized",
        "reading",
        "voice_cluster_id",
        "quality_score",
        "split",
        "game_id",
        "source_checksum",
    }
    for name in optional_fields:
        assert name not in payload

    # language は None ではないので省略されない。
    assert payload["language"] == "ja"

    restored = Utterance.from_json(payload)
    assert restored == record
    for name in optional_fields:
        assert getattr(restored, name) is None


def test_from_json_ignores_unknown_keys() -> None:
    payload = make_utterance().to_json()
    payload["future_field_added_by_another_agent"] = 123

    assert Utterance.from_json(payload) == make_utterance()


def test_from_json_rejects_missing_required_field() -> None:
    payload = make_utterance().to_json()
    del payload["speaker_id"]

    with pytest.raises(ValueError, match="speaker_id"):
        Utterance.from_json(payload)


def test_from_json_coerces_numeric_types() -> None:
    payload = make_utterance().to_json()
    payload["duration"] = 3  # int で書かれていても float にする
    payload["sample_rate"] = 44100.0  # float で書かれていても int にする

    restored = Utterance.from_json(payload)

    assert isinstance(restored.duration, float)
    assert restored.duration == 3.0
    assert isinstance(restored.sample_rate, int)
    assert restored.sample_rate == 44100


def test_split_audio_ref_handles_archive_and_plain_paths() -> None:
    assert split_audio_ref("a/b.tar::inner/0001.wav") == ("a/b.tar", "inner/0001.wav")
    assert split_audio_ref("a/b/0001.wav") == ("a/b/0001.wav", None)


# --- JSONL I/O ---------------------------------------------------------------


def test_write_then_load_manifest_roundtrip(tmp_path) -> None:
    records = [
        make_utterance(utterance_id="gol:game0001:0000001", split="train"),
        make_utterance(
            utterance_id="moe:2f8a91cc:007",
            dataset_id="moe",
            audio_ref="data/raw/moe/2f8a91cc.zip::data/2f8a91cc/wav/2f8a91cc_007.wav",
            speaker_id="2f8a91cc",
            text_raw="ごきげんよう、お嬢様。",
            duration=5.66,
            sample_rate=44100,
            quality_score=3.87,
            split="dev-seen",
        ),
    ]
    path = tmp_path / "nested" / "manifest.jsonl"

    written = write_manifest(path, records)
    loaded = list(load_manifest(path))

    assert written == 2
    assert loaded == records


def test_write_manifest_is_utf8_jsonl_without_escapes(tmp_path) -> None:
    path = tmp_path / "manifest.jsonl"
    write_manifest(path, [make_utterance(text_raw="日本語のテキスト")])

    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()

    assert len(lines) == 1
    assert "日本語のテキスト" in text  # ensure_ascii=False で書かれている
    assert json.loads(lines[0])["text_raw"] == "日本語のテキスト"


def test_load_manifest_skips_blank_lines_and_is_lazy(tmp_path) -> None:
    path = tmp_path / "manifest.jsonl"
    path.write_text(
        json.dumps(make_utterance().to_json(), ensure_ascii=False) + "\n\n",
        encoding="utf-8",
    )

    stream = load_manifest(path)
    assert not isinstance(stream, list)  # 全件をメモリに載せない
    assert len(list(stream)) == 1


def test_load_manifest_reports_line_number_on_broken_json(tmp_path) -> None:
    path = tmp_path / "manifest.jsonl"
    path.write_text(
        json.dumps(make_utterance().to_json(), ensure_ascii=False) + "\n{not json}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=":2:"):
        list(load_manifest(path))


def test_manifest_checksum_matches_sha256_of_file(tmp_path) -> None:
    import hashlib

    path = tmp_path / "manifest.jsonl"
    write_manifest(path, [make_utterance()])
    expected = hashlib.sha256(path.read_bytes()).hexdigest()

    assert manifest_checksum(path) == expected


# --- validator ---------------------------------------------------------------


def test_validate_returns_empty_list_for_clean_records() -> None:
    records = [
        make_utterance(utterance_id="gol:game0001:0000001"),
        make_utterance(
            utterance_id="moe:2f8a91cc:007",
            dataset_id="moe",
            speaker_id="2f8a91cc",
            text_raw="おはようございます、今日もよろしくお願いします。",
            duration=5.66,
            sample_rate=44100,
        ),
    ]

    assert validate(records) == []


def test_validate_detects_duplicate_id() -> None:
    records = [make_utterance(), make_utterance()]

    issues = validate(records)

    assert [issue.code for issue in issues] == ["duplicate_id"]
    assert issues[0].utterance_id == "gol:game0001:0000001"


def test_validate_detects_empty_text() -> None:
    issues = validate([make_utterance(text_raw="   ")])

    assert codes(issues) == {"empty_text"}


def test_validate_detects_too_short() -> None:
    issues = validate([make_utterance(duration=0.4)], min_duration=1.0)

    assert codes(issues) == {"too_short"}


def test_validate_detects_too_long() -> None:
    # 45秒に見合う長さのtextにして、too_long だけが出ることを見る
    issues = validate([make_utterance(duration=45.0, text_raw="あ" * 300)],
                      max_duration=30.0)

    assert codes(issues) == {"too_long"}


def test_validate_detects_unknown_dataset() -> None:
    issues = validate([make_utterance(dataset_id="jsut")])

    assert codes(issues) == {"unknown_dataset"}
    assert "jsut" in issues[0].detail


def test_validate_detects_generic_speaker() -> None:
    generic = "F" * 32
    issues = validate(
        [make_utterance(speaker_id=generic)],
        generic_speaker_ids=frozenset({generic}),
    )

    assert codes(issues) == {"generic_speaker"}


def test_validate_detects_bad_duration_and_sample_rate() -> None:
    issues = validate([make_utterance(duration=0.0, sample_rate=0)])

    assert codes(issues) == {"bad_duration", "bad_sample_rate"}


def test_validate_reports_multiple_issues_for_one_record() -> None:
    issues = validate([make_utterance(dataset_id="jsut", text_raw="", duration=0.2)])

    assert codes(issues) == {"unknown_dataset", "empty_text", "too_short"}


def test_validate_uses_text_rules_when_available(monkeypatch) -> None:
    """text_rules が後から追加されたら punctuation_only 等を判定する。"""
    import sys
    import types

    module = types.ModuleType("cutetts.training.text_rules")
    module.is_punctuation_only = lambda text: text.strip() == "…………"
    module.contains_markup = lambda text: "</r>" in text
    module.contains_name_placeholder = lambda text: "%bd" in text
    monkeypatch.setitem(sys.modules, "cutetts.training.text_rules", module)
    monkeypatch.setattr("cutetts.training.text_rules", module, raising=False)

    # duration は text の長さに見合わせる（text_audio_mismatch を誘発しない）
    assert codes(validate([make_utterance(text_raw="…………", duration=1.0)])) == {"punctuation_only"}
    assert codes(validate([make_utterance(text_raw="<rかな>仮名</r>", duration=1.5)])) == {"markup"}
    assert codes(validate([make_utterance(text_raw="%bdさん、おはよう", duration=1.8)])) == {"name_placeholder"}


def test_validate_skips_text_rule_codes_when_module_missing(monkeypatch) -> None:
    """text_rules が未実装でも例外にせず、そのcodeの判定だけskipする。"""
    import builtins

    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "cutetts.training" and fromlist and "text_rules" in fromlist:
            raise ImportError("text_rules is not implemented yet")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    assert validate([make_utterance(text_raw="…………", duration=1.0)]) == []


def test_all_reported_codes_are_declared() -> None:
    records = [
        make_utterance(dataset_id="jsut", text_raw="", duration=-1.0, sample_rate=-1),
        make_utterance(dataset_id="jsut", duration=99.0),
        make_utterance(duration=0.1, speaker_id="generic"),
    ]

    issues = validate(records, generic_speaker_ids=frozenset({"generic"}))

    assert issues
    assert codes(issues) <= set(VALIDATION_CODES)


def test_dataset_ids_are_the_two_inventoried_datasets() -> None:
    assert DATASET_IDS == ("gol", "moe")


# --- summary -----------------------------------------------------------------


def test_summarize_counts_datasets_splits_speakers_and_hours() -> None:
    records = [
        make_utterance(utterance_id="gol:g:1", duration=3600.0, split="train"),
        make_utterance(utterance_id="gol:g:2", duration=1800.0, split="train", speaker_id="other"),
        make_utterance(
            utterance_id="moe:2f8a91cc:1",
            dataset_id="moe",
            speaker_id="2f8a91cc",
            duration=1800.0,
            sample_rate=44100,
            voice_cluster_id="vc-1",
        ),
    ]

    summary = summarize(records)

    assert summary["total"] == 3
    assert summary["by_dataset"] == {"gol": 2, "moe": 1}
    assert summary["by_split"] == {"train": 2, "unassigned": 1}
    assert summary["speakers"] == 3
    assert summary["voice_clusters"] == 1
    assert summary["hours"] == pytest.approx(2.0)
    assert summary["hours_by_dataset"] == {"gol": pytest.approx(1.5), "moe": pytest.approx(0.5)}


def test_summarize_handles_empty_input() -> None:
    summary = summarize([])

    assert summary["total"] == 0
    assert summary["by_dataset"] == {}
    assert summary["by_split"] == {}
    assert summary["speakers"] == 0
    assert summary["hours"] == 0.0


def test_summarize_counts_same_speaker_id_across_datasets_separately() -> None:
    """speaker IDはdataset内でのみ一意。gol/moeで同じ文字列でも別話者として数える。"""
    records = [
        make_utterance(utterance_id="gol:g:1", speaker_id="abcd"),
        make_utterance(utterance_id="moe:x:1", dataset_id="moe", speaker_id="abcd"),
    ]

    assert summarize(records)["speakers"] == 2
