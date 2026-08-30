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

"""scripts/evaluate_japanese_vae.py（P1c）の決定的テスト。

既知の信号で既知の値になることだけを確認する。実weight・実データを使うテストは
``@pytest.mark.slow`` を付けてあり、checkpointが無ければskipする。
``-m "not slow"`` で除外できる。
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
import zipfile
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
torchaudio = pytest.importorskip("torchaudio")

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "evaluate_japanese_vae.py"
VAE_CHECKPOINT = REPO_ROOT / "model" / "CuteTTS" / "weights" / "audio_vae"


def _load_script_module():
    """scripts/ はpackageではないのでfile pathからimportする（main は実行しない）。"""
    spec = importlib.util.spec_from_file_location("evaluate_japanese_vae", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


evaluate = _load_script_module()


@pytest.fixture(scope="module")
def sine_24k() -> torch.Tensor:
    """440 Hz / 1秒 / 24 kHz の決定的な信号。"""
    t = torch.arange(24000, dtype=torch.float32) / 24000.0
    return 0.5 * torch.sin(2.0 * math.pi * 440.0 * t)


# --- SNR ----------------------------------------------------------------------


def test_snr_of_identical_signal_is_infinite(sine_24k) -> None:
    assert evaluate.snr_db(sine_24k, sine_24k.clone()) == math.inf


def test_snr_matches_hand_computed_value() -> None:
    # signal power = 100, noise power = 100 * 0.01 = 1 -> 10*log10(100) = 20 dB
    reference = torch.ones(100)
    estimate = reference + 0.1
    assert evaluate.snr_db(reference, estimate) == pytest.approx(20.0)


def test_snr_of_silent_reference_is_negative_infinite() -> None:
    assert evaluate.snr_db(torch.zeros(64), torch.ones(64)) == -math.inf


def test_snr_truncates_to_the_shorter_signal(sine_24k) -> None:
    longer = torch.cat([sine_24k, torch.ones(500)])
    assert evaluate.snr_db(sine_24k, longer) == math.inf


def test_snr_is_shape_agnostic(sine_24k) -> None:
    flat = evaluate.snr_db(sine_24k, sine_24k * 0.9)
    batched = evaluate.snr_db(sine_24k.reshape(1, -1), (sine_24k * 0.9).reshape(1, 1, -1))
    assert flat == pytest.approx(batched)


# --- log-mel ------------------------------------------------------------------


def test_log_mel_distance_of_identical_signal_is_zero(sine_24k) -> None:
    resolution = evaluate.MelResolution(n_fft=512, hop_length=128, n_mels=64)
    mel = evaluate.build_mel_transform(resolution, 24000)
    assert evaluate.log_mel_l1(sine_24k, sine_24k.clone(), mel) == 0.0


def test_log_mel_distance_of_a_scaled_signal_is_the_log_ratio(sine_24k) -> None:
    # power=2 のmelを2倍振幅で通すと、clampが効かない帯域では log(4) だけずれる。
    resolution = evaluate.MelResolution(n_fft=512, hop_length=128, n_mels=64)
    mel = evaluate.build_mel_transform(resolution, 24000)
    distance = evaluate.log_mel_l1(sine_24k, sine_24k * 2.0, mel, eps=1e-12)
    assert distance == pytest.approx(math.log(4.0), rel=1e-3)


def test_log_mel_distance_of_different_signals_is_positive(sine_24k) -> None:
    resolution = evaluate.MelResolution(n_fft=512, hop_length=128, n_mels=64)
    mel = evaluate.build_mel_transform(resolution, 24000)
    noise = torch.zeros_like(sine_24k).uniform_(-0.5, 0.5, generator=torch.Generator().manual_seed(0))
    assert evaluate.log_mel_l1(sine_24k, noise, mel) > 0.1


def test_multi_resolution_log_mel_reports_mean_over_resolutions(sine_24k) -> None:
    resolutions = [
        evaluate.MelResolution(n_fft=512, hop_length=128, n_mels=64),
        evaluate.MelResolution(n_fft=1024, hop_length=256, n_mels=80),
    ]
    mels = {r.name: evaluate.build_mel_transform(r, 24000) for r in resolutions}
    distances = evaluate.multi_resolution_log_mel_l1(sine_24k, sine_24k * 0.5, mels)

    assert set(distances) == {"n_fft512_mels64", "n_fft1024_mels80", "mean"}
    per_resolution = [value for name, value in distances.items() if name != "mean"]
    assert distances["mean"] == pytest.approx(sum(per_resolution) / len(per_resolution))


# --- spectral convergence / RMS / cosine --------------------------------------


def test_spectral_convergence_of_identical_signal_is_zero(sine_24k) -> None:
    assert evaluate.spectral_convergence(sine_24k, sine_24k.clone()) == 0.0


def test_spectral_convergence_of_doubled_signal_is_one(sine_24k) -> None:
    # |2R - R| / |R| = 1
    assert evaluate.spectral_convergence(sine_24k, sine_24k * 2.0) == pytest.approx(1.0, rel=1e-6)


def test_rms_db_of_full_scale_square_is_zero() -> None:
    assert evaluate.rms_db(torch.ones(128)) == pytest.approx(0.0)
    assert evaluate.rms_db(torch.full((128,), 0.1)) == pytest.approx(-20.0)
    assert evaluate.rms_db(torch.zeros(128)) == -math.inf


def test_cosine_similarity_known_values() -> None:
    a = torch.tensor([1.0, 0.0, 0.0])
    b = torch.tensor([0.0, 1.0, 0.0])
    assert evaluate.cosine_similarity(a, a.clone()) == pytest.approx(1.0)
    assert evaluate.cosine_similarity(a, b) == pytest.approx(0.0)
    assert evaluate.cosine_similarity(a, -a) == pytest.approx(-1.0)


def test_pairwise_cosine_baselines_splits_intra_and_inter_speaker() -> None:
    # 話者aの2本は同一ベクトル、話者bは直交ベクトル。
    embeddings = torch.tensor(
        [
            [1.0, 0.0],
            [2.0, 0.0],  # 向きは同じなのでcos=1
            [0.0, 1.0],
        ]
    )
    baselines = evaluate.pairwise_cosine_baselines(embeddings, ["a", "a", "b"])

    assert baselines["intra_speaker"] == pytest.approx([1.0])
    assert baselines["inter_speaker"] == pytest.approx([0.0, 0.0])


def test_pairwise_cosine_baselines_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError):
        evaluate.pairwise_cosine_baselines(torch.zeros(3, 2), ["a", "b"])


def test_worst_cases_orders_and_truncates() -> None:
    records = [
        {
            "utterance_id": f"a/{index}",
            "duration_sec": 1.0,
            "mel_l1": float(index),
            "speaker_cosine": 1.0 - index / 10.0,
            "snr_db": 10.0,
        }
        for index in range(5)
    ]
    lowest = evaluate.worst_cases(records, "speaker_cosine", 2, largest=False)
    highest = evaluate.worst_cases(records, "mel_l1", 3, largest=True)

    assert [item["utterance_id"] for item in lowest] == ["a/4", "a/3"]
    assert [item["utterance_id"] for item in highest] == ["a/4", "a/3", "a/2"]


# --- 集計 ---------------------------------------------------------------------


def test_percentile_matches_linear_interpolation() -> None:
    values = [1.0, 2.0, 3.0, 4.0]
    assert evaluate.percentile(values, 0.0) == 1.0
    assert evaluate.percentile(values, 50.0) == pytest.approx(2.5)
    assert evaluate.percentile(values, 100.0) == 4.0
    assert evaluate.percentile([5.0], 95.0) == 5.0


def test_summarize_known_values() -> None:
    summary = evaluate.summarize([1.0, 2.0, 3.0])

    assert summary["count"] == 3
    assert summary["finite_count"] == 3
    assert summary["non_finite_count"] == 0
    assert summary["mean"] == pytest.approx(2.0)
    assert summary["std"] == pytest.approx(1.0)  # 標本標準偏差
    assert summary["min"] == 1.0
    assert summary["p50"] == pytest.approx(2.0)
    assert summary["max"] == 3.0


def test_summarize_separates_non_finite_values() -> None:
    summary = evaluate.summarize([1.0, math.inf, 3.0])

    assert summary["count"] == 3
    assert summary["finite_count"] == 2
    assert summary["non_finite_count"] == 1
    assert summary["mean"] == pytest.approx(2.0)


def test_summarize_of_only_non_finite_values_has_no_mean() -> None:
    summary = evaluate.summarize([math.inf, -math.inf])

    assert summary["finite_count"] == 0
    assert summary["mean"] is None


def test_duration_bucket_edges() -> None:
    edges = [2.0, 4.0, 8.0]
    assert evaluate.duration_bucket(0.5, edges) == "lt2"
    assert evaluate.duration_bucket(2.0, edges) == "2-4"
    assert evaluate.duration_bucket(3.9, edges) == "2-4"
    assert evaluate.duration_bucket(4.0, edges) == "4-8"
    assert evaluate.duration_bucket(8.0, edges) == "ge8"
    assert evaluate.duration_bucket(100.0, edges) == "ge8"


def test_bucket_order_covers_every_bucket_name() -> None:
    edges = [2.0, 4.0, 8.0]
    names = evaluate.bucket_order(edges)

    assert names == ["lt2", "2-4", "4-8", "ge8"]
    for seconds in (0.1, 2.5, 5.0, 30.0):
        assert evaluate.duration_bucket(seconds, edges) in names


def test_phonetic_marks_detection() -> None:
    assert evaluate.phonetic_marks("がっこう") == {
        "sokuon": True,
        "hatsuon": False,
        "chouon": False,
    }
    assert evaluate.phonetic_marks("こんにちはー")["hatsuon"] is True
    assert evaluate.phonetic_marks("こんにちはー")["chouon"] is True
    assert evaluate.phonetic_marks("あいうえお") == {
        "sokuon": False,
        "hatsuon": False,
        "chouon": False,
    }


def test_group_summary_groups_and_counts() -> None:
    records = [
        {"speaker": "a", "mel_l1": 1.0},
        {"speaker": "a", "mel_l1": 3.0},
        {"speaker": "b", "mel_l1": 2.0},
    ]
    grouped = evaluate.group_summary(records, lambda r: r["speaker"], ["mel_l1"])

    assert grouped["a"]["utterances"] == 2
    assert grouped["a"]["mel_l1"]["mean"] == pytest.approx(2.0)
    assert grouped["b"]["mel_l1"]["mean"] == pytest.approx(2.0)


# --- 判定 ---------------------------------------------------------------------


def _checks(**overrides):
    values = {
        "mel_l1": 0.5,
        "speaker_cosine": 0.9,
        "streaming_snr_db": 60.0,
        "short_bucket_mel_ratio": 1.1,
    }
    values.update(overrides)
    return evaluate.run_threshold_checks(thresholds=evaluate.Thresholds(), **values)


def test_verdict_is_freeze_ok_when_every_check_passes() -> None:
    assert evaluate.verdict_from_checks(_checks()) == "freeze_ok"


def test_verdict_is_caveated_when_only_the_non_blocking_check_fails() -> None:
    checks = _checks(short_bucket_mel_ratio=3.0)

    assert evaluate.verdict_from_checks(checks) == "freeze_with_caveats"
    failed = [check for check in checks if not check.passed]
    assert [check.name for check in failed] == ["short_bucket_mel_ratio"]


def test_verdict_recommends_s4_when_speaker_similarity_fails() -> None:
    assert evaluate.verdict_from_checks(_checks(speaker_cosine=0.4)) == "consider_s4"


def test_verdict_recommends_s4_when_mel_distance_fails() -> None:
    assert evaluate.verdict_from_checks(_checks(mel_l1=5.0)) == "consider_s4"


def test_verdict_recommends_s4_when_streaming_decode_diverges() -> None:
    assert evaluate.verdict_from_checks(_checks(streaming_snr_db=5.0)) == "consider_s4"


def test_infinite_streaming_snr_counts_as_a_pass() -> None:
    # streaming と offline が bit 単位で一致すると SNR は +inf になる。
    checks = {check.name: check for check in _checks(streaming_snr_db=math.inf)}
    assert checks["streaming_vs_offline_snr_db"].passed is True


def test_missing_value_fails_its_check() -> None:
    checks = {check.name: check for check in _checks(speaker_cosine=None)}
    assert checks["speaker_cosine_mean"].passed is False


# --- 選択の決定性 --------------------------------------------------------------


def _references(count: int = 40):
    return [
        evaluate.UtteranceRef(
            speaker="spk",
            zip_path=Path("spk.zip"),
            wav_member=f"data/spk/wav/spk_{index:03d}.wav",
            duration_sec=1.0 + (index % 10),
            text="あ" * (index % 7 + 1),
            speech_mos=3.0,
        )
        for index in range(count)
    ]


def test_stratified_sample_is_deterministic_for_a_fixed_seed() -> None:
    import random

    references = _references()
    bucket_of = lambda ref: evaluate.duration_bucket(ref.duration_sec, [2.0, 4.0, 8.0])

    first = evaluate.stratified_sample(references, bucket_of, 8, random.Random("seed:spk"))
    second = evaluate.stratified_sample(references, bucket_of, 8, random.Random("seed:spk"))

    assert [ref.wav_member for ref in first] == [ref.wav_member for ref in second]
    assert len(first) == 8


def test_stratified_sample_spreads_across_duration_buckets() -> None:
    import random

    references = _references()
    bucket_of = lambda ref: evaluate.duration_bucket(ref.duration_sec, [2.0, 4.0, 8.0])
    picked = evaluate.stratified_sample(references, bucket_of, 8, random.Random(0))

    buckets = {bucket_of(ref) for ref in picked}
    assert len(buckets) == 4  # lt2 / 2-4 / 4-8 / ge8 のすべてが出る


def test_stratified_sample_returns_everything_when_count_exceeds_the_pool() -> None:
    import random

    references = _references(5)
    picked = evaluate.stratified_sample(references, lambda _ref: "all", 99, random.Random(0))

    assert len(picked) == 5


def test_stratified_sample_of_zero_is_empty() -> None:
    import random

    assert evaluate.stratified_sample(_references(), lambda _ref: "all", 0, random.Random(0)) == []


def test_choose_sample_records_prefers_phonetic_marks_and_round_robins() -> None:
    records = []
    for speaker in ("a", "b"):
        for index in range(3):
            records.append(
                {
                    "utterance_id": f"{speaker}/{index}",
                    "speaker": speaker,
                    "sokuon": index == 2,
                    "hatsuon": False,
                    "chouon": False,
                }
            )
    chosen = evaluate.choose_sample_records(records, 4, prefer_phonetic_marks=True)

    assert chosen[:2] == ["a/2", "b/2"]  # 促音持ちが先
    assert len(chosen) == 4
    assert len(set(chosen)) == 4


def test_choose_sample_records_respects_the_pair_limit() -> None:
    records = [
        {"utterance_id": f"a/{index}", "speaker": "a", "sokuon": False, "hatsuon": False, "chouon": False}
        for index in range(10)
    ]
    assert evaluate.choose_sample_records(records, 0, True) == []
    assert len(evaluate.choose_sample_records(records, 3, False)) == 3


# --- zip / config の入出力 -----------------------------------------------------


def _write_archive(path: Path, speaker: str, count: int) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        for index in range(count):
            member = f"data/{speaker}/wav/{speaker}_{index:03d}"
            archive.writestr(
                member + ".json",
                json.dumps(
                    {
                        "parakeet_jp_transcription": "あっという間ー",
                        "duration": 1.0 + index,
                        "speechMOS": 3.5,
                    },
                    ensure_ascii=False,
                ),
            )
            archive.writestr(member + ".wav", b"RIFF-not-a-real-wav")


def test_list_speaker_archives_skips_broken_zip(tmp_path) -> None:
    good = tmp_path / "aaaa.zip"
    _write_archive(good, "aaaa", 2)
    (tmp_path / "bbbb.zip").write_bytes(b"partial download")

    usable, skipped = evaluate.list_speaker_archives(tmp_path)

    assert [path.name for path in usable] == ["aaaa.zip"]
    assert len(skipped) == 1
    assert skipped[0]["zip"] == "bbbb.zip"
    assert "BadZipFile" in skipped[0]["reason"]


def test_read_archive_manifest_pairs_json_and_wav(tmp_path) -> None:
    archive_path = tmp_path / "aaaa.zip"
    _write_archive(archive_path, "aaaa", 3)

    references = evaluate.read_archive_manifest(archive_path)

    assert [ref.stem for ref in references] == ["aaaa_000", "aaaa_001", "aaaa_002"]
    assert references[0].speaker == "aaaa"
    assert references[0].duration_sec == 1.0
    assert references[0].utterance_id == "aaaa/aaaa_000"
    assert references[0].text == "あっという間ー"


def test_select_utterances_is_reproducible(tmp_path) -> None:
    for speaker in ("aaaa", "bbbb"):
        _write_archive(tmp_path / f"{speaker}.zip", speaker, 12)
    config = evaluate.DataConfig(
        moe_zip_dir=tmp_path,
        max_speakers=2,
        utterances_per_speaker=4,
        min_duration_sec=1.0,
        max_duration_sec=20.0,
        duration_bucket_edges_sec=(2.0, 4.0, 8.0),
    )
    archives, _ = evaluate.list_speaker_archives(tmp_path)

    first, per_speaker = evaluate.select_utterances(archives, config, seed=7)
    second, _ = evaluate.select_utterances(archives, config, seed=7)
    other_seed, _ = evaluate.select_utterances(archives, config, seed=8)

    assert [ref.utterance_id for ref in first] == [ref.utterance_id for ref in second]
    assert len(first) == 8
    assert {item["speaker"] for item in per_speaker} == {"aaaa", "bbbb"}
    assert all(item["selected"] == 4 for item in per_speaker)
    # 別seedなら（この母集団では）別の選択になる。
    assert [ref.utterance_id for ref in first] != [ref.utterance_id for ref in other_seed]


def test_select_utterances_drops_out_of_range_durations(tmp_path) -> None:
    _write_archive(tmp_path / "aaaa.zip", "aaaa", 12)  # duration = 1.0 .. 12.0
    config = evaluate.DataConfig(
        moe_zip_dir=tmp_path,
        max_speakers=1,
        utterances_per_speaker=99,
        min_duration_sec=3.0,
        max_duration_sec=6.0,
        duration_bucket_edges_sec=(2.0, 4.0, 8.0),
    )
    archives, _ = evaluate.list_speaker_archives(tmp_path)
    selected, per_speaker = evaluate.select_utterances(archives, config, seed=1)

    assert all(3.0 <= ref.duration_sec <= 6.0 for ref in selected)
    assert per_speaker[0]["eligible"] == len(selected) == 4


def test_shipped_config_loads_with_expected_shape() -> None:
    config = evaluate.load_config(REPO_ROOT / "configs" / "japanese" / "vae-reconstruction.yaml")

    assert config.phase == "p1c"
    assert config.metrics.streaming_chunk_latent_frames == 2
    assert len(config.metrics.mel_resolutions) == 3
    assert config.data.moe_zip_dir.is_absolute()
    assert config.checkpoints.audio_vae.name == "audio_vae"
    assert config.thresholds.speaker_cosine_min > 0.0


def test_config_from_mapping_fills_defaults() -> None:
    config = evaluate.config_from_mapping({}, root=REPO_ROOT)

    assert config.phase == "p1c"
    assert config.data.duration_bucket_edges_sec == (2.0, 4.0, 8.0)
    assert config.thresholds == evaluate.Thresholds()


def test_config_rejects_unsorted_duration_edges() -> None:
    with pytest.raises(ValueError):
        evaluate.config_from_mapping({"data": {"duration_bucket_edges_sec": [4.0, 2.0]}})


def test_render_report_contains_the_verdict(tmp_path) -> None:
    config = evaluate.config_from_mapping({}, root=REPO_ROOT)
    checks = _checks()
    metrics = {
        "phase": "p1c",
        "seed": 1,
        "device": "cpu",
        "config_path": "configs/japanese/vae-reconstruction.yaml",
        "checkpoints": {"audio_vae": "model/CuteTTS/weights/audio_vae"},
        "dataset": {
            "speakers": 1,
            "utterances": 1,
            "total_duration_sec": 60.0,
            "skipped_archives": [{"zip": "x.zip", "reason": "BadZipFile"}],
        },
        "overall": {
            name: evaluate.summarize([1.0])
            for name in (
                "mel_l1",
                "snr_db",
                "spectral_convergence",
                "speaker_cosine",
                "rms_delta_db",
                "streaming_snr_db",
                "streaming_max_abs_diff",
                "latent_std",
                "latent_abs_max",
                "duration_sec",
            )
        },
        "by_speaker": {"aaaa": {"utterances": 1, **{n: evaluate.summarize([1.0]) for n in ("mel_l1", "snr_db", "speaker_cosine")}}},
        "by_duration": {"lt2": {"utterances": 1, **{n: evaluate.summarize([1.0]) for n in ("mel_l1", "snr_db", "speaker_cosine")}}},
        "bucket_order": ["lt2", "2-4", "4-8", "ge8"],
        "by_phonetic_mark": {"sokuon=yes": {"utterances": 1, **{n: evaluate.summarize([1.0]) for n in ("mel_l1", "snr_db", "speaker_cosine")}}},
        "streaming": {"chunk_latent_frames": 2, "length_mismatch_count": 0},
        "latent_normalization": {"speech_scaling_factor": 0.4127, "speech_bias_factor": -0.0022},
        "verdict": {
            "verdict": evaluate.verdict_from_checks(checks),
            "checks": [check.to_dict() for check in checks],
            "short_bucket": "lt2",
        },
        "samples": {"pairs": 3},
        "speaker_reference": {
            "intra_speaker": evaluate.summarize([0.95]),
            "inter_speaker": evaluate.summarize([0.30]),
        },
        "worst_cases": {
            "lowest_speaker_cosine": [
                {
                    "utterance_id": "aaaa/aaaa_000",
                    "duration_sec": 2.0,
                    "mel_l1": 1.0,
                    "speaker_cosine": 0.5,
                    "snr_db": 3.0,
                }
            ],
            "highest_mel_l1": [],
        },
    }
    metrics["overall"]["mel_l1_per_resolution"] = {"n_fft512_mels64": evaluate.summarize([1.0])}

    report = evaluate.render_report(metrics, config)

    assert "# P1c" in report
    assert evaluate.VERDICT_LABEL_JA["freeze_ok"] in report
    assert "PESQ" in report  # 未実施metricを明記していること
    assert "x.zip" in report  # skipしたzipを明記していること
    assert "aaaa/aaaa_000" in report  # 失敗例を明記していること
    assert "別話者ペア" in report  # speaker cosineの基準線を明記していること


# --- 実weightを使うテスト（slow） ----------------------------------------------


@pytest.mark.slow
@pytest.mark.skipif(
    not (VAE_CHECKPOINT / "model.safetensors").is_file(),
    reason="model/CuteTTS/weights/audio_vae が無い",
)
def test_streaming_decode_matches_offline_decode_on_real_weights(sine_24k) -> None:
    """streaming decodeが壊れるとS1以降のstreaming評価が無効になるので固定する。"""
    from cutetts.modeling.audio_adapter import AudioAcousticVAEAdapter

    vae = AudioAcousticVAEAdapter(VAE_CHECKPOINT).eval()
    with torch.no_grad():
        latent = vae.encode(sine_24k.reshape(1, 1, -1)).mean
        offline = vae.decode(latent)
        streaming = evaluate.streaming_decode_waveform(vae, latent, chunk_frames=2)

    assert latent.shape[0] == 1
    assert latent.shape[2] == 64
    assert latent.shape[1] == math.ceil(24000 / 1920)  # 12.5 Hz
    assert streaming.shape == offline.shape
    assert evaluate.snr_db(offline, streaming) > 60.0
