import io, os, re, shutil, subprocess
os.chdir(r"C:\Users\yuta\Desktop\Private\CuteTTS")
PY = os.path.abspath(r".venv\Scripts\python.exe")
SRC = "src/cutetts/training/objectives.py"
base = io.open(SRC, encoding="utf-8").read()
MUTANTS = [
 ("stopラベルを1つ手前へ", "last_index = lengths - 1", "last_index = (lengths - 2).clamp(min=0)", "tests/training/test_stop_targets.py"),
 ("velocity符号反転", "velocity = clean - noise", "velocity = noise - clean", "tests/training/test_flow_objective.py"),
 ("補間のt反転", "x_t = (1.0 - t_view) * noise + t_view * clean", "x_t = t_view * noise + (1.0 - t_view) * clean", "tests/training/test_flow_objective.py"),
 ("joint dropout無効化", "reference = speaker.clone() if config.joint else draw(config.reference)", "reference = draw(config.reference)", "tests/training/test_condition_dropout.py"),
 ("flow lossのmask無視", "weights = batch.loss_mask.to(error.dtype)", "weights = torch.ones_like(batch.loss_mask, dtype=error.dtype)", "tests/training/test_flow_objective.py"),
 ("stopのpadding除外を無効化", "numerator = (per_position * valid).sum()", "numerator = per_position.sum()", "tests/training/test_stop_targets.py"),
 ("STOP_STOPを0にする", "STOP_STOP = 1", "STOP_STOP = 0", "tests/training/test_stop_targets.py"),
 ("copiesを無視して1固定", "clean = target_patches.repeat(copies, 1, 1)", "clean = target_patches.repeat(1, 1, 1)", "tests/training/test_flow_objective.py"),
 ("speaker dropoutを無効化", 'speaker_out[drop["speaker"]] = 0.0', "pass", "tests/training/test_condition_dropout.py"),
]
detected = 0
try:
    for label, a, b, tgt in MUTANTS:
        if a not in base:
            print(f"  {label:26s}  パターン未発見 — スキップ"); continue
        io.open(SRC, "w", encoding="utf-8", newline="\n").write(base.replace(a, b, 1))
        for root, dirs, _ in os.walk("src"):
            for d in list(dirs):
                if d == "__pycache__":
                    shutil.rmtree(os.path.join(root, d), ignore_errors=True)
        r = subprocess.run([PY, "-m", "pytest", tgt, "-p", "no:warnings", "--tb=no", "-q", "-p", "no:cacheprovider"],
                           capture_output=True, text=True, errors="replace")
        out = r.stdout + r.stderr
        hit = r.returncode != 0            # pytest は失敗時に非0を返す
        n = len(re.findall(r"^FAILED ", out, re.M))
        detected += 1 if hit else 0
        print(f"  {label:26s}  {'検出 (' + str(n) + '件失敗)' if hit else '見逃し ***'}")
finally:
    io.open(SRC, "w", encoding="utf-8", newline="\n").write(base)
print(f"\n検出率: {detected}/{len(MUTANTS)}")
