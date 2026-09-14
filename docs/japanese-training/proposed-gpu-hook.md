# GPUフックを自動判定にする提案（2026-09-14）

## いまの問題

`.claude/settings.json` の `PreToolUse` フックが、GPUを使いうるコマンドを
**すべて `ask`** にしている。そのため `evaluate_prosody.py` を回すたびに
承認を求められ、作業が止まる。

## D-023 が本当に守りたいもの

1. **知らないうちに作業機を占有しない**（他の作業ができなくなる）
2. **GPUジョブを並列起動しない**（過去に ssh のtimeoutを死亡と誤認して
   二重起動し、GPUを並列で使った実績がある）

**2は機械的に判定できる。** `nvidia-smi` を見れば、いま走っているかが分かる。
1は「止める」ことではなく「知らせる」ことで足りる。

## 提案する判定

| 状況 | 判定 |
|---|---|
| GPUが空いている（2GB超を使うプロセスが無い） | **自動で許可**。`systemMessage` で知らせるだけ |
| 既にGPUジョブが走っている | **`ask`**（並列起動を防ぐ） |
| `nvidia-smi` が無い・読めない | **`ask`**（分からないときは止める） |

遠隔実行（`ssh` / `scp` / `rsync` / `vastai`）と `git commit` のメッセージ本文は
これまでどおり判定から外す。

## 適用するなら

`.claude/settings.json` の `hooks.PreToolUse[0].hooks[0].command` を次で置き換える。
**自分の権限フックを自分で書き換えることになるので、実行は人が行うこと。**
自動では拒否される（それが正しい）。

```sh
c=$(jq -r '.tool_input.command // ""'); case "$c" in *'git commit'*) c="${c%%git commit*}";; esac; if printf '%s' "$c" | grep -qE '(^|[;&|[:space:]])(ssh|scp|rsync|vastai)([[:space:]]|$)'; then exit 0; fi; if ! printf '%s' "$c" | grep -qE '(python[^;&|]*(reproduce_baseline|evaluate_japanese_vae|cache_audio_latents|benchmark_training_memory|train_continual|evaluate_japanese_cer|diagnose_flow_loss|diagnose_generation|check_reference_following|measure_asr_floor|evaluate_forgetting|build_listening_kit|synthesize_japanese|evaluate_prosody|infer)[.]py|--device[ =]+(cuda|auto)|(^|[;&|[:space:]])cutetts(-demo)?([[:space:]]|$)|pytest[^;&|]*-m[^;&|]*gpu)'; then exit 0; fi; used=$(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits 2>/dev/null) || { jq -nc '{hookSpecificOutput:{hookEventName:"PreToolUse",permissionDecision:"ask",permissionDecisionReason:"GPUの使用状況を確認できませんでした。分からないときは止めます（D-023）。"}}'; exit 0; }; busy=$(printf '%s\n' "$used" | awk '$1>2000{n++} END{print n+0}'); if [ "$busy" -gt 0 ]; then jq -nc '{hookSpecificOutput:{hookEventName:"PreToolUse",permissionDecision:"ask",permissionDecisionReason:"**GPUジョブが既に走っています。** 並列で起動しないでください（D-023）。先行ジョブを待つか止めてから実行してください。リモートのジョブを再起動する前に必ず ps で生存を確認すること。"}}'; else jq -nc '{systemMessage:"[GPU] ローカルGPU(4070 Ti SUPER 16GB)を使います。空いていたので自動で許可しました（D-023の自動判定）。長時間かかる処理は所要時間をユーザーへ伝えること。重い学習は vast.ai へ（D-024）。",hookSpecificOutput:{hookEventName:"PreToolUse",permissionDecision:"allow",permissionDecisionReason:"GPUが空いているため自動許可"}}'; fi
```

## 残る限界

* **所要時間は判定できない。** 30秒の推論も3時間の測定も同じ扱いになる。
  長時間の処理は、これまでどおり**実行前に所要時間を伝える**運用で補う。
* `train_continual` のような重い学習も自動許可になる。止めたければ、
  スクリプト名で `ask` に振り分ける分岐を足す（下の変種）。

### 変種: 学習だけは常に確認する

`busy` の判定の前に次を挟むと、`train_continual` と `cache_audio_latents` は
GPUが空いていても `ask` のままになる。

```sh
if printf '%s' "$c" | grep -qE '(train_continual|cache_audio_latents)[.]py'; then jq -nc '{hookSpecificOutput:{hookEventName:"PreToolUse",permissionDecision:"ask",permissionDecisionReason:"**長時間の学習ジョブです。** 所要時間と費用を見積もってから実行してください（D-023 / D-024）。"}}'; exit 0; fi;
```
