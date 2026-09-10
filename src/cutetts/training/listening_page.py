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

"""聴取評価ページを組み立てる。

自然性とアクセントは日本語向けの信頼できる自動指標が無く、聴取でしか測れない。
加えて19回の学習を CER だけで判断してきたため、**CERが知覚と対応しているかが
未検証**である。

ページは3つを備える。

* **盲検A/B** — どちらが学習後かを伏せる。先に答えを見ると判断が引きずられる
* **アンカー** — 人間の実音声。ASR床10.4%を測った音源で、そこが到達点
* **書き出し** — 入力を text にして持ち出す。`file://` では clipboard API が
  使えないことがあるので、textarea へも出して手動コピーできるようにする

入力は localStorage へ自動保存する（再読み込みで消えない）。
`file://` では失敗しうるので、読み書きは必ず try/catch で囲む。
"""

from __future__ import annotations

import html
import json

CHECKS = ["自然", "抑揚が不自然", "アクセントが違う", "読み間違い",
          "途切れ・破綻", "声質が不安定"]

_STYLE = """
:root{--fg:#1a1a1a;--mut:#666;--line:#ddd;--bg:#fff;--head:#f6f6f6;--warn:#fff8e1}
body{font:14px/1.7 system-ui,-apple-system,"Segoe UI",sans-serif;margin:0;
     color:var(--fg);background:var(--bg)}
main{max-width:1500px;margin:0 auto;padding:24px}
h1{font-size:22px;margin:0 0 4px}
h2{font-size:17px;margin:34px 0 10px;padding-top:14px;border-top:2px solid var(--line)}
p.note{background:var(--warn);border-left:4px solid #f0b429;padding:12px 16px;max-width:74em}
table{border-collapse:collapse;width:100%;margin-top:10px}
th,td{border:1px solid var(--line);padding:8px;vertical-align:top;text-align:left}
th{background:var(--head);font-size:13px;position:sticky;top:0;z-index:2}
.n{width:2.6em;color:var(--mut);font-variant-numeric:tabular-nums}
.g{width:4.5em;font-size:12px;color:var(--mut)}
.t{width:20em;font-weight:600;font-size:13px}
.sp{margin-top:4px;font-weight:400;font-size:12px;color:var(--mut)}
.lb{margin-top:4px;font-size:11px;color:var(--mut)}
audio{width:210px;height:32px}
.pref{width:12em;font-size:12px;white-space:nowrap}
.pref label{margin-right:8px}
.ckcell{min-width:26em}
.ck{display:inline-block;margin:0 10px 4px 0;font-size:12px;white-space:nowrap}
.note-in{width:14em;font-size:12px;padding:3px 6px;border:1px solid var(--line);
         border-radius:3px}
button{font:inherit;padding:7px 16px;border:1px solid #999;border-radius:5px;
       background:#fafafa;cursor:pointer}
button:hover{background:#f0f0f0}
button.primary{background:#1a6feb;color:#fff;border-color:#1a6feb;font-weight:600}
button.primary:hover{background:#155ec9}
#key{margin-top:12px;font-size:13px}
.hid{display:none}
ol.steps>li{margin-bottom:6px}
code{background:#f2f2f2;padding:1px 5px;border-radius:3px;font-size:12px}
#export{width:100%;height:19em;font:12px/1.5 ui-monospace,Consolas,monospace;
        margin-top:10px;padding:10px;border:1px solid var(--line);border-radius:5px}
#saved{margin-left:10px;font-size:12px;color:var(--mut)}
"""


def _checks(prefix: str) -> str:
    boxes = " ".join(
        f'<label class=ck><input type=checkbox data-k="{prefix}.{i}"> {html.escape(c)}</label>'
        for i, c in enumerate(CHECKS))
    return boxes + f'<input class=note-in data-k="{prefix}.note" placeholder="メモ">'


def _audio(file: str) -> str:
    return f'<audio controls preload=none src="audio/{html.escape(file)}"></audio>'


def render(manifest: dict, anchors: list[dict]) -> str:
    """manifest と アンカーから1枚のHTMLを作る。"""
    entries = manifest["entries"]
    left_is_trained = manifest["left_is_trained"]

    by_index: dict[int, list[dict]] = {}
    for entry in entries:
        by_index.setdefault(entry["index"], []).append(entry)

    ab_rows: list[str] = []
    j2_rows: list[str] = []
    for index in sorted(by_index):
        group = by_index[index][0]["group"]
        text = by_index[index][0]["text"]
        if group == "数詞":
            raw = next(e for e in by_index[index] if e["model"] == "trained" and not e["j2"])
            j2 = next(e for e in by_index[index] if e["model"] == "trained" and e["j2"])
            base = next(e for e in by_index[index] if e["model"] == "base" and e["j2"])
            j2_rows.append(
                f'<tr><td class=n>{index:02d}</td>'
                f'<td class=t>{html.escape(text)}'
                f'<div class=sp>読み: {html.escape(j2["spoken"] or "")}</div></td>'
                f'<td>{_audio(base["file"])}<div class=lb>未学習＋J2</div></td>'
                f'<td>{_audio(raw["file"])}<div class=lb>学習後・J2なし</div></td>'
                f'<td>{_audio(j2["file"])}<div class=lb>学習後・J2あり</div></td>'
                f'<td class=ckcell>{_checks(f"j2.{index}")}</td></tr>')
            continue
        base = next(e for e in by_index[index] if e["model"] == "base")
        trained = next(e for e in by_index[index] if e["model"] == "trained")
        first_trained = bool(left_is_trained.get(str(index)))
        left, right = (trained, base) if first_trained else (base, trained)
        ab_rows.append(
            f'<tr data-idx="{index}" data-left="{"trained" if first_trained else "base"}">'
            f'<td class=n>{index:02d}</td><td class=g>{html.escape(group)}</td>'
            f'<td class=t>{html.escape(text)}</td>'
            f'<td>{_audio(left["file"])}</td><td>{_audio(right["file"])}</td>'
            f'<td class=pref>'
            f'<label><input type=radio name="p{index}" data-k="ab.{index}.pick" value=A> A</label>'
            f'<label><input type=radio name="p{index}" data-k="ab.{index}.pick" value=B> B</label>'
            f'<label><input type=radio name="p{index}" data-k="ab.{index}.pick" value=X> 差なし</label>'
            f'</td><td class=ckcell>{_checks(f"ab.{index}")}</td></tr>')

    anchor_rows = "".join(
        f'<tr><td class=n>{i:02d}</td><td class=t>{html.escape(a["text"])}</td>'
        f'<td>{_audio(a["file"])}</td>'
        f'<td class=ckcell>{_checks(f"anchor.{i}")}</td></tr>'
        for i, a in enumerate(anchors))

    texts = {str(i): by_index[i][0]["text"] for i in by_index}
    groups = {str(i): by_index[i][0]["group"] for i in by_index}

    return f"""<!doctype html><meta charset=utf-8>
<title>CuteTTS 日本語 聴取評価</title>
<style>{_STYLE}</style>
<main>
<h1>CuteTTS 日本語継続学習 — 聴取評価</h1>
<p class=note><b>ローカル聴取専用。</b>音声は学習データのライセンス上、公開・コミットしてはいけません
（<code>artifacts/</code> はgitignore済み）。入力はこの端末のブラウザにのみ保存されます。</p>

<p><b>なぜ聴くのか。</b>S1のゴールのうち<b>自然性とアクセントだけが未測定</b>で、
日本語向けの信頼できる自動指標が存在しません。加えて19回の学習を CER だけで
判断してきたため、<b>CERが知覚と対応しているかすら未検証</b>です。</p>

<p>参考値（評価set v3・600文）: 未学習 <b>35.86%</b> → 30,000 step <b>20.10%</b>。ASR床は 10.4%。</p>

<h2>1. 盲検A/B（{len(ab_rows)}問）</h2>
<p>A と B のどちらが未学習でどちらが学習後かは<b>伏せてあります</b>（項目ごとにランダム）。
先に全部聴いて選んでから、最後の「答えを表示」を押してください。
<b>先に答えを見ると判断が引きずられます。</b></p>
<table>
<tr><th class=n>#</th><th class=g>種別</th><th class=t>原文</th><th>A</th><th>B</th>
<th>良い方</th><th>気づいた点</th></tr>
{''.join(ab_rows)}
</table>
<p style="margin-top:14px"><button id=score class=primary>集計</button>
<button id=reveal style="margin-left:8px">答えを表示</button></p>
<div id=key class=hid></div>

<h2>2. アンカー: 人間の実音声（{len(anchors)}本）</h2>
<p>学習データに含まれる<b>実際の人間の声</b>です。TTSと聴き比べて、
どこが違うかを見てください。ASR床10.4%はこの音声から測った値で、
<b>ここが到達点</b>です。</p>
<table><tr><th class=n>#</th><th class=t>原文</th><th>音声</th><th>気づいた点</th></tr>
{anchor_rows}</table>

<h2>3. 漢数字の読み（J2の効き）（{len(j2_rows)}問）</h2>
<p>推論の直前で <code>千二百八十</code> → <code>せんにひゃくはちじゅう</code> と展開します。
専用評価set200文で <b>-11.80pt</b>（有意）。<b>耳で確かめてください。</b></p>
<table>
<tr><th class=n>#</th><th class=t>原文 / 読み</th><th>未学習＋J2</th>
<th>学習後・J2なし</th><th>学習後・J2あり</th><th>気づいた点</th></tr>
{''.join(j2_rows)}
</table>

<h2>4. 結果の書き出し</h2>
<p>下のボタンで入力内容をtextにします。<b>そのまま貼って渡してください。</b>
クリップボードが使えない場合は、textareaを全選択してコピーしてください。</p>
<p><button id=dump class=primary>結果を書き出す</button>
<button id=clear style="margin-left:8px">入力を消す</button>
<span id=saved></span></p>
<textarea id=export placeholder="ここに結果が出ます"></textarea>

<h2>5. 見ていただきたい点</h2>
<ol class=steps>
<li><b>盲検の正答率</b>。半々なら「CERは下がったが聴いて分からない」ということ</li>
<li><b>アクセントの誤り</b>があった項目番号。自動測定できないのでここでしか拾えない</li>
<li><b>人間の音声との差</b>がどこにあるか（抑揚 / 声質 / 間 / ノイズ）</li>
<li><b>J2</b> が耳でも改善して聞こえるか</li>
</ol>
</main>
<script>
const CHECKS = {json.dumps(CHECKS, ensure_ascii=False)};
const LEFT = {json.dumps(left_is_trained, ensure_ascii=False)};
const TEXTS = {json.dumps(texts, ensure_ascii=False)};
const GROUPS = {json.dumps(groups, ensure_ascii=False)};
const KEY = 'cutetts-listen-v1';

const fields = () => document.querySelectorAll('[data-k]');
function save() {{
  const data = {{}};
  fields().forEach(el => {{
    if (el.type === 'checkbox') {{ if (el.checked) data[el.dataset.k] = 1; }}
    else if (el.type === 'radio') {{ if (el.checked) data[el.dataset.k] = el.value; }}
    else if (el.value) data[el.dataset.k] = el.value;
  }});
  try {{ localStorage.setItem(KEY, JSON.stringify(data)); }} catch (e) {{}}
  const el = document.getElementById('saved');
  el.textContent = '保存しました';
  setTimeout(() => {{ el.textContent = ''; }}, 1200);
}}
function restore() {{
  let data = null;
  try {{ data = JSON.parse(localStorage.getItem(KEY) || 'null'); }} catch (e) {{}}
  if (!data) return;
  fields().forEach(el => {{
    const v = data[el.dataset.k];
    if (v === undefined) return;
    if (el.type === 'checkbox') el.checked = true;
    else if (el.type === 'radio') el.checked = (el.value === v);
    else el.value = v;
  }});
}}
restore();
document.addEventListener('change', save);
document.addEventListener('input', e => {{ if (e.target.matches('.note-in')) save(); }});

function tally() {{
  let hit = 0, total = 0, tie = 0;
  document.querySelectorAll('tr[data-idx]').forEach(tr => {{
    const picked = tr.querySelector('input[type=radio]:checked');
    if (!picked) return;
    total++;
    if (picked.value === 'X') {{ tie++; return; }}
    const leftTrained = tr.dataset.left === 'trained';
    if ((picked.value === 'A') === leftTrained) hit++;
  }});
  return {{hit, total, tie, judged: total - tie}};
}}
document.getElementById('score').onclick = () => {{
  const t = tally();
  const pct = t.judged ? (100 * t.hit / t.judged).toFixed(0) : '—';
  document.getElementById('key').className = '';
  document.getElementById('key').innerHTML =
    `<p>回答 ${{t.total}} 問（うち「差なし」${{t.tie}}）。学習後を選べたのは ` +
    `<b>${{t.hit}}/${{t.judged}}</b>（${{pct}}%）。<br>` +
    `半々（50%前後）なら、CERは下がっているが<b>聴いて区別できない</b>ということ。` +
    `70%を超えるなら知覚できる差がある。</p>`;
}};
document.getElementById('reveal').onclick = () => {{
  document.querySelectorAll('tr[data-idx]').forEach(tr => {{
    if (tr.dataset.revealed) return;
    tr.dataset.revealed = '1';
    const leftTrained = tr.dataset.left === 'trained';
    const cells = tr.querySelectorAll('td');
    cells[3].insertAdjacentHTML('beforeend',
      `<div class=lb><b>${{leftTrained ? '学習後' : '未学習'}}</b></div>`);
    cells[4].insertAdjacentHTML('beforeend',
      `<div class=lb><b>${{leftTrained ? '未学習' : '学習後'}}</b></div>`);
  }});
}};
function collect(prefix, index) {{
  const on = [];
  CHECKS.forEach((c, i) => {{
    const el = document.querySelector(`[data-k="${{prefix}}.${{index}}.${{i}}"]`);
    if (el && el.checked) on.push(c);
  }});
  const note = document.querySelector(`[data-k="${{prefix}}.${{index}}.note"]`);
  return {{on, note: note ? note.value.trim() : ''}};
}}
document.getElementById('dump').onclick = () => {{
  const t = tally();
  const pct = t.judged ? (100 * t.hit / t.judged).toFixed(0) : '—';
  const out = ['# CuteTTS 聴取評価の結果', ''];
  out.push(`## 盲検A/B: 学習後を選べたのは ${{t.hit}}/${{t.judged}}（${{pct}}%）` +
           `　回答 ${{t.total}} 問 / 差なし ${{t.tie}}`);
  document.querySelectorAll('tr[data-idx]').forEach(tr => {{
    const i = tr.dataset.idx;
    const picked = tr.querySelector('input[type=radio]:checked');
    const {{on, note}} = collect('ab', i);
    if (!picked && !on.length && !note) return;
    const leftTrained = tr.dataset.left === 'trained';
    let verdict = '未回答';
    if (picked) verdict = picked.value === 'X' ? '差なし'
      : (((picked.value === 'A') === leftTrained) ? '学習後を選択' : '未学習を選択');
    const bits = [verdict];
    if (on.length) bits.push(on.join('/'));
    if (note) bits.push(`「${{note}}」`);
    out.push(`- [${{i}}] ${{GROUPS[i]}} ${{TEXTS[i]}}  →  ${{bits.join(' | ')}}`);
  }});
  out.push('', '## アンカー（人間の実音声）');
  document.querySelectorAll('[data-k^="anchor."]').forEach(el => {{}});
  for (let i = 0; i < 99; i++) {{
    if (!document.querySelector(`[data-k="anchor.${{i}}.0"]`)) break;
    const {{on, note}} = collect('anchor', i);
    if (on.length || note) out.push(`- [${{i}}] ${{on.join('/')}}${{note ? ' 「' + note + '」' : ''}}`);
  }}
  out.push('', '## 数詞（J2）');
  Object.keys(TEXTS).forEach(i => {{
    if (GROUPS[i] !== '数詞') return;
    const {{on, note}} = collect('j2', i);
    if (on.length || note) out.push(`- [${{i}}] ${{TEXTS[i]}}  →  ${{on.join('/')}}${{note ? ' 「' + note + '」' : ''}}`);
  }});
  const text = out.join('\\n');
  const box = document.getElementById('export');
  box.value = text;
  box.select();
  try {{ navigator.clipboard.writeText(text); }} catch (e) {{}}
  try {{ document.execCommand('copy'); }} catch (e) {{}}
}};
document.getElementById('clear').onclick = () => {{
  if (!confirm('入力をすべて消します。よろしいですか')) return;
  try {{ localStorage.removeItem(KEY); }} catch (e) {{}}
  location.reload();
}};
</script>
"""
