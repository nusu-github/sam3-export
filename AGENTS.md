# AGENTS.md — sam3-export

このリポジトリで働くエージェントは **本ファイルを必ず読む**。

ワークスペース全体（計画・NOTES・`sam3/` 参照ツリー）を触る場合は、親の
[`../AGENTS.md`](../AGENTS.md) も同じ規約である。内容が衝突したら **より具体的な DoD
（今のマイルストーン）と本ファイルの禁止事項** を優先する。

---

## 0. 30 秒で守ること

1. **今のマイルストーン DoD だけ** を満たす。次 M の仕事と「将来の堅牢化」を同梱しない。  
2. **本質（誤認防止・比較可能性・最小契約）> 頑丈さ（先回り validator / 文書増殖）**。  
3. **四層を混ぜない**: Public API → Host Runtime → Components → Plans/Artifacts。  
4. **component ≠ artifact。** smoke 成功 ≠ release。legacy shipped ≠ future default。  
5. **draft は draft の厚み。** 使われない cross-rule・synthetic 語彙・巨大 negative 行列を足さない。  
6. **正本を増やさない。** 同じ exclusive list を README/Space/catalog にフルコピーしない。  
7. **v1 に無い metadata を推測で埋めない。** 不明は未記録のまま書く。  
8. 完了時は DoD 照合。fortification の量を成果に数えない。

---

## 1. 読む順番

| 優先 | 文書 |
|---|---|
| 1 | 本 `AGENTS.md` |
| 2 | `../docs/SAM3_EXPORT_IMPLEMENTATION_PLAN.md`（M の DoD・非目的） |
| 3 | `docs/GLOSSARY.md` |
| 4 | 当該タスクの catalog / schema / コード |

- `../SAM3_EXPORT_PARTITIONING_NOTES.md` … 仮説メモ。default の正本ではない。  
- `docs/EXPORT_CUTS.md` … **public** artifact のみ。  
- `docs/INTERNAL_COMPONENTS.md` … component / test-only（**tensor 契約の作業地図を消さない**）。  
- `docs/DEPLOYMENT_PLANS.md` … composition と dispatch 意図。  
- `docs/MANIFEST_V2.md` + `schemas/` … draft 契約。production loader 完成前に要塞化しない。

---

## 2. 本質 vs 過剰（このリポジトリで繰り返した失敗）

| やる（本質） | やらない（過剰な頑丈さ） |
|---|---|
| scope label と exclusions で誤認を止める | 全将来 backend 用の抽象 framework |
| M が要求する必須 field を列挙する | draft schema に path/parity/cross-class の巨大 if/then |
| public / internal / fixture を分ける | 文書を 6 本に分裂させ同じ文を複製 |
| 比較可能な測定と decision record | 単発ベンチを設計定数化 |
| 既存正本の更新 | 「inventory」など計画に無い常設正本の増殖 |
| 実装者が使う tensor I/O を internal に残す | catalog 純粋化で作業地図を削除 |

**「壊れないように厚くする」より「間違ったものを default / shipped / production と呼ばない」を優先する。**

迷ったら次の 3 問:

1. 今の DoD を満たす最小変更は何か？  
2. それを足さないと誰が・いつ・どう誤るか？  
3. 次の M で入れると遅すぎるか？  

(2) が「なんとなく不安」だけなら **入れない**。

---

## 3. アーキテクチャ

```text
Public API  ->  Host Runtime  ->  Canonical Components  ->  Deployment Plans / Artifacts
```

- Public API に `OrtValue` / backend tensor 名 / slot を晒さない。  
- Host が巨大中間 tensor を CPU/NumPy でつなぐ経路を **正規 default にしない**。  
- wrapper があることと、配布 ONNX があることを同一視しない。  
- cut の根拠は lifetime / fan-out / host policy / backend 互換のみ。クラス境界は不可。

### 現行出荷の scope（固定）

> **SAM3 text-only image PCS / legacy split v1**

- geometry/exemplar、production interactive、video、semantic、SAM3.1 は **含まない**。  
- legacy であり **future default ではない**。M1 decision 前に default 置換しない。  
- tiny `PromptEncode` / `InteractiveDecode` は **test-only fixture**。

lifecycle（shipped/candidate/…）と dispatch_role（default/optional/legacy/…）は **別軸**。

---

## 4. マイルストーン規律

| M | やってよい | やってはいけない |
|---|---|---|
| M0 | 用語・label・catalog 分離・v2 **草案** | graph 変更、破壊不能 schema、runtime 改造 |
| M1 | E1–E3 同一条件比較と decision record | 条件違いの比較で default 独断 |
| M2 | 承認済み recipe + 最小 Host ABI | backend ABI の Public 漏洩、legacy と default 混同 |
| M3+ | その M の DoD のみ | 未決を黙って default 埋め込み |

DoD 外の仕事を「品質向上」として同梱しない。次 M へ明示的に残す。

---

## 5. 文書

- 正本は少なく。新 `.md` の前に既存更新を検討。  
- 計画の分担（GLOSSARY / EXPORT_CUTS / INTERNAL / DEPLOYMENT_PLANS / manifest / decision）に無い常設文書を増やすなら、owner と「何の正本か」を一文で言えること。  
- README / Space は要約 + リンク。admission 規則の全文コピー禁止。  
- public catalog から落とした tensor 契約は **internal に移す**。削除して消さない。

---

## 6. Schema / manifest

- draft に必要なもの: 必須 block、主要型、今使う enum、id/hash 形。  
- 前倒し禁止の例: 過剰 path regex、未使用 cross-ref 完全性、parity 4 段の schema 一意盛り、巨大 classification マトリクス、production 語彙への `synthetic` 汚染。  
- doc が「semantic linter は later」と書いた厚みを、schema に先行実装しない。  
- v1 と v2 は `format` で dispatch。v1 欠落を想像で埋めない。  
- 可能なら real candidate に近い skeleton を fixture にする。

---

## 7. コード

- タスクに必要な最小 diff。無関係 refactor を混ぜない。  
- `sam3.export` は tensor-only。tokenize / NMS / loop は `sam3.runtime`。  
- 固定 profile を勝手に汎用 dynamic 化しない。  
- variant 固有値を generic 暗黙 default に隠さない。  
- SAM3 base batch ≠ SAM3.1 Multiplex。ABI を共用しない。  
- selected-K を device-resident でない backend で CPU 連結して public 化しない。

---

## 8. テスト

- リスク比例。draft に production 並み negative 行列を要求しない。  
- DoD gate と公開契約破壊を優先。  
- `export_smoke.py` は internal round-trip。**release gate ではない。**  
- 変更後: 影響範囲に応じて `make test` / `make quality`。

```bash
uv sync --all-groups --inexact
make quality
make test
PYTHONPATH=src python scripts/export_smoke.py
```

---

## 9. 実験と decision

- 同一 fixture / dtype / warmup / backend / handoff で比較。  
- 環境・commit・中央値/p95・VRAM・copy bytes・artifact size・parity 段階を記録。  
- 1 回の速さで cut を固定しない。  
- decision record に `Decision` と `Applicable profiles` が無ければ採否無効。

---

## 10. 禁止アンチパターン（短縮版）

1. 万能 ONNX / 万能 manifest / 万能 runtime  
2. component ↔ artifact 一対一  
3. smoke = 製品対応  
4. tiny fixture = production  
5. legacy = future default  
6. 文書増殖で責任をぼかす  
7. draft の production-grade 要塞化  
8. 作業用地図（tensor I/O）の削除  
9. 未測定の設計定数化  
10. selected-K の CPU round-trip 正規化  
11. TrackerStep を Multiplex 実装扱い  
12. DoD 外 fortification を「品質」として押し込む  
13. 同一文言の多面コピー  
14. v1 metadata の想像補完  

---

## 11. 終了前チェック

- [ ] DoD のみ満たしている  
- [ ] DoD 外の schema/文書/framework/テストを足していない  
- [ ] scope / exclusions がユーザー向け面で正しい  
- [ ] public / internal / fixture / legacy を混同していない  
- [ ] 新正本は不可避だった（既存更新ではダメだった）  
- [ ] 移した実用情報は別正本に残った  
- [ ] 根拠なし default 化をしていない  
- [ ] 差分は最小。graph を触る必然がある  

No があるなら削るか、次 M に明示して残す。

---

## 12. レビューで「過剰」と言われたとき

防御層をさらに足して応答しない。  
**削る案・移す M・残す最小本質** を先に出す。
