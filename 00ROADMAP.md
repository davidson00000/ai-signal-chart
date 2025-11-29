# AI 自動投資システム 開発ロードマップ v2  
**Author: Kousuke × ChatGPT（AI PM Mode）**

本ドキュメントは、  

- AI-Signal-Chart（チャート + シグナル表示アプリ）
- Auto-Strategy Lab（戦略自動設計ラボ）
- 将来の完全自動投資システム

を **最短距離で構築するための公式ロードマップ** です。

参考スレッド（進捗ログ）  
→ https://chatgpt.com/share/6928d7b6-8410-8004-8e8d-a5338dc9d75d

---

## 0. 現在地点（2025-11-28 時点）

### 完了済み / ほぼ完了

- FastAPI バックエンド（データ取得 / シグナル生成 / 紙トレード）
- Streamlit 開発者ダッシュボード（AI-Signal-Chart）
- MA クロス戦略のリアルタイムシグナル
- 紙トレードでの簡易シミュレーション

### これからやる大きなテーマ

1. **バックテストエンジンの実装（過去データでのシミュレーション）**
2. **Auto-Strategy Lab（戦略自動設計システム）の立ち上げ**
3. 将来の **完全自動運用（実弾トレード）** への橋渡し

---

## 1. 全体像：3レイヤーモデル

プロジェクト全体は、ざっくり次の3レイヤーに分けて考える。

1. **Execution Layer（運用レイヤー）**
   - 実際に発注・ポジション管理・リスク制御を行う部分
   - 現在は「紙トレード」のみ

2. **Auto-Strategy Lab Layer（戦略研究レイヤー）※今回の主役**
   - 戦略のバックテスト
   - パラメータ最適化
   - 戦略の自動生成・進化
   - LLM による分析・提案

3. **UI / Dashboard Layer（インタフェースレイヤー）**
   - Streamlit 開発者ダッシュボード
   - 将来の SvelteKit / Web UI

---

## 2. フェーズ別ロードマップ

### Phase 1：AI-Signal-Chart MVP（完了〜調整中）

- チャート + シグナル表示
- 紙トレード（現状は簡易版）
- MA クロスなど基本戦略を動かす

**目的**：  
「とりあえず動くもの」を手元に置き、  
開発のモチベーションと全体像をつかむ。

---

### Phase 2：Backtest Engine v1（最優先）

**ゴール**：  
「過去データを使った戦略のバックテスト」が  
FastAPI + Streamlit でワンクリック実行できる状態。

#### やること

1. `backtest.py` を新規作成
   - `run_backtest(symbol, timeframe, start, end, strategy_name, params)` を実装
   - 内部で `data_feed` + `strategy` + `PaperTrader` を利用

2. FastAPI に `POST /backtest` を追加
   - Request: symbol, timeframe, period, strategy, params
   - Response: PnL / drawdown / win_rate / equity_curve / trades

3. Streamlit に **Backtest タブ** を追加
   - 入力フォーム
   - 実行ボタン
   - 結果のグラフ + テーブル表示

---

### Phase 3：Auto-Strategy Lab v1（パラメータ最適化）

**ゴール**：  
「同じ戦略のパラメータを自動で振って、  
どれが一番いいかをランキングできる」状態。

#### 機能

- グリッドサーチ
  - MA クロスの short / long を範囲指定して総当たり
- ランキングテーブル
  - total_pnl / max_drawdown / win_rate / Sharpe など
- UI
  - Backtest タブの中に「Param Search」セクション追加  
    または「Optimization」タブを新設

---

### Phase 4：Auto-Strategy Lab v2（戦略ブロック化 + 進化）

**ゴール**：  
「戦略そのものを自動で生成・進化させる」土台を作る。

#### やること

1. 戦略ブロックの定義（`strategy/blocks.py`）
   - Indicator / Condition / Logic ブロックをクラス化

2. 戦略の構造化表現（`strategy/rules.py`）
   - MAクロス戦略をブロックで表現し直す

3. ランダム戦略ジェネレータ（`strategy/generator.py`）
   - 一定ルールの範囲でランダムに戦略を生成

4. Evolution Engine v1（`strategy/evolver.py`）
   - selection + mutation を実装した簡易 GA
   - 複数戦略を並列バックテストしてスコアリング

---

### Phase 5：LLM Advisor v1（研究員モード）

**ゴール**：  
「実験ログを LLM に渡し、  
次に試すべき方向性のアドバイスをもらう」仕組みを作る。

#### 機能

- 実験ログ（JSON / CSV）を書き出し
- LLM に投げるためのプロンプトテンプレートを作成
- LLM の返答を整理して保存
- UI から：
  - 「直近 N 実験の要約」
  - 「次に試すべき戦略案」

---

### Phase 6：Execution Layer 強化（将来）

**ゴール**：  
- 証券会社 API（例：楽天証券、SBI 等）が使えるようになったら  
  実運用へつなげられるように準備する。

やること（将来案）：

- リスク管理（最大ポジションサイズ、同時銘柄数）
- 取引ログの永続化（DB）
- 「本番スイッチ」の設計  
  - `paper` / `live` モード切り替え

---

## 3. ディレクトリ構成のイメージ

```text
ai-signal-chart/
  backend/
    data_feed.py
    strategy.py
    paper_trade.py
    backtest.py          # ← 新規
    strategy/
      blocks.py          # ← 戦略ブロック
      rules.py
      generator.py
      optimizer.py
      evolver.py
    llm/
      advisor.py
    main.py              # FastAPI
  dev_dashboard.py       # Streamlit
  00ROADMAP.md
  01architecture.md
  02API_SPEC.md
  03AutoStrategyLab_Blueprint.md
```

---

## 4. 今日・明日の ToDo（現実的な一歩）

### 今日

- `backtest.py` のシグネチャと中身の骨組みを決める
- FastAPI に `POST /backtest` のエンドポイント定義だけ追加
- ダミーで `{"status": "ok"}` を返すところまで

### 明日

- `run_backtest()` を実装（MAクロスのみでOK）
- Streamlit に Backtest タブを追加  
- AAPL 1年分のテストが回るところまで持っていく

---

## 5. このロードマップの使い方

- 新しい機能を思いついたときは、  
  **どの Phase / Layer に属するか** をこのドキュメントで確認する。
- 月に1回くらいの頻度でロードマップを見直し、  
  「やらなくていいもの」を減らしていく。

---

**まとめ**：  
- いま最優先は **Phase 2：Backtest Engine v1**  
- その上に **Auto-Strategy Lab v1 / v2** を乗せる  
- 将来の完全自動運用は、そのさらに先にある

この順番で進めることで、  
「まずは強い戦略を量産するラボ」を手に入れてから、  
安心して実運用に踏み出せるようになる。
