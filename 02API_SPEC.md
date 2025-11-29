# AI 自動投資システム API仕様書  
**File:** `API_SPEC.md`  
**Author:** Kousuke × ChatGPT（API Designer Mode）  

このドキュメントは、AI-Signal-Chart / 自動投資システムにおける  
バックエンド（FastAPI想定）の **REST API 仕様** をまとめたものです。

フロントエンド（Streamlit / SvelteKit）や、将来の他クライアントから  
一貫した形で呼び出せることを目的とします。

---

## 0. 共通事項

### ベースURL（想定）

- 開発環境：`http://localhost:8000`
- 本番環境例：`https://api.example.com`

### 共通ヘッダ

```http
Content-Type: application/json
Accept: application/json
```

（必要に応じて後で認証ヘッダを追加）

---

## 1. Health Check

### 1-1. `GET /health`

システムが動作しているか確認するための簡易エンドポイント。

#### Request

- Method: `GET`
- Path: `/health`
- Query: なし
- Body: なし

#### Response（例）

```json
{
  "status": "ok",
  "version": "0.1.0",
  "timestamp": "2025-11-28T12:34:56Z"
}
```

---

## 2. チャートデータ取得 API

### 2-1. `GET /chart-data`

指定銘柄・期間のチャート用データを返します。  
フロントエンドはこのデータを使ってローソク足・MAなどを描画します。

#### Request

- Method: `GET`
- Path: `/chart-data`
- Query Parameters:

| Name      | Type   | Required | Description                        |
|-----------|--------|----------|------------------------------------|
| symbol    | string | yes      | 銘柄コード（例: `AAPL`, `7203.T`）|
| start     | string | no       | 期間開始日（`YYYY-MM-DD`）        |
| end       | string | no       | 期間終了日（`YYYY-MM-DD`）        |
| interval  | string | no       | 足種別（`1d`, `1h`, `5m` など）   |
| with_ma   | bool   | no       | MAを含めるか（default: true）     |

※ `start` / `end` 未指定の場合、デフォルトは「直近〇日」（サーバ側設定）。

#### Response（例）

```json
{
  "symbol": "AAPL",
  "interval": "1d",
  "data": [
    {
      "time": "2025-11-01T00:00:00Z",
      "open": 190.12,
      "high": 192.34,
      "low": 189.50,
      "close": 191.80,
      "volume": 12345678,
      "ma_short": 190.50,
      "ma_long": 185.30
    },
    {
      "time": "2025-11-02T00:00:00Z",
      "open": 191.90,
      "high": 193.20,
      "low": 190.80,
      "close": 192.50,
      "volume": 9876543,
      "ma_short": 191.10,
      "ma_long": 185.60
    }
  ]
}
```

---

## 3. シグナル取得 API

### 3-1. `GET /signal`

Brain（戦略ロジック）が生成した **売買シグナル** を取得します。

#### Request

- Method: `GET`
- Path: `/signal`
- Query Parameters:

| Name       | Type   | Required | Description                                     |
|------------|--------|----------|-------------------------------------------------|
| symbol     | string | yes      | 銘柄コード                                     |
| date       | string | no       | 判定日（`YYYY-MM-DD`。省略時は最新）           |
| strategy   | string | no       | 戦略名（`ma_cross`, `lightgbm`, etc.）         |
| timeframe  | string | no       | `1d` / `1h` など（必要に応じて）               |

#### Response（例）

```json
{
  "symbol": "AAPL",
  "date": "2025-11-02",
  "timeframe": "1d",
  "strategy": "ma_cross",
  "signal": "BUY",
  "confidence": 0.73,
  "reason": "short_ma crossed above long_ma",
  "price": 192.50,
  "meta": {
    "short_ma": 191.10,
    "long_ma": 185.60
  }
}
```

- `signal`：`BUY` / `SELL` / `HOLD`
- `confidence`：AI 戦略のときは予測確率などを入れる

---

## 4. ペーパートレード（仮想注文）API

### 4-1. `POST /paper-order`

シグナルに基づいて **ペーパートレード注文** を記録します。  
実際の証券会社には発注せず、内部の仮想口座にのみ反映します。

#### Request

- Method: `POST`
- Path: `/paper-order`
- Body（JSON）:

```json
{
  "symbol": "AAPL",
  "side": "BUY",
  "quantity": 10,
  "price": 192.50,
  "signal_id": "2025-11-02-AAPL-ma_cross",
  "order_time": "2025-11-02T09:15:00Z",
  "mode": "market"
}
```

| Field      | Type    | Required | Description                           |
|-----------|---------|----------|---------------------------------------|
| symbol    | string  | yes      | 銘柄コード                            |
| side      | string  | yes      | `BUY` or `SELL`                       |
| quantity  | number  | yes      | 注文数量                              |
| price     | number  | no       | 指値（`mode = limit` のとき）         |
| signal_id | string  | no       | シグナルのID（ログ紐付けのため）     |
| order_time| string  | no       | 注文時間（未指定ならサーバ時刻）     |
| mode      | string  | no       | `market` or `limit`（デフォルト`market`） |

#### Response（例）

```json
{
  "order_id": "paper-20251102-0001",
  "status": "accepted",
  "symbol": "AAPL",
  "side": "BUY",
  "quantity": 10,
  "executed_price": 192.55,
  "executed_at": "2025-11-02T09:15:01Z",
  "pnl": 0.0
}
```

---

## 5. ポジション / 約定 / 損益 API

### 5-1. `GET /positions`

現在の **保有ポジション一覧** を取得します。

#### Request

- Method: `GET`
- Path: `/positions`
- Query Parameters: なし（将来的に `symbol` など追加可）

#### Response（例）

```json
{
  "positions": [
    {
      "symbol": "AAPL",
      "quantity": 30,
      "avg_price": 190.20,
      "current_price": 192.50,
      "unrealized_pnl": 69.0
    },
    {
      "symbol": "7203.T",
      "quantity": 100,
      "avg_price": 2500.0,
      "current_price": 2525.5,
      "unrealized_pnl": 5500.0
    }
  ],
  "total_unrealized_pnl": 5569.0
}
```

---

### 5-2. `GET /trades`

これまでの **ペーパートレード（仮想約定履歴）** を取得します。

#### Request

- Method: `GET`
- Path: `/trades`
- Query Parameters:

| Name    | Type   | Required | Description                    |
|---------|--------|----------|--------------------------------|
| symbol  | string | no       | 絞り込み銘柄                   |
| from    | string | no       | 開始日（`YYYY-MM-DD`）         |
| to      | string | no       | 終了日（`YYYY-MM-DD`）         |
| limit   | int    | no       | 最大件数（default: 100）       |

#### Response（例）

```json
{
  "trades": [
    {
      "order_id": "paper-20251102-0001",
      "symbol": "AAPL",
      "side": "BUY",
      "quantity": 10,
      "price": 192.55,
      "executed_at": "2025-11-02T09:15:01Z",
      "strategy": "ma_cross",
      "signal_id": "2025-11-02-AAPL-ma_cross"
    },
    {
      "order_id": "paper-20251103-0002",
      "symbol": "AAPL",
      "side": "SELL",
      "quantity": 10,
      "price": 195.10,
      "executed_at": "2025-11-03T09:20:12Z",
      "strategy": "ma_cross",
      "signal_id": "2025-11-03-AAPL-ma_cross"
    }
  ]
}
```

---

### 5-3. `GET /pnl`

日次などの損益サマリを取得します。  
ダッシュボードで累積損益グラフを描くためのAPIです。

#### Request

- Method: `GET`
- Path: `/pnl`
- Query Parameters:

| Name   | Type   | Required | Description               |
|--------|--------|----------|---------------------------|
| from   | string | no       | 開始日（`YYYY-MM-DD`）    |
| to     | string | no       | 終了日（`YYYY-MM-DD`）    |
| mode   | string | no       | `daily` or `monthly`      |

#### Response（例）

```json
{
  "mode": "daily",
  "from": "2025-11-01",
  "to": "2025-11-10",
  "pnl": [
    {
      "date": "2025-11-01",
      "realized": 0.0,
      "unrealized": 15.0,
      "equity": 100015.0
    },
    {
      "date": "2025-11-02",
      "realized": 25.0,
      "unrealized": 30.0,
      "equity": 100070.0
    }
  ]
}
```

---

## 6. 将来追加予定のAPI（案）

### 6-1. `POST /backtest`

指定戦略・期間でバックテストを実行し、その結果を返す。

### 6-2. `POST /train-model`

AIモデル（LightGBM, NN など）を再学習させるトリガー。

### 6-3. `GET /config` / `POST /config`

運用パラメータ（使用戦略・リスク許容度・銘柄リストなど）の取得・変更。

---

## 7. 実装メモ（FastAPI想定）

- FastAPI + Pydantic モデルで Request/Response 型を定義する
- エラー時には統一形式で返す：

```json
{
  "error": {
    "code": "INVALID_PARAMETER",
    "message": "symbol is required"
  }
}
```

- ログ用に各エンドポイントで：
  - `symbol`
  - `strategy`
  - `latency_ms`
  - `status_code`  
 などを記録しておくと分析に便利。

---

この `API_SPEC.md` は、  
`architecture.md` / `ROADMAP.md` とセットで、  
AI 自動投資システムの「技術仕様の柱」として運用していきます。
