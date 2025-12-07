# Cloudflare Tunnel でモバイルビューに外部アクセスする方法

## 概要

このガイドでは、Mac mini上で動作している `mobile_live_view.py`（Streamlit）に、
スマートフォンから**安全に外部アクセス**するための Cloudflare Tunnel の設定方法を説明します。

Cloudflare Tunnel を使用することで、以下が実現できます：

- 🔒 **セキュア**: ファイアウォールやポートフォワーディング不要
- 🌍 **どこからでもアクセス**: インターネット経由でスマホから閲覧可能
- 🚀 **簡単**: `cloudflared` コマンド一つでトンネル開始

## 前提条件

### 必須要件

1. **Cloudflareアカウント**
   - [Cloudflare](https://dash.cloudflare.com/sign-up) でアカウント作成（無料プランでOK）

2. **ドメイン管理**
   - Cloudflareでドメインを管理していること
   - 例: `example.com` を Cloudflare DNS で管理

3. **ローカル環境**
   - Mac mini 上で `mobile_live_view.py` が動作していること
   - Streamlit が `localhost:8501` でアクセス可能なこと

4. **ネットワーク**
   - Mac mini がインターネットに接続されていること

## ステップ1: cloudflared のインストール

### macOS の場合

Homebrew を使用してインストール：

```bash
# cloudflared をインストール
brew install cloudflared

# バージョン確認
cloudflared --version
```

**出力例**:
```
cloudflared version 2024.11.1 (built 2024-11-15-1234 UTC)
```

### 手動インストール（Homebrewを使わない場合）

```bash
# ダウンロード（バージョンは最新を確認）
curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-darwin-amd64.tgz | tar -xz

# 実行権限を付与
chmod +x cloudflared

# /usr/local/bin に移動
sudo mv cloudflared /usr/local/bin/

# 確認
cloudflared --version
```

## ステップ2: Cloudflare に認証

```bash
# ブラウザが開き、Cloudflare にログインする
cloudflared tunnel login
```

**何が起こるか**:
1. ブラウザが自動的に開く
2. Cloudflareのログイン画面が表示される
3. ログイン後、どのドメインでトンネルを使うか選択
4. 認証が完了すると、認証ファイルが `~/.cloudflared/cert.pem` に保存される

## ステップ3: トンネルの作成

```bash
# トンネルを作成（名前は好きなもの）
cloudflared tunnel create exiton-mobile

# 成功すると、トンネルIDとクレデンシャルファイルのパスが表示される
```

**出力例**:
```
Tunnel credentials written to /Users/yourname/.cloudflared/<TUNNEL_ID>.json
Created tunnel exiton-mobile with id <TUNNEL_ID>
```

**重要**: `<TUNNEL_ID>` をメモしておいてください。

## ステップ4: DNS設定

作成したトンネルにサブドメインを割り当てます。

```bash
# DNSレコードを追加
cloudflared tunnel route dns exiton-mobile signal.example.com
```

**パラメータ**:
- `exiton-mobile`: トンネル名
- `signal.example.com`: 使用したいサブドメイン（任意）

**注意**: `example.com` を実際のドメインに置き換えてください。

**出力例**:
```
Created CNAME record for signal.example.com, pointing to <TUNNEL_ID>.cfargotunnel.com
```

## ステップ5: 設定ファイルの作成

`~/.cloudflared/config.yml` を作成（または編集）：

```bash
# エディタで開く
nano ~/.cloudflared/config.yml
```

以下の内容を記述：

```yaml
# Tunnel ID（ステップ3で取得）
tunnel: <TUNNEL_ID>

# クレデンシャルファイルのパス
credentials-file: /Users/yourname/.cloudflared/<TUNNEL_ID>.json

# Ingress rules
ingress:
  # signal.example.com へのアクセスを localhost:8501 に転送
  - hostname: signal.example.com
    service: http://localhost:8501
  
  # すべての他のリクエストは404を返す（セキュリティ）
  - service: http_status:404
```

**重要な置き換え**:
- `<TUNNEL_ID>`: ステップ3で取得したID
- `/Users/yourname/`: 実際のユーザーパス
- `signal.example.com`: 実際に使用するドメイン

## ステップ6: トンネルの起動

### テスト起動（フォアグラウンド）

まずは手動で起動してテスト：

```bash
cloudflared tunnel run exiton-mobile
```

**正常起動の出力例**:
```
2024-12-07T15:00:00Z INF Starting tunnel tunnelID=<TUNNEL_ID>
2024-12-07T15:00:01Z INF Connection registered connIndex=0
2024-12-07T15:00:01Z INF Connection registered connIndex=1
2024-12-07T15:00:01Z INF Connection registered connIndex=2
2024-12-07T15:00:01Z INF Connection registered connIndex=3
```

この状態で、スマホのブラウザから `https://signal.example.com` にアクセスしてみてください。

### バックグラウンド起動

正常動作を確認したら、Ctrl+Cで停止し、バックグラウンドで起動：

```bash
# バックグラウンドで起動
nohup cloudflared tunnel run exiton-mobile > /tmp/cloudflared.log 2>&1 &

# プロセス確認
ps aux | grep cloudflared
```

### システム起動時に自動起動（推奨）

launchd を使用してシステム起動時に自動起動：

```bash
# サービスとしてインストール
sudo cloudflared service install
```

サービスの管理：

```bash
# 起動
sudo launchctl start com.cloudflare.cloudflared

# 停止
sudo launchctl stop com.cloudflare.cloudflared

# ステータス確認
sudo launchctl list | grep cloudflare
```

## ステップ7: アクセス確認

### スマホからアクセス

1. スマホのブラウザを開く
2. `https://signal.example.com` にアクセス
3. モバイルビューが表示されることを確認

### 動作確認

- ✅ シグナルカードが表示される
- ✅ 詳細情報が展開できる
- ✅ 自動更新が機能する
- ✅ サイドバーから設定変更できる

## 完全な起動手順（まとめ）

すべてセットアップ後の日常的な起動手順：

```bash
# ターミナル1: バックエンド起動
cd /path/to/ai-signal-chart
uvicorn backend.main:app --host 0.0.0.0 --port 8001

# ターミナル2: モバイルビュー起動
cd /path/to/ai-signal-chart
streamlit run mobile_live_view.py

# ターミナル3: Cloudflare Tunnel起動（自動起動していない場合）
cloudflared tunnel run exiton-mobile
```

**または、すべて自動起動する場合**:
- バックエンドとStreamlitをsystemdやlaunchdで自動起動設定
- Cloudflare Tunnelもサービスとしてインストール

## セキュリティ上の注意点

### ⚠️ 重要: 認証とアクセス制御

**現状**:
- `mobile_live_view.py` には認証機能がありません
- URLを知っている人は誰でもアクセス可能です

**推奨対策**:

#### 1. Cloudflare Access（推奨）

Cloudflareの無料プランで使える認証機能：

```bash
# Accessポリシーを設定（Cloudflare Dashboardから）
1. Zero Trust > Access > Applications
2. "Add an application" をクリック
3. Self-hosted を選択
4. Application domain: signal.example.com
5. Policyを設定:
   - ルール名: "Allowed Users"
   - Include: メールアドレスまたはドメイン
   - 例: yourname@gmail.com
```

これにより、指定したメールアドレスでのログインが必須になります。

#### 2. Streamlit認証（コード側で実装）

`mobile_live_view.py` に認証機能を追加：

```python
import hmac

def check_password():
    """パスワード認証"""
    def password_entered():
        if hmac.compare_digest(st.session_state["password"], "your_secret_password"):
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if st.session_state.get("password_correct", False):
        return True

    st.text_input("Password", type="password", on_change=password_entered, key="password")
    if "password_correct" in st.session_state:
        st.error("😕 Password incorrect")
    return False

# main()の最初で呼び出し
if not check_password():
    st.stop()
```

#### 3. IP制限

特定のIPアドレスからのみアクセス許可（Cloudflare Firewall Rules）：

1. Cloudflare Dashboard > Security > WAF
2. Create Firewall Rule:
   - Expression: `(ip.src ne YOUR_PHONE_IP)`
   - Action: Block

### その他のセキュリティベストプラクティス

1. **HTTPS必須**: Cloudflare Tunnelは自動的にHTTPSを使用
2. **クレデンシャルファイルの保護**:
   ```bash
   chmod 600 ~/.cloudflared/<TUNNEL_ID>.json
   ```
3. **ログ監視**: 定期的にアクセスログを確認
4. **定期的なパスワード変更**: 認証を実装した場合

## トラブルシューティング

### トンネルが起動しない

```bash
# ログ確認
cloudflared tunnel info exiton-mobile

# DNS設定確認
cloudflared tunnel route dns exiton-mobile
```

### 404 エラー

**原因**: Ingress rulesが間違っている

**対処**:
1. `~/.cloudflared/config.yml` を確認
2. `hostname` が正しいか確認
3. `service` のポート番号が正しいか確認（8501）

### Streamlitに接続できない

**原因**: Streamlitが起動していない、または違うポートで起動している

**対処**:
```bash
# Streamlitが8501で起動しているか確認
lsof -i :8501

# 起動していなければ起動
streamlit run mobile_live_view.py
```

### "Could not connect to Cloudflare" エラー

**原因**: ネットワーク接続の問題

**対処**:
```bash
# Cloudflareに到達できるか確認
ping cloudflare.com

# プロキシ設定が必要な場合
export HTTPS_PROXY=http://your-proxy:port
cloudflared tunnel run exiton-mobile
```

## コスト

### Cloudflare Tunnel

- ✅ **完全無料**（Free プランで利用可能）
- データ転送量に制限なし
- 帯域幅制限なし

### Cloudflare Access（認証機能）

- ✅ **最大50ユーザーまで無料**
- 必要に応じて有料プラン検討

## 参考リンク

- [Cloudflare Tunnel公式ドキュメント](https://developers.cloudflare.com/cloudflare-one/connections/connect-apps/)
- [cloudflared GitHub](https://github.com/cloudflare/cloudflared)
- [Cloudflare Zero Trust Dashboard](https://one.dash.cloudflare.com/)
- [Streamlit公式ドキュメント](https://docs.streamlit.io/)

## まとめ

Cloudflare Tunnelを使用することで：

- 🔒 **セキュア**: VPNやポートフォワーディング不要
- 🚀 **簡単**: 数コマンドで設定完了
- 💰 **無料**: 個人利用なら完全無料
- 📱 **便利**: どこからでもスマホでアクセス可能

ただし、**認証機能の追加は必須**です。Cloudflare AccessまたはStreamlit側での実装を推奨します。
