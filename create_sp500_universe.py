# create_sp500_universe.py
import pandas as pd
import requests
from pathlib import Path
from io import StringIO


def main():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    print(f"Downloading S&P 500 table from {url} ...")

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0 Safari/537.36"
        )
    }

    resp = requests.get(url, headers=headers, timeout=15)
    resp.raise_for_status()

    # FutureWarning 回避のため StringIO を使う
    html_io = StringIO(resp.text)
    tables = pd.read_html(html_io)
    if not tables:
        raise RuntimeError("No tables found on the S&P 500 page.")

    print(f"Found {len(tables)} tables on the page. Searching for Symbol/Security columns...")

    df_constituents = None

    # すべてのテーブルを走査して、「Symbol」「Security」を持つものを探す
    for idx, t in enumerate(tables):
        cols = list(t.columns)
        print(f"  - Table {idx}: columns = {cols}")

        # ケース1: すでに "Symbol" / "Security" が列名として存在している
        if {"Symbol", "Security"}.issubset(t.columns):
            df_constituents = t
            print(f"  -> Using table {idx} as constituents table (direct match).")
            break

        # ケース2: 列名が 0,1,... のようなインデックスで、
        # 1行目にヘッダーが乗っているタイプ
        if 0 in t.columns and 1 in t.columns:
            header_row = t.iloc[0]
            t2 = t[1:].reset_index(drop=True)
            t2.columns = header_row

            if {"Symbol", "Security"}.issubset(t2.columns):
                df_constituents = t2
                print(f"  -> Using table {idx} as constituents table (after header fix).")
                break

    if df_constituents is None:
        raise RuntimeError(
            "Could not find a table with 'Symbol' and 'Security' columns "
            f"among {len(tables)} tables."
        )

    # 必要な列だけ抽出
    df_out = df_constituents[["Symbol", "Security"]].copy()
    df_out.columns = ["symbol", "note"]

    # ティッカーをクリーンアップ
    df_out["symbol"] = df_out["symbol"].astype(str).str.strip().str.upper()

    tools_dir = Path("tools")
    tools_dir.mkdir(exist_ok=True)
    out_path = tools_dir / "symbols_universe_sp500.csv"

    df_out.to_csv(out_path, index=False)
    print(f"Saved {len(df_out)} symbols to {out_path}")


if __name__ == "__main__":
    main()
