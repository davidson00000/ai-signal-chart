import yfinance as yf
from datetime import datetime, timedelta

symbol = "SMCI"
interval = "15m"
end_dt = datetime.now()
start_dt = end_dt - timedelta(days=60)

print(f"Fetching {symbol} {interval} from {start_dt} to {end_dt}")

try:
    df = yf.download(
        symbol,
        start=start_dt.strftime("%Y-%m-%d"),
        end=end_dt.strftime("%Y-%m-%d"),
        interval=interval,
        progress=False,
        auto_adjust=True
    )
    print("Result:")
    print(df.head())
    print(f"Rows: {len(df)}")
except Exception as e:
    print(f"Error: {e}")

print("-" * 20)
print("Trying with period='5d'")
try:
    df2 = yf.download(
        symbol,
        period="5d",
        interval=interval,
        progress=False,
        auto_adjust=True
    )
    print("Result:")
    print(df2.head())
    print(f"Rows: {len(df2)}")
except Exception as e:
    print(f"Error: {e}")
