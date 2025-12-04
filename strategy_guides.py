from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass
class ParamGuide:
    label: str
    recommended_min: Optional[float] = None
    recommended_max: Optional[float] = None
    note: str = ""

@dataclass
class Preset:
    label: str      # e.g. "Conservative", "Balanced", "Aggressive"
    params: Dict[str, float]

@dataclass
class StrategyGuide:
    key: str               # e.g. "ma_cross"
    label: str             # e.g. "MA Cross"
    overview: str
    timeframe_note: str
    param_guides: Dict[str, ParamGuide]
    presets: List[Preset]
    tips: List[str]

STRATEGY_GUIDES: Dict[str, StrategyGuide] = {
    "ma_cross": StrategyGuide(
        key="ma_cross",
        label="MA Cross",
        overview="""
        移動平均線（MA）のクロスオーバーに基づくトレンドフォロー戦略です。
        短期MAが長期MAを上抜けた時に「買い」、下抜けた時に「売り」を行います。
        トレンドが発生している相場（トレンド相場）で利益を出しやすい一方、
        方向感のないレンジ相場では「ダマシ」に遭いやすく、損失が重なる傾向があります。
        """,
        timeframe_note="※この推奨パラメータは日足(1d)を前提としています。",
        param_guides={
            "short_window": ParamGuide(
                label="Short Window (短期MA)",
                recommended_min=5,
                recommended_max=20,
                note="小さいほど反応が早いがダマシも増える。大きいと滑らかになるが遅れる。"
            ),
            "long_window": ParamGuide(
                label="Long Window (長期MA)",
                recommended_min=50,
                recommended_max=200,
                note="大きなトレンドを捉えるための期間。40未満だと敏感になりすぎる場合がある。"
            )
        },
        presets=[
            Preset(
                label="Conservative (慎重)",
                params={"short_window": 10, "long_window": 100}
            ),
            Preset(
                label="Balanced (バランス)",
                params={"short_window": 13, "long_window": 40}
            ),
            Preset(
                label="Aggressive (積極)",
                params={"short_window": 5, "long_window": 30}
            )
        ],
        tips=[
            "一般的に、長期MA ≈ 短期MA × 3〜5 程度が良いバランスと言われています。",
            "レンジ相場ではダマシシグナルが多くなるため、注意が必要です。",
            "出来高やRSIなど別の指標と組み合わせると安定しやすくなります。",
            "トレイリングストップなどと組み合わせてリスク管理するのが一般的です。"
        ]
    ),
    "rsi_mean_reversion": StrategyGuide(
        key="rsi_mean_reversion",
        label="RSI Mean Reversion",
        overview="""
        RSI（相対力指数）を用いた平均回帰（逆張り）戦略です。
        RSIが「売られすぎ」水準を下回ったら「買い」、「買われすぎ」水準を上回ったら「売り」を行います。
        一定の範囲で価格が推移するレンジ相場で効果を発揮しますが、
        強いトレンドが発生している相場では、シグナルが出続けて含み損が拡大するリスクがあります。
        """,
        timeframe_note="※この推奨パラメータは日足(1d)を前提としています。",
        param_guides={
            "rsi_period": ParamGuide(
                label="RSI Period",
                recommended_min=7,
                recommended_max=21,
                note="期間が短いほど値動きに敏感（ノイズ多い）、長いほど滑らか（遅行する）。"
            ),
            "oversold": ParamGuide(
                label="Oversold Level (売られすぎ)",
                recommended_min=20,
                recommended_max=35,
                note="値を小さくするほど、極端に売られた時だけ買う（慎重）。大きくすると取引回数増。"
            ),
            "overbought": ParamGuide(
                label="Overbought Level (買われすぎ)",
                recommended_min=65,
                recommended_max=80,
                note="値を大きくするほど、極端に買われた時だけ売る（慎重）。小さくすると取引回数増。"
            )
        },
        presets=[
            Preset(
                label="Conservative (慎重)",
                params={"rsi_period": 14, "oversold": 25, "overbought": 75}
            ),
            Preset(
                label="Balanced (バランス)",
                params={"rsi_period": 14, "oversold": 30, "overbought": 70}
            ),
            Preset(
                label="Aggressive (積極)",
                params={"rsi_period": 10, "oversold": 35, "overbought": 65}
            )
        ],
        tips=[
            "強い上昇トレンドではRSIが長期間70以上で張り付くことがあります（逆張りは危険）。",
            "トレンドフィルター（例：長期MAが上向きの時だけ買い）と併用されることが多いです。",
            "ボラティリティが極端に低い銘柄ではシグナルが少なくなる傾向があります。"
        ]
    ),
    "ema9_dip_buy": StrategyGuide(
        key="ema9_dip_buy",
        label="EMA9 Dip Buy",
        overview="""
        9EMA押し目買い手法をベースにしたロング専用のトレンドフォロー戦略です。
        強い上昇トレンド中に、価格が9EMAまで押し目をつけた際に、
        直近高値のブレイクアウトでエントリーする手法です。
        2023年トレード世界王者が使用したとされる手法として知られています。
        """,
        timeframe_note="※この戦略は5分足など短期足での強いトレンド相場に適しています。",
        param_guides={
            "ema_fast": ParamGuide(
                label="Fast EMA Period (短期EMA)",
                recommended_min=7,
                recommended_max=12,
                note="通常は9。押し目のサポートレベルとして機能。小さいと敏感、大きいと滑らか。"
            ),
            "ema_slow": ParamGuide(
                label="Slow EMA Period (長期EMA)",
                recommended_min=18,
                recommended_max=26,
                note="通常は21。トレンド継続の判定に使用。価格がこれを下回ると決済。"
            ),
            "deviation_threshold": ParamGuide(
                label="Deviation Threshold % (乖離率)",
                recommended_min=1.0,
                recommended_max=3.0,
                note="Fast EMAからの許容乖離率。小さいほど厳密な押し目のみ検出。"
            ),
            "stop_buffer": ParamGuide(
                label="Stop Loss Buffer % (損切りバッファ)",
                recommended_min=0.3,
                recommended_max=1.0,
                note="エントリーバー前の安値からの追加バッファ。リスク管理に重要。"
            ),
            "risk_reward": ParamGuide(
                label="Risk/Reward Ratio",
                recommended_min=1.5,
                recommended_max=3.0,
                note="損失に対する利益の比率。2.0推奨。大きいほど利確目標が遠い。"
            ),
            "lookback_volume": ParamGuide(
                label="Volume Lookback Period",
                recommended_min=15,
                recommended_max=25,
                note="出来高平均を計算する期間。出来高増加の判定に使用。"
            )
        },
        presets=[
            Preset(
                label="Conservative (慎重)",
                params={
                    "ema_fast": 9,
                    "ema_slow": 21,
                    "deviation_threshold": 1.5,
                    "stop_buffer": 0.8,
                    "risk_reward": 2.5,
                    "lookback_volume": 20
                }
            ),
            Preset(
                label="Balanced (バランス)",
                params={
                    "ema_fast": 9,
                    "ema_slow": 21,
                    "deviation_threshold": 2.0,
                    "stop_buffer": 0.5,
                    "risk_reward": 2.0,
                    "lookback_volume": 20
                }
            ),
            Preset(
                label="Aggressive (積極)",
                params={
                    "ema_fast": 9,
                    "ema_slow": 21,
                    "deviation_threshold": 2.5,
                    "stop_buffer": 0.3,
                    "risk_reward": 1.5,
                    "lookback_volume": 20
                }
            )
        ],
        tips=[
            "この戦略は「ガードレール相場」（価格が9EMAに沿って上昇）で最も効果的です。",
            "レンジ相場や下降トレンドでは機能しないため、環境認識が重要です。",
            "出来高の増減を重視するため、流動性の高い銘柄で使用してください。",
            "トレーリングストップ（21EMA割れ）により、大きなトレンドに乗り続けることができます。"
        ]
    )
}
