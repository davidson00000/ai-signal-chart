// frontend/src/api/backtest.ts
/// <reference types="vite/client" />

// ======================
// 型定義
// ======================

export interface BacktestRequest {
    symbol: string;
    timeframe?: string;        // e.g. "1d", "1h"
    start_date?: string | undefined;       // ISO 形式 "2020-01-01"
    end_date?: string | undefined;         // ISO 形式
    strategy?: string;         // "MA_CROSS" など（今は未使用でもOK）

    initial_capital?: number;  // 初期資金
    commission?: number;       // 手数料率
    position_size?: number;    // 1.0 = フルポジ

    short_window?: number;     // MA短期
    long_window?: number;      // MA長期
}

export interface BacktestMetrics {
    initial_capital: number;
    final_equity: number;
    total_pnl: number;
    return_pct: number;
    max_drawdown: number;
    win_rate: number;
    trade_count: number;
    winning_trades: number;
    losing_trades: number;
    sharpe_ratio?: number;
}

export interface BacktestTrade {
    date: string;              // ISO datetime string
    side: 'BUY' | 'SELL';
    price: number;
    quantity: number;
    commission: number;
    pnl?: number | null;       // BUY時はnull
    cash_after?: number;       // Python側であれば付けているはず（なければ undefined でOK）
}

export interface EquityPoint {
    date: string;              // ISO datetime string
    equity: number;
    cash?: number;
}

export interface BacktestResponse {
    metrics: BacktestMetrics;
    trades: BacktestTrade[];
    equity_curve: EquityPoint[];
}

// ===== Experiments =====

export interface BacktestExperimentCreate {
    name: string;
    description?: string;
    request: BacktestRequest;
    result?: BacktestResponse | null;
}

export interface BacktestExperiment {
    id: string;
    name: string;
    description?: string;
    created_at: string;
    updated_at: string;
    request: BacktestRequest;
    result?: BacktestResponse | null;
}

export interface BacktestExperimentSummary {
    id: string;
    name: string;
    description?: string;
    created_at: string;
    updated_at: string;
}

// ======================
// API ベース URL
// ======================

const API_BASE = import.meta.env.VITE_API_BASE ?? 'http://localhost:8000';

// ======================
// バックテスト実行
// ======================

export async function runBacktest(
    request: BacktestRequest,
): Promise<BacktestResponse> {
    const response = await fetch(`${API_BASE}/simulate`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(request),
    });

    if (!response.ok) {
        let detail = response.statusText;
        try {
            const err = await response.json();
            if (err?.detail) detail = err.detail;
        } catch {
            // ignore
        }
        throw new Error(detail || 'Simulation failed');
    }

    return response.json();
}

// ======================
// Experiment API 群
// ======================

export async function createExperiment(
    payload: BacktestExperimentCreate,
): Promise<BacktestExperiment> {
    const res = await fetch(`${API_BASE}/experiments`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
    });

    if (!res.ok) {
        let detail = res.statusText;
        try {
            const err = await res.json();
            if (err?.detail) detail = err.detail;
        } catch {
            // ignore
        }
        throw new Error(detail || 'Create experiment failed');
    }

    return res.json();
}

export async function listExperiments(): Promise<BacktestExperimentSummary[]> {
    const res = await fetch(`${API_BASE}/experiments`);

    if (!res.ok) {
        let detail = res.statusText;
        try {
            const err = await res.json();
            if (err?.detail) detail = err.detail;
        } catch {
            // ignore
        }
        throw new Error(detail || 'List experiments failed');
    }

    return res.json();
}

export async function getExperiment(
    id: string,
): Promise<BacktestExperiment> {
    const res = await fetch(`${API_BASE}/experiments/${id}`);

    if (!res.ok) {
        let detail = res.statusText;
        try {
            const err = await res.json();
            if (err?.detail) detail = err.detail;
        } catch {
            // ignore
        }
        throw new Error(detail || 'Get experiment failed');
    }

    return res.json();
}

export async function runExperiment(
    id: string,
): Promise<BacktestResponse> {
    const res = await fetch(`${API_BASE}/experiments/${id}/run`, {
        method: 'POST',
    });

    if (!res.ok) {
        let detail = res.statusText;
        try {
            const err = await res.json();
            if (err?.detail) detail = err.detail;
        } catch {
            // ignore
        }
        throw new Error(detail || 'Run experiment failed');
    }

    return res.json();
}

// ======================
// ヘルパー（任意）
// ======================

export function createDefaultBacktestRequest(symbol: string): BacktestRequest {
    return {
        symbol,
        timeframe: '1d',
        initial_capital: 1_000_000,
        commission: 0.001,
        position_size: 1.0,
        short_window: 9,
        long_window: 21,
    };
}

// 既存の export function runExperiment(...) { ... } のあととかでOK

// 互換用ラッパー：既存コードが runSimulation を呼んでいても動くようにする
export async function runSimulation(
    request: BacktestRequest,
): Promise<BacktestResponse> {
    return runBacktest(request);
}
