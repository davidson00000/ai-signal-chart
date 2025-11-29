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
    exp: BacktestExperimentCreate,
): Promise<BacktestExperiment> {
    // Transform to match backend ExperimentCreate model
    const payload = {
        name: exp.name,
        description: exp.description,
        strategy_type: exp.request.strategy || 'ma_cross',  // Use strategy from request, default to 'ma_cross'
        symbol: exp.request.symbol,
        timeframe: exp.request.timeframe || '1d',
        start_date: exp.request.start_date || null,
        end_date: exp.request.end_date || null,
        parameters: {
            short_window: exp.request.short_window,
            long_window: exp.request.long_window,
            initial_capital: exp.request.initial_capital,
            commission: exp.request.commission,
            position_size: exp.request.position_size,
        },
        results: exp.result ? {
            total_pnl: exp.result.metrics.total_pnl,
            return_pct: exp.result.metrics.return_pct,
            win_rate: exp.result.metrics.win_rate,
            max_drawdown: exp.result.metrics.max_drawdown,
            trade_count: exp.result.metrics.trade_count,
            sharpe_ratio: exp.result.metrics.sharpe_ratio,
        } : null,
    };

    const res = await fetch(`${API_BASE}/experiments`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
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

    const data = await res.json();
    // Backend returns {experiments: [...], summary: {...}, count: N}
    // Extract the experiments array
    return data.experiments || [];
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
// Optimization API
// ======================

export interface OptimizationRequest {
    symbol: string;
    timeframe?: string;
    strategy_type?: string;
    short_window_min: number;
    short_window_max: number;
    short_window_step: number;
    long_window_min: number;
    long_window_max: number;
    long_window_step: number;
    metric?: 'total_return' | 'sharpe';
    initial_capital?: number;
    commission_rate?: number;
    position_size?: number;
    top_n?: number;
}

export interface OptimizationResultParams {
    short_window: number;
    long_window: number;
}

export interface OptimizationResultMetrics {
    total_pnl: number;
    return_pct: number;
    sharpe_ratio?: number;
    max_drawdown: number;
    win_rate: number;
    trade_count: number;
}

export interface OptimizationResult {
    rank: number;
    params: OptimizationResultParams;
    metrics: OptimizationResultMetrics;
}

export interface OptimizationResponse {
    symbol: string;
    timeframe: string;
    strategy_type: string;
    total_combinations: number;
    top_results: OptimizationResult[];
}

export async function runOptimization(
    request: OptimizationRequest
): Promise<OptimizationResponse> {
    const response = await fetch(`${API_BASE}/optimize`, {
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
        throw new Error(detail || 'Optimization failed');
    }

    return response.json();
}

// ======================
// Default factory
// ======================

export function createDefaultBacktestRequest(symbol: string): BacktestRequest {
    return {
        symbol,
        timeframe: '1d',
        start_date: '2020-01-01',  // Fixed: yyyy-MM-dd format
        end_date: '2022-12-31',    // Fixed: yyyy-MM-dd format
        initial_capital: 1000000,
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
// ======================
// Strategy Lab API
// ======================

export interface StrategyLabBatchRequest {
    study_name: string;
    symbols: string[];
    timeframe?: string;
    strategy_type?: string;
    initial_capital?: number;
    commission_rate?: number;
    position_size?: number;
    short_ma_min: number;
    short_ma_max: number;
    short_ma_step: number;
    long_ma_min: number;
    long_ma_max: number;
    long_ma_step: number;
    metric?: 'total_return' | 'sharpe';
}

export interface StrategyLabSymbolResult {
    symbol: string;
    short_window: number;
    long_window: number;
    total_return: number;
    sharpe: number | null;
    max_drawdown: number;
    win_rate: number;
    trades: number;
    metric_score: number;
    rank: number;
    error: string | null;
}

export interface StrategyLabBatchResponse {
    study_name: string;
    metric: string;
    results: StrategyLabSymbolResult[];
}

export async function runStrategyLabBatch(
    request: StrategyLabBatchRequest,
): Promise<StrategyLabBatchResponse> {
    const response = await fetch(`${API_BASE}/strategy-lab/run-batch`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(request),
    });

    if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Strategy Lab batch run failed');
    }

    return response.json();
}
