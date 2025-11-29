/**
 * Backtest API Types
 * Based on 03BACKTEST_SPEC.md
 */

export interface BacktestRequest {
    symbol: string;
    timeframe?: string;
    start_date?: string;
    end_date?: string;
    strategy?: string;
    initial_capital?: number;
    commission?: number;
    position_size?: number;
    short_window?: number;
    long_window?: number;
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
}

export interface BacktestTrade {
    date: string;
    side: 'BUY' | 'SELL';
    price: number;
    quantity: number;
    commission: number;
    pnl?: number;
    cash_after: number;
}

export interface EquityPoint {
    date: string;
    equity: number;
    cash: number;
    position_value: number;
}

export interface BacktestResponse {
    symbol: string;
    timeframe: string;
    strategy: string;
    metrics: BacktestMetrics;
    trades: BacktestTrade[];
    equity_curve: EquityPoint[];
    data_points: number;
}

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

export async function runSimulation(request: BacktestRequest): Promise<BacktestResponse> {
    const response = await fetch(`${API_BASE_URL}/simulate`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(request),
    });

    if (!response.ok) {
        const error = await response.json().catch(() => ({ detail: response.statusText }));
        throw new Error(error.detail || 'Simulation failed');
    }

    return response.json();
}
