import { BacktestMetrics } from '../api/backtest';
import './MetricsPanel.css';

interface MetricsPanelProps {
    metrics: BacktestMetrics;
    symbol: string;
    strategy: string;
}

export default function MetricsPanel({ metrics, symbol, strategy }: MetricsPanelProps) {
    const formatCurrency = (value: number) => {
        return `¥${value.toLocaleString('ja-JP', { maximumFractionDigits: 0 })}`;
    };

    const formatPercent = (value: number) => {
        return `${value.toFixed(2)}%`;
    };

    const isProfit = metrics.total_pnl >= 0;

    return (
        <div className="metrics-panel">
            <div className="metrics-header">
                <h2>📊 Simulation Results</h2>
                <div className="metrics-subtitle">
                    <span>{symbol}</span>
                    <span className="separator">•</span>
                    <span>{strategy}</span>
                </div>
            </div>

            <div className="metrics-grid">
                <div className="metric-card">
                    <div className="metric-label">Total P&L</div>
                    <div className={`metric-value ${isProfit ? 'profit' : 'loss'}`}>
                        {formatCurrency(metrics.total_pnl)}
                    </div>
                </div>

                <div className="metric-card">
                    <div className="metric-label">Return</div>
                    <div className={`metric-value ${isProfit ? 'profit' : 'loss'}`}>
                        {formatPercent(metrics.return_pct)}
                    </div>
                </div>

                <div className="metric-card">
                    <div className="metric-label">Max Drawdown</div>
                    <div className="metric-value loss">
                        {formatPercent(metrics.max_drawdown)}
                    </div>
                </div>

                <div className="metric-card">
                    <div className="metric-label">Win Rate</div>
                    <div className="metric-value">
                        {formatPercent(metrics.win_rate)}
                    </div>
                </div>

                <div className="metric-card">
                    <div className="metric-label">Total Trades</div>
                    <div className="metric-value">
                        {metrics.trade_count}
                    </div>
                    <div className="metric-detail">
                        {metrics.winning_trades}W / {metrics.losing_trades}L
                    </div>
                </div>

                <div className="metric-card">
                    <div className="metric-label">Final Equity</div>
                    <div className="metric-value">
                        {formatCurrency(metrics.final_equity)}
                    </div>
                    <div className="metric-detail">
                        Initial: {formatCurrency(metrics.initial_capital)}
                    </div>
                </div>
            </div>
        </div>
    );
}
