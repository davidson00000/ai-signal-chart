import { BacktestTrade } from '../api/backtest';
import './TradesTable.css';

interface TradesTableProps {
    trades: BacktestTrade[];
}

export default function TradesTable({ trades }: TradesTableProps) {
    const formatCurrency = (value: number) => {
        return `¥${value.toLocaleString('ja-JP', { maximumFractionDigits: 0 })}`;
    };

    const formatDate = (dateString: string) => {
        return new Date(dateString).toLocaleDateString('ja-JP', {
            year: 'numeric',
            month: 'short',
            day: 'numeric',
        });
    };

    // Show most recent trades first
    const sortedTrades = [...trades].reverse();

    return (
        <div className="trades-table">
            <h3>📋 Trade History</h3>
            <div className="table-container">
                <table>
                    <thead>
                        <tr>
                            <th>Date</th>
                            <th>Side</th>
                            <th>Price</th>
                            <th>Quantity</th>
                            <th>Commission</th>
                            <th>P&L</th>
                            <th>Cash After</th>
                        </tr>
                    </thead>
                    <tbody>
                        {sortedTrades.map((trade, index) => (
                            <tr key={index} className={trade.side === 'BUY' ? 'buy-row' : 'sell-row'}>
                                <td>{formatDate(trade.date)}</td>
                                <td>
                                    <span className={`side-badge ${trade.side.toLowerCase()}`}>
                                        {trade.side}
                                    </span>
                                </td>
                                <td>{formatCurrency(trade.price)}</td>
                                <td>{trade.quantity.toLocaleString()}</td>
                                <td>{formatCurrency(trade.commission)}</td>
                                <td>
                                    {trade.pnl !== undefined && trade.pnl !== null ? (
                                        <span className={trade.pnl >= 0 ? 'profit' : 'loss'}>
                                            {formatCurrency(trade.pnl)}
                                        </span>
                                    ) : (
                                        <span className="na">-</span>
                                    )}
                                </td>
                                <td>{trade.cash_after !== undefined ? formatCurrency(trade.cash_after) : '-'}</td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
            <div className="table-summary">
                Total {trades.length} trades
            </div>
        </div>
    );
}
