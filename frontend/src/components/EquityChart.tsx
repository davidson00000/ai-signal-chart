import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts';
import { EquityPoint } from '../api/backtest';
import './EquityChart.css';

interface EquityChartProps {
    data: EquityPoint[];
}

export default function EquityChart({ data }: EquityChartProps) {
    // Format data for Recharts
    const chartData = data.map(point => ({
        date: new Date(point.date).toLocaleDateString('ja-JP', { month: 'short', day: 'numeric' }),
        fullDate: point.date,
        equity: Math.round(point.equity),
        cash: point.cash !== undefined ? Math.round(point.cash) : 0,
    }));

    // Sample data to reduce chart complexity
    const sampledData = chartData.length > 200
        ? chartData.filter((_, index) => index % Math.ceil(chartData.length / 200) === 0)
        : chartData;

    const formatYAxis = (value: number) => {
        if (value >= 1000000) {
            return `¥${(value / 1000000).toFixed(1)}M`;
        }
        return `¥${(value / 1000).toFixed(0)}K`;
    };

    return (
        <div className="equity-chart">
            <h3>📈 Equity Curve</h3>
            <div className="chart-container">
                <ResponsiveContainer width="100%" height={400}>
                    <LineChart data={sampledData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#2a2a4e" />
                        <XAxis
                            dataKey="date"
                            stroke="#a0a0c0"
                            tick={{ fill: '#a0a0c0', fontSize: 12 }}
                        />
                        <YAxis
                            tickFormatter={formatYAxis}
                            stroke="#a0a0c0"
                            tick={{ fill: '#a0a0c0', fontSize: 12 }}
                        />
                        <Tooltip
                            contentStyle={{
                                background: '#0f0f1e',
                                border: '1px solid #2a2a4e',
                                borderRadius: '8px',
                                color: '#fff'
                            }}
                            labelStyle={{ color: '#a0a0c0' }}
                            formatter={(value: number) => [`¥${value.toLocaleString()}`, '']}
                        />
                        <Legend
                            wrapperStyle={{ color: '#a0a0c0' }}
                            iconType="line"
                        />
                        <Line
                            type="monotone"
                            dataKey="equity"
                            stroke="#4a9eff"
                            strokeWidth={2}
                            dot={false}
                            name="Equity"
                        />
                        <Line
                            type="monotone"
                            dataKey="cash"
                            stroke="#a0a0c0"
                            strokeWidth={1}
                            dot={false}
                            strokeDasharray="5 5"
                            name="Cash"
                        />
                    </LineChart>
                </ResponsiveContainer>
            </div>
        </div>
    );
}
