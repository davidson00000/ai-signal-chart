import { useEffect, useState } from 'react';
import { createChart, IChartApi, CandlestickSeries, LineSeries } from 'lightweight-charts';

interface CandlestickChartProps {
    symbol: string;
    interval?: string;
    showMA?: boolean;
    shortWindow?: number;
    longWindow?: number;
}

interface ChartDataPoint {
    time: number;
    open: number;
    high: number;
    low: number;
    close: number;
    volume: number;
    ma_short?: number | null;
    ma_long?: number | null;
}

export default function CandlestickChart({
    symbol,
    interval = '1d',
    showMA = true,
    shortWindow = 9,
    longWindow = 21
}: CandlestickChartProps) {
    const [chartData, setChartData] = useState<ChartDataPoint[]>([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        fetchChartData();
    }, [symbol, interval, showMA, shortWindow, longWindow]);

    async function fetchChartData() {
        setLoading(true);
        setError(null);

        try {
            const API_BASE = import.meta.env.VITE_API_BASE ?? 'http://localhost:8000';
            const response = await fetch(
                `${API_BASE}/chart-data?symbol=${symbol}&interval=${interval}&with_ma=${showMA}&short_window=${shortWindow}&long_window=${longWindow}&limit=500`
            );

            if (!response.ok) {
                throw new Error('Failed to fetch chart data');
            }

            const data = await response.json();
            setChartData(data.data || []);
        } catch (e: any) {
            setError(e.message || 'Failed to load chart');
        } finally {
            setLoading(false);
        }
    }

    useEffect(() => {
        if (chartData.length === 0) return;

        const chartContainer = document.getElementById('candlestick-chart');
        if (!chartContainer) return;

        // Clear previous chart
        chartContainer.innerHTML = '';

        const chart: IChartApi = createChart(chartContainer, {
            width: chartContainer.clientWidth,
            height: 400,
            layout: {
                background: { color: '#0f0f1e' },
                textColor: '#a0a0c0',
            },
            grid: {
                vertLines: { color: '#2a2a4e' },
                horzLines: { color: '#2a2a4e' },
            },
            timeScale: {
                timeVisible: true,
                secondsVisible: false,
            },
        });

        // Candlestick series  
        const candlestickSeries = chart.addSeries(CandlestickSeries, {
            upColor: '#26a69a',
            downColor: '#ef5350',
            borderVisible: false,
            wickUpColor: '#26a69a',
            wickDownColor: '#ef5350',
        });

        const candleData = chartData.map(d => ({
            time: d.time as any,
            open: d.open,
            high: d.high,
            low: d.low,
            close: d.close,
        }));

        candlestickSeries.setData(candleData);

        // MA Short Line
        if (showMA && chartData.some(d => d.ma_short !== null && d.ma_short !== undefined)) {
            const maShortSeries = chart.addSeries(LineSeries, {
                color: '#4a9eff',
                lineWidth: 2,
                title: `MA${shortWindow}`,
            });

            const maShortData = chartData
                .filter(d => d.ma_short !== null && d.ma_short !== undefined)
                .map(d => ({
                    time: d.time as any,
                    value: d.ma_short!,
                }));

            maShortSeries.setData(maShortData);
        }

        // MA Long Line
        if (showMA && chartData.some(d => d.ma_long !== null && d.ma_long !== undefined)) {
            const maLongSeries = chart.addSeries(LineSeries, {
                color: '#ff6b6b',
                lineWidth: 2,
                title: `MA${longWindow}`,
            });

            const maLongData = chartData
                .filter(d => d.ma_long !== null && d.ma_long !== undefined)
                .map(d => ({
                    time: d.time as any,
                    value: d.ma_long!,
                }));

            maLongSeries.setData(maLongData);
        }

        chart.timeScale().fitContent();

        // Handle resize
        const handleResize = () => {
            chart.applyOptions({ width: chartContainer.clientWidth });
        };

        window.addEventListener('resize', handleResize);

        return () => {
            window.removeEventListener('resize', handleResize);
            chart.remove();
        };
    }, [chartData, showMA, shortWindow, longWindow]);

    return (
        <div className="candlestick-chart-container">
            <h3 className="text-xl font-semibold mb-4">
                📈 {symbol} Chart {showMA && `(MA${shortWindow}/${longWindow})`}
            </h3>

            {loading && <p className="text-slate-400">Loading chart...</p>}
            {error && <p className="text-red-400">Error: {error}</p>}

            <div id="candlestick-chart" style={{ width: '100%', height: '400px' }}></div>

            <div className="mt-2 text-xs text-slate-400">
                Data points: {chartData.length}
            </div>
        </div>
    );
}
