// frontend/src/App.tsx
import { useEffect, useState } from 'react';
import './App.css';
import {
    BacktestRequest,
    BacktestResponse,
    BacktestExperiment,
    BacktestExperimentSummary,
    runBacktest,
    runExperiment,
    createExperiment,
    listExperiments,
    getExperiment,
    createDefaultBacktestRequest,
} from './api/backtest';
import {
    CartesianGrid,
    Line,
    LineChart,
    ResponsiveContainer,
    Tooltip,
    XAxis,
    YAxis,
} from 'recharts';

type TabKey = 'backtest' | 'experiments' | 'opt' | 'lab';

const numberFmt = new Intl.NumberFormat('ja-JP');
const percentFmt = new Intl.NumberFormat('ja-JP', {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
});

type OptConfig = {
    shortMin: number;
    shortMax: number;
    shortStep: number;
    longMin: number;
    longMax: number;
    longStep: number;
    topN: number;
};

type MAOptimizationRow = {
    short_window: number;
    long_window: number;
    metrics: BacktestResponse['metrics'];
};

function App() {
    const [activeTab, setActiveTab] = useState<TabKey>('backtest');

    // Backtest 条件
    const [request, setRequest] = useState<BacktestRequest>(
        createDefaultBacktestRequest('AAPL'),
    );

    // 結果
    const [result, setResult] = useState<BacktestResponse | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    // Experiments
    const [experiments, setExperiments] = useState<BacktestExperimentSummary[]>(
        [],
    );
    const [selectedExperiment, setSelectedExperiment] =
        useState<BacktestExperiment | null>(null);
    const [expName, setExpName] = useState('');
    const [expDescription, setExpDescription] = useState('');

    // Optimization
    const [optConfig, setOptConfig] = useState<OptConfig>({
        shortMin: 5,
        shortMax: 25,
        shortStep: 2,
        longMin: 20,
        longMax: 80,
        longStep: 5,
        topN: 20,
    });
    const [optResults, setOptResults] = useState<MAOptimizationRow[] | null>(
        null,
    );
    const [optLoading, setOptLoading] = useState(false);

    // Experimentsタブを開いたら一覧更新
    useEffect(() => {
        if (activeTab === 'experiments') {
            refreshExperiments();
        }
    }, [activeTab]);

    async function refreshExperiments() {
        try {
            const list = await listExperiments();
            setExperiments(list);
        } catch (e: any) {
            console.error(e);
            setError(e.message ?? 'Failed to load experiments');
        }
    }

    async function handleRunBacktest() {
        setLoading(true);
        setError(null);

        try {
            const res = await runBacktest(request);
            setResult(res);
        } catch (e: any) {
            console.error(e);
            setError(e.message ?? 'Failed to run backtest');
        } finally {
            setLoading(false);
        }
    }

    function updateRequest<K extends keyof BacktestRequest>(
        key: K,
        value: BacktestRequest[K],
    ) {
        setRequest((prev) => ({
            ...prev,
            [key]: value,
        }));
    }

    async function handleLoadExperiment(id: string) {
        setLoading(true);
        setError(null);
        try {
            const exp = await getExperiment(id);
            setSelectedExperiment(exp);

            setRequest(exp.request);
            setResult(exp.result ?? null);
        } catch (e: any) {
            console.error(e);
            setError(e.message ?? 'Failed to load experiment');
        } finally {
            setLoading(false);
        }
    }

    async function handleRunExperiment(id: string) {
        setLoading(true);
        setError(null);
        try {
            const res = await runExperiment(id);
            setResult(res);
        } catch (e: any) {
            console.error(e);
            setError(e.message ?? 'Failed to run experiment');
        } finally {
            setLoading(false);
        }
    }

    // Optimization 実行
    async function handleRunOptimization() {
        setOptLoading(true);
        setError(null);
        setOptResults(null);

        try {
            const rows: MAOptimizationRow[] = [];

            for (
                let sw = optConfig.shortMin;
                sw <= optConfig.shortMax;
                sw += optConfig.shortStep
            ) {
                for (
                    let lw = optConfig.longMin;
                    lw <= optConfig.longMax;
                    lw += optConfig.longStep
                ) {
                    if (sw >= lw) continue;

                    const res = await runBacktest({
                        ...request,
                        short_window: sw,
                        long_window: lw,
                    });

                    rows.push({
                        short_window: sw,
                        long_window: lw,
                        metrics: res.metrics,
                    });
                }
            }

            rows.sort((a, b) => b.metrics.total_pnl - a.metrics.total_pnl);

            const top = rows.slice(0, optConfig.topN);
            setOptResults(top);
        } catch (e: any) {
            console.error(e);
            setError(e.message ?? 'Optimization failed');
        } finally {
            setOptLoading(false);
        }
    }

    // Optimization 行から MA パラメータをフォームに反映
    function handleApplyFromOpt(row: MAOptimizationRow) {
        setRequest((prev) => ({
            ...prev,
            short_window: row.short_window,
            long_window: row.long_window,
        }));
        // ここでは自動実行はせず、「Run Simulation」はユーザーに任せる
    }

    async function handleSaveExperiment() {
        if (!expName.trim()) {
            setError('Experiment name is required');
            return;
        }
        setError(null);
        try {
            const created = await createExperiment({
                name: expName.trim(),
                description: expDescription.trim() || undefined,
                request,
                result,
            });
            setExpName('');
            setExpDescription('');
            setExperiments((prev) => [created, ...prev]);
        } catch (e: any) {
            console.error(e);
            setError(e.message ?? 'Failed to save experiment');
        }
    }

    const equityData =
        result?.equity_curve?.map((p) => ({
            date: p.date.slice(0, 10),
            equity: p.equity,
            cash: p.cash ?? null,
        })) ?? [];

    return (
        <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950 text-slate-100">
            <div className="max-w-6xl mx-auto px-4 py-8">
                {/* Header */}
                <header className="mb-8">
                    <h1 className="text-3xl font-bold mb-2 flex items-center gap-2">
                        <span className="text-sky-400">🚀 EXITON</span> Backtest Simulation
                    </h1>
                    <p className="text-slate-400 text-sm">
                        Test your trading strategies with historical data.
                    </p>
                </header>

                {/* Tabs */}
                <div className="mb-6 border-b border-slate-800/70">
                    <nav className="-mb-px flex gap-4 text-sm">
                        {[
                            { key: 'backtest' as TabKey, label: 'Backtest' },
                            { key: 'experiments' as TabKey, label: 'Experiments' },
                            { key: 'opt' as TabKey, label: 'Optimization' },
                            { key: 'lab' as TabKey, label: 'Strategy Lab (coming soon)' },
                        ].map((tab) => (
                            <button
                                key={tab.key}
                                className={`px-3 py-2 border-b-2 transition-colors ${activeTab === tab.key
                                    ? 'border-sky-400 text-sky-300'
                                    : 'border-transparent text-slate-500 hover:text-slate-200'
                                    }`}
                                onClick={() => setActiveTab(tab.key)}
                            >
                                {tab.label}
                            </button>
                        ))}
                    </nav>
                </div>

                {/* Layout */}
                <div className="grid grid-cols-1 lg:grid-cols-[minmax(0,1.2fr)_minmax(0,1.5fr)] gap-6 items-start">
                    {/* Left: Backtest form */}
                    <section className="bg-slate-900/70 rounded-2xl p-6 shadow-xl shadow-sky-900/40 border border-slate-800/70 space-y-4">
                        <h2 className="text-xl font-semibold mb-2 flex items-center gap-2">
                            <span className="text-emerald-400">🧪</span> Backtest Simulation
                        </h2>

                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
                            <div>
                                <label className="block text-xs text-slate-400 mb-1">
                                    Symbol *
                                </label>
                                <input
                                    className="w-full rounded-md bg-slate-900 border border-slate-700 px-3 py-2 text-sm focus:outline-none focus:ring-1 focus:ring-sky-500"
                                    value={request.symbol}
                                    onChange={(e) => updateRequest('symbol', e.target.value)}
                                />
                            </div>

                            {/* Symbol Preset */}
                            <div className="mb-4">
                                <label className="block text-slate-300 text-sm font-semibold mb-1">
                                    Symbol Preset
                                </label>

                                <select
                                    className="w-full bg-slate-800/60 border border-slate-700/50 rounded-lg px-3 py-2 text-slate-200 focus:ring-2 focus:ring-sky-500 focus:outline-none"
                                    onChange={(e) => {
                                        const presetSymbol = e.target.value;
                                        if (presetSymbol) {
                                            updateRequest('symbol', presetSymbol);
                                        }
                                    }}
                                >
                                    <option value="">-- Select preset --</option>

                                    <option disabled>── JP Stocks ──</option>
                                    <option value="7203.T">TOYOTA (7203.T)</option>
                                    <option value="9984.T">SoftBank G (9984.T)</option>
                                    <option value="6758.T">SONY (6758.T)</option>
                                    <option value="6861.T">Keyence (6861.T)</option>
                                    <option value="8316.T">SMBC (8316.T)</option>
                                    <option value="8306.T">MUFG (8306.T)</option>

                                    <option disabled>── JP ETFs ──</option>
                                    <option value="1306.T">TOPIX (1306.T)</option>
                                    <option value="1305.T">TOPIX Core (1305.T)</option>
                                    <option value="2558.T">S&P500 (2558.T)</option>
                                    <option value="1655.T">S&P500 (1655.T)</option>
                                    <option value="1545.T">NASDAQ100 (1545.T)</option>

                                    <option disabled>── US Stocks ──</option>
                                    <option value="AAPL">Apple (AAPL)</option>
                                    <option value="MSFT">Microsoft (MSFT)</option>
                                    <option value="NVDA">NVIDIA (NVDA)</option>
                                    <option value="META">Meta (META)</option>
                                    <option value="TSLA">Tesla (TSLA)</option>

                                    <option disabled>── Crypto ──</option>
                                    <option value="BTC-USD">Bitcoin (BTC-USD)</option>
                                    <option value="ETH-USD">Ethereum (ETH-USD)</option>
                                </select>
                            </div>


                            <div>
                                <label className="block text-xs text-slate-400 mb-1">
                                    Timeframe
                                </label>
                                <select
                                    className="w-full rounded-md bg-slate-900 border border-slate-700 px-3 py-2 text-sm focus:outline-none focus:ring-1 focus:ring-sky-500"
                                    value={request.timeframe}
                                    onChange={(e) => updateRequest('timeframe', e.target.value)}
                                >
                                    <option value="1d">1 day</option>
                                    <option value="1h">1 hour</option>
                                    <option value="4h">4 hours</option>
                                </select>
                            </div>

                            <div>
                                <label className="block text-xs text-slate-400 mb-1">
                                    Start Date
                                </label>
                                <input
                                    type="date"
                                    className="w-full rounded-md bg-slate-900 border border-slate-700 px-3 py-2 text-sm focus:outline-none focus:ring-1 focus:ring-sky-500"
                                    value={request.start_date ?? ''}
                                    onChange={(e) =>
                                        updateRequest('start_date', e.target.value || undefined)
                                    }
                                />
                            </div>

                            <div>
                                <label className="block text-xs text-slate-400 mb-1">
                                    End Date
                                </label>
                                <input
                                    type="date"
                                    className="w-full rounded-md bg-slate-900 border border-slate-700 px-3 py-2 text-sm focus:outline-none focus:ring-1 focus:ring-sky-500"
                                    value={request.end_date ?? ''}
                                    onChange={(e) =>
                                        updateRequest('end_date', e.target.value || undefined)
                                    }
                                />
                            </div>

                            <div>
                                <label className="block text-xs text-slate-400 mb-1">
                                    Strategy
                                </label>
                                <select
                                    className="w-full rounded-md bg-slate-900 border border-slate-700 px-3 py-2 text-sm focus:outline-none focus:ring-1 focus:ring-sky-500"
                                    value={request.strategy}
                                    onChange={(e) =>
                                        updateRequest(
                                            'strategy',
                                            e.target.value as BacktestRequest['strategy'],
                                        )
                                    }
                                >
                                    <option value="ma_cross">MA Cross</option>
                                </select>
                            </div>

                            <div>
                                <label className="block text-xs text-slate-400 mb-1">
                                    Initial Capital (¥)
                                </label>
                                <input
                                    type="number"
                                    className="w-full rounded-md bg-slate-900 border border-slate-700 px-3 py-2 text-sm focus:outline-none focus:ring-1 focus:ring-sky-500"
                                    value={request.initial_capital}
                                    onChange={(e) =>
                                        updateRequest('initial_capital', Number(e.target.value))
                                    }
                                />
                            </div>

                            <div>
                                <label className="block text-xs text-slate-400 mb-1">
                                    Commission Rate
                                </label>
                                <input
                                    type="number"
                                    step="0.0001"
                                    className="w-full rounded-md bg-slate-900 border border-slate-700 px-3 py-2 text-sm focus:outline-none focus:ring-1 focus:ring-sky-500"
                                    value={request.commission}
                                    onChange={(e) =>
                                        updateRequest('commission', Number(e.target.value))
                                    }
                                />
                            </div>

                            <div>
                                <label className="block text-xs text-slate-400 mb-1">
                                    Position Size (0–1)
                                </label>
                                <input
                                    type="number"
                                    step="0.1"
                                    className="w-full rounded-md bg-slate-900 border border-slate-700 px-3 py-2 text-sm focus:outline-none focus:ring-1 focus:ring-sky-500"
                                    value={request.position_size}
                                    onChange={(e) =>
                                        updateRequest('position_size', Number(e.target.value))
                                    }
                                />
                            </div>

                            <div>
                                <label className="block text-xs text-slate-400 mb-1">
                                    Short MA Window
                                </label>
                                <input
                                    type="number"
                                    className="w-full rounded-md bg-slate-900 border border-slate-700 px-3 py-2 text-sm focus:outline-none focus:ring-1 focus:ring-sky-500"
                                    value={request.short_window}
                                    onChange={(e) =>
                                        updateRequest('short_window', Number(e.target.value))
                                    }
                                />
                            </div>

                            <div>
                                <label className="block text-xs text-slate-400 mb-1">
                                    Long MA Window
                                </label>
                                <input
                                    type="number"
                                    className="w-full rounded-md bg-slate-900 border border-slate-700 px-3 py-2 text-sm focus:outline-none focus:ring-1 focus:ring-sky-500"
                                    value={request.long_window}
                                    onChange={(e) =>
                                        updateRequest('long_window', Number(e.target.value))
                                    }
                                />
                            </div>
                        </div>

                        <button
                            className="mt-4 inline-flex items-center justify-center px-4 py-2 rounded-md bg-sky-500 hover:bg-sky-600 text-sm font-semibold disabled:opacity-60 shadow-md shadow-sky-700/40"
                            onClick={handleRunBacktest}
                            disabled={loading}
                        >
                            {loading ? 'Running...' : '🚀 Run Simulation'}
                        </button>

                        {selectedExperiment && (
                            <p className="mt-2 text-xs text-slate-400">
                                Loaded from experiment:{' '}
                                <span className="font-semibold">
                                    {selectedExperiment.name}
                                </span>
                            </p>
                        )}

                        {error && activeTab === 'backtest' && (
                            <p className="mt-2 text-sm text-red-400">⚠ {error}</p>
                        )}
                    </section>

                    {/* Right side: Tabs content + Results */}
                    <div className="space-y-4">
                        {/* Experiments tab */}
                        {activeTab === 'experiments' && (
                            <section className="bg-slate-900/70 rounded-2xl p-6 shadow-xl shadow-sky-900/40 border border-slate-800/70 space-y-6">
                                <h2 className="text-xl font-semibold mb-2 flex items-center gap-2">
                                    <span className="text-emerald-400">🧪</span> Experiments
                                </h2>

                                {/* Save current config */}
                                <div className="bg-slate-950/60 rounded-xl p-4 border border-slate-800/80">
                                    <h3 className="font-semibold mb-2 text-sm text-slate-200">
                                        Save Current Backtest as Experiment
                                    </h3>
                                    <div className="grid grid-cols-1 md:grid-cols-2 gap-3 mb-3">
                                        <div>
                                            <label className="block text-xs text-slate-400 mb-1">
                                                Experiment Name
                                            </label>
                                            <input
                                                className="w-full rounded-md bg-slate-900 border border-slate-700 px-3 py-2 text-sm focus:outline-none focus:ring-1 focus:ring-sky-500"
                                                value={expName}
                                                onChange={(e) => setExpName(e.target.value)}
                                            />
                                        </div>
                                        <div>
                                            <label className="block text-xs text-slate-400 mb-1">
                                                Description (optional)
                                            </label>
                                            <input
                                                className="w-full rounded-md bg-slate-900 border border-slate-700 px-3 py-2 text-sm focus:outline-none focus:ring-1 focus:ring-sky-500"
                                                value={expDescription}
                                                onChange={(e) => setExpDescription(e.target.value)}
                                            />
                                        </div>
                                    </div>
                                    <button
                                        className="px-3 py-2 rounded-md bg-emerald-600 hover:bg-emerald-500 text-xs font-semibold shadow-sm shadow-emerald-700/50"
                                        onClick={handleSaveExperiment}
                                        disabled={loading}
                                    >
                                        💾 Save Experiment
                                    </button>
                                </div>

                                {/* Experiments list */}
                                <div className="bg-slate-950/60 rounded-xl p-4 border border-slate-800/80">
                                    <div className="flex items-center justify-between mb-2">
                                        <h3 className="font-semibold text-sm text-slate-200">
                                            Saved Experiments
                                        </h3>
                                        <button
                                            className="text-xs px-2 py-1 rounded bg-slate-800 hover:bg-slate-700"
                                            onClick={refreshExperiments}
                                        >
                                            🔄 Refresh
                                        </button>
                                    </div>
                                    {experiments.length === 0 ? (
                                        <p className="text-slate-500 text-sm">
                                            No experiments saved yet.
                                        </p>
                                    ) : (
                                        <div className="overflow-x-auto">
                                            <table className="min-w-full text-xs">
                                                <thead>
                                                    <tr className="text-slate-400 border-b border-slate-800">
                                                        <th className="text-left py-2 pr-2">Name</th>
                                                        <th className="text-left py-2 pr-2 hidden md:table-cell">
                                                            Description
                                                        </th>
                                                        <th className="text-left py-2 pr-2">Created</th>
                                                        <th className="py-2 pr-2">Actions</th>
                                                    </tr>
                                                </thead>
                                                <tbody>
                                                    {experiments.map((exp) => (
                                                        <tr
                                                            key={exp.id}
                                                            className="border-b border-slate-900/60 hover:bg-slate-900/60"
                                                        >
                                                            <td className="py-2 pr-2">{exp.name}</td>
                                                            <td className="py-2 pr-2 hidden md:table-cell text-slate-400">
                                                                {exp.description}
                                                            </td>
                                                            <td className="py-2 pr-2 text-slate-500">
                                                                {exp.created_at?.slice(0, 10)}
                                                            </td>
                                                            <td className="py-2 pr-2 space-x-2">
                                                                <button
                                                                    className="px-2 py-1 rounded bg-slate-800 text-xs hover:bg-slate-700"
                                                                    onClick={() =>
                                                                        handleLoadExperiment(exp.id)
                                                                    }
                                                                >
                                                                    Load
                                                                </button>
                                                                <button
                                                                    className="px-2 py-1 rounded bg-blue-600 text-xs hover:bg-blue-500"
                                                                    onClick={() =>
                                                                        handleRunExperiment(exp.id)
                                                                    }
                                                                >
                                                                    Run
                                                                </button>
                                                            </td>
                                                        </tr>
                                                    ))}
                                                </tbody>
                                            </table>
                                        </div>
                                    )}
                                </div>

                                {error && (
                                    <p className="text-red-400 text-sm mt-1">⚠ {error}</p>
                                )}
                            </section>
                        )}

                        {/* Optimization tab */}
                        {activeTab === 'opt' && (
                            <section className="bg-slate-900/70 rounded-2xl p-6 shadow-xl shadow-purple-900/40 border border-slate-800/70 space-y-6">
                                <h2 className="text-xl font-semibold mb-2 flex items-center gap-2">
                                    <span className="text-purple-400">⚙️</span> MA Optimization
                                </h2>

                                <div className="bg-slate-950/60 rounded-xl p-4 border border-slate-800/80">
                                    <h3 className="font-semibold mb-2 text-sm text-slate-200">
                                        Search Range
                                    </h3>
                                    <div className="grid grid-cols-1 md:grid-cols-3 gap-3 mb-3">
                                        <div>
                                            <label className="block text-xs text-slate-400 mb-1">
                                                Short MA (min)
                                            </label>
                                            <input
                                                type="number"
                                                className="w-full rounded-md bg-slate-900 border border-slate-700 px-2 py-1 text-sm focus:outline-none focus:ring-1 focus:ring-sky-500"
                                                value={optConfig.shortMin}
                                                onChange={(e) =>
                                                    setOptConfig({
                                                        ...optConfig,
                                                        shortMin: Number(e.target.value),
                                                    })
                                                }
                                            />
                                        </div>
                                        <div>
                                            <label className="block text-xs text-slate-400 mb-1">
                                                Short MA (max)
                                            </label>
                                            <input
                                                type="number"
                                                className="w-full rounded-md bg-slate-900 border border-slate-700 px-2 py-1 text-sm focus:outline-none focus:ring-1 focus:ring-sky-500"
                                                value={optConfig.shortMax}
                                                onChange={(e) =>
                                                    setOptConfig({
                                                        ...optConfig,
                                                        shortMax: Number(e.target.value),
                                                    })
                                                }
                                            />
                                        </div>
                                        <div>
                                            <label className="block text-xs text-slate-400 mb-1">
                                                Short MA step
                                            </label>
                                            <input
                                                type="number"
                                                className="w-full rounded-md bg-slate-900 border border-slate-700 px-2 py-1 text-sm focus:outline-none focus:ring-1 focus:ring-sky-500"
                                                value={optConfig.shortStep}
                                                onChange={(e) =>
                                                    setOptConfig({
                                                        ...optConfig,
                                                        shortStep: Number(e.target.value),
                                                    })
                                                }
                                            />
                                        </div>
                                    </div>

                                    <div className="grid grid-cols-1 md:grid-cols-3 gap-3 mb-3">
                                        <div>
                                            <label className="block text-xs text-slate-400 mb-1">
                                                Long MA (min)
                                            </label>
                                            <input
                                                type="number"
                                                className="w-full rounded-md bg-slate-900 border border-slate-700 px-2 py-1 text-sm focus:outline-none focus:ring-1 focus:ring-sky-500"
                                                value={optConfig.longMin}
                                                onChange={(e) =>
                                                    setOptConfig({
                                                        ...optConfig,
                                                        longMin: Number(e.target.value),
                                                    })
                                                }
                                            />
                                        </div>
                                        <div>
                                            <label className="block text-xs text-slate-400 mb-1">
                                                Long MA (max)
                                            </label>
                                            <input
                                                type="number"
                                                className="w-full rounded-md bg-slate-900 border border-slate-700 px-2 py-1 text-sm focus:outline-none focus:ring-1 focus:ring-sky-500"
                                                value={optConfig.longMax}
                                                onChange={(e) =>
                                                    setOptConfig({
                                                        ...optConfig,
                                                        longMax: Number(e.target.value),
                                                    })
                                                }
                                            />
                                        </div>
                                        <div>
                                            <label className="block text-xs text-slate-400 mb-1">
                                                Long MA step
                                            </label>
                                            <input
                                                type="number"
                                                className="w-full rounded-md bg-slate-900 border border-slate-700 px-2 py-1 text-sm focus:outline-none focus:ring-1 focus:ring-sky-500"
                                                value={optConfig.longStep}
                                                onChange={(e) =>
                                                    setOptConfig({
                                                        ...optConfig,
                                                        longStep: Number(e.target.value),
                                                    })
                                                }
                                            />
                                        </div>
                                    </div>

                                    <div className="grid grid-cols-1 md:grid-cols-3 gap-3 mb-4">
                                        <div>
                                            <label className="block text-xs text-slate-400 mb-1">
                                                Show Top N
                                            </label>
                                            <input
                                                type="number"
                                                className="w-full rounded-md bg-slate-900 border border-slate-700 px-2 py-1 text-sm focus:outline-none focus:ring-1 focus:ring-sky-500"
                                                value={optConfig.topN}
                                                onChange={(e) =>
                                                    setOptConfig({
                                                        ...optConfig,
                                                        topN: Number(e.target.value),
                                                    })
                                                }
                                            />
                                        </div>
                                    </div>

                                    <button
                                        onClick={handleRunOptimization}
                                        disabled={optLoading}
                                        className="px-4 py-2 rounded-md bg-purple-500 hover:bg-purple-600 text-xs font-semibold disabled:opacity-60 shadow-md shadow-purple-800/60"
                                    >
                                        {optLoading
                                            ? 'Running grid search...'
                                            : '🔍 Run MA Grid Search'}
                                    </button>
                                </div>

                                <div className="bg-slate-950/60 rounded-xl p-4 border border-slate-800/80">
                                    <div className="flex items-center justify-between mb-2">
                                        <h3 className="font-semibold text-sm text-slate-200">
                                            Results (sorted by TOTAL P&L)
                                        </h3>
                                        {optResults && optResults.length > 0 && (
                                            <span className="text-[11px] text-slate-400">
                                                Best row ={' '}
                                                <span className="text-emerald-400 font-semibold">
                                                    ★ rank 1
                                                </span>
                                            </span>
                                        )}
                                    </div>
                                    {(!optResults || optResults.length === 0) && !optLoading && (
                                        <p className="text-slate-500 text-sm">
                                            No results yet. Set ranges and run optimization.
                                        </p>
                                    )}
                                    {optResults && optResults.length > 0 && (
                                        <div className="overflow-x-auto">
                                            <table className="min-w-full text-xs">
                                                <thead>
                                                    <tr className="text-slate-400 border-b border-slate-800">
                                                        <th className="text-left py-2 pr-2 w-8">#</th>
                                                        <th className="text-right py-2 pr-2">Short</th>
                                                        <th className="text-right py-2 pr-2">Long</th>
                                                        <th className="text-right py-2 pr-2">
                                                            TOTAL P&L
                                                        </th>
                                                        <th className="text-right py-2 pr-2">RETURN</th>
                                                        <th className="text-right py-2 pr-2">MAX DD</th>
                                                        <th className="text-right py-2 pr-2">
                                                            WIN RATE
                                                        </th>
                                                        <th className="text-right py-2 pr-2">TRADES</th>
                                                        <th className="text-right py-2 pr-2">
                                                            Actions
                                                        </th>
                                                    </tr>
                                                </thead>
                                                <tbody>
                                                    {optResults.map((row, idx) => (
                                                        <tr
                                                            key={`${row.short_window}-${row.long_window}-${idx}`}
                                                            className={`border-b border-slate-800/60 hover:bg-slate-900 cursor-pointer ${idx === 0
                                                                ? 'bg-emerald-500/10'
                                                                : ''
                                                                }`}
                                                            onClick={() => handleApplyFromOpt(row)}
                                                        >
                                                            <td className="py-1.5 pr-2 text-left">
                                                                {idx === 0 ? '★' : idx + 1}
                                                            </td>
                                                            <td className="py-1.5 pr-2 text-right">
                                                                {row.short_window}
                                                            </td>
                                                            <td className="py-1.5 pr-2 text-right">
                                                                {row.long_window}
                                                            </td>
                                                            <td className="py-1.5 pr-2 text-right">
                                                                ¥
                                                                {numberFmt.format(row.metrics.total_pnl)}
                                                            </td>
                                                            <td className="py-1.5 pr-2 text-right">
                                                                {percentFmt.format(
                                                                    row.metrics.return_pct,
                                                                )}
                                                                %
                                                            </td>
                                                            <td className="py-1.5 pr-2 text-right">
                                                                {percentFmt.format(
                                                                    row.metrics.max_drawdown,
                                                                )}
                                                                %
                                                            </td>
                                                            <td className="py-1.5 pr-2 text-right">
                                                                {percentFmt.format(
                                                                    row.metrics.win_rate,
                                                                )}
                                                                %
                                                            </td>
                                                            <td className="py-1.5 pr-2 text-right">
                                                                {row.metrics.trade_count}
                                                            </td>
                                                            <td className="py-1.5 pr-2 text-right">
                                                                <button
                                                                    className="px-2 py-1 rounded bg-sky-600 hover:bg-sky-500 text-[11px] font-semibold"
                                                                    onClick={(e) => {
                                                                        e.stopPropagation();
                                                                        handleApplyFromOpt(row);
                                                                    }}
                                                                >
                                                                    Apply
                                                                </button>
                                                            </td>
                                                        </tr>
                                                    ))}
                                                </tbody>
                                            </table>
                                        </div>
                                    )}
                                </div>

                                {error && (
                                    <p className="text-red-400 text-sm mt-1">⚠ {error}</p>
                                )}
                            </section>
                        )}

                        {/* Results 共通表示 */}
                        {result && (
                            <section className="space-y-4">
                                {/* Metrics */}
                                <div className="bg-slate-900/80 rounded-2xl p-6 border border-slate-800 shadow-lg shadow-sky-900/40">
                                    <h2 className="text-lg font-semibold mb-4">
                                        📊 Simulation Results
                                    </h2>
                                    <div className="grid grid-cols-2 md:grid-cols-5 gap-4 text-sm">
                                        <div>
                                            <div className="text-slate-400 text-xs mb-1">
                                                TOTAL P&L
                                            </div>
                                            <div className="text-emerald-400 font-bold">
                                                ¥{numberFmt.format(result.metrics.total_pnl)}
                                            </div>
                                        </div>
                                        <div>
                                            <div className="text-slate-400 text-xs mb-1">
                                                RETURN
                                            </div>
                                            <div className="text-emerald-300 font-bold">
                                                {percentFmt.format(result.metrics.return_pct)}%
                                            </div>
                                        </div>
                                        <div>
                                            <div className="text-slate-400 text-xs mb-1">
                                                MAX DRAWDOWN
                                            </div>
                                            <div className="text-red-400 font-bold">
                                                {percentFmt.format(result.metrics.max_drawdown)}%
                                            </div>
                                        </div>
                                        <div>
                                            <div className="text-slate-400 text-xs mb-1">
                                                WIN RATE
                                            </div>
                                            <div className="text-sky-300 font-bold">
                                                {percentFmt.format(result.metrics.win_rate)}%
                                            </div>
                                        </div>
                                        <div>
                                            <div className="text-slate-400 text-xs mb-1">
                                                TOTAL TRADES
                                            </div>
                                            <div className="text-slate-100 font-bold">
                                                {result.metrics.trade_count}
                                            </div>
                                        </div>
                                    </div>
                                </div>

                                {/* Equity Curve */}
                                <div className="bg-slate-900/80 rounded-2xl p-6 border border-slate-800 shadow-lg shadow-slate-900/40">
                                    <h3 className="text-sm font-semibold mb-2">
                                        📈 Equity Curve
                                    </h3>
                                    <div className="h-72">
                                        <ResponsiveContainer width="100%" height="100%">
                                            <LineChart data={equityData}>
                                                <CartesianGrid
                                                    strokeDasharray="3 3"
                                                    stroke="#1f2937"
                                                />
                                                <XAxis dataKey="date" hide />
                                                <YAxis />
                                                <Tooltip />
                                                <Line
                                                    type="monotone"
                                                    dataKey="equity"
                                                    stroke="#38bdf8"
                                                    dot={false}
                                                />
                                                <Line
                                                    type="monotone"
                                                    dataKey="cash"
                                                    stroke="#a855f7"
                                                    dot={false}
                                                />
                                            </LineChart>
                                        </ResponsiveContainer>
                                    </div>
                                </div>

                                {/* Trade History */}
                                <div className="bg-slate-900/80 rounded-2xl p-6 border border-slate-800 shadow-lg shadow-slate-900/40">
                                    <h3 className="text-sm font-semibold mb-2">
                                        📜 Trade History
                                    </h3>
                                    {result.trades.length === 0 ? (
                                        <p className="text-slate-500 text-sm">
                                            No trades executed.
                                        </p>
                                    ) : (
                                        <div className="overflow-x-auto">
                                            <table className="min-w-full text-xs">
                                                <thead>
                                                    <tr className="text-slate-400 border-b border-slate-800">
                                                        <th className="text-left py-2 pr-3">Date</th>
                                                        <th className="text-left py-2 pr-3">Side</th>
                                                        <th className="text-right py-2 pr-3">Price</th>
                                                        <th className="text-right py-2 pr-3">Qty</th>
                                                        <th className="text-right py-2 pr-3">
                                                            Commission
                                                        </th>
                                                        <th className="text-right py-2 pr-3">P&L</th>
                                                        <th className="text-right py-2 pr-3">
                                                            Cash After
                                                        </th>
                                                    </tr>
                                                </thead>
                                                <tbody>
                                                    {result.trades.map((t, idx) => (
                                                        <tr
                                                            key={idx}
                                                            className="border-b border-slate-800/60 hover:bg-slate-900"
                                                        >
                                                            <td className="py-1.5 pr-3">
                                                                {t.date?.slice(0, 10)}
                                                            </td>
                                                            <td className="py-1.5 pr-3">
                                                                <span
                                                                    className={`px-2 py-0.5 rounded-full text-[11px] font-semibold ${t.side === 'BUY'
                                                                        ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/40'
                                                                        : 'bg-rose-500/20 text-rose-300 border border-rose-500/40'
                                                                        }`}
                                                                >
                                                                    {t.side}
                                                                </span>
                                                            </td>
                                                            <td className="py-1.5 pr-3 text-right">
                                                                ¥{numberFmt.format(t.price)}
                                                            </td>
                                                            <td className="py-1.5 pr-3 text-right">
                                                                {numberFmt.format(t.quantity)}
                                                            </td>
                                                            <td className="py-1.5 pr-3 text-right">
                                                                ¥{numberFmt.format(t.commission)}
                                                            </td>
                                                            <td
                                                                className={`py-1.5 pr-3 text-right ${t.pnl !== undefined && t.pnl !== null
                                                                        ? t.pnl > 0
                                                                            ? 'text-emerald-300'
                                                                            : 'text-rose-300'
                                                                        : 'text-slate-200'
                                                                    }`}
                                                            >
                                                                {t.pnl !== undefined && t.pnl !== null
                                                                    ? `¥${numberFmt.format(t.pnl)}`
                                                                    : '-'}
                                                            </td>
                                                            <td className="py-1.5 pr-3 text-right">
                                                                {t.cash_after !== undefined
                                                                    ? `¥${numberFmt.format(t.cash_after)}`
                                                                    : '-'}
                                                            </td>
                                                        </tr>
                                                    ))}
                                                </tbody>
                                            </table>
                                        </div>
                                    )}
                                </div>
                            </section>
                        )}
                    </div>
                </div>
            </div>
        </div>
    );
}

export default App;
