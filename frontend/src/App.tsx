import { useState } from 'react';
import SimulationForm from './components/SimulationForm';
import MetricsPanel from './components/MetricsPanel';
import EquityChart from './components/EquityChart';
import TradesTable from './components/TradesTable';
import { BacktestRequest, BacktestResponse, runSimulation } from './api/backtest';
import './App.css';

function App() {
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [result, setResult] = useState<BacktestResponse | null>(null);

    const handleSubmit = async (request: BacktestRequest) => {
        setLoading(true);
        setError(null);
        setResult(null);

        try {
            const response = await runSimulation(request);
            setResult(response);
        } catch (err) {
            setError(err instanceof Error ? err.message : 'An error occurred');
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="app">
            <header className="app-header">
                <h1>🚀 EXITON Backtest Simulation</h1>
                <p className="subtitle">Test your trading strategies with historical data</p>
            </header>

            <main className="app-main">
                <SimulationForm onSubmit={handleSubmit} loading={loading} />

                {loading && (
                    <div className="loading-container">
                        <div className="spinner"></div>
                        <p>Running simulation...</p>
                    </div>
                )}

                {error && (
                    <div className="error-container">
                        <h3>❌ Error</h3>
                        <p>{error}</p>
                    </div>
                )}

                {result && !loading && (
                    <div className="results-container">
                        <MetricsPanel
                            metrics={result.metrics}
                            symbol={result.symbol}
                            strategy={result.strategy}
                        />
                        <EquityChart data={result.equity_curve} />
                        <TradesTable trades={result.trades} />
                    </div>
                )}
            </main>

            <footer className="app-footer">
                <p>EXITON AI Trading System v0.2.0</p>
            </footer>
        </div>
    );
}

export default App;
