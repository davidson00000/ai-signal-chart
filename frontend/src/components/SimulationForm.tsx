import React, { useState } from 'react';
import { BacktestRequest } from '../api/backtest';
import './SimulationForm.css';

interface SimulationFormProps {
    onSubmit: (request: BacktestRequest) => void;
    loading: boolean;
}

export default function SimulationForm({ onSubmit, loading }: SimulationFormProps) {
    const [formData, setFormData] = useState<BacktestRequest>({
        symbol: 'AAPL',
        timeframe: '1d',
        start_date: '2020-01-01',
        end_date: '2023-12-31',
        strategy: 'ma_cross',
        initial_capital: 1000000,
        commission: 0.0005,
        position_size: 1.0,
        short_window: 9,
        long_window: 21,
    });

    const [errors, setErrors] = useState<Record<string, string>>({});

    const validate = (): boolean => {
        const newErrors: Record<string, string> = {};

        if (!formData.symbol) {
            newErrors.symbol = 'Symbol is required';
        }

        if (formData.short_window && formData.long_window && formData.short_window >= formData.long_window) {
            newErrors.short_window = 'Short window must be less than long window';
        }

        if (formData.initial_capital && formData.initial_capital <= 0) {
            newErrors.initial_capital = 'Initial capital must be positive';
        }

        if (formData.commission && (formData.commission < 0 || formData.commission > 1)) {
            newErrors.commission = 'Commission must be between 0 and 1';
        }

        setErrors(newErrors);
        return Object.keys(newErrors).length === 0;
    };

    const handleSubmit = (e: React.FormEvent) => {
        e.preventDefault();
        if (validate()) {
            onSubmit(formData);
        }
    };

    const handleChange = (field: keyof BacktestRequest, value: string | number) => {
        setFormData(prev => ({ ...prev, [field]: value }));
        // Clear error for this field
        if (errors[field]) {
            setErrors(prev => {
                const newErrors = { ...prev };
                delete newErrors[field];
                return newErrors;
            });
        }
    };

    return (
        <form onSubmit={handleSubmit} className="simulation-form">
            <h2>🎯 Backtest Simulation</h2>

            <div className="form-grid">
                <div className="form-group">
                    <label htmlFor="symbol">Symbol *</label>
                    <input
                        id="symbol"
                        type="text"
                        value={formData.symbol}
                        onChange={(e) => handleChange('symbol', e.target.value)}
                        placeholder="e.g., AAPL, 7203.T"
                        disabled={loading}
                    />
                    {errors.symbol && <span className="error">{errors.symbol}</span>}
                </div>

                <div className="form-group">
                    <label htmlFor="timeframe">Timeframe</label>
                    <select
                        id="timeframe"
                        value={formData.timeframe}
                        onChange={(e) => handleChange('timeframe', e.target.value)}
                        disabled={loading}
                    >
                        <option value="1m">1 minute</option>
                        <option value="5m">5 minutes</option>
                        <option value="15m">15 minutes</option>
                        <option value="30m">30 minutes</option>
                        <option value="1h">1 hour</option>
                        <option value="4h">4 hours</option>
                        <option value="1d">1 day</option>
                    </select>
                </div>

                <div className="form-group">
                    <label htmlFor="start_date">Start Date</label>
                    <input
                        id="start_date"
                        type="date"
                        value={formData.start_date}
                        onChange={(e) => handleChange('start_date', e.target.value)}
                        disabled={loading}
                    />
                </div>

                <div className="form-group">
                    <label htmlFor="end_date">End Date</label>
                    <input
                        id="end_date"
                        type="date"
                        value={formData.end_date}
                        onChange={(e) => handleChange('end_date', e.target.value)}
                        disabled={loading}
                    />
                </div>

                <div className="form-group">
                    <label htmlFor="strategy">Strategy</label>
                    <select
                        id="strategy"
                        value={formData.strategy}
                        onChange={(e) => handleChange('strategy', e.target.value)}
                        disabled={loading}
                    >
                        <option value="ma_cross">MA Cross</option>
                    </select>
                </div>

                <div className="form-group">
                    <label htmlFor="initial_capital">Initial Capital (¥)</label>
                    <input
                        id="initial_capital"
                        type="number"
                        value={formData.initial_capital}
                        onChange={(e) => handleChange('initial_capital', parseFloat(e.target.value))}
                        step="10000"
                        disabled={loading}
                    />
                    {errors.initial_capital && <span className="error">{errors.initial_capital}</span>}
                </div>

                <div className="form-group">
                    <label htmlFor="commission">Commission Rate</label>
                    <input
                        id="commission"
                        type="number"
                        value={formData.commission}
                        onChange={(e) => handleChange('commission', parseFloat(e.target.value))}
                        step="0.0001"
                        min="0"
                        max="1"
                        disabled={loading}
                    />
                    {errors.commission && <span className="error">{errors.commission}</span>}
                </div>

                <div className="form-group">
                    <label htmlFor="position_size">Position Size (0-1)</label>
                    <input
                        id="position_size"
                        type="number"
                        value={formData.position_size}
                        onChange={(e) => handleChange('position_size', parseFloat(e.target.value))}
                        step="0.1"
                        min="0"
                        max="1"
                        disabled={loading}
                    />
                </div>

                <div className="form-group">
                    <label htmlFor="short_window">Short MA Window</label>
                    <input
                        id="short_window"
                        type="number"
                        value={formData.short_window}
                        onChange={(e) => handleChange('short_window', parseInt(e.target.value))}
                        min="1"
                        disabled={loading}
                    />
                    {errors.short_window && <span className="error">{errors.short_window}</span>}
                </div>

                <div className="form-group">
                    <label htmlFor="long_window">Long MA Window</label>
                    <input
                        id="long_window"
                        type="number"
                        value={formData.long_window}
                        onChange={(e) => handleChange('long_window', parseInt(e.target.value))}
                        min="2"
                        disabled={loading}
                    />
                </div>
            </div>

            <button type="submit" className="submit-btn" disabled={loading}>
                {loading ? '⏳ Running Simulation...' : '🚀 Run Simulation'}
            </button>
        </form>
    );
}
