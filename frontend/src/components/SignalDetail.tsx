import React from 'react';

interface SignalExplain {
    indicators: Record<string, number | string>;
    conditions_triggered: string[];
    confidence: number;
}

interface Signal {
    index: number;
    date: string;
    type: 'BUY' | 'SELL' | 'HOLD';
    price: number;
    explain: SignalExplain;
}

interface SignalDetailProps {
    signal: Signal | null;
    onClose: () => void;
}

const SignalDetail: React.FC<SignalDetailProps> = ({ signal, onClose }) => {
    if (!signal) return null;

    const confidencePercentage = signal.explain.confidence * 100;

    // Determine color based on confidence
    const getConfidenceColor = (conf: number) => {
        if (conf >= 0.7) return '#22c55e'; // green
        if (conf >= 0.5) return '#eab308'; // yellow
        return '#ef4444'; // red
    };

    const formatIndicatorValue = (key: string, value: number | string) => {
        if (typeof value === 'number') {
            // Format based on key name
            if (key.includes('pct') || key.includes('rate')) {
                return `${value.toFixed(2)}%`;
            }
            if (key.includes('rsi') || key.includes('_level')) {
                return value.toFixed(1);
            }
            return value.toFixed(4);
        }
        return String(value);
    };

    return (
        <div
            style={{
                position: 'fixed',
                top: '50%',
                right: '20px',
                transform: 'translateY(-50%)',
                width: '380px',
                maxHeight: '80vh',
                backgroundColor: '#1a1a2e',
                border: '1px solid #3a3a5a',
                borderRadius: '12px',
                boxShadow: '0 10px 40px rgba(0, 0, 0, 0.5)',
                zIndex: 1000,
                overflow: 'hidden',
            }}
        >
            {/* Header */}
            <div
                style={{
                    background: signal.type === 'BUY'
                        ? 'linear-gradient(135deg, #22c55e 0%, #16a34a 100%)'
                        : 'linear-gradient(135deg, #ef4444 0%, #dc2626 100%)',
                    padding: '16px 20px',
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center',
                }}
            >
                <div>
                    <h3 style={{ margin: 0, fontSize: '18px', fontWeight: 600, color: 'white' }}>
                        {signal.type} Signal
                    </h3>
                    <p style={{ margin: '4px 0 0', fontSize: '13px', opacity: 0.9, color: 'white' }}>
                        {new Date(signal.date).toLocaleDateString('en-US', {
                            year: 'numeric',
                            month: 'short',
                            day: 'numeric'
                        })}
                    </p>
                </div>
                <button
                    onClick={onClose}
                    style={{
                        background: 'rgba(255, 255, 255, 0.2)',
                        border: 'none',
                        borderRadius: '50%',
                        width: '32px',
                        height: '32px',
                        cursor: 'pointer',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        color: 'white',
                        fontSize: '18px',
                    }}
                >
                    ×
                </button>
            </div>

            {/* Content */}
            <div style={{ padding: '20px', overflowY: 'auto', maxHeight: 'calc(80vh - 80px)' }}>
                {/* Price */}
                <div style={{ marginBottom: '20px' }}>
                    <label style={{ fontSize: '12px', color: '#888', textTransform: 'uppercase', letterSpacing: '1px' }}>
                        Price at Signal
                    </label>
                    <p style={{ margin: '4px 0 0', fontSize: '24px', fontWeight: 600, color: '#fff' }}>
                        ${signal.price.toFixed(2)}
                    </p>
                </div>

                {/* Confidence Gauge */}
                <div style={{ marginBottom: '20px' }}>
                    <label style={{ fontSize: '12px', color: '#888', textTransform: 'uppercase', letterSpacing: '1px' }}>
                        Confidence Score
                    </label>
                    <div style={{ marginTop: '8px', position: 'relative' }}>
                        <div
                            style={{
                                width: '100%',
                                height: '12px',
                                backgroundColor: '#2a2a4a',
                                borderRadius: '6px',
                                overflow: 'hidden',
                            }}
                        >
                            <div
                                style={{
                                    width: `${confidencePercentage}%`,
                                    height: '100%',
                                    backgroundColor: getConfidenceColor(signal.explain.confidence),
                                    borderRadius: '6px',
                                    transition: 'width 0.5s ease',
                                }}
                            />
                        </div>
                        <p style={{ margin: '6px 0 0', fontSize: '16px', fontWeight: 500, color: getConfidenceColor(signal.explain.confidence) }}>
                            {confidencePercentage.toFixed(0)}%
                        </p>
                    </div>
                </div>

                {/* Indicators */}
                <div style={{ marginBottom: '20px' }}>
                    <label style={{ fontSize: '12px', color: '#888', textTransform: 'uppercase', letterSpacing: '1px' }}>
                        Indicators
                    </label>
                    <div
                        style={{
                            marginTop: '8px',
                            backgroundColor: '#252540',
                            borderRadius: '8px',
                            padding: '12px',
                        }}
                    >
                        {Object.entries(signal.explain.indicators).map(([key, value]) => (
                            <div
                                key={key}
                                style={{
                                    display: 'flex',
                                    justifyContent: 'space-between',
                                    padding: '6px 0',
                                    borderBottom: '1px solid #3a3a5a',
                                }}
                            >
                                <span style={{ color: '#aaa', fontSize: '13px' }}>
                                    {key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                                </span>
                                <span style={{ color: '#fff', fontSize: '13px', fontWeight: 500, fontFamily: 'monospace' }}>
                                    {formatIndicatorValue(key, value)}
                                </span>
                            </div>
                        ))}
                    </div>
                </div>

                {/* Conditions Triggered */}
                <div>
                    <label style={{ fontSize: '12px', color: '#888', textTransform: 'uppercase', letterSpacing: '1px' }}>
                        Conditions Triggered
                    </label>
                    <ul
                        style={{
                            marginTop: '8px',
                            paddingLeft: '0',
                            listStyle: 'none',
                        }}
                    >
                        {signal.explain.conditions_triggered.map((condition, idx) => (
                            <li
                                key={idx}
                                style={{
                                    display: 'flex',
                                    alignItems: 'flex-start',
                                    padding: '8px 12px',
                                    backgroundColor: '#252540',
                                    borderRadius: '6px',
                                    marginBottom: '6px',
                                    fontSize: '13px',
                                    color: '#ddd',
                                }}
                            >
                                <span
                                    style={{
                                        marginRight: '10px',
                                        color: signal.type === 'BUY' ? '#22c55e' : '#ef4444',
                                    }}
                                >
                                    ✓
                                </span>
                                {condition}
                            </li>
                        ))}
                    </ul>
                </div>
            </div>
        </div>
    );
};

export default SignalDetail;
