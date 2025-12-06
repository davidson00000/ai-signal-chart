# JSON Strategy Schema Documentation

This document defines the schema for the JSON-based trading strategy engine (Meta Strategy v0.1).
It allows defining trading rules in a structured JSON format that can be executed by the backend engine.

## Schema Overview

A valid strategy JSON object must adhere to the following structure:

```json
{
  "name": "Strategy Name",
  "description": "Optional description",
  "indicators": [ ... ],
  "entry_rules": [ ... ],
  "exit_rules": [ ... ],
  "position": { ... }
}
```

### 1. Indicators (`indicators`)

Defines the technical indicators to be calculated.

| Field | Type | Description |
|---|---|---|
| `id` | string | Unique identifier for the indicator (e.g., "ma20", "rsi14"). Used in rules. |
| `type` | string | Indicator type. Supported: `"sma"`, `"ema"`, `"rsi"`, `"bollinger"`. |
| `source` | string | Input column. Usually `"close"`. |
| `period` | int | Period for SMA, EMA, RSI. |
| `std_dev` | float | Standard deviation for Bollinger Bands. |

**Example:**
```json
"indicators": [
  { "id": "ma20", "type": "sma", "source": "close", "period": 20 },
  { "id": "rsi14", "type": "rsi", "source": "close", "period": 14 }
]
```

### 2. Rules (`entry_rules`, `exit_rules`)

Defines the logic for entering and exiting positions.
Each rule group can contain `all` (AND) or `any` (OR) conditions.

#### Condition Structure
A single condition compares a `left` value with a `right` value using an `op`erator.

| Field | Type | Description |
|---|---|---|
| `left` | ValueRef | Left side of comparison. |
| `op` | string | Operator: `"=="`, `"!="`, `">"`, `">="`, `"<"`, `"<="`. |
| `right` | ValueRef | Right side of comparison. |

#### ValueRef Structure
Represents a value, either a reference to an indicator/column or a constant.

| Field | Type | Description |
|---|---|---|
| `ref` | string | Reference to an indicator ID (e.g., `"ma20"`) or column (`"close"`, `"open"`, `"high"`, `"low"`, `"volume"`). |
| `value` | float | Constant numeric value (e.g., `30`, `0.5`). |

**Example (Entry Rule):**
"Enter if Close > MA20 AND RSI < 30"
```json
"entry_rules": [
  {
    "all": [
      { "left": { "ref": "close" }, "op": ">", "right": { "ref": "ma20" } },
      { "left": { "ref": "rsi14" }, "op": "<", "right": { "value": 30 } }
    ]
  }
]
```

### 3. Position Config (`position`)

| Field | Type | Description |
|---|---|---|
| `direction` | string | `"long_only"`, `"short_only"`, `"both"`. (Currently only `long_only` fully supported in logic). |
| `max_position` | float | Max position size (multiplier of capital). Default `1.0`. |

---

## Example Strategies

### 1. Simple MA Cross
```json
{
  "name": "MA Cross",
  "indicators": [
    { "id": "ma5", "type": "sma", "source": "close", "period": 5 },
    { "id": "ma20", "type": "sma", "source": "close", "period": 20 }
  ],
  "entry_rules": [
    {
      "all": [
        { "left": { "ref": "ma5" }, "op": ">", "right": { "ref": "ma20" } }
      ]
    }
  ],
  "exit_rules": [
    {
      "any": [
        { "left": { "ref": "ma5" }, "op": "<", "right": { "ref": "ma20" } }
      ]
    }
  ],
  "position": { "direction": "long_only", "max_position": 1.0 }
}
```

### 2. RSI Mean Reversion
```json
{
  "name": "RSI Reversion",
  "indicators": [
    { "id": "rsi14", "type": "rsi", "source": "close", "period": 14 }
  ],
  "entry_rules": [
    {
      "all": [
        { "left": { "ref": "rsi14" }, "op": "<", "right": { "value": 30 } }
      ]
    }
  ],
  "exit_rules": [
    {
      "any": [
        { "left": { "ref": "rsi14" }, "op": ">", "right": { "value": 70 } }
      ]
    }
  ],
  "position": { "direction": "long_only", "max_position": 1.0 }
}
```

---

## LLM Prompt Template

Use the following prompt to generate strategies using an LLM:

```text
You are a quantitative trading strategy generator.
Your task is to create a trading strategy in a specific JSON format.

Output ONLY a valid JSON object that follows this schema. Do not include markdown formatting or explanations.

Schema constraints:
1. "indicators": List of {id, type, source, period, std_dev}. Supported types: "sma", "ema", "rsi", "bollinger".
2. "entry_rules" and "exit_rules": List of rule groups.
3. Rule group: {"all": [...]} for AND, {"any": [...]} for OR.
4. Condition: {left, op, right}.
5. ValueRef: {"ref": "indicator_id_or_column"} OR {"value": number}.
6. Operators: "==", "!=", ">", ">=", "<", "<=".

Request:
Create a strategy that [INSERT STRATEGY DESCRIPTION HERE, e.g., "buys when price is above 200 SMA and RSI is below 30, sells when RSI is above 70"].
```
