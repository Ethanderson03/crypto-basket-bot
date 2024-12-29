# Crypto Market Sentiment Trading Strategy

A Python-based cryptocurrency trading strategy that uses market sentiment analysis, technical indicators, and on-chain metrics to make trading decisions. The strategy includes backtesting capabilities and supports multiple exchanges through the CCXT library.

## Features

- Sentiment analysis using Fear & Greed Index and social media metrics
- Market analysis including price trends and volume analysis
- Order book analysis for optimal execution
- Risk management with position sizing and portfolio protection
- Backtesting framework with historical data caching
- Support for multiple exchanges via CCXT
- Real-time and paper trading capabilities

## Installation

1. Clone the repository:

```bash
git clone https://github.com/Ethanderson03/crypto-market-sentiment.git
cd crypto-market-sentiment
```

2. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Running a Backtest

```python
python src/backtest/run_backtest.py
```

Options:

- `--use-cache`: Use cached historical data if available
- `--start-date`: Start date for backtest (default: 2023-01-01)
- `--end-date`: End date for backtest (default: 2024-01-01)
- `--initial-balance`: Initial portfolio balance (default: 10000.0)

### Project Structure

```
src/
├── analysis/         # Market analysis and order book analysis
├── backtest/         # Backtesting framework
├── data/            # Data handling and price feeds
├── execution/       # Order execution
├── risk/           # Risk management
└── trading/        # Trading strategy and position management
```

## Configuration

- Exchange credentials should be stored in a `.env` file
- Trading parameters can be adjusted in the respective module configuration files
- Supported exchanges: Binance, KuCoin, Gate.io, Huobi

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

This software is for educational purposes only. Do not risk money which you are afraid to lose. USE THE SOFTWARE AT YOUR OWN RISK. THE AUTHORS AND ALL AFFILIATES ASSUME NO RESPONSIBILITY FOR YOUR TRADING RESULTS.
