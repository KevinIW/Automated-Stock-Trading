# Reinforcement Learning for Automated Trading: DQN vs PPO

This project implements and compares reinforcement learning algorithms for automated trading on S&P 500 historical data. We specifically analyze the performance differences between Deep Q-Networks (DQN) and Proximal Policy Optimization (PPO) in a trading environment with partial buy/sell actions.

## Project Overview

We developed a trading environment with:
- **State Space**: Normalized price, volatility, position ratio, cash ratio
- **Action Space**: Hold, Buy (10% of available cash), Sell (1% of current position)
- **Reward Function**: Weighted combination of ROI, Sharpe ratio, maximum drawdown, win rate, with overtrading penalties

Our analysis reveals that DQN consistently outperforms PPO in this discrete action space trading environment, with better financial metrics and learning stability.

## Key Findings

- DQN achieves significantly higher rewards and better financial performance than PPO
- DQN's value-based approach better handles the delayed rewards inherent in trading environments
- Action distribution analysis shows that DQN adopts a more balanced trading strategy compared to PPO's conservative approach
- The best DQN model achieved 4.75% ROI compared to 3.52% for the best PPO model

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd m2-2206810042

# Install required packages
pip install -r requirements.txt
```

### Dependencies

- Python 3.8+
- pandas
- numpy
- matplotlib
- gymnasium
- stable-baselines3
- torch

## Usage

### Running the Trading Simulation

```bash
cd code
python -m jupyter notebook "DQN and PPO.ipynb"
```

The notebook contains a complete implementation of:
- Trading environment with partial buy/sell actions
- DQN and PPO implementations and hyperparameter settings
- Visualization and comparative analysis tools

### Data Requirements

Place your S&P 500 historical data CSV file in the project directory. The file should contain columns for 'Date', 'Price', 'Open', 'High', and 'Low'.

## Project Structure
