# 🎯 A/B Testing with Epsilon-Greedy & Thompson Sampling

This project simulates an A/B testing scenario using two classic multi-armed bandit algorithms — **Epsilon-Greedy** and **Thompson Sampling** — to find the most rewarding ad among four options.

---

## 📌 Scenario

- **Bandits (Ads):** 4 choices with fixed rewards: `[1, 2, 3, 4]`
- **Trials:** 20,000 iterations for each algorithm  
- **Epsilon-Greedy:** Uses decaying epsilon `ε = 1 / t`  
- **Thompson Sampling:** Uses Beta distribution with known precision and binary reward conversion

---

## 📊 Outputs

Each algorithm:
- Saves results as CSV: `epsilon_greedy_rewards.csv`, `thompson_sampling_rewards.csv`
- Logs **total reward** and **regret** using `loguru`
- Plots:
  - 📈 Learning curves (average arm index)
  - 📉 Cumulative reward comparison

---

## 📁 Files

| File | Description |
|------|-------------|
| `Bandit.py` | Main experiment script with classes and visualizations |
| `epsilon_greedy_rewards.csv` | Rewards collected by Epsilon-Greedy |
| `thompson_sampling_rewards.csv` | Rewards collected by Thompson Sampling |

---

## ▶️ How to Run

```bash
pip install loguru matplotlib pandas
python Bandit.py
```

---

## 💡 Bonus Suggestion 

To improve the implementation:

- **Modularize the code**: Splitting classes into separate files (e.g., `epsilon_greedy.py`, `thompson_sampling.py`, `visualization.py`, etc.)
- Adding **unit tests** with `pytest` to ensure correctness of each component
- Using **argparse or a config file** to allow easy modification of parameters like rewards, number of trials, or epsilon formula
- Integrating with **experiment tracking tools** like `mlflow` or `wandb` for better monitoring and logging
- Considering using **cumulative moving average** plots or confidence intervals for better performance analysis# A-B-Testing