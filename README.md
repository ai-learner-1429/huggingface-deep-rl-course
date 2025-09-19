# HuggingFace Deep RL Course

Hands-on exercises for training reinforcement learning agents using Stable-Baselines3 and Gymnasium. See the official course here: [HuggingFace Deep RL Course](https://huggingface.co/learn/deep-rl-course/en/unit0/introduction)

## Installation

This project depends on `pygame==2.1.3`, which **is not compatible with Python 3.12 or later**.  

To avoid issues, please use **Python 3.11 or lower**. For example, you can create a virtual environment using Python 3.11:

```bash
uv init --python=python3.11
source .venv/bin/activate   # Linux/macOS
# or
.venv\Scripts\activate      # Windows
pip install -r unit1_hands_on/requirements.txt
```

## Unit1: Hands-on PPO
Run the training script (it took ~24min on a GPU):
```bash
python unit1_hands_on/ppo_lunarlander.py
```