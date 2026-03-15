# jax-MuJoCo

Starter template for **Python + [JAX](https://github.com/google/jax) + [MuJoCo](https://mujoco.readthedocs.io/)** (including [MJX](https://mujoco.readthedocs.io/en/stable/mjx.html) — MuJoCo on JAX).  
Package manager: **uv** (by Astral).

## Requirements

- Python 3.10–3.12
- [uv](https://docs.astral.sh/uv/) (install: `curl -LsSf https://astral.sh/uv/install.sh | sh`)

**Tested on:** Ubuntu.

## Install and run

```bash
# Virtual environment and base dependencies (JAX)
uv sync

# With MuJoCo on JAX (mujoco + mujoco-mjx)
uv sync --extra mujoco

# Run tests
uv run pytest

# MuJoCo/MJX tests (requires --extra mujoco)
uv run pytest tests/test_mujoco_mjx.py -v
```

## Project layout

```
mujoco/
├── pyproject.toml       # Dependencies: JAX, optional cuda, mujoco, dev
├── uv.lock
├── examples/
├── tests/
│   └── test_mujoco_mjx.py   # MuJoCo on JAX (MJX) sanity check
└── README.md
```

## Optional dependencies

- **cuda** — JAX with CUDA support
- **mujoco** — MuJoCo and mujoco-mjx (JAX simulation on GPU/TPU)
- **dev** — pytest, pytest-cov

Install multiple extras: `uv sync --extra mujoco --extra dev`

## Why uv

- **Speed**: much faster than pip and Poetry for installs.
- **Single tool**: replaces pip, venv, pip-tools, poetry.
- **Lockfile**: reproducible installs via `uv.lock`.
