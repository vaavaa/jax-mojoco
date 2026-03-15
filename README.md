# jax-MoJoCo

Готовая структура для старта проектов на **Python + [JAX](https://github.com/google/jax) + [MuJoCo](https://mujoco.readthedocs.io/)** (в т.ч. [MJX](https://mujoco.readthedocs.io/en/stable/mjx.html) — MuJoCo под JAX).  
Менеджер пакетов: **uv** (быстрый менеджер от Astral).

## Требования

- Python 3.10–3.12
- [uv](https://docs.astral.sh/uv/) (установка: `curl -LsSf https://astral.sh/uv/install.sh | sh`)

**Среда, на которой проверялся проект:** Ubuntu.
## Установка и запуск

```bash
# Виртуальное окружение и базовые зависимости (JAX)
uv sync

# С опцией MuJoCo под JAX (mujoco + mujoco-mjx)
uv sync --extra mujoco

# Запуск тестов
uv run pytest

# Тесты MuJoCo/MJX (нужен --extra mujoco)
uv run pytest tests/test_mujoco_mjx.py -v
```

## Структура проекта

```
mujoco/
├── pyproject.toml       # Зависимости: JAX, опционально cuda, mujoco, dev
├── uv.lock
├── examples/
├── tests/
│   └── test_mujoco_mjx.py   # Проверка MuJoCo под JAX (MJX)
└── README.md
```

## Опциональные зависимости

- **cuda** — JAX с поддержкой CUDA
- **mujoco** — MuJoCo и mujoco-mjx (симуляция под JAX на GPU/TPU)
- **dev** — pytest, pytest-cov

Установка с несколькими опциями: `uv sync --extra mujoco --extra dev`

## Почему uv

- **Скорость**: в разы быстрее pip и Poetry при установке.
- **Один инструмент**: замена pip, venv, pip-tools, poetry.
- **Lockfile**: воспроизводимые сборки через `uv.lock`.
