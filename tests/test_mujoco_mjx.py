"""Проверка работоспособности MuJoCo под JAX (mujoco-mjx).

Запуск: установите зависимости с extra mujoco и выполните pytest:
  uv sync --extra mujoco && uv run pytest tests/test_mujoco_mjx.py -v
"""

import pytest

# Пропустить все тесты модуля, если mujoco/mjx не установлены
pytest.importorskip("mujoco")
mjx = pytest.importorskip("mujoco.mjx")

import jax
import mujoco


# Минимальная MJCF-модель: один свободный шар (для шага симуляции под JAX)
BALL_XML = """
<mujoco>
  <worldbody>
    <body>
      <freejoint/>
      <geom size="0.1" mass="1" type="sphere"/>
    </body>
  </worldbody>
</mujoco>
"""


def test_mjx_import():
    """mujoco.mjx доступен (пакет mujoco-mjx установлен)."""
    assert mjx is not None


def test_mjx_put_model_and_step():
    """Модель загружается, переносится в JAX (put_model), создаётся data и делается шаг."""
    model = mujoco.MjModel.from_xml_string(BALL_XML)
    mjx_model = mjx.put_model(model)
    mjx_data = mjx.make_data(mjx_model)
    mjx_data = mjx.step(mjx_model, mjx_data)
    assert mjx_data.qpos.shape == (7,)  # freejoint: 3 pos + 4 quat
    assert mjx_data.qvel.shape == (6,)


def test_mjx_jit_step():
    """Шаг симуляции работает под JAX jit (типичный сценарий MJX)."""
    model = mujoco.MjModel.from_xml_string(BALL_XML)
    mjx_model = mjx.put_model(model)

    def step(data):
        return mjx.step(mjx_model, data)

    mjx_data = mjx.make_data(mjx_model)
    stepped = jax.jit(step)(mjx_data)
    assert stepped.qpos.shape == mjx_data.qpos.shape
    assert jax.numpy.all(jax.numpy.isfinite(stepped.qpos))
