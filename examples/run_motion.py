#!/usr/bin/env python3
"""Минимальный пример: проверка, что JAX доступен."""

import jax.numpy as jnp


def main():
    x = jnp.array([1.0, 2.0, 3.0])
    print("JAX работает. Пример:", x @ x)

if __name__ == "__main__":
    main()
