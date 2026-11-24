"""Gera as entradas do Black-Scholes em binario float32, lidas pelas duas versoes.

Mesma distribuicao do input_generator do numba-examples (blackscholes.py).
Uso: python3 gen_data.py N
"""
import os
import sys

import numpy as np

N = int(sys.argv[1]) if len(sys.argv) > 1 else 4_000_000
DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DIR, exist_ok=True)


def randfloat(rand_var, low, high):
    return (1.0 - rand_var) * low + rand_var * high


rng = np.random.default_rng(42)
dados = {
    "S": randfloat(rng.random(N), 5.0, 30.0),    # stockPrice
    "X": randfloat(rng.random(N), 1.0, 100.0),   # optionStrike
    "T": randfloat(rng.random(N), 0.25, 10.0),   # optionYears
}
for nome, arr in dados.items():
    arr.astype(np.float32).tofile(os.path.join(DIR, f"{nome}.bin"))

print(f"{N} opcoes geradas em {DIR}")
