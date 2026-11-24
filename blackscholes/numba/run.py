"""Executa o Black-Scholes original do numba-examples sem alterar o kernel.

Le as entradas de ../data, faz uma execucao de aquecimento (compilacao JIT),
mede REPS execucoes e grava os resultados em ../out.
A medicao inclui as copias host<->GPU, como na funcao original black_scholes_cuda.
Uso: python3 run.py [REPS]
"""
import os
import sys
import time

import numpy as np

from blackscholes_cuda import black_scholes_cuda  # arquivo original, sem modificacoes

RISKFREE = 0.02     # mesmos valores do input_generator original
VOLATILITY = 0.30

REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 10
RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(RAIZ, "data")
OUT = os.path.join(RAIZ, "out")
os.makedirs(OUT, exist_ok=True)

S = np.fromfile(os.path.join(DATA, "S.bin"), dtype=np.float32)
X = np.fromfile(os.path.join(DATA, "X.bin"), dtype=np.float32)
T = np.fromfile(os.path.join(DATA, "T.bin"), dtype=np.float32)
n = len(S)

# Aquecimento: a primeira chamada compila o kernel
call, put = black_scholes_cuda(S, X, T, RISKFREE, VOLATILITY)

tempos = []
for _ in range(REPS):
    t0 = time.perf_counter()
    call, put = black_scholes_cuda(S, X, T, RISKFREE, VOLATILITY)
    tempos.append((time.perf_counter() - t0) * 1000)

call.astype(np.float32).tofile(os.path.join(OUT, "numba_call.bin"))
put.astype(np.float32).tofile(os.path.join(OUT, "numba_put.bin"))

print("versao\tN\tmin_ms\tmediana_ms")
tempos.sort()
mediana = tempos[len(tempos) // 2]  # mesma definicao usada na versao PolyHok
print(f"Numba\t{n}\t{tempos[0]:.3f}\t{mediana:.3f}")
