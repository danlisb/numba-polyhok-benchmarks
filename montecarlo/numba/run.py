"""Mede a integracao de Monte Carlo em Numba (f(x) = 1/x).

1. Validacao: com os estados iniciais de data/estados.bin, integra em [1, 2] e em [2, 3]
   (os dois casos do exemplo original) e grava os resultados em ../out/numba.txt.
2. Medicao, com aquecimento antes (JIT fora da medicao) e estados ja na GPU:
     amostragem: so o kernel
     total:      kernel + reducao + resultado no host
Uso: python3 run.py [REPS]
"""
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore", message="Grid size")

import numpy as np
from numba import cuda
from numba.cuda.random import xoroshiro128p_dtype

from montecarlo_cuda import amostrar, integrar

REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 10
RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(RAIZ, "out")
os.makedirs(OUT, exist_ok=True)

estados_host = np.fromfile(os.path.join(RAIZ, "data", "estados.bin"), dtype=xoroshiro128p_dtype)
n = len(estados_host)
out = cuda.device_array(n, dtype=np.float32)

# 1. Validacao, sempre a partir dos estados iniciais
r12 = integrar(out, cuda.to_device(estados_host), 1, 2)
out.copy_to_host().tofile(os.path.join(OUT, "numba_fx.bin"))  # f(amostra), antes da reducao
r23 = integrar(out, cuda.to_device(estados_host), 2, 3)
with open(os.path.join(OUT, "numba.txt"), "w") as f:
    f.write(f"{float(r12)!r}\n{float(r23)!r}\n")

# 2. Medicao
d_estados = cuda.to_device(estados_host)


def medir(f):
    f()  # aquecimento
    tempos = []
    for _ in range(REPS):
        t0 = time.perf_counter()
        f()
        tempos.append((time.perf_counter() - t0) * 1000)
    tempos.sort()
    return tempos[0], tempos[len(tempos) // 2]


min_a, med_a = medir(lambda: amostrar(out, d_estados, 1, 2))
min_t, med_t = medir(lambda: integrar(out, d_estados, 1, 2))

print("versao\tN\tamostragem_min_ms\tamostragem_mediana_ms\ttotal_min_ms\ttotal_mediana_ms\tI[1,2]\tI[2,3]")
print(f"Numba\t{n}\t{min_a:.3f}\t{med_a:.3f}\t{min_t:.3f}\t{med_t:.3f}\t{float(r12):.7f}\t{float(r23):.7f}")
