"""Gera os dados compartilhados pelas duas versoes.

  data/estados.bin      estados iniciais do xoroshiro128+ (create_xoroshiro128p_states, seed=42,
                        como no original): para cada amostra, s0 e s1 (uint64, little-endian)
  data/amostras_ref.bin primeira amostra uniforme float32 de cada estado, gerada pelo Numba
                        (xoroshiro128p_uniform_float32); usada para conferir bit a bit o
                        gerador portado para a PolyHok
  data/tpb.txt          threads por bloco que o forall do Numba escolhe para o kernel

Uso: python3 gen_data.py [N]   (padrao: 2**24 amostras)
"""
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore", message="Grid size")

import numpy as np
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "numba"))
from montecarlo_cuda import threads_por_bloco

N = int(sys.argv[1]) if len(sys.argv) > 1 else 2**24
DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DIR, exist_ok=True)

t0 = time.perf_counter()
estados = create_xoroshiro128p_states(N, seed=42)
t_init = time.perf_counter() - t0
estados_host = estados.copy_to_host()
estados_host.tofile(os.path.join(DIR, "estados.bin"))


@cuda.jit
def primeira_amostra(u, rng_states):
    i = cuda.grid(1)
    if i < len(u):
        u[i] = xoroshiro128p_uniform_float32(rng_states, i)


u = cuda.device_array(N, dtype=np.float32)
primeira_amostra.forall(N)(u, cuda.to_device(estados_host))
u.copy_to_host().tofile(os.path.join(DIR, "amostras_ref.bin"))

tpb = threads_por_bloco(N, estados)
with open(os.path.join(DIR, "tpb.txt"), "w") as f:
    f.write(f"{tpb}\n")

print(f"{N} estados gerados em {DIR} ({t_init:.2f} s na CPU); forall usa {tpb} threads por bloco")
