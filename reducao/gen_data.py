"""Gera o vetor de entrada da reducao (int32), lido pelas duas versoes.

Valores inteiros de 0 a 9, para que a soma caiba em int32 (o tipo do vetor
em memoria compartilhada no exemplo original) e a validacao seja exata.
Uso: python3 gen_data.py [N]   (padrao: 2**25 = 33.554.432 elementos)
"""
import os
import sys

import numpy as np

N = int(sys.argv[1]) if len(sys.argv) > 1 else 2**25
DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DIR, exist_ok=True)

rng = np.random.default_rng(42)
dados = rng.integers(0, 10, size=N, dtype=np.int32)
dados.tofile(os.path.join(DIR, "dados.bin"))

print(f"{N} elementos gerados em {DIR} (soma esperada {int(dados.sum(dtype=np.int64))})")
