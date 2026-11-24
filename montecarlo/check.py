"""Confere as duas versoes.

1. Gerador: as amostras do xoroshiro128+ portado para a PolyHok devem ser identicas, bit a bit,
   as do Numba (data/amostras_ref.bin), partindo dos mesmos estados.
2. Integrais: mesmo criterio do exemplo original (atol = 0,001 em relacao aos valores
   calculados no Maple: 0,69315 para [1, 2] e 0,4055 para [2, 3]); mostra tambem a
   diferenca entre as duas versoes.
Uso: python3 check.py
"""
import os
import sys

import numpy as np

RAIZ = os.path.dirname(os.path.abspath(__file__))
ok = True

ref = np.fromfile(os.path.join(RAIZ, "data", "amostras_ref.bin"), dtype=np.uint32)
caminho = os.path.join(RAIZ, "out", "polyhok_amostras.bin")
if os.path.exists(caminho):
    ph = np.fromfile(caminho, dtype=np.uint32)
    iguais = int((ph == ref).sum())
    if len(ph) == len(ref) and iguais == len(ref):
        print(f"gerador: OK ({iguais} amostras identicas bit a bit)")
    else:
        ok = False
        print(f"gerador: DIVERGE ({iguais} de {len(ref)} amostras identicas)")
else:
    print("gerador: sem resultados da PolyHok")

esperado = {"[1, 2]": 0.69315, "[2, 3]": 0.4055}
res = {}
for versao in ("numba", "polyhok"):
    caminho = os.path.join(RAIZ, "out", f"{versao}.txt")
    if not os.path.exists(caminho):
        print(f"{versao}: sem resultados (rode a versao antes)")
        continue
    with open(caminho) as f:
        valores = [float(v) for v in f.read().split()]
    res[versao] = valores
    for (intervalo, alvo), v in zip(esperado.items(), valores):
        estado = "OK" if abs(v - alvo) <= 1e-3 else "DIVERGE"
        ok &= estado == "OK"
        print(f"{versao} {intervalo}: {estado} ({v:.9f}; Maple {alvo})")

if len(res) == 2:
    dif = max(abs(a - b) for a, b in zip(res["numba"], res["polyhok"]))
    print(f"maior diferenca entre Numba e PolyHok: {dif:.2e}")

sys.exit(0 if ok else 1)
