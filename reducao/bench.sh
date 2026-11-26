#!/bin/bash
# Roda as versoes 3 vezes (30 repeticoes cada) e valida:
#   kernel do exemplo: Numba e port para a PolyHok (mesmo algoritmo)
#   reducoes genericas: @cuda.reduce do Numba e reduce da PolyHok
# Uso: bash bench.sh   (depois de gen_data.py)
cd "$(dirname "$0")"
. ~/venv-numba/bin/activate
echo -e "versao\tN\ttotal_min_ms\ttotal_mediana_ms\tgpu_min_ms\tgpu_mediana_ms"
for i in 1 2 3; do
  (cd numba && python run.py 30 | tail -1)
  (cd polyhok && mix run reducao.exs 30 2>/dev/null | tail -1)
  (cd numba && python run_reduce.py 30 | tail -1)
  (cd polyhok && mix run reducao_ske.exs 30 2>/dev/null | tail -1)
done
python check.py
