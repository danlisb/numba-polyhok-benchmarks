# Redução (soma): Numba × PolyHok

Ponto de partida: exemplo oficial de redução do numba-cuda
(`numba/cuda/tests/doc_examples/test_reduction.py`, numba-cuda 0.30.4),
copiado sem alterações em `numba/test_reduction_original.py`.

## Por que foi preciso estender o original

O exemplo só funciona com **um bloco de 1024 elementos**: cada bloco reduziria sua parte,
mas o resultado é gravado em `data[0]` e as somas parciais de vários blocos nunca são combinadas.

Extensão mínima (`numba/reduction_cuda.py`); o resto do kernel é idêntico ao original:

1. cada bloco grava sua soma parcial em `partial[blockIdx.x]`;
2. threads além do fim do vetor carregam 0 na memória compartilhada, para todas chegarem aos `syncthreads()`;
3. o tamanho `n` vira parâmetro (no original, `len(data)`);
4. o host relança o kernel sobre as parciais até restar um valor. **Sem operações atômicas.**

Com N = 2^25, são 3 lançamentos: 33.554.432 → 32.768 → 32 → 1.

## Estrutura

| Caminho | O que é |
|---|---|
| `numba/test_reduction_original.py` | Exemplo original, sem alterações |
| `numba/reduction_cuda.py` | Versão estendida para vários blocos |
| `numba/run.py` | Mede a versão Numba |
| `polyhok/reducao.exs` | Port para a PolyHok |
| `gen_data.py` | Gera o vetor de entrada (int32, valores de 0 a 9) |
| `check.py` | Compara as somas com `np.sum` (comparação exata) |
| `bench.sh` | 3 rodadas de 30 repetições das duas versões + validação |

## Como foi feita a comparação

- **Mesmo kernel:** 1024 threads por bloco, vetor `int32` de 1024 posições em memória
  compartilhada, redução em árvore com `tid % (2*s)`. O código CUDA gerado pela PolyHok
  (veja com `POLYHOK_DEBUG=1 mix run reducao.exs 3`) é equivalente linha a linha ao kernel Numba.
- **Tipo inteiro:** como no original (memória compartilhada `int32`). Valores de 0 a 9
  mantêm a soma dentro de `int32` e permitem validação exata.
- **Duas medidas**, com aquecimento antes (JIT fora da medição):
  - `total`: cópia host→GPU + kernels + cópia do resultado;
  - `gpu`: dados já na GPU, só kernels + cópia do resultado.

## Como rodar (WSL)

```bash
python3 gen_data.py            # 2^25 elementos
bash bench.sh
```

## Observação sobre o tempo `gpu`

Na PolyHok, cada `PolyHok.spawn` (em `c_src/gpu_nifs.cu` do cuda_backend) carrega o módulo
(`cuModuleLoadDataEx`), lança o kernel, sincroniza (`cuCtxSynchronize`) e descarrega o módulo.
O NVRTC é chamado sem `-arch`, então o PTX é gerado para uma arquitetura genérica e
recompilado pelo driver para a GPU ao carregar o módulo. O Numba compila direto para a GPU
e só sincroniza ao copiar o resultado.
