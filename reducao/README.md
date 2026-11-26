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
| `numba/run_reduce.py` | Mede a redução genérica do Numba (`@cuda.reduce`) |
| `polyhok/reducao.exs` | Port para a PolyHok |
| `polyhok/reducao_ske.exs` | Redução genérica da PolyHok (`reduce` do prof. Andre Du Bois) |
| `polyhok/codigo_gerado.cu` | CUDA que a PolyHok gera para o port |
| `gen_data.py` | Gera o vetor de entrada (int32, valores de 0 a 9) |
| `check.py` | Compara as somas com `np.sum` (comparação exata) |
| `bench.sh` | 3 rodadas de 30 repetições das quatro versões + validação |

## Como foi feita a comparação

- **Mesmo kernel:** 1024 threads por bloco, vetor `int32` de 1024 posições em memória
  compartilhada, redução em árvore com `tid % (2*s)`. O código CUDA gerado pela PolyHok
  (`polyhok/codigo_gerado.cu`) é equivalente linha a linha ao kernel Numba.
- **Tipo inteiro:** como no original (memória compartilhada `int32`); todas as contas do kernel
  são inteiras nas duas versões. Valores de 0 a 9
  mantêm a soma dentro de `int32` e permitem validação exata.
- **Duas medidas**, com aquecimento antes (JIT fora da medição):
  - `total`: cópia host→GPU + kernels + cópia do resultado;
  - `gpu`: dados já na GPU, só kernels + cópia do resultado.

## Reduções genéricas

Além do kernel do exemplo, o benchmark compara as reduções genéricas de cada linguagem,
que recebem a operação como função:

- **Numba:** `@cuda.reduce`, da biblioteca;
- **PolyHok:** o `reduce` enviado pelo prof. Andre Du Bois (`polyhok/reducao_ske.exs`),
  já que o esqueleto `reduce` da PolyHok 0.1.1 está comentado. Usa 64 blocos de 256 threads,
  cada thread acumula vários elementos num laço, e termina em duas chamadas de kernel.

Os algoritmos das duas reduções genéricas são diferentes entre si e diferentes do kernel do exemplo.

Resultado com os dados já na GPU (mediana de 30 execuções, 3 rodadas):

| | Numba | PolyHok |
|---|---:|---:|
| Kernel do exemplo (mesmo algoritmo) | 1,9–2,0 ms | 3,7–3,8 ms |
| Redução genérica | 0,55–0,59 ms | 1,0–1,3 ms |

## Como rodar

Pré-requisitos e instalação: veja o [README principal](../README.md).

```bash
source ~/venv-numba/bin/activate
python gen_data.py             # 2^25 elementos
bash bench.sh
```

## Observação sobre o tempo `gpu`

Na PolyHok, cada `PolyHok.spawn` (em `c_src/gpu_nifs.cu` do cuda_backend) carrega o módulo
(`cuModuleLoadDataEx`), lança o kernel, sincroniza (`cuCtxSynchronize`) e descarrega o módulo.
O NVRTC é chamado sem `-arch`, então o PTX é gerado para uma arquitetura genérica e
recompilado pelo driver para a GPU ao carregar o módulo. O Numba compila direto para a GPU
e só sincroniza ao copiar o resultado.
