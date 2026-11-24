# Numba × PolyHok

Programas GPU escritos em [Numba](https://github.com/NVIDIA/numba-cuda), tirados de repositórios
oficiais, portados para a [PolyHok](https://github.com/lups-ufpel/poly_hok) (DSL para GPU em Elixir)
para comparar as duas linguagens com o mesmo programa.

| Programa | Origem do código Numba | Regime | Pasta |
|---|---|---|---|
| Black-Scholes | [numba-examples](https://github.com/numba/numba-examples), `examples/finance/blackscholes` | Computação, um elemento por thread | [`blackscholes`](blackscholes) |
| Redução (soma) | [numba-cuda](https://github.com/NVIDIA/numba-cuda), `tests/doc_examples/test_reduction.py` | Banda de memória, memória compartilhada | [`reducao`](reducao) |
| Integração de Monte Carlo | [numba-cuda](https://github.com/NVIDIA/numba-cuda), `tests/doc_examples/test_montecarlo.py` | Gerador aleatório no device + redução | [`montecarlo`](montecarlo) |

Cada pasta tem o código Numba original (sem alterações ou com as mudanças mínimas explicadas
no README dela), o port para a PolyHok, um gerador de entradas comum às duas versões, um
validador e um script de medição.

## Resultados

RTX 5060 (driver 621.24), WSL 2 com Ubuntu 24.04, CUDA 12.9, numba-cuda 0.30.4, Elixir 1.17.3 / OTP 25,
PolyHok no commit `32397e6`. Mediana de 30 execuções, faixa de 3 rodadas, com aquecimento antes
(compilação JIT fora da medição).

| Programa | Medida | Numba | PolyHok |
|---|---|---|---|
| Black-Scholes (4 M opções) | cópias + kernel | 18,9–20,6 ms | 14,7–17,3 ms |
| | só o kernel | 3,28–3,56 ms | 3,18–3,74 ms |
| Redução (2^25 int32) | cópia + kernels | 18,3–18,5 ms | 18,8–19,8 ms |
| | dados já na GPU | 1,71–1,73 ms | 3,33–3,42 ms |
| Monte Carlo (2^24 amostras) | só o kernel de amostragem | 1,55–1,58 ms | 2,14 ms |
| | kernel + redução | 1,77–1,81 ms | 4,68–4,74 ms |

Validação (as duas versões partem das mesmas entradas):

| Programa | Resultado |
|---|---|
| Black-Scholes | 8 milhões de saídas **idênticas bit a bit** entre Numba e PolyHok |
| Redução | soma exata nas duas versões |
| Monte Carlo | amostras do gerador e valores de f **idênticos bit a bit**; integrais diferem em ~8e-8 (ordem da soma) |

Observações:

- **Black-Scholes:** o kernel empata. O tempo com cópias é dominado pela transferência de
  ~80 MB e varia vários ms entre rodadas nas duas versões.
- **Redução:** o CUDA gerado pela PolyHok é equivalente linha a linha ao kernel Numba. A cada
  `spawn`, a PolyHok carrega o módulo, lança, sincroniza e descarrega (≈0,1–0,2 ms por chamada),
  e compila com NVRTC sem `-arch`. Isso não explica toda a diferença; o restante ainda não
  foi isolado.
- **Monte Carlo:** a PolyHok não tem gerador aleatório nem inteiros de 64 bits; o xoroshiro128+
  foi reescrito com pares de `int` de 32 bits, o que custa mais operações por amostra
  (amostragem ~35% mais lenta). A maior parte da diferença no total vem da redução: o Numba usa
  `@cuda.reduce` (biblioteca) e a PolyHok usa a redução em árvore de `reducao`, porque o
  esqueleto `reduce` está comentado na versão atual.
- **Tipos:** os ports usam os mesmos tipos que o Numba infere para o original (conferidos com
  `inspect_types`). No Black-Scholes e no Monte Carlo, isso significa contas em `float64`,
  porque escalares Python viram `float64` no Numba.

## Limitações da PolyHok encontradas

| Limitação | Onde apareceu | Contorno |
|---|---|---|
| Sem inteiros de 64 bits nem sem sinal | Monte Carlo (xoroshiro128+) | Pares de `int` de 32 bits |
| Sem gerador aleatório no device | Monte Carlo | xoroshiro128+ escrito à mão |
| Esqueleto `reduce` comentado | Monte Carlo | Kernel de redução em árvore |
| Literal sozinho não aceito em contexto `double` (`var a double = 0.5`) | Black-Scholes | Literal dentro da expressão |
| Comparação de `double` com literal (`d > 0.0`) dá erro de tipo | Black-Scholes | `d > d * 0.0` |
| Sem conversão implícita `double` → `float` ao gravar em vetor | Black-Scholes, Monte Carlo | Variável `double` intermediária |
| **Bug:** reatribuir dentro de `if` uma variável declarada com `var` gera uma nova declaração no C, e a reatribuição se perde sem erro | Black-Scholes | `type x double` + atribuição comum |

Detalhes no README de cada programa.

## Ambiente (WSL 2)

GPUs Blackwell (RTX 50xx) precisam de CUDA ≥ 12.8. O WSL 1 não tem acesso à GPU.

```bash
# No Windows (PowerShell): distro nova no WSL 2
wsl --install -d Ubuntu-24.04

# No Ubuntu: CUDA Toolkit para WSL
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb && sudo apt-get update
sudo apt-get install -y build-essential git cmake python3-venv cuda-toolkit-12-9 erlang erlang-dev

# Elixir >= 1.17 (o do apt é 1.14)
wget https://github.com/elixir-lang/elixir/releases/download/v1.17.3/elixir-otp-25.zip
sudo mkdir -p /opt/elixir && sudo unzip -q elixir-otp-25.zip -d /opt/elixir
echo 'export PATH=/usr/local/cuda/bin:/opt/elixir/bin:$PATH' >> ~/.bashrc
source ~/.bashrc
mix local.hex --force && mix local.rebar --force

# Numba (NumPy < 2.4: o numba-cuda 0.30.4 ainda usa np.row_stack)
python3 -m venv ~/venv-numba
~/venv-numba/bin/pip install "numba-cuda[cu12]" "numpy<2.4"
```

Em cada programa, na primeira vez:

```bash
cd polyhok && mix deps.get && mix compile && cd ..
```

Depois, em cada pasta:

```bash
~/venv-numba/bin/python gen_data.py
bash bench.sh
```

## Licenças

O código Numba original vem de numba-examples e numba-cuda, ambos sob licença BSD;
os arquivos de licença estão nas pastas `numba/` de cada programa.
