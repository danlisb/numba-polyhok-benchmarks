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
(compilação JIT fora da medição). As duas versões passam na validação em todos os programas.

| Programa | Medida | Numba | PolyHok |
|---|---|---|---|
| Black-Scholes (4 M opções) | cópias + kernel | 16,5–17,8 ms | 25,0–29,2 ms |
| | melhor execução | 13,1–14,2 ms | 13,1–14,0 ms |
| Redução (2^25 int32) | cópia + kernels | 17,8–18,8 ms | 19,4–19,8 ms |
| | dados já na GPU | 1,74–1,80 ms | 3,10–3,39 ms |
| Monte Carlo (2^24 amostras) | só o kernel de amostragem | 1,56–1,58 ms | 1,63–1,78 ms |
| | kernel + redução | 1,79–1,80 ms | 4,15–4,41 ms |

Observações:

- **Black-Scholes:** no melhor caso as duas empatam (o tempo é dominado pelas cópias);
  a PolyHok varia mais de uma execução para outra, e a causa não foi investigada.
- **Redução:** o CUDA gerado pela PolyHok é equivalente linha a linha ao kernel Numba. A cada
  `spawn`, a PolyHok carrega o módulo, lança, sincroniza e descarrega (≈0,1–0,2 ms por chamada),
  e compila com NVRTC sem `-arch`. Isso não explica toda a diferença; o restante ainda não
  foi isolado.
- **Monte Carlo:** a PolyHok não tem gerador aleatório nem inteiros de 64 bits; o xoroshiro128+
  foi reescrito com pares de `int` de 32 bits e gera amostras **idênticas bit a bit** às do Numba.
  A diferença no total vem da redução: o Numba usa `@cuda.reduce` (biblioteca) e a PolyHok
  usa a redução em árvore de `reducao`, porque o esqueleto `reduce` está comentado na versão atual.

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
