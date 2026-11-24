# Integracao de Monte Carlo em PolyHok, portada de numba/montecarlo_cuda.py
# (exemplo oficial do numba-cuda, tests/doc_examples/test_montecarlo.py).
#
# O que a PolyHok nao tem e foi preciso escrever:
#
# 1. Gerador xoroshiro128+ no device (no Numba: xoroshiro128p_uniform_float32).
#    A PolyHok so tem int de 32 bits com sinal, float e double; nao ha inteiro de 64 bits
#    nem sem sinal. Cada estado de 64 bits vira dois int (parte baixa e alta) e cada operacao
#    de 64 bits (soma, xor, rotacao, deslocamento) e feita sobre as duas metades. Os
#    deslocamentos mascaram o operando antes, para o C gerado nao ter comportamento indefinido
#    com int negativo. Os estados iniciais sao os mesmos do Numba (data/estados.bin), e as
#    amostras saem identicas bit a bit (conferido por check.py).
#
# 2. Reducao (no Numba: @cuda.reduce, da biblioteca). O esqueleto reduce da PolyHok esta
#    comentado nesta versao, entao a soma usa o mesmo kernel de reducao em arvore do
#    programa de reducao (../reducao), em versao float.
#
# A funcao integrada e passada ao kernel como parametro (kernel de ordem superior).
#
# Uso: mix run montecarlo.exs [REPS]

require PolyHok

PolyHok.defmodule MonteCarlo do
  # ---------- operacoes de 64 bits sobre pares de int de 32 bits ----------

  # Deslocamento logico a direita (k de 2 a 31): >> em int e aritmetico, entao mascara o sinal
  defd lshr(x, k) do
    return((x >>> k) &&& ((1 <<< (32 - k)) - 1))
  end

  # Deslocamento a esquerda (k de 2 a 31) mascarando antes, para o operando nunca ser negativo
  defd shl(x, k) do
    return((x &&& ((1 <<< (32 - k)) - 1)) <<< k)
  end

  # Parte baixa de a + b (64 bits), somando em blocos de 16 bits para nao estourar int
  defd add_lo(al, bl) do
    l16 = (al &&& 65535) + (bl &&& 65535)
    h16 = lshr(al, 16) + lshr(bl, 16) + lshr(l16, 16)
    return(shl(h16, 16) ||| (l16 &&& 65535))
  end

  # Parte alta de a + b (64 bits), com o vai-um da parte baixa
  defd add_hi(ah, al, bh, bl) do
    l16 = (al &&& 65535) + (bl &&& 65535)
    carry = lshr(lshr(al, 16) + lshr(bl, 16) + lshr(l16, 16), 16)
    m16 = (ah &&& 65535) + (bh &&& 65535) + carry
    n16 = lshr(ah, 16) + lshr(bh, 16) + lshr(m16, 16)
    return(shl(n16, 16) ||| (m16 &&& 65535))
  end

  # ---------- kernel (equivale a mc_integrator_kernel do original) ----------
  # st guarda os estados: st[4*i] = s0 baixo, st[4*i+1] = s0 alto, st[4*i+2] = s1 baixo,
  # st[4*i+3] = s1 alto (mesma ordem de bytes do vetor de estados do Numba)
  defk mc_integrator_kernel(out, st, lower_lim, upper_lim, size, f) do
    # No original os limites sao inteiros do Python (int64), e amostra float32 * int64 da
    # float64 no Numba; aqui os limites sao double, com os mesmos valores.
    type lower_lim double
    type upper_lim double
    gid = blockIdx.x * blockDim.x + threadIdx.x

    if gid < size do
      b = 4 * gid
      s0l = st[b]
      s0h = st[b + 1]
      s1l = st[b + 2]
      s1h = st[b + 3]

      # xoroshiro128p_next: result = s0 + s1
      rl = add_lo(s0l, s1l)
      rh = add_hi(s0h, s0l, s1h, s1l)

      # s1 ^= s0
      t1l = s1l +++ s0l
      t1h = s1h +++ s0h

      # s0 = rotl(s0, 55) ^ s1 ^ (s1 << 14)   [rotl 55 = troca as metades e rotl 23]
      st[b] = (shl(s0h, 23) ||| lshr(s0l, 9)) +++ t1l +++ shl(t1l, 14)
      st[b + 1] = (shl(s0l, 23) ||| lshr(s0h, 9)) +++ t1h +++ (shl(t1h, 14) ||| lshr(t1l, 18))

      # s1 = rotl(s1, 36)   [rotl 36 = troca as metades e rotl 4]
      st[b + 2] = shl(t1h, 4) ||| lshr(t1l, 28)
      st[b + 3] = shl(t1l, 4) ||| lshr(t1h, 28)

      # uint64_to_unit_float32: float32((result >> 11) * 2^-53)
      # a = 21 bits altos e c = 32 bits baixos de (result >> 11). A expressao e toda em double
      # e exata (53 bits); so a atribuicao a u arredonda para float, como no Numba.
      a = lshr(rh, 11)
      c = shl(rh, 21) ||| lshr(rl, 11)
      var u float = (a * 4294967296.0 + (lshr(c, 16) * 65536.0 + (c &&& 65535))) * 1.1102230246251565e-16

      # normaliza para o intervalo (em double, como no Numba) e avalia a funcao
      var samp double = u * (upper_lim - lower_lim) + lower_lim
      var y double = f(samp)
      out[gid] = y
    end
  end

  # ---------- reducao (mesmo kernel de ../reducao, em float) ----------
  defk soma_blocos(data, partial, n) do
    __shared__ shr[1024]

    tid = threadIdx.x
    i = blockIdx.x * blockDim.x + threadIdx.x

    if i < n do
      shr[tid] = data[i]
    else
      shr[tid] = 0.0
    end

    __syncthreads()

    s = 1

    while s < blockDim.x do
      if (tid ~>> (2 * s)) == 0 do
        shr[tid] = shr[tid] + shr[tid + s]
      end

      s = s * 2
      __syncthreads()
    end

    if tid == 0 do
      partial[blockIdx.x] = shr[0]
    end
  end

  def reduzir(d, n) when n <= 1 do
    PolyHok.get_gnx(d) |> Nx.to_flat_list() |> hd()
  end

  def reduzir(d, n) do
    nb = div(n + 1023, 1024)
    parcial = PolyHok.new_gnx({nb}, {:f, 32})
    PolyHok.spawn(&MonteCarlo.soma_blocos/3, {nb, 1, 1}, {1024, 1, 1}, [d, parcial, n])
    reduzir(parcial, nb)
  end

  # ---------- equivalentes de amostrar / integrar da versao Numba ----------
  def amostrar(out, st, n, tpb, lower, upper, f) do
    nb = div(n + tpb - 1, tpb)

    PolyHok.spawn(
      &MonteCarlo.mc_integrator_kernel/6,
      {nb, 1, 1},
      {tpb, 1, 1},
      [out, st, lower, upper, n, f]
    )
  end

  def integrar(out, st, n, tpb, lower, upper, f) do
    amostrar(out, st, n, tpb, lower, upper, f)
    # normalization factor: (b - a)/(N - 1)
    factor = (upper - lower) / (n - 1)
    reduzir(out, n) * factor
  end
end

# funcao integrada, como no original: f(x) = 1/x
func = PolyHok.phok(fn x -> 1.0 / x end)
identidade = PolyHok.phok(fn x -> x end)

reps =
  case System.argv() do
    [arg] -> String.to_integer(arg)
    _ -> 10
  end

raiz = Path.expand("..", __DIR__)
out_dir = Path.join(raiz, "out")
File.mkdir_p!(out_dir)

estados = File.read!(Path.join([raiz, "data", "estados.bin"])) |> Nx.from_binary({:s, 32})
n = div(Nx.size(estados), 4)
tpb = File.read!(Path.join([raiz, "data", "tpb.txt"])) |> String.trim() |> String.to_integer()
out = PolyHok.new_gnx({n}, {:f, 32})

# 1a. Conferencia do gerador: com f(x) = x em [0, 1), out recebe a propria amostra
MonteCarlo.amostrar(out, PolyHok.new_gnx(estados), n, tpb, 0.0, 1.0, identidade)
File.write!(Path.join(out_dir, "polyhok_amostras.bin"), out |> PolyHok.get_gnx() |> Nx.to_binary())

# 1b. Validacao, sempre a partir dos estados iniciais: [1, 2] e [2, 3], como no original
# (POLYHOK_DEBUG=1 mostra o CUDA gerado para o kernel com f(x) = 1/x e para a reducao)
if System.get_env("POLYHOK_DEBUG"), do: PolyHok.set_debug_logs(true)
r12 = MonteCarlo.integrar(out, PolyHok.new_gnx(estados), n, tpb, 1.0, 2.0, func)
PolyHok.set_debug_logs(false)
# f(amostra), antes da reducao
File.write!(Path.join(out_dir, "polyhok_fx.bin"), out |> PolyHok.get_gnx() |> Nx.to_binary())
r23 = MonteCarlo.integrar(out, PolyHok.new_gnx(estados), n, tpb, 2.0, 3.0, func)
File.write!(Path.join(out_dir, "polyhok.txt"), "#{r12}\n#{r23}\n")

# 2. Medicao, com aquecimento antes e estados ja na GPU
d_estados = PolyHok.new_gnx(estados)

medir = fn f ->
  f.()

  ms =
    for _ <- 1..reps do
      t0 = System.monotonic_time()
      f.()
      System.convert_time_unit(System.monotonic_time() - t0, :native, :microsecond) / 1000
    end
    |> Enum.sort()

  {hd(ms), Enum.at(ms, div(length(ms), 2))}
end

{min_a, med_a} = medir.(fn -> MonteCarlo.amostrar(out, d_estados, n, tpb, 1.0, 2.0, func) end)
{min_t, med_t} = medir.(fn -> MonteCarlo.integrar(out, d_estados, n, tpb, 1.0, 2.0, func) end)

fmt = fn x, d -> :erlang.float_to_binary(x / 1, decimals: d) end

IO.puts("versao\tN\tamostragem_min_ms\tamostragem_mediana_ms\ttotal_min_ms\ttotal_mediana_ms\tI[1,2]\tI[2,3]")

IO.puts(
  "PolyHok\t#{n}\t#{fmt.(min_a, 3)}\t#{fmt.(med_a, 3)}\t#{fmt.(min_t, 3)}\t#{fmt.(med_t, 3)}\t#{fmt.(r12, 7)}\t#{fmt.(r23, 7)}"
)
