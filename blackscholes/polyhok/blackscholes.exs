# Black-Scholes em PolyHok, portado de numba-examples/examples/finance/blackscholes/blackscholes_cuda.py
#
# Mantem a estrutura do original: uma funcao de device (cnd) e um kernel com uma opcao por thread,
# 1024 threads por bloco. Le as entradas de ../data, faz uma execucao de aquecimento (compilacao JIT),
# mede REPS execucoes (incluindo copias host<->GPU, como no original) e grava os resultados em ../out.
#
# Uso: mix run blackscholes.exs [REPS]

require PolyHok

PolyHok.defmodule BlackScholes do
  # Distribuicao normal acumulada (equivale a cnd_cuda do original)
  defd cnd(d) do
    a1 = 0.31938153
    a2 = -0.356563782
    a3 = 1.781477937
    a4 = -1.821255978
    a5 = 1.330274429
    rsqrt2pi = 0.3989422804014327
    k = 1.0 / (1.0 + 0.2316419 * fabs(d))

    ret_val =
      rsqrt2pi * exp(-0.5 * d * d) *
        (k * (a1 + k * (a2 + k * (a3 + k * (a4 + k * a5)))))

    if d > 0.0 do
      ret_val = 1.0 - ret_val
    end

    return(ret_val)
  end

  # Equivale a black_scholes_cuda_kernel do original
  defk black_scholes_kernel(call_result, put_result, s, x, t, r, v, n) do
    i = threadIdx.x + blockIdx.x * blockDim.x

    if i < n do
      sqrt_t = sqrt(t[i])
      d1 = (log(s[i] / x[i]) + (r + 0.5 * v * v) * t[i]) / (v * sqrt_t)
      d2 = d1 - v * sqrt_t
      cndd1 = cnd(d1)
      cndd2 = cnd(d2)

      exp_rt = exp(-1.0 * r * t[i])
      call_result[i] = s[i] * cndd1 - x[i] * exp_rt * cndd2
      put_result[i] = x[i] * exp_rt * (1.0 - cndd2) - s[i] * (1.0 - cndd1)
    end
  end

  # Equivale a funcao black_scholes_cuda do original: copia, executa e traz os resultados
  def black_scholes(s_cpu, x_cpu, t_cpu, r, v) do
    n = Nx.size(s_cpu)
    block_size = 1024
    n_blocks = div(n + block_size - 1, block_size)

    s = PolyHok.new_gnx(s_cpu)
    x = PolyHok.new_gnx(x_cpu)
    t = PolyHok.new_gnx(t_cpu)
    call_gpu = PolyHok.new_gnx({n}, {:f, 32})
    put_gpu = PolyHok.new_gnx({n}, {:f, 32})

    PolyHok.spawn(
      &BlackScholes.black_scholes_kernel/8,
      {n_blocks, 1, 1},
      {block_size, 1, 1},
      [call_gpu, put_gpu, s, x, t, r, v, n]
    )

    {PolyHok.get_gnx(call_gpu), PolyHok.get_gnx(put_gpu)}
  end
end

riskfree = 0.02
volatility = 0.30

reps =
  case System.argv() do
    [arg] -> String.to_integer(arg)
    _ -> 10
  end

data_dir = Path.expand("../data", __DIR__)
out_dir = Path.expand("../out", __DIR__)
File.mkdir_p!(out_dir)

ler = fn nome -> File.read!(Path.join(data_dir, nome)) |> Nx.from_binary({:f, 32}) end
s = ler.("S.bin")
x = ler.("X.bin")
t = ler.("T.bin")
n = Nx.size(s)

# Aquecimento: a primeira chamada compila o kernel
BlackScholes.black_scholes(s, x, t, riskfree, volatility)

tempos =
  for _ <- 1..reps do
    t0 = System.monotonic_time()
    {call, put} = BlackScholes.black_scholes(s, x, t, riskfree, volatility)
    t1 = System.monotonic_time()
    {System.convert_time_unit(t1 - t0, :native, :microsecond) / 1000, call, put}
  end

{_, call, put} = List.last(tempos)
File.write!(Path.join(out_dir, "polyhok_call.bin"), Nx.to_binary(call))
File.write!(Path.join(out_dir, "polyhok_put.bin"), Nx.to_binary(put))

ms = tempos |> Enum.map(fn {ms, _, _} -> ms end) |> Enum.sort()
mediana = Enum.at(ms, div(length(ms), 2))

IO.puts("versao\tN\tmin_ms\tmediana_ms")
IO.puts("PolyHok\t#{n}\t#{:erlang.float_to_binary(hd(ms), decimals: 3)}\t#{:erlang.float_to_binary(mediana, decimals: 3)}")
