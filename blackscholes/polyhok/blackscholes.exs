# Black-Scholes em PolyHok, portado de numba-examples/examples/finance/blackscholes/blackscholes_cuda.py
#
# Mantem a estrutura do original: uma funcao de device (cnd) e um kernel com uma opcao por thread,
# 1024 threads por bloco.
#
# Tipos iguais aos que o Numba infere para o original (conferidos com inspect_types):
#   - vetores float32;
#   - R e V sao float64 (floats do Python), entao d1, d2, cnd, exp_rt e as contas de
#     call/put sao float64;
#   - sqrt_t e float32 (sqrt de float32) e log(S/X) tambem e calculado em float32.
# A PolyHok nao converte double para float sozinha ao gravar num vetor float, entao os
# resultados passam por uma variavel double antes de ir para o vetor.
#
# Le as entradas de ../data, faz uma execucao de aquecimento (compilacao JIT) e mede:
#   total: copias host<->GPU + kernel, como a funcao black_scholes_cuda do original
#   gpu:   so o kernel, com os dados ja na GPU
# Grava os resultados em ../out.
#
# Uso: mix run blackscholes.exs [REPS]
#      POLYHOK_DEBUG=1 mix run blackscholes.exs 3   (mostra o CUDA gerado)

require PolyHok

PolyHok.defmodule BlackScholes do
  # Distribuicao normal acumulada (equivale a cnd_cuda do original).
  # As constantes A1..A5 e RSQRT2PI do original (float64) estao escritas direto na expressao:
  # o verificador de tipos da PolyHok nao aceita um literal sozinho num contexto double
  # (var a1 double = 0.31938153 da erro), mas aceita dentro de expressoes, e no CUDA gerado
  # o literal ja e double. O calculo e o mesmo.
  defd cnd(d) do
    type d double
    var k double = 1.0 / (1.0 + 0.2316419 * fabs(d))

    # ret_val e reatribuida no if abaixo. Com "var ret_val double = ...", a PolyHok declara
    # uma variavel nova dentro do if (double ret_val = 1.0 - ret_val;) e a reatribuicao se
    # perde. Com "type" + atribuicao comum, ela registra a declaracao e reatribui certo.
    type ret_val double

    # RSQRT2PI = 0.3989422804014327; A1 = 0.31938153; A2 = -0.356563782;
    # A3 = 1.781477937; A4 = -1.821255978; A5 = 1.330274429
    ret_val =
      0.3989422804014327 * exp(-0.5 * d * d) *
            (k *
               (0.31938153 +
                  k * (-0.356563782 + k * (1.781477937 + k * (-1.821255978 + k * 1.330274429)))))

    # Original: if d > 0. A PolyHok nao aceita comparar double com literal (d > 0.0 da erro de
    # tipo). d * 0.0 vale 0 e tem o tipo do primeiro operando (double), entao a condicao e a
    # mesma para d finito.
    if d > d * 0.0 do
      ret_val = 1.0 - ret_val
    end

    return(ret_val)
  end

  # Equivale a black_scholes_cuda_kernel do original
  defk black_scholes_kernel(call_result, put_result, s, x, t, r, v, n) do
    type r double
    type v double
    i = threadIdx.x + blockIdx.x * blockDim.x

    if i < n do
      var sqrt_t float = sqrt(t[i])
      var d1 double = (log(s[i] / x[i]) + (r + 0.5 * v * v) * t[i]) / (v * sqrt_t)
      var d2 double = d1 - v * sqrt_t
      var cndd1 double = cnd(d1)
      var cndd2 double = cnd(d2)

      var exp_rt double = exp(-1.0 * r * t[i])
      var c double = s[i] * cndd1 - x[i] * exp_rt * cndd2
      var p double = x[i] * exp_rt * (1.0 - cndd2) - s[i] * (1.0 - cndd1)
      call_result[i] = c
      put_result[i] = p
    end
  end

  def lancar(call_gpu, put_gpu, s, x, t, r, v, n) do
    block_size = 1024
    n_blocks = div(n + block_size - 1, block_size)

    PolyHok.spawn(
      &BlackScholes.black_scholes_kernel/8,
      {n_blocks, 1, 1},
      {block_size, 1, 1},
      [call_gpu, put_gpu, s, x, t, r, v, n]
    )
  end

  # Equivale a funcao black_scholes_cuda do original: copia, executa e traz os resultados
  def black_scholes(s_cpu, x_cpu, t_cpu, r, v) do
    n = Nx.size(s_cpu)
    s = PolyHok.new_gnx(s_cpu)
    x = PolyHok.new_gnx(x_cpu)
    t = PolyHok.new_gnx(t_cpu)
    call_gpu = PolyHok.new_gnx({n}, {:f, 32})
    put_gpu = PolyHok.new_gnx({n}, {:f, 32})
    lancar(call_gpu, put_gpu, s, x, t, r, v, n)
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

if System.get_env("POLYHOK_DEBUG"), do: PolyHok.set_debug_logs(true)

# Aquecimento (a primeira chamada compila o kernel) e resultado para a validacao
{call, put} = BlackScholes.black_scholes(s, x, t, riskfree, volatility)
PolyHok.set_debug_logs(false)
File.write!(Path.join(out_dir, "polyhok_call.bin"), Nx.to_binary(call))
File.write!(Path.join(out_dir, "polyhok_put.bin"), Nx.to_binary(put))

medir = fn f ->
  ms =
    for _ <- 1..reps do
      t0 = System.monotonic_time()
      f.()
      System.convert_time_unit(System.monotonic_time() - t0, :native, :microsecond) / 1000
    end
    |> Enum.sort()

  {hd(ms), Enum.at(ms, div(length(ms), 2))}
end

{min_t, med_t} = medir.(fn -> BlackScholes.black_scholes(s, x, t, riskfree, volatility) end)

# So o kernel: dados e saidas ja na GPU (o spawn da PolyHok sincroniza ao final)
[d_s, d_x, d_t] = Enum.map([s, x, t], &PolyHok.new_gnx/1)
d_call = PolyHok.new_gnx({n}, {:f, 32})
d_put = PolyHok.new_gnx({n}, {:f, 32})

{min_g, med_g} =
  medir.(fn -> BlackScholes.lancar(d_call, d_put, d_s, d_x, d_t, riskfree, volatility, n) end)

f = fn x -> :erlang.float_to_binary(x, decimals: 3) end
IO.puts("versao\tN\ttotal_min_ms\ttotal_mediana_ms\tgpu_min_ms\tgpu_mediana_ms")
IO.puts("PolyHok\t#{n}\t#{f.(min_t)}\t#{f.(med_t)}\t#{f.(min_g)}\t#{f.(med_g)}")
