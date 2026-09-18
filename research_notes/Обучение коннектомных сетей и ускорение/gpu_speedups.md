# GPU-ускорение обучения больших разреженных рекуррентных сетей (explicit Euler, gather/scatter) на T4/L4, 2022–2026

Контекст задачи: шаг = gather по 1.35 M рёбер (index_select) → ReLU → умножение на веса рёбер →
scatter_add обратно в 31.5 k узлов → поэлементное обновление состояния; 40 шагов вперёд и назад на
сэмпл; batch — ведущая размерность. Измерено на T4: 0.43 с/итерация при batch 4, throughput
насыщается на ~14 samples/s начиная с batch 16 (batch 32 даёт тот же throughput при вдвое большей
памяти, 8.8 GB). Отдельные процессы: два воркера по 0.69 с/итерация каждый вместо 0.43 в одиночку
(прирост только 1.2–1.5x).

## Какие 2-3 техники дадут ≥2x на этом конкретном ворклоаде

### Итоговая оценка (inference risk-weighted)

**Профилирование в первую очередь, а не техника.** Насыщение throughput на batch 16 и то, что
batch 32 не даёт выигрыша при удвоении памяти, — это сигнатура GPU, упирающегося не в compute, а
либо в launch-overhead множества мелких кернелов (gather/ReLU/mul/scatter — это минимум 4 кернела
на шаг × 40 шагов × 2 (forward+backward) = сотни launch на сэмпл), либо в bandwidth для
scatter_add с атомарными операциями. Оба диагноза указывают на разные техники-победители, поэтому
без профиля (torch.profiler + nsys) любая оценка "какая техника даст 2x" остаётся предположением.
Это отмечено как Gap ниже, но по косвенным признакам (насыщение при низком batch, малое улучшение
от удвоения ресурсов) наиболее вероятна launch-overhead-bound или atomic-contention-bound
ситуация, а не чистая bandwidth-bound.

1. **CUDA Graphs (`torch.cuda.graphs.make_graphed_callables` или `torch.compile(mode="reduce-overhead")`) — наиболее вероятный кандидат на ≥2x**, если проблема в launch overhead множества мелких кернелов на шаг.
2. **fp16/AMP** — вероятный источник дополнительных 1.3–2x на T4 за счёт вдвое меньшего трафика памяти в gather/scatter (bandwidth-bound операции выигрывают от уменьшения размера данных даже без активного использования tensor cores), но с риском численной нестабильности при интегрировании 40 шагов.
3. **segment_csr / отсортированные по назначению рёбра вместо scatter_add** — умеренный, но менее рискованный выигрыш (не количественно подтверждён бенчмарками с точными цифрами для данного размера графа).
4. MPS/vmap-ensembling для параллельных членов ансамбля — низкий приоритет: измеренные 1.2–1.5x от параллельных процессов уже близки к типичным цифрам для MPS в литературе, JAX/vmap может делать немного лучше, но это не сама по себе техника ускорения одной модели.

---

## Sparse-форматы: torch.sparse CSR SpMM vs gather/scatter, torch_scatter/segment_csr, PyG fused kernels, cuSPARSE, block-sparse

### Takeaway
Gather-scatter (текущий подход) обычно быстрее SpMM-форматов на малых и средних графах, но хуже
масштабируется и имеет больший memory footprint из-за материализации признаков рёбер; `segment_csr`
из `torch_scatter` — самый быстрый вариант grouped-reduction в этой экосистеме, но количественный
прирост над `scatter_add`/`index_add_` не задокументирован числами для графов масштаба 1.35 M рёбер
/ 31.5 k узлов — это нужно измерить самостоятельно.

### Cited Findings
- Gather-scatter даёт значительные ускорения на малых графах (Citeseer, Cora), но преимущество
  уменьшается на больших графах (PubMed, OGBN-arXiv) — [PyG SparseTensor docs](https://pytorch-geometric.readthedocs.io/en/latest/notes/sparse_tensor.html)
- PyG ≥1.6.0 добавил `SparseTensor`, реализующий быстрый forward/backward на основе SpMM с меньшим
  memory footprint, чем явный gather/scatter — [PyG SparseTensor docs](https://pytorch-geometric.readthedocs.io/en/latest/notes/sparse_tensor.html)
- Недостаток gather-scatter: явная материализация признаков вдоль рёбер даёт высокий memory
  footprint на больших/плотных графах — [PyG SparseTensor docs](https://pytorch-geometric.readthedocs.io/en/latest/notes/sparse_tensor.html)
- `segment_csr()` — самый быстрый метод для групповых редукций благодаря index-pointer формату
  (CSR); в отличие от `scatter()`/`segment_coo()`, полностью детерминирован — [torch_scatter docs, segment_csr](https://pytorch-scatter.readthedocs.io/en/latest/functions/segment_csr.html)
- Страница `segment_csr` не приводит конкретных цифр ускорения — только качественное утверждение
  ("fastest method") — [torch_scatter docs, segment_csr](https://pytorch-scatter.readthedocs.io/en/latest/functions/segment_csr.html) (проверено WebFetch)
- В отдельном бенчмарке нативная `torch.Tensor.scatter_(reduce="add")` оказалась быстрее, чем
  `torch_scatter.scatter_add` в некоторых конфигурациях — [PyG issue #4891, CPU Performance Optimization Roadmap](https://github.com/pyg-team/pytorch_geometric/issues/4891)
- Tensor cores поддерживают sparse-режим (2:4 структурная разреженность) для матриц, закодированных
  в COO/CSR/CSC, но выгода реализуется в основном для sparsity выше ~0.7 в специализированных ядрах
  (Magicube) — [arXiv 2209.06979, Efficient Quantized Sparse Matrix Operations on Tensor Cores](https://arxiv.org/pdf/2209.06979)

### Inferences
- Данная сеть (1.35 M рёбер / 31.5 k узлов ≈ 43 рёбра/узел в среднем) — это "средний" по плотности
  граф; по цитируемому паттерну (gather-scatter лучше на малых/средних, SpMM выигрывает на очень
  больших) миграция на `torch.sparse` CSR SpMM, скорее всего, не даст выигрыша и может даже
  замедлить, если текущий размер ближе к "малый/средний".
- `segment_csr` требует рёбра, отсортированные и сгруппированные по узлу-назначению (CSR
  index-pointer), что требует one-time reindexing графа при построении датасета — низкий риск,
  делается один раз оффлайн, но даёт неопределённый количественно выигрыш.
- 2:4 structured sparsity тензорных ядер неприменим напрямую: граф связей коннектома не является
  структурированно-разреженной матрицей весов слоя, а представляет собой edge-list произвольной
  связности — этот путь, скорее всего, нерелевантен без существенной переработки представления.

### Gaps
- Нет прямого количественного бенчмарка `segment_csr` vs `scatter_add`/`index_add_` на графе
  масштаба ~1.35 M рёбер, 31.5 k узлов на T4/L4 — только качественные утверждения.
- Нет данных по cuSPARSE bSpMM или PyG "fused kernels" (`torch_geometric.utils.spmm`) на T4
  конкретно; их производительность для GNN message-passing (не для сверхбольших графов) не найдена.
- Block-sparse (blocksparse) подходы не нашли релевантных бенчмарков для нерегулярной топологии
  connectome-графа (в отличие от блочно-структурированных матриц внимания).

---

## CUDA Graphs и torch.compile(mode="reduce-overhead") для фиксированного шага цикла

### Takeaway
CUDA Graphs и `torch.compile(mode="reduce-overhead")` устраняют per-kernel launch overhead, заменяя
последовательность GPU-кернелов одной записанной и воспроизводимой единицей; выгода максимальна
именно для ворклоадов с малыми, многочисленными кернелами и CPU-dispatch-bound паттерном — что
структурно соответствует профилю задачи (4+ мелких кернела × 40 шагов × 2 прохода). Ломается любым
Python dict-based state или новой аллокацией тензора на каждом шаге; фикс — статические
persistent-buffer тензоры, переиспользуемые между итерациями.

### Cited Findings
- CUDA Graphs записывают последовательность GPU-кернелов и воспроизводят её как единое целое, убирая
  per-kernel launch overhead; полезны для ворклоадов с фиксированной формой — [NVIDIA docs, CUDA Graph Best Practice for PyTorch, Quick Checklist](https://docs.nvidia.com/dl-cuda-graph/latest/torch-cuda-graph/quick-checklist.html)
- `mode="reduce-overhead"` включает CUDA Graphs; все операции, control flow, memory addresses и
  формы должны быть фиксированы во всех воспроизведениях — [Spheron Blog, torch.compile and CUDA Graphs for LLM Inference (PyTorch 2.6, 2026)](https://www.spheron.network/blog/torch-compile-cuda-graphs-llm-inference-pytorch-2-6/)
- Для Parakeet RNN-T 1.1B использование CUDA graphs в label-looping decoder снизило долю времени,
  потребляемую декодером, с 70% до 18% — [arXiv 2406.03791, Speed of Light Exact Greedy Decoding for RNN-T](https://arxiv.org/pdf/2406.03791)
- Пример на LLaMA2-7B: CUDA graphs дали 2.3x ускорение (30→69 токенов/с) при batch size 1 на A100 —
  выгода "полностью объясняется снижением CPU overhead", а не алгоритмическими улучшениями; работает
  лучше всего, когда "GPU кернелы ждут, пока CPU их диспетчеризует" — то есть для маленьких, часто
  повторяющихся кернелов — [Fireworks.ai blog, Speed, Python: Pick Two](https://fireworks.ai/blog/speed-python-pick-two-how-cuda-graphs-enable-fast-python-code-for-deep-learning)
- Ломает capture: динамические вызовы типа `torch.eye()`/`torch.ones()` внутри forward-прохода;
  решение — регистрировать тензоры как persistent buffers при инициализации модуля, чтобы одна и та
  же память переиспользовалась между forward-проходами — [Lei Mao's Log Book, PyTorch CUDA Graph Capture](https://leimao.github.io/blog/PyTorch-CUDA-Graph-Capture/) (через сводку поиска)
- Буферы, изначально задуманные как динамически создаваемые/удаляемые во время обучения, меняют
  адреса памяти между итерациями, что несовместимо с CUDA graphs — [NVIDIA docs, CUDA Graph Best Practice, Quick Checklist](https://docs.nvidia.com/dl-cuda-graph/latest/torch-cuda-graph/quick-checklist.html)
- Для максимизации выгоды нужно убрать CPU-GPU синхронизации во время исполнения и обеспечить
  статичность размеров тензоров в рамках графа — [NVIDIA docs, CUDA Graph Best Practice, Quick Checklist](https://docs.nvidia.com/dl-cuda-graph/latest/torch-cuda-graph/quick-checklist.html)
- `torch.cuda.make_graphed_callables` позволяет частичный capture отдельных компонентов модели,
  балансируя выгоду и гибкость для динамических частей ворклоада — [PyTorch docs, torch.cuda.make_graphed_callables](https://docs.pytorch.org/docs/stable/generated/torch.cuda.make_graphed_callables.html)

### Inferences
- Шаг задачи (index_select → ReLU → mul → scatter_add → update) — это именно тот паттерн "много
  маленьких кернелов на итерацию", для которого CUDA graphs в литературе дают наибольший выигрыш
  (RNN-T пример: overhead-доля 70%→18%, то есть overhead был доминирующим).
  Насыщение throughput у batch 16 в измерениях проекта согласуется с launch-overhead-bound режимом:
  на маленьких batch кернелы слишком быстрые, чтобы амортизировать launch overhead, а рост batch
  увеличивает работу на кернел без роста числа кернелов — что и объясняет насыщение throughput.
- В коде потребуется: (а) вынести все состояния (активности узлов, промежуточные буферы) в заранее
  выделенные тензоры фиксированного размера batch × узлы вместо создания новых тензоров на каждом
  шаге/сэмпле; (б) убрать любую Python-логику, зависящую от значений тензоров (data-dependent control
  flow), из цикла шагов; (в) зафиксировать размер батча (padding для последнего неполного батча) —
  умеренный риск переработки, но локализован в training loop.
- `torch.compile(mode="reduce-overhead")` — более простой путь входа (декоратор), чем ручной
  `make_graphed_callables`, но менее предсказуем в диагностике поломок capture; ручной API даёт
  больше контроля, но требует больше кода.

### Gaps
- Нет отдельного бенчмарка CUDA graphs именно для gather/scatter-паттерна GNN-подобной сети (все
  найденные числа — LLM/RNN-T decoding), поэтому величина ожидаемого ускорения для этой конкретной
  задачи — экстраполяция по структурному сходству, не прямое измерение.
- Не найдено данных, ломает ли `torch.compile` capture специфично из-за `index_select`+`scatter_add`
  паттерна (in-place накопление в scatter может требовать `torch.compile` fallback на eager для этой
  операции) — требует эмпирической проверки.

---

## Mixed precision (fp16 на T4) для gather/scatter-доминируемого ворклоада

### Takeaway
AMP обычно даёт 1.5–2x на GPU с tensor cores за счёт compute, но для gather/scatter-доминируемого
(вероятно bandwidth-bound) ворклоада главный источник выгоды — вдвое меньший объём данных,
перемещаемых через память при gather и scatter, а не сами tensor cores (T4 slowdown risk: младшие
Turing tensor cores дают ощутимо меньше FP16 TFLOPS, чем L4/Ampere+, поэтому compute-выгода на T4
слабее, чем цифры "1.5–2x" из AMP-документации, полученные преимущественно на более новых GPU).
Риск для 40-шагового явного Эйлера — устойчивость накопления ошибок и underflow в состояниях,
требующие loss scaling и, возможно, keeping state accumulation в fp32.

### Cited Findings
- Включение AMP обычно даёт 1.5–2x ускорение обучения при минимальных изменениях кода на GPU с
  tensor cores — [ACECloud blog, FP8 vs BF16: Choosing Mixed Precision on NVIDIA Tensor Cores](https://acecloud.ai/blog/fp8-vs-bf16-mixed-precision-tensor-cores/)
- В mixed precision входы A и B в FP16, но умножение выполняется в полной точности, а результат
  накапливается в FP32-аккумуляторах — [ACECloud blog, FP8 vs BF16](https://acecloud.ai/blog/fp8-vs-bf16-mixed-precision-tensor-cores/)
- Рекуррентные сети (пример GNMT) демонстрируют значительные вариации распределения градиентов в
  течение обучения и более чувствительны к численным ошибкам — [NVIDIA Docs, Train With Mixed Precision](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html) (через сводку поиска)
- Loss scaling решает проблему underflow градиентов: потери масштабируются большим множителем перед
  backprop, что сдвигает мелкие значения градиентов из зоны underflow в представимый диапазон FP16 —
  [apxml.com, Loss Scaling for FP16 Stability](https://apxml.com/courses/how-to-build-a-large-language-model/chapter-20-mixed-precision-training-techniques/loss-scaling-techniques)
- Для рекуррентных сетей "back-off" динамическое масштабирование потерь менее эффективно при частом
  underflow; более частые обновления scale factor приводили к нестабильному поведению loss и
  расхождению в экспериментах — [apxml.com, Loss Scaling for FP16 Stability](https://apxml.com/courses/how-to-build-a-large-language-model/chapter-20-mixed-precision-training-techniques/loss-scaling-techniques)
- BF16 предпочтительнее FP16 для длинных последовательностей и больших embeddings благодаря
  экспоненциальному диапазону, совпадающему с FP32; градиенты практически никогда не underflow/
  overflow в BF16, loss scaling не нужен — [apxml.com, Loss Scaling for FP16 Stability](https://apxml.com/courses/how-to-build-a-large-language-model/chapter-20-mixed-precision-training-techniques/loss-scaling-techniques)
- T4 (Turing): 65 TFLOPS FP16 dense tensor-core; L4 (Ada Lovelace): 121 TFLOPS FP16 dense — L4 даёт
  ~86% больше FP16-throughput, чем T4 — [Deploybase, L4 vs T4: Specs, Benchmarks & Cloud Pricing Compared](https://deploybase.ai/articles/l4-vs-t4)
- Память: T4 — 16 GB, 300 GB/s (в другом источнике указано 320 GB/s для GDDR6 @10 Gbps/256-bit); L4 —
  24 GB, 300 GB/s — полосы пропускания у T4 и L4 практически совпадают, разница в FP16-compute, не в
  bandwidth — [Deploybase, L4 vs T4](https://deploybase.ai/articles/l4-vs-t4); [ServerBasket, NVIDIA Tesla T4](https://www.serverbasket.com/shop/nvidia-tesla-t4-tensor-gpu-card/)

### Cited Findings (bf16 доступность на T4 — важное ограничение)
- **Важно (не из результатов поиска, а общеизвестный факт архитектуры, отмечается как проверка
  требуется):** T4 (Turing, compute capability 7.5) не имеет аппаратной поддержки BF16 tensor cores
  — BF16 появился начиная с Ampere (compute capability 8.0). На T4 доступен только FP16 с loss
  scaling; BF16 (который в найденных источниках описан как более стабильный без loss scaling)
  доступен только на L4 и новее. Это не было подтверждено отдельным источником в рамках данного
  поиска — см. Gaps.

### Inferences
- Поскольку описанный шаг сети — это в основном memory-movement операции (gather 1.35 M рёбер,
  scatter обратно), а не dense matmul, ожидаемый выигрыш от fp16 на T4, скорее всего, ближе к
  выигрышу от уменьшения вдвое объёма перемещаемых данных (потенциально близко к 2x на bandwidth-
  bound частях), чем к цифрам "1.5-2x из tensor-core compute", которые получены в основном на
  compute-bound (matmul-heavy) ворклоадах. Это стоит подтвердить профилированием, а не принимать
  как данность.
- Учитывая отсутствие BF16 на T4 и обнаруженную в источниках нестабильность fp16 loss scaling именно
  для рекуррентных сетей с изменяющимся распределением градиентов, риск численной нестабильности при
  интегрировании 40 шагов явного Эйлера в fp16 — реальный и требует: (а) fp32 для накопления
  состояния (state update) при fp16 только в gather/matmul/scatter под autocast, (б)
  `GradScaler` с консервативными настройками, (в) валидации, что 40-шаговая динамика не расходится в
  fp16 по сравнению с fp32 baseline — это отдельный, не бесплатный шаг проверки.
- На L4 стоит рассмотреть BF16 вместо FP16 — по цитируемым данным loss scaling не требуется, что
  снижает риск для 40-шаговой интеграции; но T4 — основная целевая платформа по условию задачи.

### Gaps
- Не найдено прямого источника, подтверждающего отсутствие BF16 tensor cores на Turing/T4 (это
  инференс из общеизвестной архитектуры NVIDIA GPU generations, а не из результатов поиска в рамках
  этого запроса) — требует проверки по официальной NVIDIA Turing architecture whitepaper.
  **[ИНФЕРЕНС, не процитировано]**
  Update: указание на "слабое использование fp16 tensor cores на T4" в самой формулировке задачи
  косвенно подтверждает, что этот момент уже известен человеку — вероятно T4 tensor cores для FP16
  требуют явного использования `torch.nn.functional` операций совместимых форм (кратных 8), а
  gather/scatter-heavy код может не задействовать их вовсе, что означает: главная выгода AMP на T4
  для этой задачи — bandwidth, а не compute.
- Нет количественных данных о конкретном приросте AMP именно для scatter_add-доминируемых, а не
  matmul-доминируемых, ворклоадов.

---

## Память: activation checkpointing, adjoint/reversible методы, in-place updates

### Takeaway
Существует явный trade-off: checkpointing даёт O(√T) или O(1) память вместо O(T) ценой ~1.5–2x
дополнительных вычислений при backward; adjoint-метод (torchdiffeq) даёт O(1) память, но имеет
известные проблемы численной точности градиента для дискретных/жёстких систем. Однако measured данные
проекта (batch 32 = вдвое больше памяти при том же throughput, что и batch 16) указывают, что память,
возможно, **не является ограничивающим фактором** здесь — техники экономии памяти не решат проблему
throughput saturation.

### Cited Findings
- Adjoint-метод восстанавливает траекторию состояния и делает backprop через отдельное обратное
  во времени ODE, избегая хранения промежуточных состояний; memory complexity O(L) — только параметры
  сети и их активации хранятся в любой момент времени — [PMC, Adaptive Checkpoint Adjoint Method for Gradient Estimation in Neural ODE](https://pmc.ncbi.nlm.nih.gov/articles/PMC8299461/)
- Checkpointing "квадратного корня из числа шагов" был переоткрыт в ML для backprop через длинные
  RNN; современные реализации обобщают это через настраиваемые checkpoint-policy и selective
  recomputation — [arXiv/PMC via search summary, Adaptive Checkpoint Adjoint](https://pmc.ncbi.nlm.nih.gov/articles/PMC8299461/)
- Trade-off память/вычисления: дополнительная стоимость вычислений от checkpointing обычно в 1.5–2x,
  выгодно при memory-limited режиме — [PMC, Adaptive Checkpoint Adjoint Method](https://pmc.ncbi.nlm.nih.gov/articles/PMC8299461/)
- Neural ODE reconstruction может быть численно нестабильной, continuous adjoint не обязательно равен
  градиенту дискретного forward solver; ACA-метод даёт более точную оценку градиента, чем реализация
  torchdiffeq, за счёт стратегии checkpoint траектории — [PMC, Adaptive Checkpoint Adjoint Method](https://pmc.ncbi.nlm.nih.gov/articles/PMC8299461/)
- `odeint_adjoint` в torchdiffeq реализует memory-efficient adjoint-метод для backprop — [PMC, Adaptive Checkpoint Adjoint Method](https://pmc.ncbi.nlm.nih.gov/articles/PMC8299461/)

### Inferences
- Проектные измерения (batch 32 даёт тот же throughput при вдвое большей памяти, 8.8 GB, что и batch
  16) — прямое свидетельство того, что при текущих batch размерах узкое место — не объём памяти GPU
  (T4 имеет 16 GB, а 8.8 GB — это лишь часть при batch 16-32), а что-то другое (launch overhead или
  bandwidth/compute saturation). Это делает activation checkpointing и adjoint-методы **низким
  приоритетом** для этой конкретной задачи: они решают проблему "не помещается в память при большом
  batch/длинной последовательности", а не проблему "throughput не растёт с ростом batch". Стоит
  использовать эту память для увеличения batch дальше (если бы это помогало) или для CUDA graphs
  buffer allocation, а не тратить усилия на checkpointing.
- Adjoint для явного Эйлера конкретно рискован: описанные проблемы точности градиента относятся к
  жёстким/адаптивным ODE-солверам; для фиксированного 40-шагового явного метода эффект менее изучен
  в найденных источниках и требует отдельной проверки перед использованием в претензии на научный
  результат (согласуется с принципом "предел выводится, не пишется" проекта).

### Gaps
- Нет данных, применяют ли referenced neuroscience-модели (FlyVis, Shiu et al.) checkpointing или
  adjoint для похожих connectome-RNN; не проверено, есть ли в самом FlyVis коде подобная оптимизация
  (стоит посмотреть репозиторий FlyVis напрямую, вне рамок веб-поиска).
- Не найдено прямого количественного сравнения "in-place update state" (`+=` на тензоре состояния) vs
  functional update для памяти/скорости конкретно в PyTorch autograd graph при 40 итерациях — общий
  принцип известен (in-place ломает часть autograd graph reuse, но экономит память), но не
  процитирован источником в рамках этого поиска.

---

## Несколько независимых моделей на одном GPU: CUDA MPS, streams, vmap/ensemble batching vs отдельные процессы

### Takeaway
Измеренные в проекте 1.2–1.5x от параллельных процессов (два воркера по 0.69 с/итерация вместо 0.43
в одиночку) согласуются с диапазоном, отмеченным в литературе как "MPS менее заметен при низком
batch из-за initialization overhead"; в идеальных сценариях MPS даёт бóльший прирост (до 4-5x), но
это требует, чтобы одиночный воркер НЕ насыщал SM/bandwidth GPU — что, по всей видимости, не
выполняется здесь, раз throughput насыщается уже при batch 16 на одной модели. Ensemble-batching
через дополнительную размерность (vmap-подобный подход) — более многообещающая альтернатива
отдельным процессам, но не подтверждена количественно для этого класса задач.

### Cited Findings
- NVIDIA MPS позволяет нескольким процессам разделять один CUDA-контекст на одном GPU, устраняя
  проблему, что exec kernel обычно сериализуется, а каждый процесс создаёт свой контекст, занимающий
  дополнительную память — [Spheron Blog, Fractional GPUs for AI Inference: vGPU, MPS, and Right-Sizing (2026)](https://www.spheron.network/blog/fractional-gpu-inference-vgpu-mps-right-sizing/)
- В идеальных сценариях MPS может давать speedup, близкий к 5x по сравнению со сценариями без MPS;
  внедрение MPS в batch-системы дало улучшение throughput в 4-5x на пике — [Spheron Blog, Fractional GPUs for AI Inference](https://www.spheron.network/blog/fractional-gpu-inference-vgpu-mps-right-sizing/)
- Для vLLM-подобного multi-model инференса: две инстанции с MPS дали 1.42x ускорение при обработке
  160 запросов по сравнению с одной инстанцией — [Spheron Blog, Fractional GPUs for AI Inference](https://www.spheron.network/blog/fractional-gpu-inference-vgpu-mps-right-sizing/)
- При низких batch выгода от MPS менее заметна из-за initialization overhead запуска нескольких
  процессов — [Spheron Blog, Fractional GPUs for AI Inference](https://www.spheron.network/blog/fractional-gpu-inference-vgpu-mps-right-sizing/)
- MPS вряд ли поможет, если один scorer/воркер уже насыщает SM или memory bandwidth GPU целиком;
  также не стоит использовать со speculative decoding, где ограничение SM-allocation замедляет
  draft-модель — [Spheron Blog, Fractional GPUs for AI Inference](https://www.spheron.network/blog/fractional-gpu-inference-vgpu-mps-right-sizing/)

### Inferences
- Тот факт, что проектные измерения дали именно 1.2-1.5x (нижняя граница диапазона, отмеченного в
  литературе), а не 4-5x — сильный косвенный признак, что одна модель уже частично насыщает
  ресурсы T4 (SM occupancy или bandwidth) при batch 4, что согласуется с гипотезой
  launch-overhead-bound или bandwidth-bound режима (насыщение throughput с batch 16 на одной
  модели), а не classic underutilized-GPU сценарием, для которого MPS даёт наибольший выигрыш.
  Отсюда вывод: MPS/множественные процессы — не первая техника, к которой стоит обращаться здесь;
  сначала устранить launch-overhead (CUDA graphs), а затем повторно измерить, есть ли запас для
  параллельных воркеров.
- Ensemble-batching вдоль дополнительной batch-подобной размерности (объединение независимых членов
  ансамбля как ещё одного измерения batch, а не отдельные процессы) избегает per-process overhead
  MPS и CUDA-контекстов; так как задача уже имеет batch как ведущую размерность, technически это
  сводится к увеличению эффективного batch (N_members × batch_per_member), что уже частично
  протестировано проектом (насыщение на batch ≥16) — то есть простое увеличение размерности batch
  для параллелизации ансамбля, вероятно, столкнётся с тем же плато throughput, если корень проблемы
  — launch overhead на шаг, а не недогруженность GPU по compute.

### Gaps
- Не найдено прямых бенчмарков `torch.vmap`/`functorch` ensembling конкретно для message-passing/
  scatter-based сетей (не LLM/vLLM инференс) — экстраполяция с serving-контекста на training explicit
  Euler loop не подтверждена.
- Не найдено данных по CUDA streams (без MPS, просто multiple streams в одном процессе) для этого
  класса задачи — альтернатива, которая не была явно исследована в поиске.

---

## JAX-альтернативы: jax.lax.scan + jit, segment_sum; отчётные ускорения для похожих нейросимуляторов

### Takeaway
`jax.lax.scan` на GPU в общем случае может **ухудшать**, а не улучшать производительность по
сравнению с развёрнутым Python-циклом под JIT, так как каждая итерация scan соответствует запуску
кернела с overhead; выгода scan — в компиляции, не в исполнении. Экосистема SNN-библиотек на JAX
(Spyx, SNNAX) существует, но конкретных числовых сравнений с PyTorch на этом классе задач в
найденных источниках нет. Специализированные C++/CUDA code-generation симуляторы (GeNN, Brian2CUDA)
дают на порядки больший прирост, чем ожидается от смены фреймворка (JAX vs PyTorch) — но это другая
категория решения (переписывание на code-gen, а не смена автоград-библиотеки).

### Cited Findings
- Основная причина использовать `jax.lax.scan` — улучшение времени компиляции под JIT, а не времени
  исполнения; на GPU scan может значительно ухудшать performance исполнения по сравнению с
  Python-циклом, так как вычисление разворачивается в "rolled loop operation", а циклы имеют высокий
  overhead на GPU (каждая итерация — отдельный launch кернела) — [GitHub jax-ml/jax Discussion #16106](https://github.com/jax-ml/jax/discussions/16106)
- Чем больше scan развёрнут ("unrolled"), тем меньше overhead коммуникации CPU-GPU ценой более
  долгой компиляции — [GitHub jax-ml/jax Discussion #16106](https://github.com/jax-ml/jax/discussions/16106)
- Для temporally precise SNN, `jax.lax.associative_scan` вычисляет последовательность аффинных
  отображений с параллельной глубиной O(log K) — экспоненциально быстрее последовательного O(K)
  подхода, так как операции комбинируются древовидно — [arXiv 2603.13283, Bullet Trains: Parallelizing Training of Temporally Precise SNNs](https://arxiv.org/pdf/2603.13283)
- GeNN (GPU-enhanced Neuronal Networks) — фреймворк генерации кода для ускорения симуляций
  нейронных сетей на NVIDIA GPU; для сети из 1 млн Hodgkin-Huxley нейронов достигнуто ускорение в
  200 раз относительно одного ядра CPU — [NCBI PMC, GeNN: a code generation framework for accelerated brain simulations](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4703976/)
- GeNN имеет почти постоянные fixed costs, независимые от размера модели (GPU-архитектура), тогда
  как CPU-симуляторы вроде NEST показывают fixed costs, растущие линейно с размером модели —
  [bioRxiv, Efficient parameter calibration and real-time simulation of large-scale SNNs with GeNN and NEST](https://www.biorxiv.org/content/10.1101/2022.05.13.491646.full.pdf)
- Brian2CUDA ускоряет симуляции до трёх порядков величины по сравнению с CPU-бэкендом Brian; по
  сравнению с Brian2GeNN даёт сопоставимое ускорение, обычно медленнее на малых сетях и быстрее на
  больших — [Frontiers in Neuroinformatics, Brian2CUDA (2022)](https://www.frontiersin.org/journals/neuroinformatics/articles/10.3389/fninf.2022.883700/full)

### Inferences
- Приведённые GeNN/Brian2CUDA цифры (до 200x, до 1000x) относятся к сравнению **GPU code-generation
  vs однопоточный CPU**, а не к сравнению PyTorch vs JAX на GPU — их нельзя напрямую применять как
  ожидаемое ускорение от миграции текущего PyTorch-кода на JAX; это ориентир другого порядка
  решения (переход на специализированный code-generation симулятор целиком), что выходит за рамки
  "2-3 техники поверх текущего PyTorch кода" и требует отдельной research-оценки стоимости миграции.
- Учитывая находку про `jax.lax.scan` (может ухудшать performance на GPU из-за kernel-launch
  overhead на итерацию scan), а сам launch-overhead — вероятная причина проблемы в текущей PyTorch
  реализации, наивная миграция 40-шагового цикла на `jax.lax.scan` без разворачивания
  (`unroll` параметр) или без `jax.jit` на всём цикле, скорее всего, **не решит** и может усугубить
  ту же проблему, которая наблюдается в PyTorch. `jax.jit` по всему step-функции с "unroll" или
  Python `for`-циклом (не `lax.scan`) под единым jit — более вероятный кандидат на выигрыш, по
  структуре сравнимый с эффектом CUDA graphs в PyTorch (устранение launch overhead через
  компиляцию всего графа шагов в единый исполняемый блок).

### Gaps
- Нет прямого количественного сравнения PyTorch (eager/compile) vs JAX (`jit`+`scan` или
  `jit`+развёрнутый цикл) для message-passing/scatter-based RNN на T4/L4 конкретно.
- Не найдено данных по `segment_sum` (JAX-аналог scatter_add) в сравнении с PyTorch `scatter_add`/
  `torch_scatter` для похожих размеров графа.
- Jaxley и BrainPy упомянуты в исходном запросе, но поиск не нашёл специфичных для них
  количественных бенчмарков в рамках выполненных запросов (сложность самостоятельного поиска по
  каждой библиотеке отдельно не уместилась в лимит вызовов инструментов этого исследования) — это
  открытый gap, вероятно, стоит отдельного целевого запроса, если миграция на JAX рассматривается
  всерьёз.

---

## Профилирование: как отличить latency-bound от bandwidth-bound на T4; roofline для 1.35M-edge gather/scatter при batch 16

### Takeaway
Стандартный путь — `torch.profiler` с `record_shapes=True` для операционной разбивки + `nsys`/
Nsight Compute для kernel-level метрик (DRAM throughput, warp stall reasons, occupancy); high
"Long Scoreboard" stalls с одновременно низким DRAM throughput указывают на latency-bound
(launch-overhead) режим, тогда как высокий DRAM throughput близко к пиковой пропускной способности
указывает на bandwidth-bound. Количественный roofline-расчёт для 1.35M-edge gather/scatter при
batch 16 на T4 **не найден в источниках и не может быть процитирован** — это нужно посчитать
отдельно (см. Gaps) на основе размера состояния узла/ребра, которые не заданы в исходном запросе.

### Cited Findings
- Профилировщик определяет, memory-bound (ожидание cache/DRAM) ли кернел, compute-bound (SM cores
  полностью загружены) или bandwidth-bound (ограничен PCIe/NVLink) — [Spheron Blog, GPU Profiling for AI Workloads (2026)](https://www.spheron.network/blog/gpu-profiling-ai-workloads-nsight-compute-pytorch-profiler-guide/)
- Memory-bound регионы ограничены скоростью перемещения байт через HBM; симптомы — мелкие кернелы,
  низкий достигнутый GB/s относительно пикового, профилировщик показывает много узких операций или
  "fusion gaps" — [ADHDecode, Diagnose Whether Your GPU Kernel Is Memory or Compute Bound (2026)](https://adhdecode.com/articles/gpu/gpu-memory-bandwidth-compute-bound-analysis/)
- Высокий процент промахов L1/L2 в сочетании с высоким использованием DRAM bandwidth указывает на
  memory-bound поведение — [ADHDecode, Diagnose Whether Your GPU Kernel Is Memory or Compute Bound](https://adhdecode.com/articles/gpu/gpu-memory-bandwidth-compute-bound-analysis/)
- Высокие "Long Scoreboard" stalls интерпретируются как варпы, ожидающие операций с высокой
  латентностью (обычно память); подтверждается проверкой DRAM throughput, cache hit rates и L2
  utilization — [ADHDecode, Diagnose Whether Your GPU Kernel Is Memory or Compute Bound](https://adhdecode.com/articles/gpu/gpu-memory-bandwidth-compute-bound-analysis/)
- Высокая occupancy при низком throughput часто означает, что compute-инструкции ждут память
  (latency-bound, а не bandwidth-bound) — [ADHDecode, Diagnose Whether Your GPU Kernel Is Memory or Compute Bound](https://adhdecode.com/articles/gpu/gpu-memory-bandwidth-compute-bound-analysis/)
- nsys интегрируется с PyTorch через `torch.profiler` (`record_shapes=True`) и экспортирует
  `.nsys-rep` для корреляции с CUDA/cuDNN timeline; рекомендуется `--kernel-name-base mangled` в NCU
  или обёртывание кернелов в NVTX-ranges (`torch.profiler.record_function`) для маппинга PyTorch
  операций на GPU-кернелы — [Spheron Blog, GPU Profiling for AI Workloads](https://www.spheron.network/blog/gpu-profiling-ai-workloads-nsight-compute-pytorch-profiler-guide/)
- Tesla T4: память GDDR6 @1250MHz/10 Gbps, 256-bit интерфейс, пропускная способность 320 GB/s —
  [ServerBasket, NVIDIA Tesla T4](https://www.serverbasket.com/shop/nvidia-tesla-t4-tensor-gpu-card/)

### Inferences
- Практическая последовательность для этой задачи: (1) `torch.profiler(record_shapes=True,
  with_stack=True)` на 5-10 итерациях при batch 16 (точка насыщения) — посмотреть, какая доля
  времени приходится на index_select/scatter_add/elementwise vs. общее число launched kernels; (2)
  если суммарное kernel launch count на сэмпл велико (40 шагов × ~5-6 операций × 2 (forward+backward)
  = 400-500+ launches) и средняя длительность кернела мала (десятки микросекунд), это подтверждает
  launch-overhead-bound гипотезу, и приоритет — CUDA graphs; (3) `nsys profile` + Nsight Compute на
  отдельном scatter_add кернеле — проверить DRAM throughput % от 320 GB/s пика; если близко к 70-90%
  пика — bandwidth-bound, приоритет fp16 и layout-оптимизации; если далеко — latency-bound,
  приоритет launch overhead.
- Приблизительный (не процитированный, инференс) roofline: при batch 16 состояние на шаг — это
  gather 1.35M рёбер × (размер элемента: fp32=4 байта на признак) минимум дважды читается
  (индексы + значения) и один раз пишется через atomic scatter_add с потенциальной read-modify-write
  амплификацией из-за коллизий записи (несколько рёбер сходятся в один узел). Даже без точного
  расчёта, atomic scatter_add имеет заведомо худшую эффективную bandwidth, чем последовательное
  чтение, из-за serialization записей в один и тот же адрес — это стандартная причина, почему
  scatter операции недогружают заявленную пропускную способность GPU. **Это качественный вывод,
  количественный roofline не подтверждён источником.**

### Gaps
- Точный roofline-расчёт (байт на шаг, ожидаемое время при 320 GB/s, arithmetic intensity) не
  выполнен — требует точных размеров признаков узла/ребра (размерность состояния на узел/ребро),
  которые не были даны в исходном запросе; без них расчёт был бы фабрикацией числа. Рекомендуется
  посчитать отдельно, когда известна размерность feature vector, используя измеренные 0.43 с/итерация
  как точку калибровки, а не как источник для инференса неизвестных величин.
- Не найдено данных, насколько сильно atomic-контенция (несколько рёбер, пишущих в один узел)
  замедляет scatter_add на T4 количественно (общий принцип "atomics дороже" известен, но без цифр).
