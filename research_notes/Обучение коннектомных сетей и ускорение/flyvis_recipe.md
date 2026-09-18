# FlyVis: рецепт обучения и его цена

Область: Lappalainen et al., "Connectome-constrained networks predict neural
activity across the fly visual system", Nature (2024) [далее — "статья"];
код github.com/TuragaLab/flyvis, пакет flyvis (текущий релиз 1.1.3 на
2026-09-18). Все даты источников указаны явно; сегодня 2026-09-18.

## Какое железо и сколько часов на модель/ансамбль (GPU, wall time per model, wall time per 50-model ensemble)?

### Takeaway
Модель GPU и wall-clock время на одну модель и на ансамбль **не найдены** ни
в доступном тексте статьи/PMC, ни в README/документации репозитория, ни в
найденных issues. Это ключевой пробел данного отчёта — возможно, эти цифры
есть только в PDF-версии Extended Data/Supplementary Information Nature,
которая не была доступна для полнотекстового фетча.

### Cited Findings
- Полнотекстовая версия статьи на PMC (открытый доступ) описывает
  оптимизатор, число итераций, dt, длину последовательности, регуляризацию
  и данные, но **не содержит** упоминания модели GPU, часов на модель или
  на ансамбль, CUDA, mixed precision или распараллеливания членов ансамбля
  — при целенаправленном запросе именно этих фактов ничего не было найдено.
  — [Connectome-constrained networks predict neural activity across the fly visual system, PMC (открытый доступ к Nature 2024)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11525180/)
- Основная страница nature.com отдаёт редирект на форму логина (идентификация
  издателя), полный текст через неё получить не удалось в рамках этого
  запроса — [Nature, статья](https://www.nature.com/articles/s41586-024-07939-3)
- Ансамбль в статье — 50 моделей, "constrained with the same connectome,
  and optimized to perform the same task" (одинаковая инициализация схемы,
  разные случайные веса/seed); конкретное время обучения ансамбля в найденных
  фрагментах не указано — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11525180/)
- Один из открытых GitHub issues (#13, "Training causes slurm OOM error",
  закрыт 2025-09-11) показывает практический контекст обучения: пользователь
  запускал обучение на SLURM-кластере с **1 GPU**, 4 CPU-процесса (`nP`),
  очередь `gpu`, задача `flow` с `ensemble_id 0001`; job падал по OOM даже
  при `--mem 32G`/`64G`, то есть проблема была не просто в лимите памяти —
  "the training runs for longer but only until memory overflows and the same
  error occurs" — [GitHub issue #13, TuragaLab/flyvis](https://github.com/TuragaLab/flyvis/issues/13)

### Inferences
- Раз пользователи обучают по одной модели на 1 GPU через SLURM-джобы (issue
  #13), базовый паттерн обучения в сообществе — один GPU-процесс на одну
  сеть ансамбля, без штатного упоминания упаковки нескольких моделей на одну
  карту в найденных материалах.

### Gaps
- Модель GPU (V100/A100/др.), часы на модель, часы на ансамбль из 50 моделей
  — не найдены; вероятно, нужны Methods/Extended Data PDF статьи напрямую
  с nature.com (доступ не открылся) или Supplementary Information.
- Нет данных о примерной денежной/энергетической стоимости обучения от
  авторов.

## Почему batch size = 4 и 250 000 итераций: даётся ли обоснование, тестировались ли другие batch size?

### Takeaway
Статья формально не приводит явного обоснования именно этих чисел (в
доступном фрагменте методов нет фразы вида "we chose batch size 4 because…");
указаны сами значения и связанный процесс инициализации через отдельную
150 000-итерационную оптимизацию "shared resting potentials".

### Cited Findings
- Оптимизатор: "stochastic gradient descent with adaptive moment estimation
  (β1 = 0.9, β2 = 0.999, learning rate decreased from 5 × 10⁻⁵ to 5 × 10⁻⁶ in
  ten steps over iterations, batch size of four)" — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11525180/)
- Основное обучение сети шло "after about 250,000 iterations"; отдельно,
  до/параллельно, "shared resting potentials" (общие потенциалы покоя)
  оптимизировались 150 000 итераций стохастическим градиентным спуском без
  момента (momentum) — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11525180/)
- Явного текста про выбор batch size = 4 или про тестирование других
  batch size не найдено ни в статье, ни в issues репозитория (целевой
  поиск issues по training time/batch не дал релевантных открытых или
  закрытых обсуждений batch size) — [GitHub issues, TuragaLab/flyvis](https://github.com/TuragaLab/flyvis/issues)
- Конфигурационный файл `solver.yaml` в репозитории — это Hydra-конфиг,
  собирающий подконфиги для архитектуры сети, задачи, оптимизации, штрафов
  (penalties) и расписания learning rate; сам просмотренный файл не содержит
  текстового обоснования численных значений (это компоновочный YAML, не
  документ с комментариями-обоснованиями) — [solver.yaml, TuragaLab/flyvis](https://raw.githubusercontent.com/TuragaLab/flyvis/main/flyvis/config/solver.yaml)
- В отдельном фолк-репозитории/учебном примере (документационный тьюториал
  "02_flyvision_optic_flow_task") показан игрушечный прогон на batch size 4,
  но также отдельно демонстрационный прогон на 1000 эпох с "single-batch
  overfitting" (переобучение на одном батче) — это учебный пример, не
  свидетельство перебора batch size в исследовании — [Flyvis docs, tutorial 02](https://turagalab.github.io/flyvis/examples/02_flyvision_optic_flow_task/)

### Inferences
- Отсутствие обоснования в открытых источниках позволяет предположить, что
  batch size = 4 и ~250k итераций — эмпирически подобранные авторами статьи
  значения (возможно, ограниченные памятью GPU при полном connectome ~45k
  нейронов на кадр), но это предположение, не подтверждённый факт.

### Gaps
- Нет прямого высказывания авторов, почему выбраны именно batch=4 и 250k
  итераций (например, ограничение по видеопамяти или по времени сходимости).
- Не найдено ни одного issue/обсуждения, где кто-то тестировал бы другие
  batch size и сообщал бы numbers (скорость/качество/память).

## Сколько тренировочных и валидационных последовательностей в задаче Sintel во flyvis (train loader с drop_last=True; наблюдали 16 валидационных и меньше 64 тренировочных)?

### Takeaway
Источники подтверждают dt = 20 мс (частота кадров 50 Гц), длину
последовательности 19 кадров (~792 мс) и то, что Sintel даёт 23 исходных
сцены, которые расширяются до 69 последовательностей за счёт вертикального
разбиения (3 подразбиения на сцену) — но точный численный сплит train/val
(в частности объяснение наблюдаемых 16 val / <64 train) не подтверждён
источниками.

### Cited Findings
- "not smaller than 20 ms (that is, a frame rate of 50 Hz after
  resampling)" — dt/частота кадров задачи — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11525180/)
- Длина последовательности при обучении — "19 consecutive frames",
  соответствующие "792 ms" симуляции; перед этим 500 мс серого стимула для
  инициализации сети в стационарном состоянии на каждый минибатч —
  [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11525180/)
- Тренировочные данные — "23 sequences from the publicly available
  computer-animated film Sintel"; отдельно упомянут "held-out validation
  set" без указанного числа — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11525180/)
- Документационный тьюториал сайта flyvis отдельно указывает: "The Sintel
  dataset comprises 23 original sequences, expanded to 69 total sequences
  through vertical splitting (3 splits per sequence)" — [Flyvis docs, tutorial 02 "flyvision_optic_flow_task"](https://turagalab.github.io/flyvis/examples/02_flyvision_optic_flow_task/)
- Аугментации, упомянутые в статье: случайные отражения (random flips),
  случайные повороты (random rotation), гауссов шум, изменение
  контраста/яркости, пространственный страйдинг (spatial striding) —
  [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11525180/)
- Регуляризация активности: λV=0.1, γ=1, δ=0.01, целевая активность a=5
  (условные единицы) — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11525180/)
- Чекпойнтинг: "We regularly checkpointed the error measure L_Y,Ŷ averaged
  across a held-out validation set" — периодичность чекпойнтинга (число
  итераций между чекпойнтами) в найденном тексте не указана —
  [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11525180/)

### Inferences
- 69 последовательностей (23 сцены × 3 вертикальных сплита) — по всей
  видимости, общий пул, из которого нарезаются train/val; наблюдение "16
  валидационных, меньше 64 тренировочных" арифметически совместимо с общим
  пулом ~69-85 последовательностей при разбиении по сценам (а не по
  сабклипам, чтобы избежать протечки между train/val одной и той же сцены),
  но это не подтверждено источником напрямую — вывод не проверен.

### Gaps
- Точное число train- и val-последовательностей (а не общий пул 69) не
  найдено ни в статье, ни в документации в доступном тексте.
- Не найден код `datasets/sintel.py` с параметрами `n_train`/`n_val`/логикой
  `drop_last=True` — попытка через GitHub code search вернула требование
  логина и не дала результатов.
- Нет прямого подтверждения/опровержения наблюдения "16 val, <64 train" —
  требуется прямое чтение исходного кода датасета в репозитории (не
  выполнено в рамках веб-поиска).

## Задокументированные ускорения, известные узкие места, JAX/компилированная версия?

### Takeaway
Прямых официальных ускорений от авторов (JAX-порт, torch.compile, mixed
precision) в найденных материалах **не обнаружено**; единственные найденные
конкретные сигналы о скорости/памяти — практическая проблема OOM на SLURM
(issue #13) и отчёт о накладных расходах на построение коннектома в
стороннем форке.

### Cited Findings
- GitHub issue #13 ("Training causes slurm OOM error", закрыт 2025-09-11)
  документирует, что обучение одной сети (`flow`, `ensemble_id 0001`) на
  1 GPU регулярно падало по нехватке памяти, и увеличение `--mem` лишь
  откладывало, а не устраняло проблему — [issue #13](https://github.com/TuragaLab/flyvis/issues/13)
- В стороннем форке `KedoKudo/flyvis` (не официальный репозиторий Turaga
  lab) отмечена проблема производительности: "Connectome build spends 5.4 s
  in sleep(), LayerActivity another 1.8 s per construction (missing `[:]`
  at 8 sites)" — то есть накладные расходы на построение объекта коннектома
  и LayerActivity при каждом создании — [issue #2, KedoKudo/flyvis (форк, не официальный репозиторий)](https://github.com/KedoKudo/flyvis/issues/2)
- Официальный репозиторий TuragaLab/flyvis реализован на PyTorch ("A
  connectome-constrained deep mechanistic network (DMN) model of the fruit
  fly visual system in PyTorch") — никакого упоминания JAX-порта в самом
  названии/описании репозитория не найдено — [GitHub, TuragaLab/flyvis](https://github.com/TuragaLab/flyvis)
- Целевой поиск "TuragaLab flyvis GitHub issue slow training speed up JAX"
  не нашёл релевантного issue про JAX или про ускорение обучения в самом
  репозитории TuragaLab; выдача вместо этого возвращала общие issues из
  проекта JAX (jax-ml/jax), не относящиеся к flyvis — [поиск, см. запрос выше]
- Список открытых issues на момент проверки (2026-09-18) — только два:
  #26 "License scope for published pretrained models and derived sparse
  arrays" (открыт 2026-09-16) и #23 "`solver.recover()` calls
  `resolve_checkpoints()` with 4 args but signature accepts 1 — resume
  broken in 1.1.3" (открыт 2026-06-28); ни один не о скорости/памяти —
  [Issues, TuragaLab/flyvis](https://github.com/TuragaLab/flyvis/issues)

### Inferences
- Отсутствие найденных обсуждений про JAX/compile/mixed precision наводит
  на вывод, что официальный проект остаётся на "ванильном" PyTorch без
  задокументированного публичного порта на JAX по состоянию на 2026-09-18 —
  но это вывод по отсутствию находок, не по явному отрицанию авторов.

### Gaps
- Нет подтверждения ни наличия, ни отсутствия официального JAX-порта —
  только отсутствие найденных публичных следов такого порта.
- Нет данных про sparse-операции/CUDA-кастомизацию/mixed precision в
  обучающем коде — не удалось получить исходный код тренировочного цикла
  напрямую (GitHub code search требует авторизации).
- Неясно, устранена ли проблема OOM из issue #13 и как (issue не содержал
  явного решения в извлечённом тексте).

## Ретренинги/фоллоу-апы 2025–2026 (Turaga lab, FlyWire/MaleCNS, "flyvis 2"?) — что изменили в рецепте обучения?

### Takeaway
Найдены два конкретных фоллоу-апа 2026 года, которые переобучают модели по
рецепту flyvis **без изменения** гиперпараметров/архитектуры декодера —
то есть ни один из найденных фоллоу-апов не заявляет "flyvis 2" или
переход на FlyWire/MaleCNS-коннектом в обучающем цикле; они используют тот
же connectome scaffold ("fixed flyvis network scaffold") оригинальной
статьи.

### Cited Findings
- "Reproducibility and model-selection stability in connectome-constrained
  circuit modeling" (bioRxiv, опубликовано 2026-04-21, DOI
  10.64898/2026.04.18.717873): "The retraining experiments were performed
  using the publicly available flyvis repository and training procedures
  described in Lappalainen et al. (2024), **without modification** to the
  training objective, hyperparameters, or decoder architecture. Two new
  ensembles of 50 networks each were trained with different random
  initialization seeds under otherwise identical conditions." Основной
  результат работы — что отбор "биологически правдоподобных" моделей по
  наименьшей validation-ошибке задачи нестабилен между независимыми
  ретренингами (перегруппировка кластеров моделей при небольших вариациях
  метрик) — [bioRxiv, Reproducibility and model-selection stability](https://www.biorxiv.org/content/10.64898/2026.04.18.717873v1)
- "Topological Sensitivity in Connectome-Constrained Neural Networks"
  (arXiv:2604.04033v1, 2026-04-05): использует тот же "fixed flyvis network
  scaffold" (45 669 узлов, 1 513 231 направленных рёбер — то есть тот же
  MaleCNS-производный коннектом, что и в оригинале flyvis, без явного
  указания версии FlyWire/MaleCNS); задача — MovingEdge direction decoding
  (Stage 3 канонической задачи flyvis); batch = 12 стимулов по 269 кадров
  (форма (12,269,1,721)); оптимизатор Adam, learning rate 10⁻³; 3 сида
  {0,1,2}; 734 обучаемых и 2959 фиксированных параметров сети; сообщается
  только elapsed wall time для отдельных прогонов (например, "252 s" на 5
  шагов оптимизации), без общего времени обучения или железа —
  [arXiv 2604.04033v1](https://arxiv.org/html/2604.04033v1)
- Ни одна из двух найденных работ не называется "flyvis 2" и не заявляет
  переобучение на новом коннектоме (FlyWire женской особи или отдельной
  версии MaleCNS) — обе используют существующий flyvis-коннектом как есть.

### Inferences
- Судя по обеим работам, основное направление фоллоу-апов на 2026 год —
  не ускорение/масштабирование обучения, а изучение устойчивости и
  топологической чувствительности уже существующего рецепта; ни ускорения,
  ни изменения гиперпараметров эти статьи не предлагают.

### Gaps
- Не найдено публикации, которая явно переобучала бы модель flyvis на
  MaleCNS (в отличие от коннектома, уже используемого в оригинальной
  статье/репозитории) или на FlyWire как на новом источнике коннектома —
  это может означать, что такой работы ещё нет в открытом доступе на
  2026-09-18, либо что она не была найдена этим поиском.
- Не найдено никакого "flyvis 2" анонса, релиза или CHANGELOG-записи с
  таким названием.
- Не удалось получить полный текст bioRxiv PDF (получен только сработавший
  через WebSearch фрагмент, не полнотекстовый фетч) — данные о времени/
  железе ретренинга в этой работе не извлечены, только факт "без изменения
  гиперпараметров".
