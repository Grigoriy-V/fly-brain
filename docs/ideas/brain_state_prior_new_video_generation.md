# Brain-State Prior: генерация новых видео через пространство состояний мозга

## Статус идеи

Это следующий промежуточный этап проекта **«Что снится мухе?»**.

Текущая цель этого этапа — **научиться создавать новые видео без исходного видеоклипа**, используя состояние зрительной модели мозга как промежуточное генеративное пространство.

Это ещё не попытка генерировать видео из «мыслей» или сна. Сейчас источник нового состояния будет искусственным — случайный latent / noise. В дальнейшем тот же интерфейс должен позволить заменить искусственный источник на состояния более глубоких отделов мозга и, в конечном счёте, на внутреннюю активность модели мозга без внешнего видео.

---

# 1. Что уже есть

Сейчас работает цепочка:

```text
video
→ frozen MaleCNS-derived visual model
→ T4/T5 state
→ 13B SiT
→ video
→ frozen brain
→ round-trip verification
```

13B уже умеет создавать видео, обусловленное состоянием восьми типов:

```text
T4a T4b T4c T4d
T5a T5b T5c T5d
```

При полном достижимом T4/T5-state генератор в основном ведёт себя как learned inverse: состояние содержит достаточно информации, чтобы сильно ограничить возможное видео.

Step 14 показал важное ограничение:

```text
random numbers → T4/T5 → 13B
```

работает плохо.

Произвольный вектор активности обычно не соответствует состоянию, которое frozen brain вообще способен породить от какого-либо визуального входа.

Иными словами, пространство всех возможных чисел T4/T5 гораздо больше, чем пространство **достижимых нейронных состояний**.

---

# 2. Текущая проблема

Сейчас 13B уже является генеративной видеомоделью:

```text
T4/T5 state + z_video → video
```

Но у нас нет хорошего источника **новых валидных T4/T5 states**.

Для обычного режима состояние берётся так:

```text
source video → brain → T4/T5 state
```

То есть для того, чтобы получить conditioning, сначала требуется видео.

Это ограничивает систему:

```text
video → brain state → video
```

и не даёт простого режима:

```text
random seed → new brain state → new video
```

Именно этот недостающий генеративный источник состояния мы хотим построить.

---

# 3. Основная идея

Добавить отдельную генеративную модель распределения brain states.

Минимальная схема:

```text
T4/T5 state
    ↓
Brain-State Encoder
    ↓
z_brain
    ↓
Brain-State Decoder
    ↓
T4/T5 state'
```

В первом варианте это может быть небольшой **VAE**.

После обучения:

```text
z_brain ~ N(0, I)
        ↓
Brain-State Decoder
        ↓
new T4/T5 state
        ↓
existing 13B SiT
        ↓
new video
```

Полная генеративная цепочка:

```text
random seed
   ↓
z_brain
   ↓
brain-state generator
   ↓
plausible T4/T5 state
   ↓
13B
   ↓
new video
   ↓
frozen brain
   ↓
round-trip state
```

Главное отличие от Step 14:

```text
НЕ:
random numbers → напрямую T4/T5

А:
random noise
→ learned distribution of valid brain states
→ T4/T5
```

То есть noise сначала преобразуется в состояние, похожее на те состояния, которые реально встречались у frozen brain.

---

# 4. Что именно создаётся

По сути получится генератор:

```text
noise → brain latent → brain state → video
```

На входе:

```text
seed / random noise
```

На выходе:

```text
новое видео
```

Но промежуточное пространство генерации будет не обычным arbitrary image/video latent, а пространством активности connectome-constrained visual model.

То есть:

```text
обычная видеомодель:
noise → learned visual latent → video

наш вариант:
noise → learned brain-state latent
      → simulated neural state
      → video
```

Сейчас преимущество этого подхода не в качестве обычного T2V.

Ценность в том, что генеративное пространство связано с состояниями конкретной модели мозга и позже может быть заменено реальной внутренней активностью этой модели.

---

# 5. Зачем нужен VAE / дополнительный encoder

Сам 13B решает задачу:

```text
brain state → video
```

Но он не моделирует распределение:

```text
p(brain state)
```

Новая модель должна научиться именно этому распределению.

В простом варианте:

```text
state → VAE encoder → z_brain
z_brain → VAE decoder → state
```

VAE заставляет latent-space быть компактным и пригодным для sampling.

После обучения можно делать:

```text
z_brain ~ N(0, I)
```

и получать новые состояния.

Кроме random sampling появляются:

```text
z_A → z_B interpolation
```

плавные переходы между состояниями;

```text
z + direction
```

изменения по найденным latent directions;

```text
sample around z
```

вариации вокруг одного состояния;

```text
cluster / latent editing
```

поиск осей движения, optic flow и других свойств.

---

# 6. Важное ограничение простого VAE

Обычный Gaussian VAE может хорошо реконструировать состояния, но плохо генерировать новые.

Возможный сценарий:

```text
real state
→ encoder
→ latent
→ decoder
→ good reconstruction
```

но:

```text
random N(0,1)
→ decoder
→ state outside the true reachable manifold
```

Если это произойдёт, идея не считается неудачной.

Следующий вариант:

```text
state
→ deterministic / weakly regularized autoencoder
→ z_brain

z_brain distribution
→ small latent flow / diffusion model
```

Тогда полная схема:

```text
random noise
→ latent flow / diffusion
→ valid z_brain
→ state decoder
→ T4/T5 state
→ 13B
→ video
```

То есть можно разделить две задачи:

1. хорошо представить brain state;
2. хорошо моделировать распределение brain latents.

Это потенциально сильнее обычного VAE.

---

# 7. Данные для первого эксперимента

Дополнительный прогон MaleCNS сейчас не нужен.

Уже существуют пары из Step 13:

```text
9,468 clips
× 40 frames
× 8 T4/T5 types
× 721 columns
```

Состояния уже сохранены и использовались для обучения 13B.

Это значит, что первый brain-state generator можно обучить непосредственно на существующем dataset.

Новый dataset нужен только после проверки базовой гипотезы.

---

# 8. Предлагаемая первая архитектура

Не нужен большой VAE.

Исходный tensor:

```text
T × K × N
40 × 8 × 721
```

где:

- `T = 40` временных отсчётов;
- `K = 8` типов T4/T5;
- `N = 721` гексагональная колонка.

Вариант:

```text
state
→ spatial / hex encoder
→ temporal compression
→ latent z_brain (128–256 dims)
→ temporal decoder
→ spatial / hex decoder
→ reconstructed state
```

Ориентир для первой модели:

```text
~1–5M parameters
```

Первый latent:

```text
128 или 256 dimensions
```

Этого достаточно, чтобы проверить сам принцип.

---

# 9. Что не нужно менять на первом этапе

На первом тесте не нужно:

- переобучать 13B;
- менять frozen brain;
- расширять MaleCNS;
- генерировать новый большой dataset;
- добавлять text conditioning;
- добавлять central brain;
- обучать whole-brain model.

Используем существующую систему как фиксированный downstream verifier:

```text
new brain state
→ existing 13B
→ generated video
→ existing frozen brain
→ compatibility score
```

---

# 10. Три главные проверки

## 10.1 Reconstruction

Проверить:

```text
real state
→ VAE
→ reconstructed state
```

Метрики:

- normalized MSE по T4/T5;
- correlation;
- ошибка отдельно по типам;
- temporal structure;
- spatial structure.

Главный вопрос:

> Сохраняет ли latent достаточную информацию о brain state?

---

## 10.2 Sampling

Проверить:

```text
random z_brain
→ generated state
→ 13B
→ video
```

Нужно смотреть:

- разнообразие states;
- разнообразие videos;
- нет ли collapse к одному среднему состоянию;
- не превращается ли всё в grey/noise/prior texture.

Главный вопрос:

> Даёт ли random latent действительно новые состояния и новые видео?

---

## 10.3 Brain round-trip — главная проверка

Для каждого sampled state:

```text
sampled state S
→ 13B
→ generated video V
→ frozen brain
→ reconstructed state S'
```

Сравнить:

```text
S ↔ S'
```

Это определяет, действительно ли brain-state generator создаёт состояния, совместимые с нашим brain model.

Если:

```text
round-trip low
```

то sampled state лежит близко к reachable neural manifold.

Если:

```text
round-trip high
```

то VAE научился генерировать числа, но не валидные состояния мозга.

Это важнее визуального качества ролика.

---

# 11. Дополнительные проверки

## Diversity

Разные:

```text
z_brain
```

должны давать разные:

```text
states
videos
```

Можно измерять:

- pairwise correlation states;
- pairwise correlation videos;
- motion statistics;
- distribution по T4/T5 directions.

---

## Interpolation

```text
z_A
→ lerp
→ z_B
```

Проверить:

```text
z(t)
→ state(t)
→ video(t)
```

Если latent-space хороший, изменения должны быть плавными.

---

## Nearest-neighbour check

Для sampled state найти ближайший training state.

Нужно убедиться, что модель не просто копирует training examples.

Идеальный результат:

```text
sampled state
≈ plausible
but != exact training state
```

---

# 12. Что будет считаться успехом

Минимальный успех:

1. VAE хорошо реконструирует held-out brain states.
2. Random latent produces diverse T4/T5 states.
3. Эти states имеют существенно меньший round-trip error, чем raw random T4/T5 из Step 14.
4. 13B создаёт из них разные видео.
5. Generated states не являются точными nearest-neighbour copies training set.

Сильный успех:

```text
random seed
→ never-seen brain state
→ never-seen video
→ frozen brain
→ approximately same state
```

То есть система действительно умеет создавать **новые brain-compatible videos без source video**.

---

# 13. Два noise в будущей системе

У системы потенциально будут два независимых источника случайности:

```text
z_brain
```

выбирает, **какое состояние мозга** создать;

```text
z_video
```

выбирает, **как именно визуально реализовать это состояние**.

Текущий 13B показывает, что при полном T4/T5 conditioning:

```text
z_video
```

влияет мало, потому что state сильно ограничивает результат.

Поэтому основным генеративным seed на текущем этапе станет:

```text
z_brain
```

---

# 14. Почему это не просто ещё один обычный VAE

Если целью было бы просто сделать хороший video generator, использование MaleCNS не даёт очевидного преимущества перед обычным learned video latent.

Уникальность здесь в другой постановке:

```text
video
→ connectome-constrained visual model
→ neural state
```

Этот state не был обучен специально для удобства генерации видео.

Он определяется структурой и динамикой модели зрительной системы.

Поэтому можно исследовать:

- какие видео совместимы с конкретным brain state;
- какие neural states являются достижимыми;
- как меняется визуальная информация с глубиной brain model;
- какие внутренние состояния можно перевести обратно в визуальное пространство.

Brain-state prior нужен не для того, чтобы обогнать обычный T2V, а чтобы сделать neural state полноценным генеративным интерфейсом.

---

# 15. Оценка стоимости первого эксперимента

По текущим измерениям проекта:

- 13B: ~1.4M параметров;
- 20,000 training steps;
- T4;
- ~1,117 секунд;
- оценка стоимости ~`$0.22`.

Небольшой brain-state VAE должен быть того же порядка или дешевле.

Грубая оценка:

| Работа | Ориентир |
|---|---:|
| smoke test / architecture test | `$0.03–0.10` |
| полноценное обучение VAE | `$0.15–0.40` |
| sampling + 13B generation + brain round-trip | `$0.05–0.15` |
| 1–2 дополнительных latent / β варианта | `$0.20–0.50` |

Ожидаемый бюджет первого полноценного этапа:

```text
примерно $0.3–1.0
```

Это оценка, а не измеренная стоимость нового эксперимента.

Если потребуется отдельный latent flow/diffusion prior, он всё равно должен оставаться дешёвым относительно обучения большой видеомодели, поскольку работает в компактном `z_brain`.

---

# 16. Если первый тест работает

Следующий шаг — расширить training distribution.

Сейчас:

```text
~9.5k brain states
```

Далее можно создать:

```text
100k+
```

более разнообразных video → brain pairs:

- natural videos;
- procedural motion;
- multiple local motions;
- optic flow;
- translation;
- rotation;
- expansion / contraction;
- occlusion;
- moving objects;
- dynamic textures;
- mixtures.

Цель большого dataset:

> не просто увеличить количество видео, а плотнее покрыть множество достижимых neural states.

После этого переобучаются:

```text
brain-state prior
и при необходимости 13B
```

и повторяется Step 14 / round-trip evaluation.

---

# 17. Возможный будущий text conditioning

После появления working brain-state generator можно добавить текст:

```text
text embedding + noise
→ brain-state prior
→ neural state
→ video
```

Но текущий T4/T5 state в первую очередь связан с visual motion.

Поэтому команды вроде:

```text
move right
looming
camera forward
rotation
```

естественнее связывать с T4/T5, чем:

```text
forest
face
car
```

Для полноценного semantic T2V текст, вероятно, должен также напрямую conditioning'ить video generator.

Например:

```text
text ───────────────┐
                    ↓
noise → brain prior → neural state ─→ video generator → video
                    ↑                     ↑
                    └─────────────────────┘
```

Текст может отвечать за семантику, brain state — за neural-compatible motion/visual dynamics.

Но это отдельный будущий этап.

---

# 18. Главная цель текущего этапа

Сейчас задача не T2V и не «сон».

Задача максимально конкретная:

> **Научить систему создавать новые видео без исходного видео, используя генеративно созданное, но brain-compatible T4/T5 состояние.**

То есть перейти от:

```text
source video
→ brain
→ state
→ 13B
→ video
```

к:

```text
random seed
→ learned brain-state distribution
→ new T4/T5 state
→ 13B
→ new video
```

Это будет первый случай, когда источником generated video является не конкретный исходный клип, а sampled neural state.

---

# 19. Долгосрочное продолжение: идти глубже в мозг

T4/T5 — только текущий уровень.

В будущем идея проекта — использовать states на разных уровнях visual pathway:

```text
photoreceptors
→ lamina
→ medulla
→ T4/T5
→ LPi / VS / HS
→ visual projection neurons
→ central brain
```

Для каждого уровня можно строить собственный:

```text
state → generator → video
```

или общий multi-level generator.

Это позволит сравнивать, какое визуальное содержание остаётся доступным на разных глубинах.

Примерно:

```text
early visual state
→ appearance / local detail

T4/T5
→ local motion / direction

VS/HS / downstream optic-flow circuitry
→ global motion / egomotion

deeper central-brain state
→ более абстрактное и поведенчески значимое представление
```

Это гипотеза направления, а не уже доказанный результат.

---

# 20. Конечная идея проекта

Brain-state VAE/prior — промежуточный источник neural states.

Сейчас:

```text
random noise
→ brain-state generator
→ neural state
→ video
```

В будущем этот искусственный источник должен быть заменён состояниями, которые возникают **внутри более глубокой модели мозга**:

```text
internal brain dynamics
→ deeper neural state
→ video generator
→ video
```

То есть конечная исследовательская линия:

```text
внешний визуальный стимул
→ brain activity
→ понять и декодировать state

затем

внутренняя brain activity без исходного видео
→ neural state
→ generated video
```

В популярной формулировке это можно описывать как движение к генерации видео из «мыслей» или «снов» модели мозга.

В строгой формулировке:

> **генерация визуального содержания, совместимого с внутренним нейронным состоянием динамической модели мозга.**

Это и есть долгосрочная цель.

---

# 21. Предлагаемый порядок реализации

## Phase 1 — Brain-State VAE

```text
existing T4/T5 dataset
→ VAE
→ reconstruction
```

Проверить held-out reconstruction.

---

## Phase 2 — Random neural sampling

```text
random z
→ VAE decoder
→ state
```

Проверить distribution и diversity.

---

## Phase 3 — Video generation

```text
sampled state
→ existing 13B
→ video
```

Без переобучения 13B.

---

## Phase 4 — Neural round-trip

```text
sampled state
→ 13B
→ video
→ frozen brain
→ state'
```

Это главный gate.

---

## Phase 5 — If VAE prior fails

```text
state
→ autoencoder
→ z_brain

noise
→ latent flow / diffusion
→ z_brain
→ decoder
→ state
```

---

## Phase 6 — Larger dataset

Если sampling уже принципиально работает:

```text
9.5k
→ 100k+ diverse brain states
```

и переобучить prior / generator.

---

## Phase 7 — Deeper states

После того как pipeline:

```text
state distribution
→ sampled state
→ video
→ brain verification
```

работает стабильно на T4/T5, переносить ту же идею на более глубокие уровни MaleCNS.

---

# Итог

Текущая система уже умеет:

```text
T4/T5 state → video
```

Следующая задача:

```text
random seed → plausible T4/T5 state
```

После объединения:

```text
random seed
→ plausible neural state
→ new video
→ frozen brain verification
```

Это превращает существующий 13B из генератора, которому нужен внешний neural condition, в полноценный pipeline создания **новых brain-compatible videos**.

А в дальнейшем источник состояния можно заменить:

```text
random latent
```

на:

```text
более глубокую внутреннюю активность модели мозга
```

и использовать не только T4/T5, но и другие visual layers и более глубокие brain states.

**Текущая цель: сначала научиться создавать новые видео.**
