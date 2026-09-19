# Полная архитектура проекта: от видео и T4/T5 до генерации новых видео и более глубоких состояний мозга

## Общая идея

Проект строится как система, в которой модель мозга служит не просто анализатором видео, а **структурированным нейронным пространством**, через которое можно:

1. пропускать внешний визуальный сигнал;
2. снимать внутренние состояния;
3. генерировать новые состояния;
4. создавать из этих состояний новые видео;
5. проверять результат повторным прогоном через ту же модель мозга;
6. в будущем заменить искусственный источник состояний на более глубокую внутреннюю активность модели мозга.

Ключевая долгосрочная идея:

```text
internal brain activity
→ neural state
→ video generator
→ generated video
```

В строгой формулировке:

> Генерация визуального содержания, совместимого с внутренним нейронным состоянием динамической модели мозга.

---

# 1. Текущий training / data path

Сегодня система начинается с видео:

```text
video
  ↓
frozen MaleCNS-derived visual model
  ↓
T4/T5 activity
  ↓
────────────────────────────────────────────────────────
│                                                      │
│                                                      │
↓                                                      ↓
Brain-State Model                                  13B SiT
state → latent z_brain                           state + z_video
latent → state                                       ↓
│                                                   video
│
learn distribution
p(z_brain), p(state)
```

Здесь две разные задачи:

- модель мозга превращает видео в neural state;
- 13B превращает neural state обратно в видео.

Следующий этап добавляет третий компонент:

- генератор самих neural states.

---

# 2. Текущий brain encoder

Сейчас используется существующая MaleCNS-derived visual model.

Схема примерно:

```text
video
→ photoreceptors
→ lamina
→ medulla
→ T4/T5
```

Основа модели:

```text
MaleCNS wiring
+
FlyVis-compatible dynamics
+
transplanted FlyVis parameters
```

Текущий deep state состоит из восьми типов:

```text
T4a
T4b
T4c
T4d
T5a
T5b
T5c
T5d
```

Форма данных:

```text
40 frames
×
8 types
×
721 visual columns
```

То есть видео превращается не в произвольный learned latent, а в состояние конкретных типов нейронов connectome-constrained visual model.

---

# 3. Текущий video generator — 13B

Уже работает:

```text
T4/T5 state
+
z_video
↓
conditional SiT
↓
video
```

13B — условная flow-модель, которая создаёт 40 кадров за 20 Euler steps.

При полном достижимом T4/T5-state состояние очень сильно ограничивает возможный результат.

Поэтому:

```text
state → video
```

сейчас почти однозначен, а `z_video` влияет сравнительно мало.

Это уже показывает:

> Из состояния T4/T5 можно создавать видео, совместимое с этим состоянием.

Но состояние пока в основном берётся из существующего видео.

---

# 4. Главная проблема текущей системы

Сейчас полный pipeline фактически такой:

```text
source video
→ brain
→ T4/T5
→ 13B
→ video
```

То есть для получения neural condition сначала нужен source video.

Step 14 показал, что нельзя просто заменить его на:

```text
random numbers
→ T4/T5
```

потому что произвольные значения активности чаще всего находятся вне пространства состояний, которые frozen brain способен реально создать.

Значит системе не хватает:

```text
random seed
→ valid brain state
```

---

# 5. Текущий следующий этап — Brain-State Prior

Добавляется отдельная генеративная модель пространства T4/T5.

Первый вариант:

```text
T4/T5 state
    ↓
Brain-State Encoder
    ↓
z_brain
    ↓
Brain-State Decoder
    ↓
T4/T5 state
```

Например небольшой VAE.

Во время обучения:

```text
video
→ frozen brain
→ real T4/T5 state
→ VAE
→ reconstructed T4/T5 state
```

VAE учит компактное пространство:

```text
z_brain
```

из которого потом можно семплировать новые neural states.

После обучения source video уже не нужен:

```text
random z_brain
→ state decoder
→ NEW T4/T5 state
→ 13B
→ NEW video
```

---

# 6. Полная схема ближайшей версии

```text
               random seed
                    ↓
                z_brain
                    ↓
           Brain-State Prior
            VAE / latent flow
                    ↓
          generated T4/T5 state
                    ↓
             ┌─────────────┐
             │   13B SiT   │ ← z_video
             └──────┬──────┘
                    ↓
                new video
                    ↓
              frozen brain
                    ↓
            resulting T4/T5
                    ↓
             round-trip loss
```

Главный критерий успеха:

```text
generated state ≈ brain(generated video)
```

Видео должно не просто выглядеть правдоподобно.

Оно должно действительно вызывать в brain model примерно то neural state, из которого было создано.

---

# 7. Два разных latent-space

После добавления Brain-State Prior появляются два независимых источника случайности.

## `z_brain`

Определяет:

> Какое состояние мозга создать.

```text
z_brain
→ neural structure
→ motion / optic-flow / state identity
```

## `z_video`

Определяет:

> Как именно визуально реализовать это состояние.

```text
z_video
→ ambiguity
→ visual details
→ конкретная реализация видео
```

На текущем полном T4/T5 conditioning основную генеративную роль, вероятно, будет играть именно `z_brain`.

---

# 8. Если обычный VAE окажется недостаточным

Простой Gaussian VAE может хорошо реконструировать states, но плохо семплировать новые.

Тогда архитектура разделяется:

```text
T4/T5
↓
Autoencoder
↓
compact z_brain
```

А отдельно обучается генератор распределения latent:

```text
noise
↓
latent flow / diffusion
↓
z_brain
```

Полностью:

```text
noise
→ latent diffusion / flow
→ z_brain
→ state decoder
→ T4/T5
→ 13B
→ video
```

Это уже почти обычная latent generative architecture.

Но latent представляет не RGB/video features, а activity brain model.

---

# 9. Проверка нового генеративного pipeline

Главная проверка:

```text
sampled state S
→ 13B
→ generated video V
→ frozen brain
→ reconstructed state S'
```

Сравниваем:

```text
S ↔ S'
```

Если ошибка низкая, sampled state находится близко к reachable neural manifold.

Если ошибка высокая, Brain-State Prior научился создавать числа, но не валидные состояния мозга.

Кроме round-trip нужно проверять:

- diversity sampled states;
- diversity videos;
- отсутствие collapse;
- nearest-neighbour к training set;
- interpolation в `z_brain`;
- temporal coherence;
- распределение активности по T4/T5 types.

---

# 10. Расширение данных

Первый эксперимент можно сделать на текущих:

```text
~9,468 clips
```

Если идея работает, следующий шаг:

```text
100k+
video clips
→ brain
→ neural states
```

Важно расширять не просто количество видео, а **покрытие пространства достижимых neural states**.

Нужны:

```text
translation
rotation
looming
optic flow
local motion
multiple moving regions
moving objects
occlusions
dynamic textures
natural scenes
mixtures
```

Цель:

> Сделать reachable neural-state manifold более плотным и разнообразным.

После этого можно переобучать:

```text
Brain-State Prior
+
13B
```

---

# 11. Что получается после текущего этапа

До этого:

```text
source video
→ brain
→ state
→ video
```

После Brain-State Prior:

```text
random seed
→ plausible neural state
→ new video
```

То есть система впервые сможет создавать новое видео **без исходного source video**.

Это главный смысл текущего этапа.

---

# 12. Следующий большой переход — идти глубже T4/T5

T4/T5 — только текущий neural bottleneck.

В будущем архитектура мозга должна расширяться:

```text
video
↓
photoreceptors
↓
lamina
↓
medulla
↓
T4/T5
↓
LPi / VS / HS
↓
visual projection neurons
↓
central brain
↓
descending / behavioral pathways
```

На каждом уровне можно снимать:

```text
state_X
```

и строить:

```text
state_X → video generator
```

---

# 13. Multi-level brain representation

В будущем необязательно использовать только один neural layer.

Можно собирать multi-level conditioning:

```text
early visual state
+
T4/T5
+
deep visual state
+
central brain state
↓
video generator
```

Разные уровни потенциально несут разную информацию.

Примерная гипотеза:

```text
early visual layers
→ brightness / contrast / local structure

T4/T5
→ local motion / direction

LPi / VS / HS
→ optic flow / global motion

visual projection neurons
→ selected visual information

central brain
→ heading / behavioral context / internal state
```

Это будущая гипотеза, а не уже доказанный результат.

---

# 14. Переход от FlyVis-based модели к более глубокому MaleCNS

Текущий FlyVis-based model хорошо работает как откалиброванный visual baseline, но ограничивает глубину.

Для более глубоких уровней возможен переход к:

```text
MaleCNS connectome
+
generic Shiu-style LIF dynamics
```

или позже:

```text
MaleCNS connectome
+
собственная обученная dynamical model
```

Тогда можно симулировать гораздо большую часть CNS:

```text
whole / larger MaleCNS
→ deeper neural activity
```

FlyVis-версию при этом можно сохранить как:

```text
calibrated visual reference
```

а LIF / learned MaleCNS использовать как:

```text
deeper anatomical model
```

---

# 15. Главный переход всей архитектуры

Проект проходит три фундаментальных стадии.

## Стадия 1

```text
VIDEO
→ brain
→ neural state
→ video
```

Это уже работает.

## Стадия 2

```text
RANDOM LATENT
→ neural state
→ video
```

Это текущий следующий этап.

## Стадия 3

```text
INTERNAL BRAIN ACTIVITY
→ neural state
→ video
```

Это долгосрочная цель.

---

# 16. Конечная архитектура

В наиболее полной форме:

```text
                    ┌───────────────────────────┐
                    │   Dynamic MaleCNS model  │
                    │                           │
external video ────→│ sensory visual pathway   │
                    │                           │
internal dynamics ─→│ central/recurrent brain  │
                    └─────────────┬─────────────┘
                                  ↓
                         multi-level brain state
                                  ↓
                  neural representation / latent
                                  ↓
                         conditional video model
                                  ↓
                           generated video
                                  ↓
                         MaleCNS model again
                                  ↓
                       neural compatibility check
```

State может быть:

```text
T4/T5 only
```

или:

```text
T4/T5
+ deeper optic-lobe activity
+ visual projection neurons
+ central-brain state
```

---

# 17. Эволюция проекта в одной линии

```text
ЭТАП 1 — уже есть

video
→ T4/T5
→ inversion
→ video
```

```text
ЭТАП 2 — уже есть

video
→ T4/T5
→ learned 13B
→ video
```

```text
ЭТАП 3 — сейчас

noise
→ Brain-State Prior
→ NEW T4/T5
→ 13B
→ NEW video
```

```text
ЭТАП 4

больше данных
→ richer reachable neural-state manifold
→ более разнообразные новые видео
```

```text
ЭТАП 5

video
→ deeper MaleCNS layers
→ deeper states
→ video
```

```text
ЭТАП 6

multi-level brain conditioning
→ richer neural representation
→ video
```

```text
ЭТАП 7

internal / spontaneous brain dynamics
→ state без внешнего video
→ generator
→ video
```

Финальная идея:

```text
внутреннее состояние модели мозга
→ визуальное содержание,
совместимое с этим состоянием
```

---

# 18. Роль текущего Brain-State Prior в общей картине

Brain-State Prior — не конечная цель.

Это промежуточный механизм, который должен доказать:

> Neural state можно использовать как полноценное генеративное пространство, а не только как representation, снятое с уже существующего видео.

Сейчас:

```text
random noise
→ Brain-State Prior
→ neural state
→ video
```

В будущем:

```text
internal brain dynamics
→ neural state
→ video
```

То есть Brain-State Prior временно играет роль искусственного источника внутренних состояний.

Позже этот источник можно заменить самой динамикой более глубокой модели мозга.

---

# 19. Почему это важно для долгосрочной идеи

Если сразу перейти к deeper brain state, но не иметь рабочего механизма:

```text
state → generated video
```

то будет непонятно, проблема находится:

- в brain dynamics;
- в state representation;
- в генераторе;
- в данных;
- в sampling.

Поэтому текущий этап специально разделяет задачи.

Сначала:

```text
научиться создавать новые valid states
→ научиться создавать из них новые видео
```

И только потом:

```text
заменить synthetic state source
на internal brain source
```

---

# 20. Текущая цель

Текущая цель проекта максимально конкретная:

> **Научиться создавать новые видео без исходного видео, используя сгенерированное, но brain-compatible T4/T5 состояние.**

Pipeline:

```text
random seed
→ learned brain-state distribution
→ new T4/T5 state
→ 13B
→ new video
→ frozen brain
→ neural compatibility check
```

После того как эта схема станет устойчивой, тот же интерфейс можно переносить:

```text
T4/T5
→ deeper visual layers
→ central brain
→ internal brain states
```

---

# Итог

Сегодня проект уже умеет:

```text
T4/T5 state
→ video
```

Следующий этап должен добавить:

```text
random seed
→ plausible T4/T5 state
```

После объединения получится:

```text
random seed
→ plausible neural state
→ new video
→ brain verification
```

А долгосрочно:

```text
internal brain activity
→ deeper / multi-level neural state
→ generated video
```

То есть текущий Brain-State Prior — это мост между сегодняшним генератором и будущей системой визуализации внутренних состояний более глубокой модели мозга.

**Сейчас главная цель — научиться создавать новые видео.**
