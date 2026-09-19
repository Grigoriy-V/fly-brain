# Step 16 — источник внутреннего brain-state без видео

Дата: 2026-09-19/20  
Статус: пересмотренный дизайн после 13B и анализа текущего MaleCNS/FlyVis-compatible model zero.

## 1. Цель

Цель шага 16 — **разорвать зависимость от цепочки `video → brain state`** и получить T4/T5-состояния, которые возникают внутри модели без исходного видео.

13B уже умеет превращать T4/T5-state в видео:

```text
brain state + z → 13B SiT → generated video
```

Поэтому задача шага 16 — не строить ещё один генератор, а получить **источник brain-state внутри модели**.

Итоговый контур:

```text
internal brain dynamics
        ↓
T4/T5 state
        ↓
13B generator
        ↓
new video
        ↓
frozen brain
        ↓
state'
```

Главная проверка — насколько `state'` соответствует исходному внутреннему state.

Важно: generated video здесь не считается буквальным субъективным «сном» мухи. Корректная формулировка — **видео, совместимое с внутренне возникшим состоянием модели мозга**.

---

## 2. Что уже известно

Текущий model zero — MaleCNS-derived FlyVis-compatible optic-lobe model. Он не является полной MaleCNS-моделью и не содержит central brain.

13B показал, что T4/T5-state можно использовать как conditioning генеративной video-модели. На обычных reachable states результат близок к deterministic decoder, а при partial conditioning seed начинает играть роль.

Пункт 11 уже показал, что:
- белый шум в нейроны даёт слабую рябь;
- без визуального входа сеть быстро затухает к стационарному состоянию;
- собственной богатой автономной динамики у текущего model zero практически нет.

Новый аудит connectome показывает важное ограничение текущей сети:
- около 25% synaptic input на 14 ladder types приходит от типов, которых нет в model zero;
- около 91% output T4/T5 уходит в типы, которых model zero не содержит;
- в частности, отсутствуют важные LPi/Dm/Pm/TmY/Y-pathways и часть recurrent/downstream loops.

Это означает, что текущая модель обрывается примерно в той точке, где начинается значимая downstream/recurrent visual circuitry.

---

## 3. Важное уточнение

**Добавление recurrent loops само по себе не гарантирует автономную активность.**

Стабильная recurrent network без внешнего или внутреннего drive может всё равно:

```text
activity → decay → fixed point
```

Поэтому нельзя считать, что восстановление LPi/Dm/Pm автоматически создаст «сон».

Также нельзя интерпретировать осцилляции или persistent activity как свойство MaleCNS, если они появляются только из-за произвольно выбранных `bias`, `tau` или gains новых типов.

Connectome определяет wiring, но не полностью задаёт dynamics.

---

## 4. План

### 16.1 — artificial internal drive baseline

Цель: проверить весь pipeline на brain-state, который **не получен из видео**, но ещё искусственно создаётся внутри модели.

Не использовать белый независимый шум как в пункте 11. Вместо этого подавать пространственно-временно структурированный drive:

- correlated Gaussian field;
- temporal correlation `τ ≈ 50–200 ms`;
- spatial scale 2–5 columns;
- amplitude относительно sd активности на обычных клипах.

Точки введения:
- photoreceptors;
- L1/L2/L3;
- T4/T5 напрямую.

Pipeline:

```text
structured internal drive
        ↓
model zero
        ↓
T4/T5 state
        ↓
13B
        ↓
generated video
        ↓
round-trip
```

Контроли:
- shuffled cells;
- shuffled time;
- white-noise drive;
- same drive with different seeds.

Это **не autonomous brain activity** и не следует называть «мозг сам породил состояние». Это baseline искусственного internal drive.

Ожидаемая стоимость: только inference, порядка центов.

---

### 16.2 — восстановить недостающую optic-lobe recurrence

Цель: расширить model zero так, чтобы T4/T5 больше не были почти terminal layer.

Добавлять типы, которые замыкают значимые loops вокруг T4/T5 и ladder:

- LPi;
- Dm12 / Dm4 / Dm1 / Dm9 / Dm10;
- Pm2b;
- TmY16 / TmY19a;
- Y11–Y13;
- при необходимости следующие downstream visual types, если они критичны для замыкания петли.

Синапсы и topology — из MaleCNS connectome.

### Параметры новых типов

Нельзя просто назначить arbitrary параметры и затем считать появившуюся динамику биологическим результатом.

Нужно:

- sign брать из реальных neurotransmitter annotations / MaleCNS data, а не из общего предположения по классу;
- `tau`, `bias`, gain задавать как **диапазон разумных значений**, а не одну произвольную точку;
- делать sensitivity sweep;
- считать эффект надёжным только если он сохраняется в широком диапазоне параметров.

Перед генерацией видео обязательны два gate:

#### Gate A — стабильность

Длинный прогон без visual input:

- затухание;
- fixed point;
- устойчивые oscillations;
- waves;
- divergence.

Измерять:
- lifetime activity;
- variance;
- spectrum;
- spatial autocorrelation;
- T4/T5 activity over time.

#### Gate B — не сломали зрительную модель

Повторить текущие validation tests:

- DSI;
- flash polarity;
- direction response;
- stability;
- reference clips.

Если recurrence ломает базовую visual function, такая конфигурация не используется дальше.

После этого повторить 16.1 на расширенной сети.

Главное сравнение:

```text
same internal drive
loops OFF
vs
loops ON
```

Дополнительный контроль:

```text
real topology
vs
degree/sign-preserving shuffled connectivity
```

Если structured/persistent activity возникает только при настоящей topology и сохраняется при разумном sweep параметров — это уже сильный результат.

---

### 16.3 — настоящий автономный internal source

Если 16.2 не создаёт богатую самоподдерживающуюся активность, следующий уровень — internal drive из более глубокого мозга.

Кандидаты:

- central complex;
- EPG / helicon pathways;
- R5 / dFSB sleep-related circuitry;
- other central-brain recurrent populations with projections back toward visual pathways.

Pipeline:

```text
central brain spontaneous dynamics
        ↓
visual / feedback pathways
        ↓
T4/T5 or downstream visual state
        ↓
13B
        ↓
generated video
```

Это уже наиболее близко к исходной идее «What Does a Fly Dream Of?».

Требует расширения модели за optic lobe и поэтому существенно крупнее 16.1/16.2.

---

## 5. Что считать результатом

Для любого state без исходного видео нельзя оценивать результат по «красивости».

Основная метрика:

```text
internal state
→ 13B
→ generated video
→ frozen brain
→ reconstructed state
```

Считать:

- round-trip error;
- error per T4/T5 type;
- stability across generator seeds;
- distance to training-state manifold;
- comparison with shuffled-state control;
- comparison with Adam inversion reference.

Красивая картинка при плохом round-trip = prior hallucination.

---

## 6. Что является настоящим прогрессом

Уровни результата:

### A. Synthetic internal-state generation
State задан внутри модели искусственным drive, но не происходит из видео.

Это уже разрывает `video → state → video`, но ещё не является автономной brain dynamics.

### B. Recurrent optic-lobe dynamics
Real connectome loops меняют dynamics и создают persistent / structured patterns, которых нет без loops и в shuffled controls.

Это уже свойство network topology, если эффект стабилен по параметрам.

### C. Autonomous brain-generated state
State возникает из central-brain / recurrent internal dynamics без visual stimulus.

Это наиболее сильный вариант для dream-ветки.

---

## 7. Главный риск

Самый опасный false positive:

```text
неизвестные tau/bias/gain
        ↓
искусственно созданные oscillations
        ↓
13B рисует красивое видео
        ↓
ошибочный вывод «мозг сгенерировал сон»
```

Поэтому любая автономная динамика должна проходить:

- parameter sensitivity;
- real-vs-shuffled topology control;
- loops ON/OFF control;
- round-trip through the same frozen brain;
- comparison with white/structured-noise baselines.

---

## 8. Практический порядок

1. **16.1** — дешёвый non-video-state baseline.
2. **16.2a** — добавить недостающие recurrent/downstream optic-lobe types.
3. **16.2b** — stability + visual validation gates.
4. **16.2c** — loops ON/OFF + shuffled topology + structured drive.
5. Если автономной dynamics нет — **16.3 central brain source**.
6. Любой полученный internal state → 13B → round-trip validation.

---

## 9. Главный вывод

13B уже решает задачу:

```text
brain state → generated video
```

Шаг 16 должен решить другую:

```text
откуда взять brain state без видео?
```

Самое важное открытие текущего аудита — model zero сильно обрывает circuitry после T4/T5, поэтому расширение recurrent/downstream visual network оправдано не только ради «сна», но и для более полного моделирования visual dynamics вообще.
