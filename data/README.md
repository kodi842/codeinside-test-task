# Каталог данных (`data/`)

## Структура

- `raw/`: место для полного исходного датасета (не коммитится в Git, сохраняется только `.gitkeep`).
- `curated_labels/`: выверенные YOLO-аннотации (`.txt`) для всех 40 изображений из `test_data/traffic_v5_mini/images/`. Исходные файлы в `test_data/traffic_v5_mini/` не изменяются.
- `annotation_changelog.json`: журнал правок с построчным diff (`diff.added`, `diff.removed`, `diff.modified`) относительно исходной разметки и статусом каждого кадра (`newly_labeled`, `fixed_errors`, `supplemented`, `detector_checked_unchanged`).
- `scene_groups.yaml`: ручные группы сцен (`groups:`), объединяемые через Union-Find с `dHash` и шаблонами сессий, чтобы кадры одной камеры при разном освещении или смазе не расходились по разным сплитам.
- `processed/`: сплит `train / val / test`, RGB-изображения и `dataset.yaml` (генерируется заново и не коммитится в Git).

## Соглашения о классах (`classes.txt`)

Соглашения выведены из исходной разметки `test_data/traffic_v5_mini/labels/` (`screenshot_1529.txt`, `0 - 2025-04-16..._00026790.txt`, `10_9aug202317009.txt`):

| ID | Класс | Правило разметки |
|---:|---|---|
| 0 | `bicycle` | Велосипед отдельным боксом; водитель размечается отдельным боксом `pedestrian` |
| 1 | `car` | Легковые автомобили, кроссоверы, минивэны и малые коммерческие фургоны |
| 2 | `motorcycle` | Мотоцикл или скутер отдельным боксом; водитель отдельным боксом `pedestrian` (как в `screenshot_1529.txt`) |
| 3 | `bus` | Пассажирские автобусы и маршрутные микроавтобусы (трамваи в выборке не размечаются) |
| 4 | `truck` | Среднетоннажные грузовики, фургоны с будкой и пикапы с открытым кузовом |
| 5 | `lorry` | Крупные магистральные грузовики (как в `0 - 2025-04-16..._00026790.txt`) |
| 6 | `trailer` | Автоприцепы и полуприцепы отдельно от тягача или автомобиля |
| 7 | `police` | Автомобили полиции и ДПС со спецраскраской |
| 8 | `ambulance` | Автомобили скорой помощи |
| 9 | `fire_truck` | Пожарные автомобили |
| 10 | `pedestrian` | Пешеходы, а также водители велосипедов и мотоциклов |

## Воспроизведение `data/processed/`

Из корня репозитория выполните:

```bash
python -m src.prepare_dataset \
  --input test_data/traffic_v5_mini \
  --output data/processed \
  --seed 42
```

Проверка контрольных сумм:

```bash
python -m src.prepare_dataset \
  --input test_data/traffic_v5_mini \
  --output data/processed \
  --seed 42 \
  --verify-manifest

shasum -a 256 -c reports/manifest.sha256
```
