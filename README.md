# STS CV Test Assignment (`sts-cv-test`)

Пайплайн подготовки данных и запуска экспериментов детекции объектов на базе Ultralytics YOLO с логированием в TensorBoard (`SummaryWriter`) для мини-выборки `test_data/traffic_v5_mini/`.

## Что сделано

1. Аудит и правка разметки. Исходные файлы в `test_data/traffic_v5_mini/` не изменены. Валидатор [`src/validate_labels.py`](src/validate_labels.py) нашёл в сырой разметке 16 ошибок: 10 отсутствующих `.txt` (`missing_label`), 2 бокса с несуществующим `class_id=11` (`unknown_class`) и 4 подозрительных выхода за границы кадра больше чем на 2% (`bbox_suspicious`).
   - Вручную проверены и размечены все 10 кадров без `.txt` (включая класс `bicycle` и отсев ложных срабатываний на рекламных щитах), оба бокса `class_id=11` перенесены в класс `10: pedestrian`, исправлены смещённые на ~150 px боксы и класс `police` на кадре `16-05-35`, поправлены `fire_truck` и ошибочный `ambulance` на `11-38-52`, а также добавлены пропущенные объекты.
   - Остальные 20 кадров без ошибок валидатора прошли проверку предобученной `yolo11m` (`conf >= 0.45`, статус `detector_checked_unchanged`).
   - Итог: 811 валидных боксов по всем 11 классам, `0 errors, 0 warnings`. Аннотации и построчный diff лежат в [`data/curated_labels/`](data/curated_labels/) и [`data/annotation_changelog.json`](data/annotation_changelog.json).
2. Сплит без утечки сцен. Union-Find группирует кадры по трём признакам: `dHash <= 12`, шаблонам сессий в именах файлов и ручным группам в [`data/scene_groups.yaml`](data/scene_groups.yaml), который связывает дневной и ночной кадры одной камеры и смазанные пары. Кластеры больше половины целевого размера `val` или `test` остаются в `train`, поэтому `val` (6 кадров) и `test` (6 кадров) собраны из независимых сцен, а `train` получает 28 кадров.
3. Экспорт и контроль целостности. 13 PNG с альфа-каналом конвертированы из `RGBA` в `RGB`, остальные `RGB`-файлы скопированы побайтово. Перед очисткой выходного каталога проверяется, что это директория датасета; сгенерированный [`data/processed/dataset.yaml`](data/processed/dataset.yaml) не содержит абсолютных путей, а хеши файлов лежат в [`reports/manifest.sha256`](reports/manifest.sha256).
4. Эксперимент YOLO и логи TensorBoard. В [`configs/yolo.yaml`](configs/yolo.yaml) явно заданы `optimizer: AdamW`, `lr0: 0.001` и `nbs: 8` (совпадает с `batch: 8`, чтобы оптимизатор и EMA обновлялись на каждой эпохе). Эпохи обучения `1..epochs` и финальная оценка `best.pt` на `val` и `test` разнесены по разным тегам (`metrics/*`, `final/*`, `test/*`). Базовые веса кэшируются в `weights/`, а флаг `--require-clean` блокирует запуск при незакоммиченных правках.

## Структура репозитория

```text
sts-cv-test/
├── README.md
├── pyproject.toml               # Настройки pytest и версия Python (>=3.11)
├── requirements.txt             # Зафиксированные версии прямых зависимостей
├── .gitignore
├── test_data/traffic_v5_mini/   # Исходная выборка (в Git, без изменений)
├── configs/yolo.yaml            # Конфиг эксперимента YOLO
├── src/
│   ├── __init__.py
│   ├── utils.py                 # Seed, выбор устройства, git-метаданные, SHA-256, YAML
│   ├── viz.py                   # Отрисовка боксов и транслитерация подписей
│   ├── validate_labels.py       # Валидатор изображений и YOLO-разметки
│   ├── prepare_dataset.py       # Групповой сплит, RGB-экспорт, отчёт, manifest
│   ├── tensorboard_logger.py    # Обёртка над SummaryWriter и коллажи GT vs Pred
│   └── run_experiment.py        # CLI-раннер (--dry-run, обучение, оценка val и test)
├── data/
│   ├── README.md                # Описание каталога и правила классов
│   ├── curated_labels/          # Выверенные .txt для 40 кадров (в Git)
│   ├── annotation_changelog.json# Построчный diff аннотаций (в Git)
│   ├── scene_groups.yaml        # Ручные группы сцен (в Git)
│   ├── raw/.gitkeep
│   └── processed/.gitkeep       # Генерируется prepare_dataset.py (в .gitignore)
├── reports/
│   ├── dataset_report.md        # Отчёт по аудиту и сплиту
│   ├── manifest.sha256          # Контрольные суммы data/processed/
│   └── samples/                 # Визуализации боксов (до/после и сплиты)
├── runs/.gitkeep                # Логи TensorBoard и чекпоинты (в .gitignore)
└── tests/
    ├── test_labels.py
    ├── test_dataset_split.py
    └── test_tensorboard.py
```

## Установка из чистого окружения

Нужен Python >= 3.11 (проверено на `Python 3.12.13`, `PyTorch 2.15.0`, `Ultralytics 8.4.219`).

Через стандартный `venv` и `pip`:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Через `uv`:

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Команды запуска

Все команды выполняются из корня `sts-cv-test/`.

Проверка исходной и исправленной разметки:

```bash
# Аудит исходной разметки (находит 16 ошибок и завершается с кодом 1)
python -m src.validate_labels \
  --images test_data/traffic_v5_mini/images \
  --labels test_data/traffic_v5_mini/labels \
  --classes test_data/traffic_v5_mini/classes.txt

# Проверка выверенной разметки (0 ошибок, код 0)
python -m src.validate_labels \
  --images test_data/traffic_v5_mini/images \
  --labels data/curated_labels \
  --classes test_data/traffic_v5_mini/classes.txt
```

Подготовка датасета и проверка манифеста. Скрипт сам подключает [`data/curated_labels/`](data/curated_labels/) и [`data/scene_groups.yaml`](data/scene_groups.yaml), когда имена файлов во входной папке совпадают с курированной выборкой (для работы только по сырой разметке есть флаг `--raw-only`):

```bash
python -m src.prepare_dataset \
  --input test_data/traffic_v5_mini \
  --output data/processed \
  --seed 42

python -m src.prepare_dataset \
  --input test_data/traffic_v5_mini \
  --output data/processed \
  --seed 42 \
  --verify-manifest

shasum -a 256 -c reports/manifest.sha256
```

Запуск эксперимента YOLO. В режиме `--dry-run` модель обучается 1 эпоху при `imgsz=320`, проверяет сохранение чекпоинтов, оценку на `val` и `test` и запись всех тегов TensorBoard. Обычный запуск использует [`configs/yolo.yaml`](configs/yolo.yaml) (5 эпох, `imgsz=640`, `AdamW`, `lr0=0.001`, `nbs=8`, выбор устройства `cuda`, `mps` или `cpu`; для строгого контроля версии кода добавьте `--require-clean`):

```bash
python -m src.run_experiment \
  --config configs/yolo.yaml \
  --data data/processed/dataset.yaml \
  --dry-run

python -m src.run_experiment \
  --config configs/yolo.yaml \
  --data data/processed/dataset.yaml
```

Каждый запуск пишет данные в отдельную папку `runs/<YYYYMMDD_HHMMSS>_<experiment_name>[_dryrun]/`: события TensorBoard в `tb/`, чекпоинты `best.pt` и `last.pt` в `yolo/train/weights/` и сводку параметров с метриками и версией кода в `run_meta.json`.

Просмотр логов в TensorBoard (после запуска откройте `http://localhost:6006`):

```bash
tensorboard --logdir runs
```

Запуск тестов:

```bash
pytest -v
```

## Состав логов TensorBoard

| Вкладка TensorBoard | Тег | Что логируется |
|---|---|---|
| SCALARS | `loss/train_box`, `loss/train_cls`, `loss/train_dfl` | Потери на обучении по эпохам (`1..epochs`) |
| SCALARS | `loss/val_box`, `loss/val_cls`, `loss/val_dfl` | Потери на валидации по эпохам (`1..epochs`) |
| SCALARS | `metrics/precision`, `metrics/recall`, `metrics/mAP50`, `metrics/mAP50_95` | Метрики на `val` по эпохам обучения |
| SCALARS | `final/precision`, `final/recall`, `final/mAP50`, `final/mAP50_95` | Итоговые метрики `best.pt` на `val` |
| SCALARS | `test/precision`, `test/recall`, `test/mAP50`, `test/mAP50_95` | Оценка `best.pt` на отложенном сплите `test` |
| SCALARS | `lr/pg0` | Learning rate оптимизатора по эпохам |
| IMAGES | `eval/confusion_matrix` | Матрица ошибок (11 классов и `background`) для `best.pt` |
| IMAGES | `predictions/val_sample_0..5` | Коллажи Ground Truth (слева) и Predictions (справа) |
| TEXT | `run/config`, `run/environment`, `run/dataset` | Конфиг, режим (`dry_run` или `train`), `git` commit/branch/dirty, версии пакетов, устройство и SHA-256 файла `manifest.sha256` |
| HPARAMS | `hparam/precision`, `hparam/recall`, `hparam/mAP50`, `hparam/mAP50_95`, `hparam/test_mAP50` | Сводная таблица гиперпараметров и метрик `val`/`test` |

## Результаты короткого прогона и ограничения

Выборка из 40 кадров (28 train, 6 val, 6 test) за 5 эпох проверяет корректность пайплайна, а не сходимость до высокого `mAP`. Из-за малого объёма и дисбаланса классов (единичные экземпляры `bicycle`, `lorry`, `trailer`, `fire_truck`) метрики на `val` и `test` сильно зависят от разбиения, а `val/cls_loss` может расти уже на первых эпохах из-за переобучения под `train`. Точные значения каждого прогона сохраняются в `runs/<run_id>/run_meta.json`.

Для улучшения качества на этом наборе нужно дособрать редкие классы, увеличить число эпох с cosine LR и ранней остановкой, заморозить backbone (`freeze`) и оценивать качество кросс-валидацией по группам сцен.

## Отчёт по данным и политика `runs/` в Git

Статистика аудита, кластеры утечки и распределение классов собраны в [`reports/dataset_report.md`](reports/dataset_report.md), примеры визуализаций до/после и сплитов лежат в [`reports/samples/`](reports/samples/).

По регламенту Git файлы `data/processed/*`, веса (`*.pt`, `weights/`) и логи запусков (`runs/*`) исключены из репозитория (каждый каталог `tb/` со встроенными изображениями предсказаний весит около 3.3 МБ). Логи `--dry-run` и 5-эпохового обучения с чистого коммита (`is_dirty=false`) сохранены локально в `runs/` и при необходимости передаются отдельным архивом.
