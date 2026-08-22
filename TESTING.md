# Тестирование UZYRO

Основная команда полной проверки:

```powershell
.\run_tests.ps1
```

Она последовательно запускает core, security и UI correctness suites. UI-сценарии получают отдельные Tk-процессы и выполняются до четырёх одновременно: это исключает утечки состояния Tcl между окнами, но не возвращает прежнюю длительность запуска. Реальное оборудование и benchmark-like проверки не входят в default: отсутствие GPU, CMYK-профиля или локальной AI-модели не считается ошибкой UZYRO.

## Режимы

```powershell
.\run_tests.ps1 -Mode Smoke
.\run_tests.ps1 -Mode Fast
.\run_tests.ps1 -Mode Core
.\run_tests.ps1 -Mode Ui
.\run_tests.ps1 -Mode Security
.\run_tests.ps1 -Mode Performance
.\run_tests.ps1 -Mode Core -Coverage
```

`Smoke` ловит невозможность импортировать и инициализировать базовые части приложения. `Fast` исключает UI, performance, slow и hardware. `Performance` запускается отдельно и использует только широкие пороги катастрофической регрессии.

## Markers

- `ui` — создаёт Tk root или полное приложение.
- `ui_isolated` — нативная Windows/Tcl-граница, которой нужен отдельный процесс.
- `security` — проверяет trust boundaries и повреждённые данные.
- `performance` — измеряет время или память.
- `slow` — детерминированный, но дорогой сценарий.
- `hardware` — требует реального GPU, профиля печати или OS-возможности.
- `integration` — проходит через несколько production-модулей или внешнюю границу.
- `smoke` — минимальная проверка работоспособности.

Необозначенные тесты считаются обычными unit/core checks. Неизвестный marker является ошибкой collection.

## Изоляция

Autouse fixtures перенаправляют `%LOCALAPPDATA%` в `tmp_path`, запрещают внешние сетевые соединения и оставляют доступным только loopback для локальных protocol tests. UI cleanup уничтожает оставшийся Tk root. Реальные модели не скачиваются, subprocess и plugin boundaries имеют ограниченные timeout.

## Coverage

Branch coverage измеряется для `uzyro`. Regression floor равен `37%` для быстрого core-suite и отражает фактический baseline, а не искусственную цель. Полный функциональный замер складывается из core, security и performance; UI-покрытие не требуется для прохождения порога. Отчёт показывает пропущенные строки; повышать floor следует только после подтверждённого роста meaningful coverage.
