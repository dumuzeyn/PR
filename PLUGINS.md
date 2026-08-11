# PhotoRedactor Plugin SDK v2

Плагины API v2 находятся в отдельных папках внутри `%APPDATA%\PhotoRedactor\plugins`. Каждый плагин состоит из `plugin.json` и указанного в нём Python-файла. Код выполняется в отдельном процессе с таймаутом; авария плагина не завершает редактор.

## Манифест

```json
{
  "format": "PhotoRedactor plugin v1",
  "id": "example.invert",
  "name": "Example Invert",
  "version": "1.0.0",
  "api_version": 2,
  "entrypoint": "plugin.py",
  "description": "Инвертирует изображение",
  "permissions": ["pixels"]
}
```

`id` должен быть постоянным и уникальным. `entrypoint` обязан находиться в папке плагина. Редактор не активирует новый плагин, пока пользователь не выдаст все запрошенные разрешения через `Действия -> Управление плагинами`.

Доступные разрешения:

- `pixels` - регистрация растровых фильтров;
- `document` - действия, импорт и экспорт документов;
- `filesystem.read` и `filesystem.write` - произвольное чтение и запись файлов;
- `network` - сетевые подключения;
- `process` - запуск внешних программ;
- `native` - загрузка нативных библиотек через `ctypes`.

Переданные редактором входные и выходные файлы доступны конкретной операции без выдачи произвольного доступа к файловой системе. Ограничения дополнительно контролируются audit hook внутри отдельного процесса. Это защищает основное приложение от обычных ошибок и несанкционированного использования SDK, но устанавливать следует только плагины из доверенных источников: Python-плагин не является системной виртуальной машиной.

## Фильтр

```python
def register(api):
    def invert(pixels, params):
        result = pixels.copy()
        result[:, :, :3] = 255 - result[:, :, :3]
        return result

    api.register_filter("Инверсия", invert, "Инвертирует каналы RGB")
```

Фильтр получает копию `H x W x 4 uint8` и JSON-объект параметров. Результат должен иметь тот же размер и тип. Требуется `pixels`.

## Действие документа

```python
def register(api):
    def set_opacity(document, params):
        document.layer.opacity = float(params["value"])
        document.dirty = True

    api.register_action_command("example.opacity", set_opacity)
```

Действие получает полноценный `Document` и может использоваться в Action JSON v3 и пакетной очереди. Изменённый документ передаётся обратно редактору через временный `.prdx`. Требуется `document`.

## Импорт и экспорт

```python
def register(api):
    api.register_importer("Мой формат", [".mine"], load_document)
    api.register_exporter("Мой формат", [".mine"], save_document)
```

Импортёр вызывается как `load_document(source_path, params)` и возвращает `Document`. Экспортёр вызывается как `save_document(document, target_path, params)`. Нужны `document` и соответствующее разрешение файловой системы.

## Внешний фильтр

`api.register_external_filter(name, executable, description, timeout)` запускает программу без shell и передаёт ей пути входного PNG, выходного PNG и JSON-параметры. Требуются `pixels` и `process`. Ненулевой код, таймаут и отсутствие результата считаются ошибкой.

## Совместимость

Одиночные `.py` из старого API v1 продолжают работать только как фильтры с разрешением `pixels`. Для действий, импорта, экспорта и дополнительных разрешений нужен манифест API v2.
