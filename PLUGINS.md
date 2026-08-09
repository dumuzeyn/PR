# PhotoRedactor Plugin API

PhotoRedactor loads Python plugins from `%APPDATA%\PhotoRedactor\plugins` and the repository `plugins` folder. A plugin is a `.py` file that exports `register(api)`.

```python
def register(api):
    def filter_image(pixels, params):
        # pixels is an H x W x 4 uint8 NumPy array.
        result = pixels.copy()
        result[:, :, :3] = 255 - result[:, :, :3]
        return result

    api.register_filter("My filter", filter_image, "Short description")
```

Filters must return an RGBA `uint8` array with the same dimensions. Parameters entered in the editor are passed as a JSON object. Plugin errors are isolated during discovery and shown through `Действия -> Ошибки плагинов`.

Action commands can be added with `api.register_action_command(name, callback)`. The callback receives `(document, params)` and can then be referenced by a PhotoRedactor action v2 JSON file.

External executables can be registered from Python with `registry.register_external_filter(...)`. PhotoRedactor starts the executable without a shell and passes three arguments: input PNG, output PNG and JSON parameters. A non-zero exit code, timeout or missing output is reported as a filter error.
