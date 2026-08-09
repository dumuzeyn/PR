import cv2


def register(api):
    def grayscale(pixels, params):
        result = pixels.copy()
        gray = cv2.cvtColor(result[:, :, :3], cv2.COLOR_RGB2GRAY)
        amount = max(0.0, min(1.0, float(params.get("amount", 1.0))))
        converted = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        result[:, :, :3] = (
            result[:, :, :3].astype("float32") * (1.0 - amount)
            + converted.astype("float32") * amount
        ).clip(0, 255).astype("uint8")
        return result

    api.register_filter("Пример: оттенки серого", grayscale, "Параметр amount задаёт силу от 0 до 1")
