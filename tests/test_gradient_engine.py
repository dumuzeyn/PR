import numpy as np

from photoredactor.core import Document, GradientEngine, Layer, apply_gradient


BLACK = (0, 0, 0, 255)
WHITE = (255, 255, 255, 255)


def test_linear_gradient_uses_horizontal_drag_vector() -> None:
    image = GradientEngine.render(5, 3, (0, 1), (4, 1), [(0.0, BLACK), (1.0, WHITE)], "linear")
    assert image[1, 0, 0] == 0
    assert 120 <= image[1, 2, 0] <= 135
    assert image[1, 4, 0] == 255


def test_linear_gradient_uses_vertical_drag_vector() -> None:
    image = GradientEngine.render(3, 5, (1, 0), (1, 4), [(0.0, BLACK), (1.0, WHITE)], "linear")
    assert image[0, 1, 0] == 0
    assert 120 <= image[2, 1, 0] <= 135
    assert image[4, 1, 0] == 255
    assert np.array_equal(image[:, 0], image[:, 2])


def test_linear_gradient_uses_diagonal_projection() -> None:
    image = GradientEngine.render(5, 5, (0, 0), (4, 4), [(0.0, BLACK), (1.0, WHITE)], "linear")
    assert image[0, 0, 0] == 0
    assert 120 <= image[2, 2, 0] <= 135
    assert image[4, 4, 0] == 255
    assert image[0, 4, 0] < image[4, 4, 0]


def test_radial_gradient_uses_drag_length_as_radius() -> None:
    image = GradientEngine.render(7, 7, (3, 3), (6, 3), [(0.0, BLACK), (1.0, WHITE)], "radial")
    assert image[3, 3, 0] == 0
    assert image[3, 6, 0] == 255
    assert image[0, 3, 0] == 255


def test_gradient_stops_and_selection_coverage() -> None:
    layer = Layer("Gradient", np.zeros((3, 5, 4), dtype=np.uint8))
    layer.pixels[:, :, 3] = 255
    selection = np.zeros((3, 5), dtype=np.uint8)
    selection[:, 1:4] = 255
    stops = [(0.0, BLACK), (0.5, (255, 0, 0, 255)), (1.0, WHITE)]
    apply_gradient(layer, (0, 1, 4, 1), BLACK, WHITE, selection, "linear", stops)
    assert tuple(layer.pixels[1, 0]) == BLACK
    assert layer.pixels[1, 2, 0] == 255
    assert layer.pixels[1, 2, 1] == 0
    assert tuple(layer.pixels[1, 4]) == BLACK


def test_gradient_and_texture_shape_objects_stay_editable() -> None:
    document = Document.new(64, 48, (0, 0, 0, 0))
    gradient = {
        "type": "linear",
        "start": [8, 8],
        "end": [50, 30],
        "stops": [{"position": 0.0, "color": list(BLACK)}, {"position": 1.0, "color": list(WHITE)}],
    }
    gradient_layer = document.add_shape_layer("ellipse", (8, 8, 50, 36), BLACK, WHITE, 1, gradient=gradient)
    assert gradient_layer.shape_data["gradient"]["type"] == "linear"
    assert gradient_layer.pixels[22, 28, 3] > 0
    assert gradient_layer.pixels[2, 2, 3] == 0
    texture = {"type": "checker", "size": 4, "color_a": [255, 0, 0, 255], "color_b": [0, 0, 255, 255]}
    texture_layer = document.add_shape_layer("rectangle", (4, 4, 24, 24), BLACK, None, 0, texture=texture)
    assert texture_layer.shape_data["texture"]["type"] == "checker"
    colors = np.unique(texture_layer.pixels[6:20, 6:20, :3].reshape(-1, 3), axis=0)
    assert len(colors) == 2
