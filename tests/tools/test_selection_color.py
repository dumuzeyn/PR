from __future__ import annotations

import cv2
import numpy as np

from uzyro.app_mixins.select_mask import SelectMaskMixin
from uzyro.app_shared import SELECT_MASK_PREVIEW_MODES
from uzyro.core import Document, Layer, smart_radius_refine
from uzyro.segmentation import OpenCvCpuSegmentationBackend, SegmentationService
from uzyro.selection_color import combine_sample_masks


def test_color_range_combines_positive_and_negative_samples_with_soft_alpha() -> None:
    pixels = np.zeros((40, 120, 4), dtype=np.uint8)
    pixels[:, :, 3] = 255
    pixels[:, :, 0] = np.linspace(20, 230, 120, dtype=np.uint8)[None, :]
    pixels[:, :, 1] = 65
    pixels[:, :, 2] = 55
    included = [tuple(int(value) for value in pixels[20, 35]), tuple(int(value) for value in pixels[20, 85])]
    excluded = [tuple(int(value) for value in pixels[20, 60])]
    mask = combine_sample_masks(pixels, included, excluded, 20, True)
    assert mask[20, 35] > 220
    assert mask[20, 85] > 220
    assert mask[20, 60] < 20
    assert np.count_nonzero((mask > 0) & (mask < 255)) > 100


def test_magic_wand_can_sample_composited_layers() -> None:
    document = Document.new(80, 50, (245, 245, 245, 255))
    overlay = np.zeros((50, 80, 4), dtype=np.uint8)
    overlay[15:35, 28:52] = (210, 30, 25, 255)
    document.add_layer("Overlay", overlay)
    document.magic_wand_selection(document.layer, 4, 4, 4, contiguous=False, sample_all_layers=True)
    assert document.selection_mask is not None
    assert document.selection_mask[4, 4] > 220
    assert document.selection_mask[25, 40] == 0


def test_magic_wand_antialias_switch_controls_fractional_values() -> None:
    document = Document.new(100, 30, (0, 0, 0, 255))
    gradient = np.linspace(80, 160, 100, dtype=np.uint8)
    document.layer.pixels[:, :, :3] = gradient[None, :, None]
    document.magic_wand_selection(document.layer, 45, 15, 32, contiguous=False, antialias=True)
    assert np.any((document.selection_mask > 0) & (document.selection_mask < 255))
    document.magic_wand_selection(document.layer, 45, 15, 32, contiguous=False, antialias=False)
    assert not np.any((document.selection_mask > 0) & (document.selection_mask < 255))


def test_cpu_object_segmentation_respects_roi_and_keeps_soft_edge() -> None:
    image = np.full((180, 260, 4), (55, 115, 175, 255), dtype=np.uint8)
    cv2.ellipse(image, (80, 100), (40, 55), 0, 0, 360, (195, 55, 40, 255), -1)
    cv2.rectangle(image, (175, 55), (235, 145), (195, 55, 40, 255), -1)
    backend = OpenCvCpuSegmentationBackend()
    mask = backend.select_object(image, (25, 30, 135, 165), 0.65)
    assert mask[100, 80] > 180
    assert mask[100, 205] == 0
    assert np.count_nonzero((mask > 0) & (mask < 255)) > 20


def test_segmentation_service_uses_real_cpu_fallback_without_model() -> None:
    image = np.full((80, 100, 4), (80, 130, 180, 255), dtype=np.uint8)
    cv2.circle(image, (50, 45), 22, (200, 60, 45, 255), -1)
    service = SegmentationService(primary=None, fallback=OpenCvCpuSegmentationBackend())
    result = service.select(image, "Объект", 0.6)
    assert result.backend == "OpenCV CPU"
    assert not result.fallback
    assert result.mask[45, 50] > 128


def test_smart_radius_preserves_fractional_selection_values() -> None:
    image = np.full((90, 120, 4), (30, 90, 170, 255), dtype=np.uint8)
    image[20:70, 35:85, :3] = (190, 60, 45)
    mask = np.zeros((90, 120), dtype=np.uint8)
    mask[22:68, 37:83] = 255
    mask = cv2.GaussianBlur(mask, (9, 9), 3)
    refined = smart_radius_refine(mask, image, 5, 0.8)
    assert refined[45, 60] == 255
    assert refined[5, 5] == 0
    assert np.count_nonzero((refined > 0) & (refined < 255)) > 40


def test_all_select_and_mask_preview_modes_render_rgba() -> None:
    image = np.full((50, 70, 4), (110, 150, 200, 255), dtype=np.uint8)
    mask = np.zeros((50, 70), dtype=np.uint8)
    cv2.circle(mask, (35, 25), 17, 255, -1, cv2.LINE_AA)
    for mode in SELECT_MASK_PREVIEW_MODES:
        preview = SelectMaskMixin.render_select_mask_preview(image, mask, mode, 96)
        assert preview.mode == "RGBA"
        assert preview.size == (96, 96)
        assert np.asarray(preview).std() > 0
