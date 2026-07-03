# PhotoRedactor Roadmap

## Implemented in this foundation

- Local documents: new, open, save, save as, export.
- Document geometry: resize image, resize canvas, rotate, flip, crop.
- Navigation: zoom, fit to screen, pan with scrollbars.
- Raster layers: create, duplicate, delete, reorder, visibility, opacity, merge, flatten.
- Raster selections: rectangular selection, select all, invert, feather, grow, shrink.
- Raster layer masks: reveal all, hide all, from selection, invert, toggle, apply, delete.
- Blend modes: Normal, Multiply, Screen, Overlay, Soft Light, Darken, Lighten, Difference, Color, Luminosity.
- Tools: move, brush, eraser, fill, gradient, text, rectangular selection, crop.
- Adjustments: brightness, contrast, saturation, levels, curves, grayscale, invert.
- Filters: Gaussian blur, sharpen, noise.
- History: undo and redo snapshots.
- Automation: simple batch resize/convert.
- Performance baseline: numpy arrays, OpenCV filters, cached composite preview.
- Export: PNG, JPEG, WebP, BMP, TIFF, plus `.prdx` project files.

## Next milestones toward the full list

1. Selection engine: lasso, polygonal lasso, magic wand, color range, saved alpha selections.
2. Editable masks: direct mask painting, density/feather controls and mask thumbnails.
3. Non-destructive adjustments: adjustment layers and editable filter stack.
4. Free Transform: scaling, rotation, numeric coordinates and transform handles.
5. Text engine: editable text layers instead of rasterized text.
6. Shapes and paths: vector layer model with Bezier editing.
7. Retouching: clone stamp, healing brush, patch, red-eye.
8. Smart objects: embedded image payloads, re-editable transforms, linked file updates.
9. RAW/color management: ICC profiles, 16/32-bit channels, Lab/CMYK workflows.
10. GPU/tiled rendering: tile cache, dirty-region recomposition, scratch disk, background save.
11. Plugin API and actions: command recording, batch execution, external filters.
12. Advanced AI/generative tools: subject/sky selection, content-aware fill, generative expand.
