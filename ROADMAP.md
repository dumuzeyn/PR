# PhotoRedactor Roadmap

## Implemented in this foundation

- Local documents: new, open, save, save as, export.
- Workflow: document presets, recent files, layer export and project metadata.
- Document geometry: resize image, resize canvas, trim transparent pixels, reveal all layers, rotate, flip, crop.
- Navigation: zoom, fit to screen, pan with scrollbars.
- Raster layers: create, duplicate, delete, reorder, visibility, opacity, merge, flatten.
- Layer compositing: clipping masks including adjustment-layer clipping, basic layer styles, opacity and blend modes.
- Non-destructive filters: per-layer editable filter stack with visual editor, built-in/importable/exportable presets, preview, reordering, per-filter enable/opacity/blend modes/masks, blur, sharpen, noise, median, edge detect and emboss.
- Raster selections: rectangular, elliptical, freehand lasso, magnetic lasso, polygon lasso, persistent replace/add/subtract/intersect modes, quick selection brush with a live result-mask preview and image-aware edge refinement controls, magic wand, color range, single row/column, opaque pixels, subject/background/sky selection, select all, invert, feather, smooth, grow, shrink, border/refine, smart edge cleanup/correction, richer Select and Mask previews/output including edge confidence, saved alpha selections.
- Raster layer masks: reveal all, hide all, from selection, mask thumbnails, thumbnail-click mask channel editing, red-overlay/black-white mask preview, linked/unlinked movement, paint on mask, density, feather, live edge refinement with smooth/contrast/shift and smart image-aware correction, invert, toggle, apply, delete.
- Blend modes: Normal, Multiply, Screen, Overlay, Soft Light, Darken, Lighten, Difference, Color, Luminosity.
- Tools: move, brush, eraser, fill, gradient, editable text, rectangular/elliptical/lasso/magnetic/polygon selection, magic wand, color range, brush/quick-selection canvas previews, interactive patch and crop.
- Text layers: font selection, multiline editing, paragraph width, word wrapping, alignment, line spacing and tracking.
- Local retouching: blur, sharpen, dodge, burn, clone stamp, source-based healing, automatic spot healing, retouch presets, interactive patch selection, edge-aware cleanup, red-eye reduction and content-aware fill.
- Advanced retouching: frequency separation into editable tone/texture layers with Linear Light reconstruction, plus skin-aware portrait cleanup with live preview and controls for smoothing, texture, tone and redness.
- Layer import: place embedded images, place linked images, update/relink linked layers and load several image files as layers.
- Shape layers: editable rectangle, ellipse, line, Bezier curve, polygon, star and boolean-combination layers.
- View helpers: grid overlay, numeric guides and channel preview for RGB/Red/Green/Blue/Alpha.
- UI: Russian main menu/tool labels, hover explanations, scrollable configurable tool palette, contextual tool options and centered canvas viewport navigation.
- Analysis: eyedropper, metadata/EXIF viewer, histogram and image statistics.
- Recovery and automation: autosave recovery file, manual recovery opening and action recording to JSON logs.
- Adjustments: destructive and previewable non-destructive brightness, contrast, saturation, vibrance, temperature/tint, hue/saturation, exposure, color balance, levels, curves, threshold, posterize, grayscale and invert, with importable/exportable preset libraries.
- Filters: Gaussian blur, sharpen, noise, median, edge detect, emboss, content-aware fill and red-eye reduction.
- History: undo and redo snapshots.
- Automation: simple batch resize/convert.
- Transform: visual Free Transform with eight handles for scale/move, rotate, position, flips, selected pixels, perspective corner transforms and live warp previews.
- Text: editable paragraph and path text, bold/italic/underline, baseline shift, tracking and transform handles that preserve editability.
- Shapes: custom-shape presets and visual four-point Bezier editing with live handles.
- Tool previews: real ellipse, line, polygon, star, custom shape and Bezier previews before creation.
- Performance baseline: numpy arrays, OpenCV filters, cached composite preview, cached per-stroke selection masks and optimized brush spacing.
- Export: PNG, JPEG, WebP, BMP, TIFF, plus `.prdx` project files.

## Next milestones toward the full list

1. Smart objects: re-editable transforms, embedded payload management and richer linked-file status/conflict handling.
2. RAW/color management: ICC profiles, 16/32-bit channels, Lab/CMYK workflows and richer metadata editing.
3. GPU/tiled rendering: tile cache, dirty-region recomposition, scratch disk, background save.
4. Plugin API and actions: replayable command actions, batch execution, external filters.
5. Advanced AI/generative tools: content-aware fill and generative expand.
