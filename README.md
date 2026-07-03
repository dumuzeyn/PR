# PhotoRedactor

Personal Photoshop-like raster editor for Windows.

This repository starts with a real working editor and an architecture intended to grow toward the full feature list:

- multi-layer raster documents
- custom project format (`.prdx`)
- PNG/JPEG/WebP/BMP/TIFF import and export
- brush, eraser, fill, gradient, text, crop, rectangular selection, move
- layer opacity, visibility, duplication, merge, flatten
- undo/redo
- resize, canvas resize, rotate, flip
- brightness, contrast, saturation, levels, curves, blur, sharpen, noise, grayscale, invert
- batch processing for folders
- executable packaging with PyInstaller

## Run from source

```powershell
.\run.ps1
```

## Build exe

```powershell
.\build_exe.ps1
```

The executable will be created in `dist\PhotoRedactor.exe`.

## Navigation

- Mouse wheel scrolls vertically.
- `Shift` + mouse wheel scrolls horizontally.
- Middle mouse button drags the canvas.
- `Space` + left mouse button drags the canvas.
- The `Hand` tool also drags the canvas.

## Scope

The long requirements file describes a Photoshop-scale application. This codebase treats that as the product target, not a small demo. Features that are not implemented yet are tracked in `ROADMAP.md` instead of being faked in the UI.
