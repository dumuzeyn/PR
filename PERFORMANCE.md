# PhotoRedactor performance report

Measured on 2026-08-07 with the CPU renderer and 256 px tiles. Times vary by hardware; use `benchmarks\benchmark_rendering.py` to repeat them.

| Document | Layers | Previous full composite | Cached initial full composite | One dirty tile |
| --- | ---: | ---: | ---: | ---: |
| 1920x1080 | 1 | 209 ms | 221 ms | 8 ms |
| 1920x1080 | 10 | 2101 ms | 2177 ms | 73 ms |
| 3840x2160 | 1 | 973 ms | 924 ms | 7 ms |
| 3840x2160 | 10 | 8487 ms | 8657 ms | 72 ms |

Initial full composition remains close to the reference renderer because it must visit every visible pixel. Interactive edits avoid that cost: brush, eraser, mask painting, blur/sharpen/dodge/burn, clone, healing and spot healing invalidate only affected tiles.

## Implemented

- 256 px dirty-region compositor with exact reference-renderer equivalence tests.
- Separate transparent and checkerboard caches, created only when needed.
- Cached filter stacks, feathered masks and layer effects.
- Local halo updates for blur, median, edge and emboss filters.
- Local feathered-mask updates.
- Tiled PIL/ImageTk conversion and canvas items, including zoomed views.
- 128 px sparse stroke history; each touched tile is captured once.
- Compact field/property/list history commands for ordinary layer edits.
- Incremental quick-selection preview and bounded proxy previews.
- Revision guards for background filter results and saves.
- Opt-in timings through `PHOTO_REDACTOR_DEBUG_PERF=1`.

## Correctness fallbacks

Full recomposition is retained for document resize/crop/rotation, flatten/merge, layer structure changes and operations whose result can affect the whole document. Unsupported local filter combinations also use a full filtered-layer refresh. These fallbacks preserve image quality and project compatibility.

## Remaining large-document work

The renderer is CPU-based and keeps active composite caches in RAM. A future stage can add a mipmap pyramid, GPU compositing and scratch-disk eviction for documents that exceed the configured memory budget.
