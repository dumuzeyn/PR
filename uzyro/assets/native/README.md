# FriBidi runtime

`fribidi-0.dll` is FriBidi 1.0.10, used by Pillow RAQM for bidirectional and
OpenType-aware text shaping on Windows.

- Upstream: https://github.com/fribidi/fribidi
- Distribution: Anaconda `fribidi-1.0.10-h62dcd97_0`
- SHA-256: `463c7479d434a8681a9dbde16d0675e28d09d38fa43e3e18d826e345584ba18d`
- License: LGPL-2.1-or-later; see `LICENSE.fribidi.txt`.

The DLL is loaded dynamically and remains a separate library inside the
PyInstaller bundle.
