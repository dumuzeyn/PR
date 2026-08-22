# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['uzyro_launcher.py'],
    pathex=[],
    binaries=[('uzyro/assets/native/fribidi-0.dll', '.')],
    datas=[('uzyro/assets/native/LICENSE.fribidi.txt', 'licenses'), ('uzyro/assets/tool_demos', 'uzyro/assets/tool_demos'), ('uzyro/assets/branding', 'uzyro/assets/branding')],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['scipy', 'skimage', 'aggdraw', 'matplotlib', 'PyQt5', 'PySide6'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='UZYRO',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['uzyro\\assets\\branding\\uzyro.ico'],
)
