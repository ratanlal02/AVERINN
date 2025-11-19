# -*- mode: python ; coding: utf-8 -*-

hiddenimports = [
    'gurobipy._util',
    'gurobipy._attrutil',
    'gurobipy._callback',
    'gurobipy._modelutil'
]

a = Analysis(
    ['averinn.py'],
    pathex=[],
    binaries=[('/Library/gurobi1201/macos_universal2/lib/libgurobi120.dylib*', '.')],
    datas=[],
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
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
    name='averinn',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
