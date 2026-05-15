# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all

datas = [('res', 'res'), ('assets', 'assets'), ('Informer2020', 'Informer2020'), ('data', 'data')]
binaries = []
hiddenimports = ['PySide6', 'PySide6.QtCore', 'PySide6.QtGui', 'PySide6.QtWidgets', 'matplotlib', 'matplotlib.backends.backend_qt5agg', 'matplotlib.figure', 'pandas', 'numpy', 'numpy._core', 'scipy', 'scipy._lib', 'scipy.sparse', 'scipy.linalg', 'scipy.special', 'scipy.integrate', 'scipy.interpolate', 'scipy.ndimage', 'scipy.optimize', 'scipy.stats', 'scipy.signal', 'scipy.fft', 'torch', 'torch.nn', 'torch.nn.functional', 'torch.utils.data', 'torch.distributed', 'torch.serialization', 'torch._C', 'torch.cuda', 'torch.backends', 'torch.optim', 'unittest', 'xml.etree', 'http.client', 'joblib', 'sklearn', 'sklearn.utils', 'sklearn.preprocessing', 'sklearn.metrics', 'openpyxl', 'gui_config', 'prediction_controller', 'chart_renderer', 'data_loader_module', 'api_v8', 'Informer2020.models', 'Informer2020.utils', 'Informer2020.data', 'Informer2020.exp', 'assets.wind_ceemdan_lgbm_trans.preprocessors', 'assets.wind_ceemdan_lgbm_trans.models', 'assets.pv_tcn_informer.preprocessors', 'assets.pv_tcn_informer.models']
tmp_ret = collect_all('numpy')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('pandas')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('torch')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('sklearn')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('scipy')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]


a = Analysis(
    ['GUI.py', 'api_v8.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['tkinter', 'pdb', 'test', 'pytest', 'setuptools', 'IPython', 'jupyter', 'tensorflow', 'keras'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='APredict',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['res\\icon.ico'],
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='APredict',
)
