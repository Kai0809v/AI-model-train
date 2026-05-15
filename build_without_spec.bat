@echo off
chcp 65001 >nul
echo ========================================
echo 综合能源预测系统 - PyInstaller 打包脚本
echo ========================================
echo.

REM 检查并激活虚拟环境
if exist ".venv\Scripts\activate.bat" (
    echo [提示] 检测到虚拟环境，正在激活...
    call .venv\Scripts\activate.bat
    echo ✓ 虚拟环境已激活
    echo.
)

REM 检查 Python 环境
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未找到 Python，请先安装 Python 或激活虚拟环境
    echo [提示] 如果使用了虚拟环境，请运行: .venv\Scripts\activate.bat
    pause
    exit /b 1
)

echo [1/5] 清理旧的打包文件...
if exist "build" rmdir /s /q "build"
if exist "dist" rmdir /s /q "dist"
if exist "APredict.spec" del /q "APredict.spec"
echo ✓ 清理完成
echo.

echo [2/5] 检查依赖包...
pip show pyinstaller >nul 2>&1
if errorlevel 1 (
    echo [警告] 未安装 PyInstaller，使用这个命令安装：pip install pyinstaller
)
echo ✓ PyInstaller 已就绪
echo.

echo [3/5] 开始打包（目录模式）...
echo 这可能需要几分钟时间，请耐心等待...
echo.

python -m PyInstaller ^
    --name=APredict ^
    --windowed ^
    --noconsole ^
    --icon=res/icon.ico ^
    --add-data "res;res" ^
    --add-data "assets;assets" ^
    --add-data "Informer2020;Informer2020" ^
    --add-data "data;data" ^
    --hidden-import=PySide6 ^
    --hidden-import=PySide6.QtCore ^
    --hidden-import=PySide6.QtGui ^
    --hidden-import=PySide6.QtWidgets ^
    --hidden-import=matplotlib ^
    --hidden-import=matplotlib.backends.backend_qt5agg ^
    --hidden-import=matplotlib.figure ^
    --hidden-import=pandas ^
    --hidden-import=numpy ^
    --hidden-import=numpy._core ^
    --hidden-import=scipy ^
    --hidden-import=scipy._lib ^
    --hidden-import=scipy.sparse ^
    --hidden-import=scipy.linalg ^
    --hidden-import=scipy.special ^
    --hidden-import=scipy.integrate ^
    --hidden-import=scipy.interpolate ^
    --hidden-import=scipy.ndimage ^
    --hidden-import=scipy.optimize ^
    --hidden-import=scipy.stats ^
    --hidden-import=scipy.signal ^
    --hidden-import=scipy.fft ^
    --hidden-import=torch ^
    --hidden-import=torch.nn ^
    --hidden-import=torch.nn.functional ^
    --hidden-import=torch.utils.data ^
    --hidden-import=torch.distributed ^
    --hidden-import=torch.serialization ^
    --hidden-import=torch._C ^
    --hidden-import=torch.cuda ^
    --hidden-import=torch.backends ^
    --hidden-import=torch.optim ^
    --hidden-import=unittest ^
    --hidden-import=xml.etree ^
    --hidden-import=http.client ^
    --hidden-import=joblib ^
    --hidden-import=sklearn ^
    --hidden-import=sklearn.utils ^
    --hidden-import=sklearn.preprocessing ^
    --hidden-import=sklearn.metrics ^
    --hidden-import=openpyxl ^
    --hidden-import=gui_config ^
    --hidden-import=prediction_controller ^
    --hidden-import=chart_renderer ^
    --hidden-import=data_loader_module ^
    --hidden-import=api_v8 ^
    --hidden-import=Informer2020.models ^
    --hidden-import=Informer2020.utils ^
    --hidden-import=Informer2020.data ^
    --hidden-import=Informer2020.exp ^
    --hidden-import=assets.wind_ceemdan_lgbm_trans.preprocessors ^
    --hidden-import=assets.wind_ceemdan_lgbm_trans.models ^
    --hidden-import=assets.pv_tcn_informer.preprocessors ^
    --hidden-import=assets.pv_tcn_informer.models ^
    --exclude-module=tkinter ^
    --exclude-module=pdb ^
    --exclude-module=test ^
    --exclude-module=pytest ^
    --exclude-module=setuptools ^
    --exclude-module=IPython ^
    --exclude-module=jupyter ^
    --exclude-module=tensorflow ^
    --exclude-module=keras ^
    --collect-all numpy ^
    --collect-all pandas ^
    --collect-all torch ^
    --collect-all sklearn ^
    --collect-all scipy ^
    --noconfirm ^
    GUI.py api_v8.py

if errorlevel 1 (
    echo.
    echo [错误] 打包失败！请检查上面的错误信息
    pause
    exit /b 1
)

echo.
echo [4/5] 打包完成！
echo.
echo [5/5] 验证输出文件...
if exist "dist\APredict\APredict.exe" (
    echo ✓ 可执行文件: dist\APredict\APredict.exe
    
    REM 显示文件大小
    for %%A in ("dist\APredict\APredict.exe") do (
        set size=%%~zA
        set /a size_mb=!size! / 1048576
        echo   大小: !size_mb! MB
    )
    
    echo.
    echo ✓ 所有依赖文件位于: dist\APredict\
    echo.
    echo ========================================
    echo 打包成功！
    echo ========================================
    echo.
    echo 使用说明:
    echo   1. 分发时需要复制整个 dist\APredict 目录
    echo   2. 运行: dist\APredict\APredict.exe
    echo   3. 模型文件和资源已包含在目录中
    echo.
) else (
    echo [错误] 未找到生成的可执行文件
    echo 请检查 dist\APredict 目录
)

pause
