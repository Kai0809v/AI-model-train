import sys
import os
import pandas as pd
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                               QHBoxLayout, QLabel, QComboBox, QPushButton,
                               QLineEdit, QFileDialog, QTextEdit, QFrame,
                               QProgressBar, QMessageBox, QStackedWidget)

# ============================================
# 资源路径处理工具（解决打包后路径失效问题）
# ============================================
def resource_path(relative_path):
    """获取资源的绝对路径，兼容开发环境和 PyInstaller 打包环境"""
    if hasattr(sys, '_MEIPASS'):
        base = sys._MEIPASS
    else:
        base = os.path.abspath(".")
    
    # 🔧 将 Windows 的反斜杠 \ 替换为正斜杠 /，因为 Qt 样式表只认 /
    path = os.path.join(base, relative_path).replace('\\', '/')
    return path

# 导入配置常量
from gui_config import (
    APP_NAME, APP_VERSION, APP_DESCRIPTION,
    MAIN_WINDOW_WIDTH, MAIN_WINDOW_HEIGHT,
    ANALYSIS_WINDOW_WIDTH, ANALYSIS_WINDOW_HEIGHT,
    PREDICTION_SCENARIOS,
    FILE_FILTER_STRING,
    PV_POWER_COLUMNS, WIND_POWER_COLUMNS, TIME_COLUMNS,
    COLOR_PRIMARY, COLOR_PRIMARY_HOVER, COLOR_SECONDARY,
    COLOR_BG_CARD, COLOR_BG_LOG, COLOR_TEXT_SUCCESS,
    CHART_MIN_WIDTH, CHART_MIN_HEIGHT,
    ICON_PATH, BACKGROUND_PATH
)

# 导入解耦模块
from prediction_controller import PredictionController
from chart_renderer import ChartRenderer
from data_loader_module import DataLoader 

# 应用版本
MLG_VERSION = APP_VERSION

# ============================================
# 登录页面
# ============================================
class LoginPage(QWidget):
    def __init__(self, on_login_success, switch_to_register_callback=None):
        super().__init__()
        self.on_login_success = on_login_success
        self.switch_to_register_callback = switch_to_register_callback
        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)

        # 创建主卡片容器 (限制宽度，不超过窗口的 2/3)
        container = QFrame()
        container.setFixedWidth(420)
        container.setStyleSheet("""
            QFrame#MainAuthCard {
                background-color: rgba(255, 255, 255, 0.95);
                border-radius: 15px;
                border: 1px solid rgba(255, 255, 255, 0.5);
            }
        """)
        container.setObjectName("MainAuthCard")

        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(30, 40, 30, 40)
        container_layout.setSpacing(20)

        # 1. 标题方框容器
        title_container = QFrame()
        title_container.setStyleSheet("""
            QFrame {
                background-color: rgba(232, 245, 233, 0.6); /* 极淡的新能源绿 */
                border-radius: 10px;
                border: 1px dashed #a5d6a7;
            }
        """)
        title_layout = QVBoxLayout(title_container)
        title_layout.setContentsMargins(15, 20, 15, 20)
        title_layout.setSpacing(5)

        title = QLabel("综合能源预测系统")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet(
            "font-size: 26px; font-weight: bold; color: #1b5e20; border: none; background: transparent;")
        title_layout.addWidget(title)

        subtitle = QLabel(f"{MLG_VERSION} · 智能预测引擎")
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet(
            "font-size: 13px; color: #4caf50; font-weight: bold; border: none; background: transparent;")
        title_layout.addWidget(subtitle)

        container_layout.addWidget(title_container)

        # 2. 用户名输入方框
        username_container = QFrame()
        username_container.setStyleSheet("""
            QFrame {
                background-color: #f1f8e9;
                border: 1px solid #c5e1a5;
                border-radius: 8px;
            }
        """)
        username_layout = QVBoxLayout(username_container)
        username_layout.setContentsMargins(15, 12, 15, 12)
        username_layout.setSpacing(8)

        username_label = QLabel("用户名:")
        username_label.setStyleSheet(
            "font-size: 13px; color: #33691e; font-weight: bold; border: none; background: transparent;")
        username_layout.addWidget(username_label)

        self.username_input = QLineEdit()
        self.username_input.setPlaceholderText("请输入账户名")
        self.username_input.setStyleSheet("""
            QLineEdit {
                padding: 8px;
                border: 1px solid #dcedc8;
                border-radius: 5px;
                font-size: 14px;
                background-color: #ffffff;
                color: #333;
            }
            QLineEdit:focus {
                border: 2px solid #4caf50;
            }
        """)
        username_layout.addWidget(self.username_input)

        container_layout.addWidget(username_container)

        # 3. 密码输入方框
        password_container = QFrame()
        password_container.setStyleSheet("""
            QFrame {
                background-color: #f1f8e9;
                border: 1px solid #c5e1a5;
                border-radius: 8px;
            }
        """)
        password_layout = QVBoxLayout(password_container)
        password_layout.setContentsMargins(15, 12, 15, 12)
        password_layout.setSpacing(8)

        password_label = QLabel("密码:")
        password_label.setStyleSheet(
            "font-size: 13px; color: #33691e; font-weight: bold; border: none; background: transparent;")
        password_layout.addWidget(password_label)

        self.password_input = QLineEdit()
        self.password_input.setPlaceholderText("请输入密码")
        self.password_input.setEchoMode(QLineEdit.Password)
        self.password_input.setStyleSheet("""
            QLineEdit {
                padding: 8px;
                border: 1px solid #dcedc8;
                border-radius: 5px;
                font-size: 14px;
                background-color: #ffffff;
                color: #333;
            }
            QLineEdit:focus {
                border: 2px solid #4caf50;
            }
        """)
        password_layout.addWidget(self.password_input)

        container_layout.addWidget(password_container)
        container_layout.addSpacing(5)

        # 4. 登录按钮 (新能源绿)
        login_btn = QPushButton("登 录 系 统")
        login_btn.setMinimumHeight(45)
        login_btn.setStyleSheet("""
            QPushButton {
                background-color: #2e7d32;
                color: white;
                font-size: 16px;
                font-weight: bold;
                border-radius: 8px;
                letter-spacing: 2px;
            }
            QPushButton:hover {
                background-color: #388e3c;
            }
            QPushButton:pressed {
                background-color: #1b5e20;
            }
        """)
        login_btn.clicked.connect(self.handle_login)
        container_layout.addWidget(login_btn)

        # 5. 注册链接
        register_label = QLabel(
            "新研发人员？ <a href='#' style='color: #2e7d32; text-decoration: none; font-weight: bold;'>立即注册</a>")
        register_label.setAlignment(Qt.AlignCenter)
        register_label.setStyleSheet("font-size: 13px; background: transparent;")
        register_label.linkActivated.connect(self.switch_to_register)
        container_layout.addWidget(register_label)

        # 外部包装器，用于居中
        main_layout.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(container)

        self.setLayout(main_layout)

    def handle_login(self):
        username = self.username_input.text()
        password = self.password_input.text()

        if not username or not password:
            QMessageBox.warning(self, "提示", "请输入用户名和密码！")
            return

        self.on_login_success(username)

    def switch_to_register(self):
        if self.switch_to_register_callback:
            self.switch_to_register_callback()


# ============================================
# 注册页面
# ============================================
class RegisterPage(QWidget):
    def __init__(self, on_register_success, switch_to_login_callback=None):
        super().__init__()
        self.on_register_success = on_register_success
        self.switch_to_login_callback = switch_to_login_callback
        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)

        # 创建主卡片容器
        container = QFrame()
        container.setFixedWidth(420)
        container.setStyleSheet("""
            QFrame#MainAuthCard {
                background-color: rgba(255, 255, 255, 0.95);
                border-radius: 15px;
                border: 1px solid rgba(255, 255, 255, 0.5);
            }
        """)
        container.setObjectName("MainAuthCard")

        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(30, 30, 30, 30)
        container_layout.setSpacing(15)

        # 1. 标题方框
        title_container = QFrame()
        title_container.setStyleSheet("""
            QFrame {
                background-color: rgba(232, 245, 233, 0.6);
                border-radius: 10px;
                border: 1px dashed #a5d6a7;
            }
        """)
        title_layout = QVBoxLayout(title_container)
        title_layout.setContentsMargins(15, 15, 15, 15)
        title_layout.setSpacing(5)

        title = QLabel("注册预测系统")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet(
            "font-size: 24px; font-weight: bold; color: #1b5e20; border: none; background: transparent;")
        title_layout.addWidget(title)

        subtitle = QLabel("配置研发人员账号")
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet(
            "font-size: 13px; color: #4caf50; font-weight: bold; border: none; background: transparent;")
        title_layout.addWidget(subtitle)

        container_layout.addWidget(title_container)

        # 辅助函数：创建输入方框
        def create_input_box(label_text, placeholder, is_password=False):
            box = QFrame()
            box.setStyleSheet("""
                QFrame { background-color: #f1f8e9; border: 1px solid #c5e1a5; border-radius: 8px; }
            """)
            box_layout = QVBoxLayout(box)
            box_layout.setContentsMargins(15, 10, 15, 10)
            box_layout.setSpacing(5)

            lbl = QLabel(label_text)
            lbl.setStyleSheet(
                "font-size: 13px; color: #33691e; font-weight: bold; border: none; background: transparent;")

            inp = QLineEdit()
            inp.setPlaceholderText(placeholder)
            if is_password:
                inp.setEchoMode(QLineEdit.Password)
            inp.setStyleSheet("""
                QLineEdit { padding: 6px; border: 1px solid #dcedc8; border-radius: 5px; font-size: 14px; background-color: #ffffff; color: #333; }
                QLineEdit:focus { border: 2px solid #4caf50; }
            """)

            box_layout.addWidget(lbl)
            box_layout.addWidget(inp)
            return box, inp

        # 2/3/4. 输入区
        user_box, self.username_input = create_input_box("用户名:", "请输入要注册的账户名")
        pwd_box, self.password_input = create_input_box("密码:", "设置安全密码", True)
        confirm_box, self.confirm_password_input = create_input_box("确认密码:", "请再次输入密码", True)

        container_layout.addWidget(user_box)
        container_layout.addWidget(pwd_box)
        container_layout.addWidget(confirm_box)

        # 5. 注册按钮
        register_btn = QPushButton("完 成 注 册")
        register_btn.setMinimumHeight(45)
        register_btn.setStyleSheet("""
            QPushButton { background-color: #00897b; color: white; font-size: 16px; font-weight: bold; border-radius: 8px; }
            QPushButton:hover { background-color: #00796b; }
            QPushButton:pressed { background-color: #00695c; }
        """)
        register_btn.clicked.connect(self.handle_register)
        container_layout.addWidget(register_btn)

        # 6. 返回登录
        back_label = QLabel(
            "已有账号？ <a href='#' style='color: #00897b; text-decoration: none; font-weight: bold;'>返回登录</a>")
        back_label.setAlignment(Qt.AlignCenter)
        back_label.setStyleSheet("font-size: 13px; background: transparent;")
        back_label.linkActivated.connect(self.switch_to_login)
        container_layout.addWidget(back_label)

        # 外部包装器
        main_layout.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(container)

        self.setLayout(main_layout)

    def handle_register(self):
        username = self.username_input.text()
        password = self.password_input.text()
        confirm_password = self.confirm_password_input.text()

        if not username or not password:
            QMessageBox.warning(self, "提示", "用户名和密码不能为空！")
            return

        if password != confirm_password:
            QMessageBox.warning(self, "提示", "两次输入的密码不一致！")
            return

        QMessageBox.information(self, "成功", "注册成功！请登录。")
        self.on_register_success(username)

    def switch_to_login(self):
        if self.switch_to_login_callback:
            self.switch_to_login_callback()


# ============================================
# 后台预测线程 (保持不变)
# ============================================
class PredictionWorker(QThread):
    progress = Signal(int, str)
    finished = Signal(dict)
    error = Signal(str)

    def __init__(self, controller, file_path, model_backend_name, steps=1, mode="auto", future_weather_path=None):
        super().__init__()
        self.controller = controller
        self.file_path = file_path
        self.model_backend_name = model_backend_name
        self.steps = steps
        self.mode = mode
        self.future_weather_path = future_weather_path

    def run(self):
        try:
            self.progress.emit(10, "正在读取历史数据...")
            
            # 调用控制器执行完整预测流程
            self.progress.emit(30, "数据校验通过。正在加载模型...")
            self.progress.emit(50, f"正在加载模型 [{self.model_backend_name}]...")
            self.progress.emit(70, "模型就绪，正在执行核心预测算法...")

            # 🔧 构建额外参数
            kwargs = {}
            if self.model_backend_name == "PV_TCN_Informer":
                kwargs["mode"] = self.mode
                if self.mode == "with_future" and self.future_weather_path:
                    self.progress.emit(75, "正在加载未来气象数据...")

            result = self.controller.execute_prediction(
                self.file_path,
                self.model_backend_name,
                self.steps,
                mode=self.mode,
                future_weather_path=self.future_weather_path
            )

            if result.get("success"):
                self.progress.emit(100, "✅ 预测计算完成！")
                self.finished.emit(result)
            else:
                self.error.emit(result.get("error", "未知预测错误"))
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error.emit(f"预测过程中发生严重系统异常：{str(e)}")


# ============================================
# 主 GUI 界面
# ============================================
class EnergyForecastApp(QMainWindow):
    def __init__(self):
        super().__init__()
        # 调整了窗口尺寸，高度不再过于冗长
        self.setWindowTitle(f"{APP_NAME} {MLG_VERSION}")
        self.resize(MAIN_WINDOW_WIDTH, MAIN_WINDOW_HEIGHT)

        self.current_username = None
        # 使用解耦的控制器
        self.controller = PredictionController()
        # 图表渲染器
        self.chart_renderer = ChartRenderer()
        self.data_loader = DataLoader()
        self.init_ui()

    def init_ui(self):
        self.stacked_widget = QStackedWidget()
        self.setCentralWidget(self.stacked_widget)

        auth_container = QWidget()
        auth_layout = QVBoxLayout(auth_container)
        auth_layout.setContentsMargins(0, 0, 0, 0)

        self.auth_stacked = QStackedWidget()

        def go_to_register():
            self.auth_stacked.setCurrentIndex(1)

        def go_to_login():
            self.auth_stacked.setCurrentIndex(0)

        self.login_page = LoginPage(self.on_login_success, go_to_register)
        self.auth_stacked.addWidget(self.login_page)

        self.register_page = RegisterPage(self.on_register_success, go_to_login)
        self.auth_stacked.addWidget(self.register_page)

        auth_layout.addWidget(self.auth_stacked)
        self.stacked_widget.addWidget(auth_container)

        self.main_widget = QWidget()
        self.init_main_ui()
        self.stacked_widget.addWidget(self.main_widget)

    def init_main_ui(self):
        main_layout = QHBoxLayout(self.main_widget)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(20)

        # ================= 左侧控制栏 =================
        left_panel = QFrame()
        left_panel.setFixedWidth(300)
        left_panel.setStyleSheet("QFrame { background-color: rgba(255, 255, 255, 0.95); border-radius: 12px; }")
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(20, 20, 20, 20)
        left_layout.setSpacing(15)

        self.user_welcome_label = QLabel("欢迎用户：--")
        self.user_welcome_label.setStyleSheet(
            "font-size: 14px; font-weight: bold; color: #1b5e20; padding: 10px; background-color: #e8f5e9; border-radius: 6px; border: 1px solid #c8e6c9;")
        self.user_welcome_label.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(self.user_welcome_label)

        title_label = QLabel("控制面板")
        title_label.setStyleSheet(
            "background: transparent; font-size: 18px; font-weight: bold; color: #2c3e50; margin-top: 10px;")
        left_layout.addWidget(title_label)

        # 辅助函数优化控件样式
        def style_combo(combo):
            combo.setStyleSheet("padding: 6px; border: 1px solid #cfd8dc; border-radius: 4px; background: white;")

        # ===== 预测场景 =====
        scene_label = QLabel("预测场景:")
        scene_label.setStyleSheet("background-color: transparent; color: #37474f; font-size: 13px;")
        left_layout.addWidget(scene_label)
        
        self.type_combo = QComboBox()
        self.type_combo.addItems(["风电功率预测", "光伏功率预测", "电网负荷预测（后续更新）"])
        style_combo(self.type_combo)
        self.type_combo.currentTextChanged.connect(self.on_scene_changed)
        left_layout.addWidget(self.type_combo)

        # ===== 历史数据 (CSV/xlsx) =====
        data_label = QLabel("历史数据 (CSV/xlsx):")
        data_label.setStyleSheet("background-color: transparent; color: #37474f; font-size: 13px;")
        left_layout.addWidget(data_label)
        
        file_layout = QHBoxLayout()
        self.file_input = QLineEdit()
        self.file_input.setPlaceholderText("请选择数据...")
        self.file_input.setReadOnly(True)
        self.file_input.setStyleSheet(
            "padding: 6px; border: 1px solid #cfd8dc; border-radius: 4px; background: #f8f9fa;")

        self.browse_btn = QPushButton("浏览")
        self.browse_btn.setStyleSheet(
            "padding: 6px 12px; background: #00897b; color: white; border-radius: 4px; font-weight: bold;")
        self.browse_btn.clicked.connect(self.load_file)

        file_layout.addWidget(self.file_input)
        file_layout.addWidget(self.browse_btn)
        left_layout.addLayout(file_layout)

        # ===== 算法模型 =====
        model_label = QLabel("算法模型:")
        model_label.setStyleSheet("background-color: transparent; color: #37474f; font-size: 13px;")
        left_layout.addWidget(model_label)
        
        self.model_combo = QComboBox()
        style_combo(self.model_combo)
        self.model_combo.currentTextChanged.connect(self.on_model_changed)  # 添加信号连接
        left_layout.addWidget(self.model_combo)
        
        # ===== 未来气象数据（仅“有未来气象数据”模型可见）=====
        self.future_weather_label = QLabel("未来气象数据 (CSV/xlsx):")
        self.future_weather_label.setStyleSheet("background-color: transparent; color: #37474f; font-size: 13px;")
        left_layout.addWidget(self.future_weather_label)

        future_file_layout = QHBoxLayout()
        self.future_weather_input = QLineEdit()
        self.future_weather_input.setPlaceholderText("选择未来24步气象数据...")
        self.future_weather_input.setReadOnly(True)
        self.future_weather_input.setStyleSheet(
            "padding: 6px; border: 1px solid #cfd8dc; border-radius: 4px; background: #f8f9fa;")

        self.future_browse_btn = QPushButton("浏览")
        self.future_browse_btn.setStyleSheet(
            "padding: 6px 12px; background: #00897b; color: white; border-radius: 4px; font-weight: bold;")
        self.future_browse_btn.clicked.connect(self.load_future_weather_file)

        future_file_layout.addWidget(self.future_weather_input)
        future_file_layout.addWidget(self.future_browse_btn)
        left_layout.addLayout(future_file_layout)

        # ===== 预测步长 =====
        steps_label = QLabel("预测步长:")
        steps_label.setStyleSheet("background-color: transparent; color: #37474f; font-size: 13px;")
        left_layout.addWidget(steps_label)
        
        self.steps_combo = QComboBox()
        style_combo(self.steps_combo)
        left_layout.addWidget(self.steps_combo)

        # 初始化场景（触发模型列表和步长更新）
        # 从配置常量动态加载，而非硬编码
        self.on_scene_changed(self.type_combo.currentText())

        left_layout.addStretch()

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setStyleSheet("""
            QProgressBar { border: 1px solid #b0bec5; border-radius: 4px; text-align: center; color: #333; background: #eceff1; }
            QProgressBar::chunk { background-color: #4caf50; width: 10px; margin: 0.5px; }
        """)
        self.progress_bar.hide()
        left_layout.addWidget(self.progress_bar)

        self.start_btn = QPushButton("🚀 开始智能预测")
        self.start_btn.setMinimumHeight(45)
        self.start_btn.setStyleSheet("""
            QPushButton { background-color: #2e7d32; color: white; font-weight: bold; border-radius: 8px; font-size: 15px; }
            QPushButton:hover { background-color: #388e3c; }
            QPushButton:disabled { background-color: #a5d6a7; color: #f1f8e9;}
        """)
        self.start_btn.clicked.connect(self.run_prediction)
        left_layout.addWidget(self.start_btn)

        # 历史数据分析
        self.analysis_btn = QPushButton("🔍 历史数据分析")
        self.analysis_btn.setMinimumHeight(40)
        self.analysis_btn.setStyleSheet("""
            QPushButton { 
                background-color: #00897b; 
                color: white; 
                font-weight: bold; 
                border-radius: 8px; 
                font-size: 15px;
            }
            QPushButton:hover { 
                background-color: #00796b; 
            }
        """)
        self.analysis_btn.clicked.connect(self.open_analysis_window)
        left_layout.addWidget(self.analysis_btn)

        # ================= 控制面板界面右侧图表显示区域 =================
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(15)

        # 🔧 标题区域
        result_title = QLabel("预测结果可视化")
        result_title.setStyleSheet(
            "background: transparent; font-size: 20px; font-weight: bold; color: #2c3e50; margin-bottom: 10px;")
        result_title.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(result_title)

        # 🔧 主图表区域（使用 Matplotlib 嵌入）
        # 🚀 延迟加载：仅在需要时导入
        self._matplotlib_imported = False
        self.figure = None
        self.canvas = None
        self.placeholder_widget = None  # 保存占位符引用
        
        # 初始显示占位符
        self.placeholder_widget = QFrame()
        self.placeholder_widget.setObjectName("ChartPlaceholder")
        self.placeholder_widget.setStyleSheet("""
            background-color: rgba(255, 255, 255, 0.95); 
            border: 2px solid #b2dfdb; 
            border-radius: 12px;
        """)
        self.placeholder_widget.setMinimumSize(600, 400)
        placeholder_layout = QVBoxLayout(self.placeholder_widget)
        placeholder_layout.setAlignment(Qt.AlignCenter)
        
        placeholder_text = QLabel("📊 预测结果将在此处显示")
        placeholder_text.setStyleSheet("font-size: 16px; color: #78909c;")
        placeholder_text.setAlignment(Qt.AlignCenter)
        placeholder_layout.addWidget(placeholder_text)
        
        right_layout.addWidget(self.placeholder_widget, 3)

        # 🔧 当前预测值显示卡片
        self.pred_value_card = QFrame()
        self.pred_value_card.setStyleSheet("""
            QFrame { 
                background-color: rgba(255, 255, 255, 0.80); 
                border-radius: 10px; 
                padding: 8px;
                border: 2px solid #4caf50;
            }
        """)
        pred_layout = QHBoxLayout(self.pred_value_card)
        pred_layout.setAlignment(Qt.AlignCenter)
        pred_layout.setSpacing(6)
        pred_layout.setContentsMargins(8, 4, 8, 4)
        
        self.pred_desc_label = QLabel("等待预测...")
        self.pred_desc_label.setStyleSheet(
            "font-size: 14px; color: #78909c;")
        self.pred_desc_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        pred_layout.addWidget(self.pred_desc_label)
        
        self.pred_value_label = QLabel("-- MW")
        self.pred_value_label.setStyleSheet(
            "font-size: 36px; font-weight: bold; color: #2e7d32;")
        self.pred_value_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        pred_layout.addWidget(self.pred_value_label)
        
        right_layout.addWidget(self.pred_value_card, stretch=1)

        # 日志区域
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setMaximumHeight(130)
        self.log_output.setStyleSheet(
            "background-color: rgba(38, 50, 56, 0.95); color: #69f0ae; font-family: Consolas; font-size: 13px; border-radius: 8px; padding: 10px;")
        self.append_log("新能源预测系统内核初始化完成...")
        right_layout.addWidget(self.log_output, stretch=1)

        main_layout.addWidget(left_panel)
        main_layout.addWidget(right_panel)

    def create_metric_card(self, title, value):
        card = QLabel(f"{title}\n{value}")
        card.setAlignment(Qt.AlignCenter)
        card.setStyleSheet("""
            background-color: rgba(255, 255, 255, 0.95); border-radius: 10px; padding: 12px; 
            font-size: 15px; font-weight: bold; border: 1px solid #e0e0e0; color: #37474f;
        """)
        return card

    def on_scene_changed(self, scene_text):
        """预测场景切换时，更新可选模型列表和步长选项"""
        # 从控制器获取模型列表
        self.model_combo.clear()
        model_map = self.controller.get_models_for_scenario(scene_text)
        self.model_combo.addItems(list(model_map.keys()))

        # 从控制器获取步长选项
        self.steps_combo.clear()
        steps_options = self.controller.get_steps_options_for_scenario(scene_text)
        self.steps_combo.addItems(steps_options)

        # 控制未来气象数据的可见性（根据选择的模型）
        is_pv = scene_text == "光伏功率预测"
        if is_pv:
            # 直接根据模型判断是否显示未来气象数据输入
            self._update_future_weather_visibility_by_model()
        else:
            self.future_weather_label.setVisible(False)
            self.future_weather_input.setVisible(False)
            self.future_browse_btn.setVisible(False)

    def on_model_changed(self, model_text):
        """模型切换时，更新未来气象数据的可见性"""
        self._update_future_weather_visibility_by_model()

    def _update_future_weather_visibility_by_model(self):
        """根据当前选择的模型决定是否显示未来气象数据输入"""
        model_name = self.model_combo.currentText()
        # 只有选择“有未来气象数据”版本时才显示输入框
        show = "有未来气象数据" in model_name
        self.future_weather_label.setVisible(show)
        self.future_weather_input.setVisible(show)
        self.future_browse_btn.setVisible(show)

    def load_future_weather_file(self):
        """选择未来气象数据文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "选择未来气象数据文件", 
            "", 
            "数据文件 (*.csv *.xlsx *.xls);;CSV Files (*.csv);;Excel Files (*.xlsx *.xls);;All Files (*)"
        )
        if file_path:
            self.future_weather_input.setText(file_path)
            ext = os.path.splitext(file_path)[1].upper()
            self.append_log(f"已装载{ext}格式未来气象数据：{file_path}")

    def append_log(self, text):
        self.log_output.append(f"⚡ {text}")
        scrollbar = self.log_output.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def load_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "选择历史数据文件", 
            "", 
            FILE_FILTER_STRING
        )
        if file_path:
            self.file_input.setText(file_path)
            # 显示文件格式
            format_desc = self.data_loader.get_file_format_description(file_path)
            self.append_log(f"已装载{format_desc}格式历史数据：{file_path}")

    def run_prediction(self):
        file_path = self.file_input.text()
        if not file_path:
            QMessageBox.warning(self, "提示", "请先选择历史数据文件！")
            return

        energy_type = self.type_combo.currentText()
        if energy_type == "电网负荷预测（后续更新）":
            QMessageBox.information(self, "提示", "电网负荷预测功能敬请期待！")
            return

        # 从控制器获取当前场景的模型映射
        model_map = self.controller.get_models_for_scenario(energy_type)
        if not model_map:
            QMessageBox.warning(self, "提示", f"当前场景 [{energy_type}] 暂无可用模型！")
            return

        display_model = self.model_combo.currentText()
        backend_model_name = model_map.get(display_model)
        if not backend_model_name:
            QMessageBox.warning(self, "提示", "请选择一个算法模型！")
            return

        # 🔧 根据模型名称自动判断模式（不再使用mode_combo）
        if backend_model_name == "PV_TCN_Informer":
            mode = "with_future"
        elif backend_model_name == "PV_TCN_Informer_NoWeather":
            mode = "without_future"
        else:
            mode = "auto"  # 其他模型不需要此参数

        # 校验：with_future 模式必须提供未来气象数据
        future_weather_path = None
        if mode == "with_future":
            future_weather_path = self.future_weather_input.text()
            if not future_weather_path:
                QMessageBox.warning(self, "提示", "当前为【有未来气象数据】模式，请先选择未来气象数据文件！")
                return
                        
            # 🔧 验证未来气象数据文件格式
            is_valid, error_msg = self.controller.validate_future_weather_file(future_weather_path)
            if not is_valid:
                QMessageBox.warning(self, "提示", error_msg)
                return

        # 🔧 根据场景和步长选项映射步数
        steps_text = self.steps_combo.currentText()
        steps = self.controller.parse_steps_from_text(steps_text)
        if steps == 1 and "单步" not in steps_text and "1 步" not in steps_text:
            self.append_log("⚠️ 未知步长选项，默认使用单步预测")

        self.start_btn.setEnabled(False)
        self.start_btn.setText("⏳ 模型推理中...")
        self.progress_bar.show()
        self.progress_bar.setValue(0)
        self.append_log("-" * 40)
        mode_desc = f"模式: {mode}" if energy_type == "光伏功率预测" else ""
        self.append_log(f"启动新任务：{display_model} (预测步长：{steps}) {mode_desc}")

        self.worker = PredictionWorker(
            self.controller, file_path, backend_model_name, steps,
            mode=mode, future_weather_path=future_weather_path
        )
        self.worker.progress.connect(self.update_progress)
        self.worker.finished.connect(self.on_prediction_success)
        self.worker.error.connect(self.on_prediction_error)
        self.worker.start()

    def update_progress(self, percent, text):
        self.progress_bar.setValue(percent)
        self.append_log(text)

    def on_prediction_success(self, result):
        self.reset_ui_state()
        
        # 🔧 修改：支持单值和多值显示的逻辑
        if "prediction" in result:
            # 单步预测
            val = result["prediction"]
            self.pred_value_label.setText(f"{val:.2f} MW")
            self.pred_desc_label.setText("下一时刻预测功率")
            self.pred_value_card.setStyleSheet("""
                QFrame { 
                    background-color: #e8f5e9; 
                    border-radius: 12px; 
                    padding: 15px;
                    border: 2px solid #4caf50;
                }
            """)
            
            # 绘制单点图表
            self.plot_prediction([val], "单步预测结果")
            
        elif "predictions" in result:
            # 多步预测
            vals = result["predictions"]
            # 兼容嵌套列表格式: [[0.37, 0.46, ...]] -> [0.37, 0.46, ...]
            if isinstance(vals, list) and len(vals) > 0 and isinstance(vals[0], list):
                vals = [v for sublist in vals for v in sublist]
            avg_val = sum(vals) / len(vals)
            
            # 计算时间跨度描述
            steps = len(vals)
            if steps == 4:
                time_desc = "未来 1 小时平均预测功率"
            elif steps == 8:
                time_desc = "未来 2 小时平均预测功率"
            else:
                time_desc = f"未来{steps}步平均预测功率"
            
            self.pred_value_label.setText(f"{avg_val:.2f} MW")
            self.pred_desc_label.setText(time_desc)
            self.pred_value_card.setStyleSheet("""
                QFrame { 
                    background-color: #e0f2f1; 
                    border-radius: 12px; 
                    padding: 15px;
                    border: 2px solid #009688;
                }
            """)
            
            # 绘制多点折线图
            self.plot_prediction(vals, f"{steps}步直接预测序列")
            self.append_log(f"详细多步序列：{[round(v, 2) for v in vals]}")
        
        # 🔧 新增：记录使用的模型和步长信息
        model_info = result.get("model_name", "未知模型")
        self.append_log(f"✓ 预测完成 | 模型: {model_info} | 步长: {result.get('steps', 1)}")

        QMessageBox.information(self, "成功", "预测任务已成功完成！")

    # 🔧 新增：Matplotlib 绘图方法
    def plot_prediction(self, values, title):
        """绘制预测结果的折线图（委托给ChartRenderer）"""
        canvas = self.chart_renderer.create_prediction_chart(values, title)
        
        # 如果是首次调用，需要将canvas嵌入到界面中
        if not self.chart_renderer.is_initialized():
            return
        
        # 替换占位符
        for child in self.main_widget.findChildren(QWidget):
            if hasattr(child, 'layout'):
                layout = child.layout()
                if layout and layout.count() > 0:
                    # 检查是否包含占位符
                    for i in range(layout.count()):
                        item = layout.itemAt(i)
                        if item.widget() == self.placeholder_widget:
                            # 在占位符的位置插入画布（保持标题在上方）
                            layout.removeWidget(self.placeholder_widget)
                            self.placeholder_widget.deleteLater()
                            layout.insertWidget(i, canvas, 3)  # 在原来占位符的位置插入
                            break

    def on_prediction_error(self, err_msg):
        self.reset_ui_state()
        self.progress_bar.hide()
        self.append_log(f"[异常终止] {err_msg}")
        QMessageBox.critical(self, "预测失败", f"计算错误:\n\n{err_msg}")

    def reset_ui_state(self):
        self.start_btn.setEnabled(True)
        self.start_btn.setText("🚀 开始智能预测")

    def on_login_success(self, username):
        self.current_username = username
        self.user_welcome_label.setText(f"欢迎研发工程师：{username}")
        self.stacked_widget.setCurrentIndex(1)
        self.append_log(f"研发账号 {username} 成功接入系统")

    def on_register_success(self, username):
        self.auth_stacked.setCurrentIndex(0)

    def open_analysis_window(self):
        """
        打开历史数据分析窗口
        1. 获取当前控制面板的场景和数据路径
        2. 传递给新窗口
        3. 后续在新窗口内的更改不影响控制面板
        """
        # 获取当前主控面板的值
        current_scene = self.type_combo.currentText()
        current_data_path = self.file_input.text()
        
        # 创建新窗口实例并传入数据
        # 注意：这里传递的是值的副本，新窗口内部的状态是独立的
        self.data_window = DataAnalysisWindow(init_scene=current_scene, init_data_path=current_data_path)
        self.data_window.show()



# ============================================
# 新增：历史数据分析窗口 (严格遵循设计方案.md)
# ============================================
class DataAnalysisWindow(QMainWindow):
    def __init__(self, init_scene="光伏功率预测", init_data_path=""):
        super().__init__()
        self.setWindowTitle("📊 历史光伏/风电数据分析")
        self.resize(ANALYSIS_WINDOW_WIDTH, ANALYSIS_WINDOW_HEIGHT) # 设计方案要求尺寸
        
        # 🚀 使用解耦的ChartRenderer
        self.chart_renderer = ChartRenderer()
        self.df_loaded = None
        self.current_scene = init_scene
        self.current_data_path = init_data_path  # 保存当前数据路径
        self.data_loader = DataLoader()  # 使用解耦的数据加载器
        
        # 🎨 设置背景图片
        bg_path = resource_path('./res/background.png')
        if os.path.exists(bg_path):
            self.setStyleSheet(f"""
                QMainWindow {{
                    border-image: url('{bg_path}') 0 0 0 0 stretch stretch;
                }}
            """)
        else:
            self.setStyleSheet("QMainWindow { background-color: #f5f5f5; }")
        
        self.init_ui(init_scene, init_data_path)

    def init_ui(self, init_scene, init_data_path):
        # 🎨 顶部控制栏
        top_bar = QFrame()
        top_bar.setStyleSheet("""
            QFrame {
                background-color: rgba(255, 255, 255, 0.98); 
                border-bottom: 2px solid #4caf50;
                border-radius: 8px;
                margin: 10px;
            }
        """)
        top_layout = QHBoxLayout(top_bar)

        # 🔧 新设计：数据路径显示（最左边）
        self.data_path_label = QLabel(f"📁 {os.path.basename(init_data_path) if init_data_path else '未选择数据文件'}")
        self.data_path_label.setStyleSheet("""
            QLabel {
                color: #37474f;
                font-size: 12px;
                padding: 5px 10px;
                background-color: #f5f5f5;
                border-radius: 4px;
                border: 1px solid #e0e0e0;
            }
        """)
        self.data_path_label.setMinimumWidth(300)
        top_layout.addWidget(self.data_path_label)

        # 🔧 新设计：数据粒度选择器
        granularity_label = QLabel("⏱️ 数据粒度:")
        granularity_label.setStyleSheet("color: #546e7a; font-size: 12px; margin-left: 10px;")

        self.granularity_combo = QComboBox()
        self.granularity_combo.addItems(["15分钟", "30分钟", "1小时", "5分钟"])
        self.granularity_combo.setCurrentText("15分钟")  # 默认15分钟
        # 🔧 设置字体避免警告
        from PySide6.QtGui import QFont
        font = QFont()
        font.setPointSize(12)
        self.granularity_combo.setFont(font)
        
        self.granularity_combo.setStyleSheet("""
            QComboBox {
                padding: 5px 10px; 
                background-color: white; 
                border: 1px solid #cfd8dc;
                border-radius: 4px;
                min-width: 100px;
            }
        """)

        top_layout.addWidget(granularity_label)
        top_layout.addWidget(self.granularity_combo)

        # 🔧 新设计：开始分析按钮
        start_analysis_btn = QPushButton("▶️ 开始分析")
        start_analysis_btn.setStyleSheet("""
            QPushButton {
                padding: 6px 20px;
                background-color: #1976d2;
                color: white;
                border-radius: 4px;
                font-weight: bold;
                font-size: 13px;
                margin-left: 10px;
            }
            QPushButton:hover { background-color: #1565c0; }
            QPushButton:pressed { background-color: #0d47a1; }
        """)
        start_analysis_btn.clicked.connect(self.start_analysis)
        top_layout.addWidget(start_analysis_btn)

        # 🔧 新设计：导出报告按钮
        export_btn = QPushButton("📥 导出分析报告")
        export_btn.setStyleSheet("""
            QPushButton {
                padding: 6px 15px;
                background-color: #2e7d32;
                color: white;
                border-radius: 4px;
                font-size: 12px;
                margin-left: 10px;
            }
            QPushButton:hover { background-color: #388e3c; }
        """)
        export_btn.clicked.connect(self.export_report)
        top_layout.addWidget(export_btn)

        top_layout.addStretch()  # 右侧留白

        # 🔧 新设计：进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #cfd8dc;
                border-radius: 4px;
                text-align: center;
                background-color: #f5f5f5;
                height: 20px;
            }
            QProgressBar::chunk {
                background-color: #1976d2;
                border-radius: 3px;
            }
        """)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)  # 初始隐藏

        # 主内容区 (左右布局)
        main_content = QFrame()
        main_content.setStyleSheet("QFrame { background: transparent; }")
        main_layout = QHBoxLayout(main_content)

        # 🔧 新设计：左侧图表区域 (60%)
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        # 左侧上方：功率曲线图
        self.chart_card = QFrame()
        self.chart_card.setObjectName("ChartCard")
        self.chart_card.setStyleSheet("""
            QFrame#ChartCard {
                background-color: rgba(255, 255, 255, 0.95);
                border-radius: 12px;
                border: 1px solid #e0f2f1;
                padding: 15px;
            }
        """)
        chart_layout = QVBoxLayout(self.chart_card)

        self.chart_title_label = QLabel("功率曲线分析")
        self.chart_title_label.setObjectName("ChartTitle")
        self.chart_title_label.setStyleSheet("font-weight: bold; font-size: 14px; color: #1b5e20; margin-bottom: 10px;")

        self.chart_placeholder = QLabel("请选择数据并点击'开始分析'")
        self.chart_placeholder.setObjectName("ChartPlaceholder")
        self.chart_placeholder.setAlignment(Qt.AlignCenter)
        self.chart_placeholder.setStyleSheet("color: #757575; font-size: 13px; padding: 50px;")

        chart_layout.addWidget(self.chart_title_label)
        chart_layout.addWidget(self.chart_placeholder)

        left_layout.addWidget(self.chart_card)

        # 🔧 新设计：右侧信息面板 (40%)
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        # 右上：月度日发电峰值
        self.peak_card = QFrame()
        self.peak_card.setObjectName("PeakCard")
        self.peak_card.setStyleSheet("""
            QFrame#PeakCard {
                background-color: rgba(255, 255, 255, 0.95);
                border-radius: 12px;
                border: 1px solid #e0f2f1;
                padding: 15px;
            }
        """)
        peak_layout = QVBoxLayout(self.peak_card)

        self.peak_title_label = QLabel("🏆 月度日发电峰值")
        self.peak_title_label.setObjectName("PeakTitle")
        self.peak_title_label.setStyleSheet("font-weight: bold; font-size: 14px; color: #1b5e20; margin-bottom: 10px;")

        self.peak_content_label = QLabel("等待分析...")
        self.peak_content_label.setObjectName("PeakContent")
        self.peak_content_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.peak_content_label.setStyleSheet("color: #37474f; font-size: 14px; line-height: 1.8;")
        self.peak_content_label.setWordWrap(True)

        peak_layout.addWidget(self.peak_title_label)
        peak_layout.addWidget(self.peak_content_label)

        # 右下：数据质量诊断
        self.quality_card = QFrame()
        self.quality_card.setObjectName("QualityCard")
        self.quality_card.setStyleSheet("""
            QFrame#QualityCard {
                background-color: rgba(255, 255, 255, 0.95);
                border-radius: 12px;
                border: 1px solid #e0f2f1;
                padding: 15px;
            }
        """)
        quality_layout = QVBoxLayout(self.quality_card)

        self.quality_title_label = QLabel("⚠️ 数据质量诊断")
        self.quality_title_label.setObjectName("QualityTitle")
        self.quality_title_label.setStyleSheet(
            "font-weight: bold; font-size: 14px; color: #1b5e20; margin-bottom: 10px;")

        self.quality_content_label = QLabel("等待分析...")
        self.quality_content_label.setObjectName("QualityContent")
        self.quality_content_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.quality_content_label.setStyleSheet("color: #37474f; font-size: 14px;")
        self.quality_content_label.setWordWrap(True)
        # 🔧 启用富文本渲染（支持HTML标签和换行）
        self.quality_content_label.setTextFormat(Qt.RichText)

        quality_layout.addWidget(self.quality_title_label)
        quality_layout.addWidget(self.quality_content_label)

        right_layout.addWidget(self.peak_card)
        right_layout.addWidget(self.quality_card)
        right_layout.setStretch(0, 1)
        right_layout.setStretch(1, 1)

        main_layout.addWidget(left_panel, 3)  # 左侧占60%
        main_layout.addWidget(right_panel, 2)  # 右侧占40%

        # 整体布局组装
        container = QWidget()
        container_layout = QVBoxLayout(container)
        container_layout.addWidget(top_bar)
        container_layout.addWidget(self.progress_bar)
        container_layout.addWidget(main_content)
        self.setCentralWidget(container)

        # 初始化逻辑（简化，不再调用update_content）
        self.current_scene = init_scene
        self.current_data_path = init_data_path

    def select_data_file(self):
        """在分析窗口中选择新的数据文件（不影响主窗口）"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择历史数据文件",
            "",
            FILE_FILTER_STRING
        )
        if file_path:
            self.current_data_path = file_path
            self.data_path_label.setText(f"📁 {os.path.basename(file_path)}")
            self.statusBar().showMessage(f"已加载: {os.path.basename(file_path)}", 3000)

            # 重置显示状态
            self.df_loaded = None
            self.chart_placeholder.setText("✅ 数据已加载，请点击'开始分析'按钮")
            self.peak_content_label.setText("等待分析...")
            self.quality_content_label.setText("等待分析...")

    def start_analysis(self):
        """点击'开始分析'按钮时执行分析"""
        if not self.current_data_path or not os.path.exists(self.current_data_path):
            QMessageBox.warning(self, "提示", "请先选择数据文件！")
            return

        try:
            # 显示进度条
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(10)
            self.statusBar().showMessage("正在加载数据...", 2000)

            # 🔧 使用解耦的DataLoader加载数据
            df = self.data_loader.load_file(self.current_data_path)
            self.progress_bar.setValue(30)

            self.df_loaded = df
            self.statusBar().showMessage("正在分析数据...", 2000)

            # 🚀 执行完整分析
            self.perform_full_analysis(df)

            self.progress_bar.setValue(100)
            self.statusBar().showMessage("✅ 分析完成！", 3000)

            # 延迟隐藏进度条
            from PySide6.QtCore import QTimer
            QTimer.singleShot(1000, lambda: self.progress_bar.setVisible(False))

        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "分析失败", f"数据分析出错：\n{str(e)}")
            self.statusBar().showMessage(f"❌ 分析失败: {str(e)}", 5000)
            self.progress_bar.setVisible(False)

    def perform_full_analysis(self, df):
        """执行完整的分析流程"""
        # 1. 确定功率列名
        power_col = None

        print(f"\n🔍 数据列名检测:")
        print(f"   场景: {self.current_scene}")
        print(f"   所有列: {list(df.columns)}")

        if "光伏" in self.current_scene:
            possible_cols = ['Power (MW)', 'Power(MW)', '实际发电功率（mw）', '功率(MW)', 'power']
        else:
            possible_cols = ['实际发电功率（mw）', '功率(MW)', 'Power (MW)', 'Power(MW)', 'power']

        for col in possible_cols:
            if col in df.columns:
                power_col = col
                print(f"   ✅ 找到功率列: '{col}'")
                break

        if not power_col:
            print(f"   ❌ 未找到功率列！可用列: {list(df.columns)}")
            error_msg = f"❌ 未找到功率列\n\n可用列:\n" + "\n".join([f"  • {col}" for col in df.columns[:10]])
            if len(df.columns) > 10:
                error_msg += f"\n  ... 等{len(df.columns)}列"

            self.quality_content_label.setText(error_msg)
            return

        # 2. 查找时间列
        time_col = None
        possible_time_cols = [
            '时间', 'Time', 'datetime', 'date', 'DateTime',
            '日期', 'Date', 'Timestamp', 'timestamp', '采样时间',
            '数据时间', '记录时间', 'TIME', 'DATE'
        ]

        for col in possible_time_cols:
            if col in df.columns:
                time_col = col
                break

        if not time_col:
            for col in df.columns:
                if '时间' in col or 'time' in col.lower() or 'date' in col.lower():
                    time_col = col
                    break

        if not time_col:
            print("⚠️ 未找到时间列，无法进行时间分析")
            self.chart_placeholder.setText("⚠️ 未找到时间列，无法进行分析")
            return

        # 3. 数据清洗
        df_clean = df.dropna(subset=[power_col]).copy()
        df_clean[time_col] = pd.to_datetime(df_clean[time_col], errors='coerce')
        df_clean = df_clean.dropna(subset=[time_col])
        missing_count = len(df) - len(df_clean)

        print(f"   📊 数据总量: {len(df)}, 有效数据: {len(df_clean)}, 缺失: {missing_count}")

        # 4. 绘制功率曲线（新逻辑）
        self.plot_power_curve(df_clean, power_col, time_col)

        # 5. 分析日发电峰值
        self.analyze_daily_peak(df_clean, power_col, time_col)

        # 6. 数据质量诊断（简化版）
        self.diagnose_quality_simple(df, df_clean, power_col, missing_count)

    def plot_power_curve(self, df, power_col, time_col):
        """绘制两张功率曲线图：近30天日曲线 + 近24小时小时曲线"""
        try:
            # 🔧 根据粒度获取重采样频率
            granularity_text = self.granularity_combo.currentText()
            granularity_map = {
                "5分钟": "5min",
                "15分钟": "15min",
                "30分钟": "30min",
                "1小时": "1h"
            }
            freq = granularity_map.get(granularity_text, "15min")
                
            # 设置时间列为索引
            df_indexed = df.set_index(time_col)
            
            print(f"🔍 数据时间范围: {df_indexed.index.min()} ~ {df_indexed.index.max()}")
            print(f" 数据总量: {len(df_indexed)} 条")
            
            # 🔧 初始化图表渲染器并清除（只清除一次）
            self.chart_renderer._ensure_matplotlib()
            self.chart_renderer.figure.clear()
                
            # 📊 图1：近30天的日功率曲线
            max_time = df_indexed.index.max()
            min_time_30d = max_time - pd.Timedelta(days=30)
            df_last_30_days = df_indexed[df_indexed.index >= min_time_30d]
            
            print(f" 近30天数据量: {len(df_last_30_days)} 条")
            
            has_daily = False
            has_hourly = False
            
            if len(df_last_30_days) > 0:
                df_daily = df_last_30_days[[power_col]].resample('1D').sum()
                # 移除空值
                df_daily = df_daily.dropna()
                # 取最近30天
                df_daily_30 = df_daily.tail(30)
                
                print(f"🔍 日聚合后数据量: {len(df_daily_30)} 天")
                    
                dates = df_daily_30.index.strftime('%m-%d').tolist()
                x_ticks_daily = list(range(len(dates)))
                x_labels_daily = dates
                    
                self._render_power_chart_subplot(
                    df_daily_30, power_col, 
                    "近30天日功率曲线", 
                    "日期", 
                    x_ticks_daily, x_labels_daily,
                    chart_index=0  # 第一个图表
                )
                has_daily = True
                print(f"✅ 日曲线绘制完成: {len(df_daily_30)}个点")
            else:
                print("⚠️ 近30天无数据，跳过日曲线绘制")
                
            #  图2：近24小时的小时功率曲线
            min_time_24h = max_time - pd.Timedelta(hours=24)
            df_last_24h = df_indexed[df_indexed.index >= min_time_24h]
            
            print(f"🔍 近24小时数据量: {len(df_last_24h)} 条")
            
            if len(df_last_24h) > 0:
                df_hourly = df_last_24h[[power_col]].resample('1h').sum()
                # 移除空值
                df_hourly = df_hourly.dropna()
                
                print(f"🔍 小时聚合后数据量: {len(df_hourly)} 小时")
                
                # 🔧 根据实际数据生成X轴标签（不固定为0-23点）
                if len(df_hourly) > 0:
                    actual_times = df_hourly.index.strftime('%H:%M').tolist()
                    
                    x_ticks_hourly = list(range(len(actual_times)))
                    x_labels_hourly = actual_times
                    
                    # 如果数据点较多，每隔几个显示一个标签
                    if len(x_labels_hourly) > 12:
                        step = len(x_labels_hourly) // 12
                        display_labels = ['' if i % step != 0 else label for i, label in enumerate(x_labels_hourly)]
                        x_labels_hourly = display_labels
                    
                    self._render_power_chart_subplot(
                        df_hourly, power_col,
                        "近24小时功率曲线",
                        "时间",
                        x_ticks_hourly, x_labels_hourly,
                        chart_index=1  # 第二个图表
                    )
                    has_hourly = True
                    print(f"✅ 小时曲线绘制完成: {len(df_hourly)}个点")
                else:
                    print("️ 小时聚合后无数据")
            else:
                print("⚠️ 近24小时无数据，跳过小时曲线绘制")
            
            # 🔧 调整布局并嵌入界面（只在有图表时执行）
            if has_daily or has_hourly:
                self.chart_renderer.figure.tight_layout(pad=2.0, rect=[0, 0.03, 1, 0.97])
                self.chart_renderer.canvas.draw()
                
                # 嵌入到界面
                old_layout = self.chart_card.layout()
                if old_layout:
                    while old_layout.count():
                        item = old_layout.takeAt(0)
                        if item.widget():
                            item.widget().deleteLater()
                else:
                    old_layout = QVBoxLayout(self.chart_card)
                
                old_layout.addWidget(self.chart_title_label)
                old_layout.addWidget(self.chart_renderer.canvas)
                print("✅ 图表已嵌入界面")
                
        except Exception as e:
            print(f"⚠️ 功率曲线绘制失败: {e}")
            import traceback
            traceback.print_exc()

    def _render_power_chart_subplot(self, df, power_col, title, x_label, x_ticks, x_labels, chart_index=0):
        """渲染单个功率柱状图子图（不执行clear和布局调整）"""
        # 创建子图布局：如果是第一个图表，显示在上方；第二个图表显示在下方
        if chart_index == 0:
            ax = self.chart_renderer.figure.add_subplot(211)  # 2行1列，第1个
        else:
            ax = self.chart_renderer.figure.add_subplot(212)  # 2行1列，第2个
            
        x = range(len(df))
        y = df[power_col].values
            
        # 绘制柱状图
        from gui_config import CHART_COLOR_BAR
        ax.bar(x, y, color=CHART_COLOR_BAR, alpha=0.7, edgecolor='#2e7d32', linewidth=2)
            
        # 设置x轴刻度
        if len(x_ticks) <= 31:  # 如果点数不多，显示所有标签
            ax.set_xticks(x_ticks)
            ax.set_xticklabels(x_labels, rotation=45, ha='right')
            
        # 使用纯文本标题
        ax.set_title(title, fontsize=12, fontweight='bold', color='#2c3e50', pad=10)
        ax.set_xlabel(x_label, fontsize=10, color='#546e7a', labelpad=8)
        ax.set_ylabel('功率 (MW)', fontsize=10, color='#546e7a', labelpad=8)
        ax.grid(True, linestyle='--', alpha=0.6, color='#e0e0e0')
        ax.set_facecolor('#f5f5f5')

    def analyze_daily_peak(self, df, power_col, time_col):
        """分析月度日发电峰值"""
        try:
            # 按日汇总发电量
            df_copy = df.copy()
            df_copy['date'] = df_copy[time_col].dt.date
            daily_generation = df_copy.groupby('date')[power_col].sum()

            if len(daily_generation) == 0:
                self.peak_content_label.setText("⚠️ 无有效数据")
                return

            # 找出最高日
            peak_date = daily_generation.idxmax()
            peak_value = daily_generation.max()

            # 计算日均
            avg_daily = daily_generation.mean()

            # 找出峰值功率时刻
            peak_day_data = df_copy[df_copy[time_col].dt.date == peak_date]
            if len(peak_day_data) > 0:
                peak_power_idx = peak_day_data[power_col].idxmax()
                peak_power_time = peak_day_data.loc[peak_power_idx, time_col].strftime('%H:%M')
                peak_power_value = peak_day_data.loc[peak_power_idx, power_col]
            else:
                peak_power_time = "N/A"
                peak_power_value = 0

            result_text = (
                f"📅 最高发电日: {peak_date}\n"
                f"⚡ 当日发电量: {peak_value:.2f} MW\n"
                f"📊 日均发电量: {avg_daily:.2f} MW\n"
                f"🔝 峰值功率: {peak_power_value:.2f} MW ({peak_power_time})"
            )

            self.peak_content_label.setText(result_text)

        except Exception as e:
            print(f"⚠️ 日峰值分析失败: {e}")
            self.peak_content_label.setText(f"⚠️ 分析失败: {str(e)}")

    def diagnose_quality_simple(self, df_original, df_clean, power_col, missing_count):
        """简化版数据质量诊断"""
        total_rows = len(df_original)
        valid_rows = len(df_clean)
        missing_pct = (missing_count / total_rows * 100) if total_rows > 0 else 0

        # 检测异常值（超过3倍标准差）
        mean_val = df_clean[power_col].mean()
        std_val = df_clean[power_col].std()
        outlier_mask = (df_clean[power_col] < mean_val - 3 * std_val) | (df_clean[power_col] > mean_val + 3 * std_val)
        outlier_count = outlier_mask.sum()
        outlier_pct = (outlier_count / valid_rows * 100) if valid_rows > 0 else 0

        # 评估数据质量等级
        if missing_pct < 1 and outlier_pct < 1:
            quality_level = "✅ 优秀"
            quality_color = "#2e7d32"
        elif missing_pct < 5 and outlier_pct < 5:
            quality_level = "⚠️ 良好"
            quality_color = "#f57c00"
        else:
            quality_level = "❌ 需改进"
            quality_color = "#d32f2f"

        result_text = (
            f"<p style='margin: 16px 0;'>📋 数据总量: {total_rows} 条</p>"
            f"<p style='margin: 16px 0;'>✅ 有效数据: {valid_rows} 条 ({100 - missing_pct:.1f}%)</p>"
            f"<p style='margin: 16px 0;'>❌ 缺失数据: {missing_count} 条 ({missing_pct:.1f}%)</p>"
            f"<p style='margin: 16px 0;'>⚠️ 异常值: {outlier_count} 条 ({outlier_pct:.1f}%)</p>"
            f"<hr style='margin: 10px 0; border: 1px solid #e0e0e0;'>"
            f"<p style='margin: 16px 0; font-weight: bold;'>🎯 质量评级: {quality_level}</p>"
        )
                
        self.quality_content_label.setText(result_text)
    
    def reset_analysis(self):
        """重置分析视图"""
        if self.df_loaded is not None:
            self.perform_full_analysis(self.df_loaded)
            self.statusBar().showMessage("视图已重置", 2000)
        else:
            self.statusBar().showMessage("没有可重置的数据", 2000)
    
    def export_report(self):
        """导出分析报告"""
        if self.df_loaded is None:
            QMessageBox.warning(self, "提示", "请先加载数据进行分析！")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, 
            "导出分析报告", 
            "",
            "CSV Files (*.csv);;Excel Files (*.xlsx);;All Files (*)"
        )
        
        if file_path:
            try:
                ext = os.path.splitext(file_path)[1].lower()
                if ext == '.csv':
                    self.df_loaded.to_csv(file_path, index=False, encoding='utf-8-sig')
                elif ext == '.xlsx':
                    self.df_loaded.to_excel(file_path, index=False, engine='openpyxl')
                
                QMessageBox.information(self, "成功", f"分析报告已导出至:\n{file_path}")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"导出失败: {str(e)}")


if __name__ == "__main__":
    app = QApplication(sys.argv)

    # 设置应用图标（使用资源路径工具）
    icon_path = resource_path(ICON_PATH)
    if os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path))

    window = EnergyForecastApp()

    # 使用背景图片（使用资源路径工具）
    bg_path = resource_path(BACKGROUND_PATH)
    if os.path.exists(bg_path):
        # 🔧 关键修复：使用 border-image 代替 background-image，它能更好地处理拉伸和路径
        window.setStyleSheet(f"""
            QMainWindow {{
                border-image: url('{bg_path}') 0 0 0 0 stretch stretch;
            }}
        """)


    window.show()
    sys.exit(app.exec())