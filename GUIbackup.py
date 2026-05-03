import sys
import pandas as pd
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                               QHBoxLayout, QLabel, QComboBox, QPushButton,
                               QLineEdit, QFileDialog, QTextEdit, QFrame,
                               QProgressBar, QMessageBox, QStackedWidget,
                               QGridLayout, QSpacerItem, QSizePolicy)
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QPixmap, QIcon
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# 导入预测服务接口
from api_v6 import ForecastService


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

        subtitle = QLabel("V0.2 · 智能预测引擎")
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

    def __init__(self, service, file_path, model_name, steps=1):
        super().__init__()
        self.service = service
        self.file_path = file_path
        self.model_name = model_name
        self.steps = steps

    def run(self):
        try:
            self.progress.emit(10, "正在读取 CSV 历史数据...")
            try:
                df = pd.read_csv(self.file_path)
            except Exception as e:
                self.error.emit(f"文件读取失败，请检查文件格式。\n详情：{str(e)}")
                return

            self.progress.emit(30, f"数据读取成功，共 {len(df)} 行。正在校验数据...")
            self.progress.emit(50, f"正在加载模型 [{self.model_name}]...")
            self.progress.emit(70, "模型就绪，正在执行核心预测算法...")

            result = self.service.run(self.model_name, df, self.steps)

            if result.get("success"):
                self.progress.emit(100, "✅ 预测计算完成！")
                self.finished.emit(result)
            else:
                self.error.emit(result.get("error", "未知预测错误"))
        except Exception as e:
            self.error.emit(f"预测过程中发生严重系统异常：{str(e)}")


# ============================================
# 主 GUI 界面
# ============================================
class EnergyForecastApp(QMainWindow):
    def __init__(self):
        super().__init__()
        # 调整了窗口尺寸，高度不再过于冗长
        self.setWindowTitle("综合能源预测系统 V0.2")
        self.resize(1000, 600)

        self.current_username = None
        self.forecast_service = ForecastService()
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

        # 预测场景标签
        scene_label = QLabel("预测场景:")
        scene_label.setStyleSheet("background-color: transparent; color: #37474f; font-size: 13px;")
        left_layout.addWidget(scene_label)
        
        self.type_combo = QComboBox()
        self.type_combo.addItems(["风电功率预测", "光伏功率预测", "电网负荷预测"])
        style_combo(self.type_combo)
        left_layout.addWidget(self.type_combo)

        # 历史数据标签
        data_label = QLabel("历史数据 (CSV):")
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

        # 算法模型标签
        model_label = QLabel("算法模型:")
        model_label.setStyleSheet("background-color: transparent; color: #37474f; font-size: 13px;")
        left_layout.addWidget(model_label)
        
        self.model_combo = QComboBox()
        self.model_map = {
            "CEEMDAN-LGBM-Transformer": "CEEMDAN_LGBM_Transformer",
        }
        self.model_combo.addItems(list(self.model_map.keys()))
        style_combo(self.model_combo)
        left_layout.addWidget(self.model_combo)

        # 预测步长标签
        steps_label = QLabel("预测步长:")
        steps_label.setStyleSheet("background-color: transparent; color: #37474f; font-size: 13px;")
        left_layout.addWidget(steps_label)
        
        self.steps_combo = QComboBox()
        self.steps_combo.addItems([
            "下一时刻（单步）",
            "一小时（4 步）",
            "两小时（8 步）",
            "四小时（16 步）"
        ])
        style_combo(self.steps_combo)
        left_layout.addWidget(self.steps_combo)

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

        # ================= 右侧展示区 - 🔧 重构为图表显示区域 =================
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
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
        from matplotlib.figure import Figure
        
        self.figure = Figure(figsize=(8, 6), dpi=100, facecolor='#ffffff')
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setStyleSheet("""
            background-color: rgba(255, 255, 255, 0.95); 
            border: 2px solid #b2dfdb; 
            border-radius: 12px;
        """)
        self.canvas.setMinimumSize(600, 400)
        right_layout.addWidget(self.canvas, stretch=3)

        # 🔧 当前预测值显示卡片
        self.pred_value_card = QFrame()
        self.pred_value_card.setStyleSheet("""
            QFrame { 
                background-color: rgba(255, 255, 255, 0.95); 
                border-radius: 12px; 
                padding: 15px;
                border: 2px solid #4caf50;
            }
        """)
        pred_layout = QVBoxLayout(self.pred_value_card)
        pred_layout.setAlignment(Qt.AlignCenter)
        
        self.pred_value_label = QLabel("-- MW")
        self.pred_value_label.setStyleSheet(
            "font-size: 36px; font-weight: bold; color: #2e7d32; background: transparent;")
        self.pred_value_label.setAlignment(Qt.AlignCenter)
        pred_layout.addWidget(self.pred_value_label)
        
        self.pred_desc_label = QLabel("等待预测...")
        self.pred_desc_label.setStyleSheet(
            "font-size: 14px; color: #78909c; background: transparent;")
        self.pred_desc_label.setAlignment(Qt.AlignCenter)
        pred_layout.addWidget(self.pred_desc_label)
        
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

    def append_log(self, text):
        self.log_output.append(f"⚡ {text}")
        scrollbar = self.log_output.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def load_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择数据文件", "", "CSV Files (*.csv);;All Files (*)")
        if file_path:
            self.file_input.setText(file_path)
            self.append_log(f"已装载历史数据：{file_path}")

    def run_prediction(self):
        file_path = self.file_input.text()
        if not file_path:
            QMessageBox.warning(self, "提示", "请先选择历史数据文件！")
            return

        energy_type = self.type_combo.currentText()
        if "风电" not in energy_type:
            QMessageBox.information(self, "提示", f"当前 V0.2 版本专注【风电功率预测】，{energy_type} 敬请期待！")
            return

        display_model = self.model_combo.currentText()
        backend_model_name = self.model_map.get(display_model)
        
        # 🔧 修改：根据新的选项映射步数
        steps_text = self.steps_combo.currentText()
        if "单步" in steps_text:
            steps = 1
        elif "4 步" in steps_text:
            steps = 4
        elif "8 步" in steps_text:
            steps = 8
        elif "16 步" in steps_text:
            steps = 16
        else:
            steps = 1

        self.start_btn.setEnabled(False)
        self.start_btn.setText("⏳ 模型推理中...")
        self.progress_bar.show()
        self.progress_bar.setValue(0)
        self.append_log("-" * 40)
        self.append_log(f"启动新任务：{display_model} (预测步长：{steps})")

        self.worker = PredictionWorker(self.forecast_service, file_path, backend_model_name, steps)
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
            avg_val = sum(vals) / len(vals)
            
            # 计算时间跨度描述
            steps = len(vals)
            if steps == 4:
                time_desc = "未来 1 小时平均预测功率"
            elif steps == 8:
                time_desc = "未来 2 小时平均预测功率"
            elif steps == 16:
                time_desc = "未来 4 小时平均预测功率"
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
            self.plot_prediction(vals, f"{steps}步滚动预测序列")
            self.append_log(f"详细多步序列：{[round(v, 2) for v in vals]}")

        QMessageBox.information(self, "成功", "预测任务已成功完成！")

    # 🔧 新增：Matplotlib 绘图方法
    def plot_prediction(self, values, title):
        """绘制预测结果的折线图"""
        from matplotlib import rcParams
        
        # 🔧 配置中文字体和负号显示
        rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei']
        rcParams['axes.unicode_minus'] = False
        
        self.figure.clear()
        
        ax = self.figure.add_subplot(111)
        
        # 生成 x 轴（时间点）
        x = list(range(1, len(values) + 1))
        
        # 绘制折线图
        if len(values) == 1:
            # 单步预测：用柱状图
            bars = ax.bar(x, values, color='#4caf50', alpha=0.7, edgecolor='#2e7d32', linewidth=2)
            ax.set_xticks(x)
            ax.set_xticklabels(['下一时刻'])
        else:
            # 多步预测：用折线图
            ax.plot(x, values, marker='o', linestyle='-', color='#00897b', 
                   linewidth=2, markersize=8, markerfacecolor='#ffffff', 
                   markeredgewidth=2, markeredgecolor='#00897b')
            ax.fill_between(x, values, alpha=0.3, color='#00897b')
            
            # 设置 x 轴标签
            step_labels = [f'T+{i}' for i in x]
            ax.set_xticks(x)
            ax.set_xticklabels(step_labels, rotation=45, ha='right')
        
        # 设置标题和标签
        ax.set_title(title, fontsize=14, fontweight='bold', color='#2c3e50', pad=15)
        ax.set_xlabel('预测步长', fontsize=11, color='#546e7a')
        ax.set_ylabel('功率 (MW)', fontsize=11, color='#546e7a')
        
        # 添加网格
        ax.grid(True, linestyle='--', alpha=0.6, color='#b0bec5')
        
        # 设置背景色
        ax.set_facecolor('#f5f5f5')
        self.figure.patch.set_facecolor('#ffffff')
        
        # 调整布局
        self.figure.tight_layout()
        
        # 刷新画布
        self.canvas.draw()

    def on_prediction_error(self, err_msg):
        self.reset_ui_state()
        self.progress_bar.hide()
        self.append_log(f"[异常终止] {err_msg}")
        QMessageBox.critical(self, "预测失败", f"计算错误:\n\n{err_msg}")
        self.chart_frame.setText("预测中断，请检查数据异常。")
        self.chart_frame.setStyleSheet(
            "background-color: #ffebee; border: 2px solid #f44336; border-radius: 12px; font-size: 20px; color: #c62828;")

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


if __name__ == "__main__":
    app = QApplication(sys.argv)

    # 设置应用图标（后续替换）
    app.setWindowIcon(QIcon("./res/icon.png"))

    window = EnergyForecastApp()

#======================================================================
    # 应用代表“清洁能源与自然生态”的现代渐变背景
    window.setStyleSheet("""
        QMainWindow {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 #00b4db, stop:0.5 #0083b0, stop:1 #005c97); /* 深空蓝向青绿渐变 */
        }
    """)

    # 方式 2: 使用背景图片（取消下面的注释并替换路径）
    # window.setStyleSheet(f"""
    #     QMainWindow {{
    #         background-image: url('background.jpg');
    #         background-position: center;
    #         background-repeat: no-repeat;
    #         background-size: cover;
    #     }
    # """)
# ======================================================================

    window.show()
    sys.exit(app.exec())