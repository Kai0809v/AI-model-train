"""
GUI配置常量模块
集中管理所有硬编码的配置项，便于维护和扩展
"""

# ============================================
# 应用元信息
# ============================================
APP_NAME = "综合能源预测系统"
APP_VERSION = "V1.6"
APP_DESCRIPTION = "智能预测引擎"

# ============================================
# 窗口尺寸配置
# ============================================
MAIN_WINDOW_WIDTH = 1000
MAIN_WINDOW_HEIGHT = 600
ANALYSIS_WINDOW_WIDTH = 1200
ANALYSIS_WINDOW_HEIGHT = 800

# ============================================
# 预测场景配置
# ============================================
PREDICTION_SCENARIOS = {
    "风电功率预测": {
        "models": {
            "CEEMDAN-LGBM-Transformer": "CEEMDAN_LGBM_Transformer",
        },
        "steps_options": [
            "下一时刻（单步）",
            "一小时（4 步）",
            "两小时（8 步）"
        ],
        "min_data_rows": 96,
        "time_interval_minutes": 15
    },
    "光伏功率预测": {
        "models": {
            "BP-TCN-Informer（有未来气象数据）": "PV_TCN_Informer",
            "BP-TCN-Informer（无未来气象数据）": "PV_TCN_Informer_NoWeather",
        },
        "steps_options": [
            "下一时刻（1 步）",
            "一小时（4 步）",
            "两小时（8 步）",
            "三小时（12 步）",
            "六小时（24 步）",
        ],
        "min_data_rows": 192,
        "time_interval_minutes": 15
    },
    "电网负荷预测（后续更新）": {
        "models": {},
        "steps_options": [],
        "min_data_rows": 0,
        "time_interval_minutes": 15
    }
}

# ============================================
# 文件支持格式
# ============================================
SUPPORTED_FILE_FORMATS = {
    ".csv": "CSV Files (*.csv)",
    ".xlsx": "Excel Files (*.xlsx)",
    ".xls": "Excel Files (*.xls)"
}

FILE_FILTER_STRING = "数据文件 (*.csv *.xlsx *.xls);;CSV Files (*.csv);;Excel Files (*.xlsx *.xls);;All Files (*)"

# ============================================
# 数据列名映射
# ============================================
# 光伏功率列名候选
PV_POWER_COLUMNS = ['Power (MW)', 'Power(MW)', '实际发电功率（mw）', '功率(MW)', 'power']

# 风电功率列名候选
WIND_POWER_COLUMNS = ['实际发电功率（mw）', '功率(MW)', 'Power (MW)', 'Power(MW)', 'power']

# 时间列名候选
TIME_COLUMNS = [
    '时间', 'Time', 'datetime', 'date', 'DateTime', 
    '日期', 'Date', 'Timestamp', 'timestamp', '采样时间', 
    '数据时间', '记录时间', 'TIME', 'DATE'
]

# ============================================
# UI样式配置
# ============================================
# 主题色
COLOR_PRIMARY = "#2e7d32"      # 主绿色
COLOR_PRIMARY_HOVER = "#388e3c"
COLOR_PRIMARY_PRESSED = "#1b5e20"
COLOR_SECONDARY = "#00897b"    # 次要青色
COLOR_SECONDARY_HOVER = "#00796b"

# 背景色
COLOR_BG_LIGHT = "#f1f8e9"     # 浅绿背景
COLOR_BG_CARD = "rgba(255, 255, 255, 0.95)"
COLOR_BG_LOG = "rgba(38, 50, 56, 0.95)"

# 边框色
COLOR_BORDER = "#cfd8dc"
COLOR_BORDER_GREEN = "#c5e1a5"

# 文字色
COLOR_TEXT_PRIMARY = "#2c3e50"
COLOR_TEXT_SECONDARY = "#37474f"
COLOR_TEXT_HINT = "#78909c"
COLOR_TEXT_SUCCESS = "#69f0ae"
COLOR_TEXT_ERROR = "#d32f2f"

# ============================================
# 图表配置
# ============================================
CHART_FIGURE_SIZE = (8, 6)
CHART_DPI = 100
CHART_MIN_WIDTH = 600
CHART_MIN_HEIGHT = 400

# 图表颜色
CHART_COLOR_LINE = "#00897b"
CHART_COLOR_FILL = "#00897b"
CHART_COLOR_BAR = "#4caf50"
CHART_COLOR_GRID = "#b0bec5"
CHART_ALPHA = 0.3

# ============================================
# 物理约束
# ============================================
PV_MAX_CAPACITY_MW = 130.0     # 光伏最大装机容量
WIND_CUT_IN_SPEED = 3.0        # 风电切入风速 (m/s)
NIGHT_THRESHOLD_MW = 0.05      # 夜间功率阈值

# ============================================
# 资源路径
# ============================================
ICON_PATH = "./res/icon.png"
BACKGROUND_PATH = "./res/background.png"
