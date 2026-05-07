"""
图表渲染器模块
负责Matplotlib图表的创建、配置和绘制
完全独立于GUI，可单独测试和复用
"""

from typing import List, Optional
import numpy as np
from gui_config import (
    CHART_FIGURE_SIZE,
    CHART_DPI,
    CHART_COLOR_LINE,
    CHART_COLOR_FILL,
    CHART_COLOR_BAR,
    CHART_COLOR_GRID,
    CHART_ALPHA
)


class ChartRenderer:
    """
    统一图表渲染器
    支持单步预测（柱状图）和多步预测（折线图）
    """
    
    def __init__(self):
        self.figure = None
        self.canvas = None
        self._matplotlib_imported = False
    
    def _ensure_matplotlib(self):
        """延迟加载Matplotlib"""
        if not self._matplotlib_imported:
            from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
            from matplotlib.figure import Figure
            from matplotlib import rcParams
            
            # 配置中文字体和负号显示
            rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei']
            rcParams['axes.unicode_minus'] = False
            
            # 创建Figure和Canvas
            self.figure = Figure(figsize=CHART_FIGURE_SIZE, dpi=CHART_DPI, facecolor='#ffffff')
            self.canvas = FigureCanvas(self.figure)
            
            self._matplotlib_imported = True
    
    def create_prediction_chart(self, values: List[float], title: str = "预测结果"):
        """
        创建预测结果图表
        
        Args:
            values: 预测值列表
            title: 图表标题
            
        Returns:
            Canvas对象（首次调用时创建，后续复用）
        """
        self._ensure_matplotlib()
        
        # 清除旧图表
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        # 生成x轴（时间点）
        x = list(range(1, len(values) + 1))
        
        # 根据数据长度选择图表类型
        if len(values) == 1:
            # 单步预测：用柱状图
            self._plot_single_step(ax, x, values)
        else:
            # 多步预测：用折线图
            self._plot_multi_step(ax, x, values)
        
        # 设置标题和标签
        ax.set_title(title, fontsize=14, fontweight='bold', color='#2c3e50', pad=15)
        ax.set_xlabel('预测步长', fontsize=11, color='#546e7a', labelpad=10)
        ax.set_ylabel('功率 (MW)', fontsize=11, color='#546e7a', labelpad=10)
        
        # 添加网格
        ax.grid(True, linestyle='--', alpha=0.6, color=CHART_COLOR_GRID)
        
        # 设置背景色
        ax.set_facecolor('#f5f5f5')
        self.figure.patch.set_facecolor('#ffffff')
        
        # 调整布局（增加边距确保标签完整显示）
        self.figure.tight_layout(pad=2.0, rect=[0, 0.05, 1, 0.95])
        
        # 刷新画布
        self.canvas.draw()
        
        return self.canvas
    
    def _plot_single_step(self, ax, x: List[int], values: List[float]):
        """绘制单步预测柱状图"""
        ax.bar(x, values, color=CHART_COLOR_BAR, alpha=0.7, 
               edgecolor='#2e7d32', linewidth=2)
        ax.set_xticks(x)
        ax.set_xticklabels(['下一时刻'])
    
    def _plot_multi_step(self, ax, x: List[int], values: List[float]):
        """绘制多步预测折线图"""
        ax.plot(x, values, marker='o', linestyle='-', color=CHART_COLOR_LINE, 
                linewidth=2, markersize=8, markerfacecolor='#ffffff', 
                markeredgewidth=2, markeredgecolor=CHART_COLOR_LINE)
        ax.fill_between(x, values, alpha=CHART_ALPHA, color=CHART_COLOR_FILL)
        
        # 设置x轴标签
        step_labels = [f'T+{i}' for i in x]
        ax.set_xticks(x)
        ax.set_xticklabels(step_labels, rotation=45, ha='right')
    
    def create_time_series_chart(self, df, power_col: str, 
                                 sample_size: Optional[int] = None,
                                 title: str = "功率变化趋势"):
        """
        创建时序曲线图（用于历史数据分析）
        
        Args:
            df: DataFrame对象（已筛选后的数据）
            power_col: 功率列名
            sample_size: 采样点数（None表示不限制，显示全部数据）
            title: 图表标题
            
        Returns:
            Canvas对象
        """
        self._ensure_matplotlib()
        
        # 清除旧图表
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        # 🔧 修复：如果sample_size为None，显示全部数据；否则取最近N个点
        if sample_size is not None:
            actual_size = min(sample_size, len(df))
            df_plot = df.tail(actual_size)
        else:
            actual_size = len(df)
            df_plot = df
        
        x = range(len(df_plot))
        y = df_plot[power_col].values
        
        # 绘制曲线
        ax.plot(x, y, color=CHART_COLOR_LINE, linewidth=1.5, alpha=0.8)
        ax.fill_between(x, y, alpha=CHART_ALPHA, color=CHART_COLOR_FILL)
        
        # 设置标题和标签
        if sample_size is not None:
            title_text = f'最近{actual_size}个时间点{title}'
        else:
            title_text = f'{title} (共{actual_size}个数据点)'
        
        ax.set_title(title_text, 
                     fontsize=12, fontweight='bold', color='#2c3e50', pad=10)
        ax.set_xlabel('时间点', fontsize=10, color='#546e7a', labelpad=8)
        ax.set_ylabel('功率 (MW)', fontsize=10, color='#546e7a', labelpad=8)
        ax.grid(True, linestyle='--', alpha=0.5, color=CHART_COLOR_GRID)
        ax.set_facecolor('#f5f5f5')
        
        # 调整布局
        self.figure.tight_layout(pad=1.5, rect=[0, 0.03, 1, 0.97])
        self.canvas.draw()
        
        return self.canvas
    
    def get_canvas_widget(self):
        """获取Canvas Widget（用于嵌入到Qt布局中）"""
        self._ensure_matplotlib()
        return self.canvas
    
    def is_initialized(self) -> bool:
        """检查是否已初始化"""
        return self._matplotlib_imported
