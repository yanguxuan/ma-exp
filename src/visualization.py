"""
可视化工具模块 - 统一的中文字体和美观样式配置
"""
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
from pathlib import Path

# ============================================================
# 中文字体配置
# ============================================================

def _find_chinese_font():
    """自动查找系统中可用中文字体"""
    # 优先级顺序：Noto Sans SC > 微软雅黑 > 宋体 > 楷体
    preferred = [
        "Noto Sans SC",
        "Microsoft YaHei",
        "SimHei",
        "KaiTi",
        "SimSun",
        "FangSong",
    ]
    # 获取系统所有字体
    system_fonts = {f.name: f.fname for f in fm.fontManager.ttflist}
    
    for font_name in preferred:
        if font_name in system_fonts:
            return font_name
    
    # 兜底：查找包含中文的字体
    for f in fm.fontManager.ttflist:
        if any(kw in f.name for kw in ["CJK", "Chinese", "SC", "YaHei", "Gothic"]):
            return f.name
    
    return "DejaVu Sans"  # 最终兜底

# 查找并设置中文字体
_CHINESE_FONT = _find_chinese_font()

# Matplotlib 全局配置
def setup_matplotlib():
    """配置 matplotlib 全局样式（中文字体 + 美观主题）"""
    plt.rcParams.update({
        # 字体配置
        "font.family": "sans-serif",
        "font.sans-serif": [_CHINESE_FONT, "DejaVu Sans"],
        "axes.unicode_minus": False,  # 解决负号显示问题
        
        # 全局尺寸
        "figure.dpi": 150,
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 16,
        
        # 颜色循环（色盲友好）
        "axes.prop_cycle": plt.cycler(
            "color",
            ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860", "#DA8BC3", "#8C8C8C"]
        ),
        
        # 网格
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "grid.linewidth": 0.8,
        
        # 线条
        "lines.linewidth": 2.0,
        "lines.markersize": 6,
        
        # 图例
        "legend.framealpha": 0.9,
        "legend.edgecolor": "#CCCCCC",
    })
    
    return _CHINESE_FONT

# ============================================================
# 颜色主题
# ============================================================

# 美观配色方案（参考 Tableau / Seaborn 风格）
COLORS = {
    "blue": "#4C72B0",
    "orange": "#DD8452",
    "green": "#55A868",
    "red": "#C44E52",
    "purple": "#8172B3",
    "brown": "#937860",
    "pink": "#DA8BC3",
    "gray": "#8C8C8C",
    
    # 语义色
    "primary": "#4C72B0",
    "success": "#55A868",
    "warning": "#DD8452",
    "danger": "#C44E52",
    "info": "#8172B3",
}

# 算法对比配色
ALGO_COLORS = ["#4C72B0", "#C44E52", "#55A868", "#DD8452", "#8172B3"]

# 浅色背景
BG_COLOR = "#F7F9FC"
GRID_COLOR = "#E8ECF1"

# ============================================================
# 美观工具函数
# ============================================================

def apply_chinese_font(ax=None):
    """确保当前图表使用中文字体"""
    import matplotlib.pyplot as plt
    if ax is None:
        ax = plt.gca()
    plt.rcParams["font.sans-serif"] = [_CHINESE_FONT, "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def set_spine_style(ax, hide_top_right=True):
    """美化坐标轴样式"""
    ax.spines[["top", "right"]].set_visible(not hide_top_right)
    ax.spines[["left", "bottom"]].set_color("#AAAAAA")
    ax.spines[["left", "bottom"]].set_linewidth(1.2)
    ax.tick_params(axis="both", colors="#555555")
    ax.yaxis.label.set_color("#333333")
    ax.xaxis.label.set_color("#333333")
    ax.title.set_color("#222222")


def add_value_labels(ax, bars=None, fmt="{:.1f}", offset=3, color="#333333"):
    """在柱状图/条形图上添加数值标签"""
    if bars is None:
        patches = ax.patches
    else:
        patches = bars
    
    for patch in patches:
        height = patch.get_height()
        width = patch.get_width()
        x = patch.get_x() + width / 2
        ax.annotate(
            fmt.format(height),
            xy=(x, height),
            xytext=(0, offset),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            color=color,
            fontweight="medium",
        )


def save_fig(fig, path, dpi=150, bg_color=None):
    """保存图片（带美观背景）"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    actual_bg = bg_color if bg_color is not None else BG_COLOR
    fig.patch.set_facecolor(actual_bg)
    for ax in fig.axes:
        ax.set_facecolor(actual_bg)
    
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    return path


# ============================================================
# 电梯调度专用可视化
# ============================================================

def plot_algorithm_comparison(results_dict, save_path=None):
    """
    绘制算法对比图（美观版）
    
    Args:
        results_dict: {算法名: {metric_name: value}} 的字典
        save_path: 保存路径（可选）
    
    Returns:
        matplotlib.figure.Figure
    """
    setup_matplotlib()
    
    names = list(results_dict.keys())
    metrics = [
        ("平均送达人数", [results_dict[n].get("served", 0) for n in names], COLORS["green"]),
        ("平均放弃人数", [results_dict[n].get("abandoned", 0) for n in names], COLORS["danger"]),
        ("平均等待时间(秒)", [results_dict[n].get("avg_wait", 0) for n in names], COLORS["blue"]),
        ("平均行程时间(秒)", [results_dict[n].get("avg_ride", 0) for n in names], COLORS["orange"]),
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor(BG_COLOR)
    fig.suptitle("电梯调度算法对比实验", fontsize=16, fontweight="bold", color="#222222", y=0.98)
    
    for ax, (title, values, color) in zip(axes.flat, metrics):
        ax.set_facecolor(BG_COLOR)
        
        bars = ax.bar(
            names,
            values,
            color=color,
            width=0.5,
            edgecolor="white",
            linewidth=1.5,
            zorder=3,
        )
        
        # 数值标签
        add_value_labels(ax, bars, fmt="{:.1f}", offset=5)
        
        # 标题和样式
        ax.set_title(title, fontsize=12, fontweight="600", color="#333333", pad=12)
        ax.set_ylabel("数值", fontsize=10, color="#555555")
        ax.grid(axis="y", alpha=0.4, linestyle="--", zorder=0)
        ax.set_axisbelow(True)
        set_spine_style(ax)
        
        # Y轴从0开始
        ax.set_ylim(bottom=0)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if save_path:
        save_fig(fig, save_path)
    
    return fig


def plot_single_algorithm_detail(stats, algo_name, save_path=None):
    """
    绘制单算法详细指标图
    
    Args:
        stats: 统计数据字典
        algo_name: 算法名称
        save_path: 保存路径
    """
    setup_matplotlib()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor(BG_COLOR)
    fig.suptitle(f"{algo_name} — 详细性能指标", fontsize=16, fontweight="bold", color="#222222")
    
    # 指标数据
    labels = ["送达", "放弃", "生成", "上车"]
    values = [
        stats.get("served", 0),
        stats.get("abandoned", 0),
        stats.get("generated", 0),
        stats.get("boarded", 0),
    ]
    colors_bar = [COLORS["green"], COLORS["danger"], COLORS["blue"], COLORS["orange"]]
    
    # 图1：送达/放弃/生成/上车 柱状图
    ax = axes[0, 0]
    ax.set_facecolor(BG_COLOR)
    bars = ax.bar(labels, values, color=colors_bar, edgecolor="white", linewidth=1.5, zorder=3)
    add_value_labels(ax, bars)
    ax.set_title("乘客流量统计", fontsize=12, fontweight="600", pad=12)
    ax.grid(axis="y", alpha=0.4, linestyle="--", zorder=0)
    ax.set_axisbelow(True)
    set_spine_style(ax)
    ax.set_ylim(bottom=0)
    
    # 图2：等待时间分布（模拟）
    ax = axes[0, 1]
    ax.set_facecolor(BG_COLOR)
    wait_times = stats.get("wait_time_list", [])
    if wait_times:
        ax.hist(wait_times, bins=20, color=COLORS["blue"], alpha=0.7, edgecolor="white", linewidth=1)
    else:
        # 无数据时画占位
        ax.text(0.5, 0.5, "暂无数据", ha="center", va="center", transform=ax.transAxes, color="#AAAAAA")
    ax.set_title("等待时间分布", fontsize=12, fontweight="600", pad=12)
    ax.set_xlabel("等待时间（秒）", fontsize=10)
    ax.set_ylabel("人数", fontsize=10)
    set_spine_style(ax)
    
    # 图3：行程时间分布
    ax = axes[1, 0]
    ax.set_facecolor(BG_COLOR)
    ride_times = stats.get("ride_time_list", [])
    if ride_times:
        ax.hist(ride_times, bins=20, color=COLORS["orange"], alpha=0.7, edgecolor="white", linewidth=1)
    else:
        ax.text(0.5, 0.5, "暂无数据", ha="center", va="center", transform=ax.transAxes, color="#AAAAAA")
    ax.set_title("行程时间分布", fontsize=12, fontweight="600", pad=12)
    ax.set_xlabel("行程时间（秒）", fontsize=10)
    ax.set_ylabel("人数", fontsize=10)
    set_spine_style(ax)
    
    # 图4：综合指标仪表盘
    ax = axes[1, 1]
    ax.set_facecolor(BG_COLOR)
    ax.axis("off")
    
    abandon_rate = stats.get("abandon_rate", 0) * 100
    avg_wait = stats.get("avg_wait", 0)
    avg_ride = stats.get("avg_ride", 0)
    
    info_text = (
        f"送达人数：{stats.get('served', 0):.0f}\n"
        f"放弃人数：{stats.get('abandoned', 0):.0f}\n"
        f"放弃率：{abandon_rate:.1f}%\n"
        f"平均等待：{avg_wait:.1f} 秒\n"
        f"平均行程：{avg_ride:.1f} 秒"
    )
    ax.text(
        0.1, 0.6, info_text,
        fontsize=12,
        color="#333333",
        va="center",
        bbox=dict(boxstyle="round,pad=1", facecolor="#EEF2FF", edgecolor="#4C72B0", linewidth=2),
    )
    ax.set_title("综合指标", fontsize=12, fontweight="600", pad=12)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    if save_path:
        save_fig(fig, save_path)
    
    return fig


# 初始化时自动配置
setup_matplotlib()


def plot_metrics_comparison(results_dict, save_path=None):
    """
    绘制算法对比图（接受 Metrics 对象格式）
    
    Args:
        results_dict: {算法名: Metrics对象} 的字典
        save_path: 保存路径（可选）
    
    Returns:
        matplotlib.figure.Figure
    """
    setup_matplotlib()
    
    names = list(results_dict.keys())
    metrics = [
        ("平均等待时间(AWT)", [results_dict[n].awt for n in names], COLORS["blue"]),
        ("平均行程时间(ART)", [results_dict[n].art for n in names], COLORS["orange"]),
        ("放弃率(%)", [results_dict[n].abandon_rate * 100 for n in names], COLORS["danger"]),
        ("吞吐量(人/分钟)", [results_dict[n].throughput for n in names], COLORS["purple"]),
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor(BG_COLOR)
    fig.suptitle("电梯调度算法对比实验", fontsize=16, fontweight="bold", color="#222222", y=0.98)
    
    for ax, (title, values, color) in zip(axes.flat, metrics):
        ax.set_facecolor(BG_COLOR)
        
        bars = ax.bar(
            names,
            values,
            color=color,
            width=0.5,
            edgecolor="white",
            linewidth=1.5,
            zorder=3,
        )
        
        # 数值标签
        fmt = "{:.1f}%" if "率" in title else "{:.1f}"
        add_value_labels(ax, bars, fmt=fmt, offset=5)
        
        # 标题和样式
        ax.set_title(title, fontsize=12, fontweight="600", color="#333333", pad=12)
        ax.set_ylabel("数值", fontsize=10, color="#555555")
        ax.grid(axis="y", alpha=0.4, linestyle="--", zorder=0)
        ax.set_axisbelow(True)
        set_spine_style(ax)
        
        # Y轴从0开始
        ax.set_ylim(bottom=0)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if save_path:
        save_fig(fig, save_path)
    
    return fig
