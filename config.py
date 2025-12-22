"""
多因子综合评分+机器学习选股策略 - 配置文件
Version: 4.0
Author: Claude
Date: 2025-12-22
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime

@dataclass
class DataConfig:
    """数据配置"""
    # Tushare配置
    tushare_token: str = "2876ea85cb005fb5fa17c809a98174f2d5aae8b1f830110a5ead6211"  # 请替换为您的token

    # 数据源
    use_cache: bool = True
    cache_dir: str = "./data_cache"
    incremental_update: bool = True

    # 数据过滤
    exclude_st: bool = True  # 排除ST股票
    exclude_new_stock_days: int = 252  # 排除上市不足N天的新股
    exclude_suspended: bool = True  # 排除停牌股票
    min_market_cap: float = 10  # 最小市值(亿元)
    min_liquidity: float = 5000000  # 最小日成交额(元)

    # 数据范围
    start_date: str = "20230101"
    end_date: str = None  # None表示当前日期

    # 更新频率
    update_frequency: str = "daily"  # daily, weekly, monthly


@dataclass
class FactorConfig:
    """因子配置"""
    # 技术因子
    technical_factors: List[str] = field(default_factory=lambda: [
        'momentum_20',  # 20日动量
        'momentum_60',  # 60日动量
        'rsi_14',       # RSI指标
        'macd',         # MACD
        'bbands_width', # 布林带宽度
        'atr_14',       # ATR波动率
        'adx_14',       # ADX趋势强度
        'cci_20',       # CCI指标
        'willr_14',     # 威廉指标
        'stoch_k',      # 随机指标K
        'volume_ratio', # 量比
    ])

    # 资金流因子
    money_flow_factors: List[str] = field(default_factory=lambda: [
        'turnover_rate',      # 换手率
        'amount_change',      # 成交额变化
        'volume_price_corr',  # 量价相关性
        'big_order_ratio',    # 大单比例
        'main_force_inflow',  # 主力资金流入
        'vwap_ratio',         # VWAP比率
    ])

    # 基本面因子
    fundamental_factors: List[str] = field(default_factory=lambda: [
        'pe_ttm',        # 市盈率TTM
        'pb',            # 市净率
        'ps_ttm',        # 市销率TTM
        'pcf_ttm',       # 市现率TTM
        'roe',           # ROE
        'roa',           # ROA
        'gross_margin',  # 毛利率
        'net_margin',    # 净利率
        'debt_to_asset', # 资产负债率
        'current_ratio', # 流动比率
    ])

    # Alpha因子
    alpha_factors: List[str] = field(default_factory=lambda: [
        'alpha_001',  # 自定义Alpha因子1
        'alpha_002',  # 自定义Alpha因子2
        'alpha_003',  # 自定义Alpha因子3
    ])

    # 因子处理
    winsorize: bool = True  # 极值处理
    winsorize_limits: tuple = (0.01, 0.99)  # 缩尾限制
    standardize: bool = True  # 标准化
    neutralize: bool = True  # 行业中性化
    orthogonalize: bool = True  # 正交化

    # 因子选择
    use_feature_importance: bool = True  # 使用特征重要性
    importance_threshold: float = 0.01  # 重要性阈值
    max_factors: int = 30  # 最大因子数


@dataclass
class ModelConfig:
    """模型配置"""
    # 模型类型
    model_type: str = "xgboost"  # xgboost, lightgbm, catboost, ensemble

    # XGBoost参数
    xgb_params: Dict = field(default_factory=lambda: {
        'objective': 'reg:squarederror',
        'max_depth': 6,
        'learning_rate': 0.05,
        'n_estimators': 200,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 3,
        'gamma': 0.1,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'random_state': 42,
        'n_jobs': -1,
        'tree_method': 'hist',  # 使用histogram优化
    })

    # LightGBM参数
    lgb_params: Dict = field(default_factory=lambda: {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'n_estimators': 200,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_samples': 20,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'random_state': 42,
        'n_jobs': -1,
    })

    # 训练配置
    walk_forward: bool = True  # 滚动训练
    train_window: int = 252  # 训练窗口(交易日)
    retrain_frequency: int = 20  # 重训练频率(交易日)

    # 标签配置
    forward_return_days: int = 5  # 未来收益期数
    label_type: str = "excess_return"  # return, excess_return, rank

    # 验证配置
    validation_split: float = 0.2  # 验证集比例
    use_early_stopping: bool = True
    early_stopping_rounds: int = 50

    # 集成学习
    use_ensemble: bool = False
    ensemble_models: List[str] = field(default_factory=lambda: ['xgboost', 'lightgbm'])
    ensemble_weights: List[float] = field(default_factory=lambda: [0.6, 0.4])


@dataclass
class StrategyConfig:
    """策略配置"""
    # 选股
    top_n: int = 10  # 持仓数量
    selection_method: str = "score"  # score, rank, threshold
    score_threshold: float = 0.6  # 分数阈值(当method='threshold'时使用)

    # 权重分配
    weight_method: str = "equal"  # equal, score_weighted, risk_parity, optimize
    max_single_weight: float = 0.15  # 单只股票最大权重

    # 调仓
    rebalance_frequency: str = "n_days"  # daily, weekly, monthly, n_days
    rebalance_day: int = 5  # 调仓日: weekly时5=周五, monthly时为每月日期, n_days时为每N个交易日

    # 行业配置
    max_industry_weight: float = 0.35  # 单行业最大权重
    min_industry_count: int = 3  # 最少行业数量

    # 交易成本
    commission_rate: float = 0.0003  # 佣金费率
    stamp_tax: float = 0.001  # 印花税(仅卖出)
    slippage: float = 0.001  # 滑点


@dataclass
class RiskConfig:
    """风控配置"""
    # 止损
    use_stop_loss: bool = True
    stop_loss_pct: float = -0.10  # 个股止损线(-10%)
    portfolio_stop_loss_pct: float = -0.15  # 组合止损线(-15%)

    # 止盈
    use_take_profit: bool = True
    take_profit_pct: float = 0.25  # 个股止盈线(+25%)

    # 持仓期限
    max_holding_days: int = 60  # 最大持仓天数
    min_holding_days: int = 5  # 最小持仓天数

    # 择时
    use_timing: bool = True
    timing_indicator: str = "rsrs"  # rsrs, ma, macd
    rsrs_threshold: float = 0.7  # RSRS阈值
    cash_ratio_when_bear: float = 0.5  # 熊市现金比例

    # 风险限制
    max_drawdown_alert: float = -0.20  # 最大回撤警戒线
    max_volatility: float = 0.30  # 最大波动率
    max_turnover: float = 2.0  # 最大换手率(年化)

    # 舆情风控
    use_sentiment: bool = True
    sentiment_veto_threshold: float = -0.5  # 舆情一票否决阈值
    sentiment_bonus: float = 0.1  # 利好加分


@dataclass
class BacktestConfig:
    """回测配置"""
    initial_capital: float = 1000000  # 初始资金
    benchmark: str = "000300.SH"  # 基准指数(沪深300)

    # 回测参数
    start_date: str = "20200101"
    end_date: str = None  # None表示当前日期

    # 报告
    output_dir: str = "./output"
    save_trades: bool = True
    save_positions: bool = True
    save_metrics: bool = True

    # 可视化
    plot_equity_curve: bool = True
    plot_drawdown: bool = True
    plot_returns_dist: bool = True
    plot_factor_importance: bool = True


@dataclass
class SystemConfig:
    """系统配置"""
    # 日志
    log_level: str = "INFO"  # DEBUG, INFO, WARNING, ERROR
    log_file: str = "./logs/strategy.log"

    # 性能
    use_multiprocessing: bool = True
    n_jobs: int = -1  # -1表示使用所有CPU核心
    use_gpu: bool = False  # GPU加速(需要安装相应库)

    # 内存管理
    max_memory_gb: float = 8.0  # 最大内存使用(GB)
    chunk_size: int = 500  # 分块处理大小

    # 缓存
    cache_factors: bool = True
    cache_models: bool = True

    # 实盘
    live_trading: bool = False
    trade_api: str = "easytrader"  # easytrader, xtquant, 等
    dry_run: bool = True  # 模拟运行


class Config:
    """全局配置类"""

    def __init__(self):
        self.data = DataConfig()
        self.factor = FactorConfig()
        self.model = ModelConfig()
        self.strategy = StrategyConfig()
        self.risk = RiskConfig()
        self.backtest = BacktestConfig()
        self.system = SystemConfig()

        # 版本信息
        self.version = "4.0"
        self.update_date = "2025-12-22"

    def validate(self) -> bool:
        """验证配置的合法性"""
        # 验证日期
        if self.data.start_date and self.data.end_date:
            if self.data.start_date >= self.data.end_date:
                raise ValueError("start_date must be before end_date")

        # 验证持仓数量
        if self.strategy.top_n <= 0:
            raise ValueError("top_n must be positive")

        # 验证权重
        if not (0 < self.strategy.max_single_weight <= 1.0):
            raise ValueError("max_single_weight must be between 0 and 1")

        # 验证止损止盈
        if self.risk.stop_loss_pct >= 0:
            raise ValueError("stop_loss_pct must be negative")
        if self.risk.take_profit_pct <= 0:
            raise ValueError("take_profit_pct must be positive")

        return True

    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'data': self.data.__dict__,
            'factor': self.factor.__dict__,
            'model': self.model.__dict__,
            'strategy': self.strategy.__dict__,
            'risk': self.risk.__dict__,
            'backtest': self.backtest.__dict__,
            'system': self.system.__dict__,
        }

    def save(self, filepath: str):
        """保存配置到文件"""
        import json
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=4, ensure_ascii=False)

    @classmethod
    def load(cls, filepath: str) -> 'Config':
        """从文件加载配置"""
        import json
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        config = cls()
        for key, value in data.items():
            if hasattr(config, key):
                section = getattr(config, key)
                for k, v in value.items():
                    setattr(section, k, v)
        return config


# 创建全局配置实例
config = Config()