# 多因子综合评分 + 机器学习选股策略系统 V4.0

专业级量化交易策略系统，基于机器学习的A股选股策略，支持回测和实盘交易。

## 📋 目录

- [系统特性](#系统特性)
- [系统架构](#系统架构)
- [快速开始](#快速开始)
- [详细说明](#详细说明)
- [配置指南](#配置指南)
- [使用示例](#使用示例)
- [性能优化](#性能优化)
- [常见问题](#常见问题)

## 🌟 系统特性

### 核心功能

1. **多因子体系**
   - 技术因子：动量、RSI、MACD、布林带、ATR、ADX等11个指标
   - 资金流因子：换手率、成交额变化、量价相关性、主力资金流向等6个指标
   - 基本面因子：PE、PB、ROE、ROA、毛利率、净利率等10个指标
   - Alpha因子：自定义Alpha因子，可扩展

2. **机器学习模型**
   - 支持XGBoost、LightGBM、CatBoost
   - Walk-Forward滚动训练，避免未来函数
   - 集成学习支持，提升预测稳定性
   - 特征重要性分析

3. **策略与风控**
   - 智能选股：基于ML评分的Top N选股
   - 择时系统：RSRS、均线、MACD等择时指标
   - 风险管理：止损止盈、持仓期限、行业限制
   - 舆情风控：新闻舆情分析(可扩展)

4. **回测系统**
   - 完整的回测引擎，支持佣金、印花税、滑点
   - 详细的性能指标：夏普比率、最大回撤、信息比率等
   - 可视化报告：权益曲线、回撤图、收益分布等
   - 交易记录和持仓明细

5. **实盘交易**
   - 支持实盘交易接口(EasyTrader等)
   - 模拟运行模式(Dry Run)
   - 自动生成交易指令

6. **性能优化**
   - 增量数据更新
   - 多进程并行计算
   - 智能缓存机制
   - 内存使用优化

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                        用户界面                               │
│            (命令行 / Web界面 / API接口)                        │
└────────────────────┬────────────────────────────────────────┘
                     │
        ┌────────────┴────────────┐
        │      主控程序            │
        │      (main.py)          │
        └────────────┬────────────┘
                     │
    ┌────────────────┼────────────────┐
    │                │                │
┌───▼────┐    ┌─────▼──────┐   ┌────▼─────┐
│数据层  │    │  策略层    │   │  执行层  │
└───┬────┘    └─────┬──────┘   └────┬─────┘
    │               │               │
┌───▼─────────────┐ │  ┌───────────▼──────┐
│ DataManager     │ │  │ BacktestEngine   │
│ - Tushare接口   │ │  │ - 模拟撮合       │
│ - 数据清洗      │ │  │ - 性能分析       │
│ - 增量更新      │ │  │ - 报告生成       │
│ - 缓存管理      │ │  └──────────────────┘
└─────────────────┘ │
                    │  ┌──────────────────┐
┌─────────────────┐ │  │ RealTimeTrader   │
│ FactorEngine    │ │  │ - 交易接口       │
│ - 技术因子      │ │  │ - 订单管理       │
│ - 资金流因子    │ │  │ - 风险控制       │
│ - 基本面因子    │ │  └──────────────────┘
│ - Alpha因子     │ │
│ - 因子处理      │ │
└─────────────────┘ │
                    │
┌─────────────────┐ │
│ MLModel         │ │
│ - XGBoost       │ │
│ - LightGBM      │ │
│ - CatBoost      │ │
│ - Walk-Forward  │ │
└─────────────────┘ │
                    │
┌─────────────────┐ │
│ Strategy        │◄┘
│ - 选股逻辑      │
│ - 权重分配      │
│ - 调仓规则      │
│ - 风险管理      │
│ - 择时系统      │
└─────────────────┘
```

## 🚀 快速开始

### 1. 环境准备

```bash
# Python 3.8+
python --version

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置Tushare

在 `config.py` 中设置你的Tushare Token:

```python
tushare_token: str = "YOUR_TUSHARE_TOKEN"
```

获取Token: https://tushare.pro/register

### 3. 运行回测

```bash
# 基础回测
python main.py --mode backtest

# 使用自定义配置
python main.py --mode backtest --config my_config.json

# 仅训练模型
python main.py --mode train
```

### 4. 查看结果

回测结果保存在 `./output/` 目录:
- `equity_curve.png` - 权益曲线
- `drawdown.png` - 回撤曲线
- `returns_distribution.png` - 收益分布
- `metrics_table.png` - 性能指标
- `trades.csv` - 交易记录
- `positions.csv` - 持仓记录

## 📚 详细说明

### 数据层 (Data Layer)

**DataManager** 负责所有数据相关操作:

```python
from data_manager import DataManager
from config import Config

config = Config()
dm = DataManager(config)

# 获取日线数据
df_daily = dm.get_daily_data(start_date='20200101', end_date='20231231')

# 获取基本面数据
df_fundamental = dm.get_fundamental_data('000001.SZ')

# 增量更新
df_new = dm.incremental_update()
```

**特点**:
- 自动数据清洗(去除ST、新股、停牌等)
- 智能缓存机制
- 增量更新支持
- 数据质量检查

### 因子层 (Factor Layer)

**FactorEngine** 计算多种因子:

```python
from factor_engine import FactorEngine

fe = FactorEngine(config)

# 计算所有因子
df_with_factors = fe.calculate_all_factors(df_daily)

# 因子自动处理
# - 极值处理 (Winsorize)
# - 标准化 (Standardize)
# - 行业中性化 (Neutralize)
# - 正交化 (Orthogonalize)
```

**支持的因子**:

| 类别 | 因子 | 说明 |
|------|------|------|
| 技术 | momentum_20/60 | 动量因子 |
| 技术 | rsi_14 | 相对强弱指标 |
| 技术 | macd | MACD指标 |
| 技术 | bbands_width | 布林带宽度 |
| 技术 | atr_14 | 平均真实波幅 |
| 资金流 | turnover_rate | 换手率 |
| 资金流 | volume_price_corr | 量价相关性 |
| 资金流 | main_force_inflow | 主力资金流入 |
| 基本面 | pe_ttm, pb, ps_ttm | 估值因子 |
| 基本面 | roe, roa | 盈利能力 |
| Alpha | alpha_001/002/003 | 自定义因子 |

### 模型层 (Model Layer)

**MLModel** 提供机器学习能力:

```python
from ml_model import MLModel, WalkForwardTrainer

# 单模型
model = MLModel(config, model_type='xgboost')
model.train(X_train, y_train, X_val, y_val)
predictions = model.predict(X_test)

# Walk-Forward训练
trainer = WalkForwardTrainer(config)
df_predictions = trainer.walk_forward_train(df_with_labels)
```

**Walk-Forward流程**:
```
时间轴: [====训练窗口====][预测]
        [====训练窗口====][预测]
               [====训练窗口====][预测]
                      ...
```

### 策略层 (Strategy Layer)

**Strategy** 实现选股和风控逻辑:

```python
from strategy import Strategy, RiskManager, MarketTiming

strategy = Strategy(config)

# 选股
selected = strategy.select_stocks(df_with_scores, date='20231201')

# 权重分配
selected = strategy.calculate_weights(selected)

# 风险管理
risk_manager = RiskManager(config)
to_sell = risk_manager.check_stop_loss(positions, current_prices)

# 择时
timing = MarketTiming(config)
market_signal = timing.get_market_signal(index_data, date)
```

### 回测层 (Backtest Layer)

**BacktestEngine** 执行回测:

```python
from backtest import BacktestEngine

engine = BacktestEngine(config)
results = engine.run(df_with_scores, price_data, index_data)

# 查看指标
print(results['metrics'])

# 生成报告
engine.generate_report()
engine.plot_results(results['metrics'])
```

## ⚙️ 配置指南

配置文件 `config.py` 包含所有参数:

### 数据配置 (DataConfig)

```python
# 数据源
tushare_token: str = "YOUR_TOKEN"
use_cache: bool = True
incremental_update: bool = True

# 数据过滤
exclude_st: bool = True  # 排除ST
exclude_new_stock_days: int = 252  # 排除上市不足1年的新股
min_market_cap: float = 10  # 最小市值10亿
min_liquidity: float = 5000000  # 最小成交额500万

# 日期范围
start_date: str = "20200101"
end_date: str = None  # None表示当前日期
```

### 因子配置 (FactorConfig)

```python
# 启用的因子
technical_factors: List[str] = [
    'momentum_20', 'rsi_14', 'macd', ...
]

# 因子处理
winsorize: bool = True  # 极值处理
winsorize_limits: tuple = (0.01, 0.99)
standardize: bool = True  # 标准化
neutralize: bool = True  # 行业中性化
orthogonalize: bool = True  # 正交化

# 因子选择
use_feature_importance: bool = True
max_factors: int = 30
```

### 模型配置 (ModelConfig)

```python
# 模型类型
model_type: str = "xgboost"  # xgboost, lightgbm, catboost

# XGBoost参数
xgb_params: Dict = {
    'max_depth': 6,
    'learning_rate': 0.05,
    'n_estimators': 200,
    ...
}

# Walk-Forward
walk_forward: bool = True
train_window: int = 252  # 训练窗口(交易日)
retrain_frequency: int = 20  # 每20天重训练

# 标签
forward_return_days: int = 5  # 预测未来5日收益
label_type: str = "excess_return"  # 超额收益
```

### 策略配置 (StrategyConfig)

```python
# 选股
top_n: int = 10  # 持仓数量
selection_method: str = "score"  # 选股方法

# 权重
weight_method: str = "equal"  # equal, score_weighted, risk_parity
max_single_weight: float = 0.15  # 单股最大权重15%

# 调仓
rebalance_frequency: str = "weekly"  # daily, weekly, monthly
rebalance_day: int = 5  # 周五调仓

# 行业限制
max_industry_weight: float = 0.35  # 单行业最大35%
min_industry_count: int = 3  # 至少3个行业
```

### 风控配置 (RiskConfig)

```python
# 止损止盈
use_stop_loss: bool = True
stop_loss_pct: float = -0.10  # -10%
take_profit_pct: float = 0.25  # +25%

# 持仓期限
max_holding_days: int = 60
min_holding_days: int = 5

# 择时
use_timing: bool = True
timing_indicator: str = "rsrs"
rsrs_threshold: float = 0.7
cash_ratio_when_bear: float = 0.5  # 熊市50%仓位
```

## 💡 使用示例

### 示例1: 基础回测

```python
from config import Config
from main import StrategySystem

# 创建配置
config = Config()
config.backtest.start_date = "20200101"
config.strategy.top_n = 10
config.strategy.rebalance_frequency = "weekly"

# 运行回测
system = StrategySystem(config)
results = system.run_backtest()

# 查看结果
print(f"年化收益: {results['metrics']['annual_return']:.2%}")
print(f"夏普比率: {results['metrics']['sharpe_ratio']:.2f}")
print(f"最大回撤: {results['metrics']['max_drawdown']:.2%}")
```

### 示例2: 参数优化

```python
# 测试不同持仓数量
best_sharpe = 0
best_n = 0

for n in [5, 10, 15, 20]:
    config = Config()
    config.strategy.top_n = n
    
    system = StrategySystem(config)
    results = system.run_backtest()
    
    sharpe = results['metrics']['sharpe_ratio']
    if sharpe > best_sharpe:
        best_sharpe = sharpe
        best_n = n
    
    print(f"N={n}, Sharpe={sharpe:.2f}")

print(f"最优持仓数量: {best_n}")
```

### 示例3: 因子分析

```python
from factor_engine import FactorEngine
from data_manager import DataManager

# 加载数据
dm = DataManager(config)
df = dm.get_daily_data()

# 计算因子
fe = FactorEngine(config)
df_with_factors = fe.calculate_all_factors(df)

# 计算IC值
from ml_model import LabelGenerator
df_with_labels = LabelGenerator.add_labels(df_with_factors, config)
ic_df = fe.calculate_ic(df_with_labels)

# 分析因子表现
ic_stats = ic_df.groupby('factor')['ic'].agg(['mean', 'std'])
ic_stats['ic_ir'] = ic_stats['mean'] / ic_stats['std']
print(ic_stats.sort_values('ic_ir', ascending=False))
```

### 示例4: 实盘运行

```python
# 启用实盘
config = Config()
config.system.live_trading = True
config.system.dry_run = False  # 关闭模拟模式

# 设置交易接口
config.system.trade_api = "easytrader"

# 运行
system = StrategySystem(config)
results = system.run_live()
```

## 🔧 性能优化

### 内存优化

```python
# 配置内存限制
config.system.max_memory_gb = 8.0
config.system.chunk_size = 500

# 使用增量更新
config.data.incremental_update = True

# 启用缓存
config.data.use_cache = True
config.system.cache_factors = True
```

### 并行计算

```python
# 启用多进程
config.system.use_multiprocessing = True
config.system.n_jobs = -1  # 使用所有CPU核心

# GPU加速(需要安装相应库)
config.system.use_gpu = True
```

### 数据优化

```python
# 限制数据范围
config.data.start_date = "20220101"  # 只用近期数据

# 减少因子数量
config.factor.max_factors = 20

# 降低训练频率
config.model.retrain_frequency = 40  # 每40天重训练一次
```

## 🐛 常见问题

### Q1: Tushare积分不够怎么办?

A: Tushare部分数据需要积分。可以:
1. 升级Tushare会员
2. 使用其他数据源(如AKShare)
3. 减少数据请求频率

### Q2: 模型训练太慢?

A: 尝试:
1. 减少训练数据量(`train_window`)
2. 降低模型复杂度(`max_depth`, `n_estimators`)
3. 使用更快的模型(LightGBM)
4. 启用GPU加速

### Q3: 内存不足?

A: 
1. 减少数据范围
2. 启用分块处理(`chunk_size`)
3. 关闭不必要的缓存
4. 使用增量更新

### Q4: 回测结果不理想?

A:
1. 检查因子有效性(IC值)
2. 调整模型参数
3. 优化风控参数
4. 增加训练数据
5. 使用集成学习

### Q5: 如何添加自定义因子?

A: 在 `factor_engine.py` 中添加:

```python
def _calculate_custom_factor(self, df: pd.DataFrame) -> pd.DataFrame:
    # 计算你的因子
    df['my_factor'] = ...
    return df

# 在 calculate_all_factors 中调用
df = self._calculate_custom_factor(df)
```

## 📝 版本历史

- **V4.0** (2025-12-22)
  - 完全重构代码架构
  - 增加模型自适应更新
  - 优化内存管理
  - 完善文档

- **V3.0** 
  - 修复Walk-Forward数据泄露问题
  - 增加集成学习支持

- **V2.0**
  - 增加实盘交易功能
  - 优化因子计算

- **V1.0**
  - 初始版本

## 📞 联系方式

- Issues: 通过GitHub Issues提交问题
- Email: your.email@example.com

## 📄 许可证

MIT License

---

**免责声明**: 本策略仅供学习研究使用,不构成任何投资建议。股市有风险,投资需谨慎。