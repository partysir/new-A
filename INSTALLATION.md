# 安装和快速使用指南

## 🚀 5分钟快速开始

### 步骤1: 安装Python环境

确保你的系统已安装Python 3.8或更高版本:

```bash
python --version
# 应该显示 Python 3.8.x 或更高
```

### 步骤2: 下载代码

将所有文件下载到一个目录,例如 `ml_strategy/`:

```
ml_strategy/
├── config.py
├── data_manager.py
├── factor_engine.py
├── ml_model.py
├── strategy.py
├── backtest.py
├── main.py
├── examples.py
├── requirements.txt
├── README.md
└── PROJECT_STRUCTURE.md
```

### 步骤3: 安装依赖

```bash
cd ml_strategy
pip install -r requirements.txt
```

如果速度慢,可以使用国内镜像:

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 步骤4: 获取Tushare Token

1. 访问 https://tushare.pro/register
2. 注册账号(免费)
3. 获取你的Token

### 步骤5: 配置Token

打开 `config.py`,找到这一行:

```python
tushare_token: str = "YOUR_TUSHARE_TOKEN"
```

替换为你的Token:

```python
tushare_token: str = "你的实际token"
```

### 步骤6: 运行第一个回测

```bash
python examples.py
```

选择 `1` (基础回测),程序会:
1. 下载数据
2. 计算因子
3. 训练模型
4. 运行回测
5. 生成报告

结果保存在 `./output/` 目录。

## 📖 详细安装说明

### Windows系统

1. **安装Python**
   - 下载: https://www.python.org/downloads/
   - 安装时勾选 "Add Python to PATH"

2. **安装依赖**
   ```cmd
   cd ml_strategy
   pip install -r requirements.txt
   ```

3. **运行程序**
   ```cmd
   python main.py --mode backtest
   ```

### macOS系统

1. **安装Python** (如果没有)
   ```bash
   brew install python3
   ```

2. **安装依赖**
   ```bash
   cd ml_strategy
   pip3 install -r requirements.txt
   ```

3. **运行程序**
   ```bash
   python3 main.py --mode backtest
   ```

### Linux系统

1. **安装Python** (通常已预装)
   ```bash
   sudo apt-get update
   sudo apt-get install python3 python3-pip
   ```

2. **安装依赖**
   ```bash
   cd ml_strategy
   pip3 install -r requirements.txt
   ```

3. **运行程序**
   ```bash
   python3 main.py --mode backtest
   ```

## 🔧 依赖说明

### 必需依赖

| 包名 | 版本 | 用途 |
|------|------|------|
| pandas | >=1.5.0 | 数据处理 |
| numpy | >=1.23.0 | 数值计算 |
| scikit-learn | >=1.1.0 | 机器学习基础 |
| xgboost | >=1.7.0 | XGBoost模型 |
| lightgbm | >=3.3.0 | LightGBM模型 |
| tushare | >=1.2.89 | 数据获取 |
| matplotlib | >=3.6.0 | 绘图 |

### 可选依赖

| 包名 | 用途 |
|------|------|
| catboost | CatBoost模型(可选) |
| easytrader | 实盘交易接口(可选) |
| cvxpy | 组合优化(可选) |

## 🎯 使用场景

### 场景1: 简单回测

```bash
# 使用默认配置回测
python main.py --mode backtest
```

查看结果:
- `./output/equity_curve.png` - 权益曲线
- `./output/metrics_table.png` - 性能指标

### 场景2: 自定义配置回测

```python
# 创建自定义配置
from config import Config

config = Config()
config.strategy.top_n = 20  # 持仓20只
config.strategy.rebalance_frequency = "monthly"  # 月度调仓
config.save("my_config.json")
```

```bash
# 使用自定义配置
python main.py --mode backtest --config my_config.json
```

### 场景3: 因子分析

```bash
python examples.py
# 选择 3 - 因子分析
```

输出:
- 各因子的IC值
- 因子重要性排名
- 因子表现统计

### 场景4: 模型训练

```bash
# 仅训练模型,不运行回测
python main.py --mode train
```

模型保存在 `./data_cache/latest_model.pkl`

### 场景5: 参数优化

```bash
python examples.py
# 选择 5 - 参数优化
```

自动测试不同参数组合,找出最优配置。

### 场景6: 模拟实盘

```bash
python examples.py
# 选择 6 - 模拟实盘
```

生成今日选股结果和交易指令。

## 🐛 常见问题解决

### 问题1: ImportError: No module named 'xxx'

**原因**: 依赖未安装

**解决**:
```bash
pip install xxx
# 或
pip install -r requirements.txt
```

### 问题2: Tushare权限不足

**原因**: Token无效或积分不够

**解决**:
1. 检查Token是否正确
2. 访问 https://tushare.pro 查看积分
3. 升级会员或减少数据请求

### 问题3: 内存不足 (MemoryError)

**原因**: 数据量太大

**解决**: 在 `config.py` 中:
```python
# 减少数据范围
config.data.start_date = "20220101"  # 只用近期数据

# 或减少因子数量
config.factor.max_factors = 20
```

### 问题4: 运行速度慢

**原因**: 数据量大或模型复杂

**解决**: 
```python
# 启用缓存
config.data.use_cache = True

# 减少训练频率
config.model.retrain_frequency = 40

# 使用更快的模型
config.model.model_type = "lightgbm"
```

### 问题5: 回测结果不理想

**可能原因**:
1. 数据质量问题
2. 因子失效
3. 参数不合适
4. 过拟合

**解决步骤**:
1. 检查数据质量
2. 分析因子IC值
3. 调整参数
4. 使用更长的训练窗口

### 问题6: ModuleNotFoundError: No module named 'config'

**原因**: 当前目录不正确

**解决**:
```bash
cd ml_strategy  # 确保在项目目录
python main.py --mode backtest
```

## 📊 输出文件说明

### 图片文件

| 文件名 | 说明 |
|--------|------|
| equity_curve.png | 权益曲线图,显示组合净值变化 |
| drawdown.png | 回撤曲线图,显示最大回撤 |
| returns_distribution.png | 收益分布图,直方图+QQ图 |
| metrics_table.png | 性能指标表格 |
| feature_importance.png | 特征重要性图 |

### 数据文件

| 文件名 | 说明 |
|--------|------|
| trades.csv | 所有交易记录 |
| positions.csv | 每日持仓明细 |
| equity_curve.csv | 每日净值数据 |
| feature_importance.csv | 特征重要性数据 |

### 日志文件

| 文件名 | 说明 |
|--------|------|
| logs/strategy.log | 运行日志,包含所有操作记录 |

## 🎓 学习路径

### 新手 (第1周)

1. **了解基础概念**
   - 阅读 README.md
   - 理解多因子选股原理
   - 学习机器学习基础

2. **运行示例**
   - 运行 examples.py 中的所有示例
   - 查看输出结果
   - 理解每个步骤

3. **修改参数**
   - 尝试修改持仓数量
   - 改变调仓频率
   - 调整止损止盈

### 进阶 (第2-4周)

1. **因子研究**
   - 分析现有因子的IC值
   - 尝试添加新因子
   - 进行因子组合

2. **模型调优**
   - 测试不同模型
   - 调整模型参数
   - 对比模型效果

3. **策略优化**
   - 优化选股逻辑
   - 完善风控规则
   - 增加择时模块

### 高级 (第5周+)

1. **系统改进**
   - 优化代码性能
   - 添加新功能
   - 集成其他数据源

2. **实盘准备**
   - 模拟实盘运行
   - 风险测试
   - 接入交易接口

3. **持续优化**
   - 跟踪策略表现
   - 定期重新训练
   - 根据市场变化调整

## 💡 最佳实践

### 1. 数据管理

```python
# 使用缓存加速
config.data.use_cache = True

# 定期清理缓存
dm.clear_cache()

# 使用增量更新
df = dm.incremental_update()
```

### 2. 模型训练

```python
# 使用合适的训练窗口
config.model.train_window = 252  # 1年数据

# 定期重新训练
config.model.retrain_frequency = 20  # 每月重训练

# 使用验证集
config.model.validation_split = 0.2
```

### 3. 风险控制

```python
# 设置止损止盈
config.risk.stop_loss_pct = -0.08  # -8%
config.risk.take_profit_pct = 0.20  # +20%

# 限制行业集中度
config.strategy.max_industry_weight = 0.30

# 使用择时
config.risk.use_timing = True
```

### 4. 参数优化

```python
# 使用网格搜索
for top_n in [5, 10, 15, 20]:
    for freq in ['weekly', 'monthly']:
        config.strategy.top_n = top_n
        config.strategy.rebalance_frequency = freq
        # 运行回测
        # 记录结果
```

### 5. 结果分析

```python
# 关注多个指标
- 年化收益 (目标: >20%)
- 夏普比率 (目标: >1.5)
- 最大回撤 (目标: <20%)
- 胜率 (目标: >55%)
- 信息比率 (目标: >1.0)
```

## 🔗 相关资源

- **Tushare文档**: https://tushare.pro/document/2
- **XGBoost教程**: https://xgboost.readthedocs.io/
- **机器学习**: https://scikit-learn.org/
- **量化社区**: 
  - JoinQuant: https://www.joinquant.com/
  - RiceQuant: https://www.ricequant.com/
  - 聚宽: https://www.joinquant.com/

## 📞 获取帮助

如果遇到问题:

1. **查看日志**: `logs/strategy.log`
2. **检查文档**: README.md 和 PROJECT_STRUCTURE.md
3. **运行示例**: examples.py
4. **搜索错误信息**: 复制错误信息到搜索引擎

## 📝 下一步

现在你已经完成了安装和基础使用,建议:

1. ✅ 运行所有示例代码
2. ✅ 理解系统架构
3. ✅ 尝试修改配置
4. ✅ 分析回测结果
5. ✅ 研究因子表现
6. ✅ 优化策略参数
7. ✅ 添加自定义功能

祝你在量化投资之路上取得成功! 🎉