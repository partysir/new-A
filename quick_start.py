"""
快速启动脚本 - 解决常见问题
1. 自动训练模型（如果不存在）
2. 优化数据获取（使用缓存）
3. 智能日期选择

使用方法:
    python quick_start.py
"""

import logging
from pathlib import Path
from datetime import datetime, timedelta
import joblib

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_model_exists():
    """检查模型是否存在"""
    possible_paths = [
        Path('data_cache/latest_model.pkl'),
        Path('latest_model.pkl'),
        Path('./cache/latest_model.pkl'),
    ]

    for path in possible_paths:
        if path.exists():
            logger.info(f"✓ 找到模型文件: {path}")
            return path

    return None


def train_model_quick():
    """快速训练模型"""
    logger.info("\n" + "=" * 60)
    logger.info("🤖 开始训练模型...")
    logger.info("=" * 60)

    from config import Config
    from data_manager import DataManager
    from factor_engine import FactorEngine
    from ml_model import MLModel, WalkForwardTrainer, LabelGenerator

    config = Config()

    # 使用缓存数据
    logger.info("1. 加载历史数据（使用缓存）...")
    dm = DataManager(config)

    # 查找最新的缓存文件
    cache_dir = Path(config.data.cache_dir)
    cache_files = list(cache_dir.glob('daily_all_*.pkl'))

    if cache_files:
        # 使用最新的缓存
        latest_cache = max(cache_files, key=lambda p: p.stat().st_mtime)
        logger.info(f"   使用缓存: {latest_cache.name}")

        import pandas as pd
        df_daily = pd.read_pickle(latest_cache)

        # 添加行业信息
        try:
            industry_df = dm.get_industry_data()
            df_daily = df_daily.merge(
                industry_df[['ts_code', 'industry']],
                on='ts_code',
                how='left'
            )
        except:
            pass

        logger.info(f"   ✓ 加载完成: {len(df_daily)} 条记录")
    else:
        logger.error("❌ 未找到缓存数据")
        logger.info("请先运行一次获取数据:")
        logger.info("  python run_live_strategy_enhanced.py")
        return False

    # 只使用最近的数据来训练（加快速度）
    logger.info("2. 准备训练数据...")
    recent_dates = sorted(df_daily['trade_date'].unique())[-60:]  # 最近60天
    df_recent = df_daily[df_daily['trade_date'].isin(recent_dates)]
    logger.info(f"   使用最近 {len(recent_dates)} 天数据")

    # 计算因子
    logger.info("3. 计算因子...")
    fe = FactorEngine(config)
    df_with_factors = fe.calculate_all_factors(df_recent)
    logger.info("   ✓ 因子计算完成")

    # 添加标签
    logger.info("4. 生成标签...")
    df_with_labels = LabelGenerator.add_labels(df_with_factors, config)
    logger.info("   ✓ 标签生成完成")

    # 训练模型
    logger.info("5. 训练模型...")
    trainer = WalkForwardTrainer(config)
    X, y = trainer.prepare_data(df_with_labels)

    # 去除NaN
    mask = ~(X.isna().any(axis=1) | y.isna())
    X = X[mask]
    y = y[mask]

    if len(X) < 100:
        logger.error("❌ 训练数据不足")
        return False

    logger.info(f"   训练样本数: {len(X)}")

    # 快速训练
    model = MLModel(config)
    model.build_model()

    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model.train(X_train, y_train, X_val, y_val)

    # 保存模型
    model_path = Path(config.data.cache_dir) / 'latest_model.pkl'
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(model_path))

    logger.info(f"✅ 模型已保存: {model_path}")
    return True


def optimize_data_fetching():
    """优化数据获取策略"""
    logger.info("\n" + "=" * 60)
    logger.info("📊 数据获取优化建议")
    logger.info("=" * 60)

    print("""
Tushare限流问题解决方案:

1. 【推荐】使用缓存数据
   - 你已经有缓存: data_cache/daily_all_20250925_20251224.pkl
   - 直接使用缓存，无需重新获取

2. 减少获取天数
   - 当前: --lookback 90 (太多)
   - 建议: --lookback 30 (足够)

3. 升级Tushare账户
   - 免费账户: 120次/分钟（太慢）
   - 付费账户: 500次/分钟（快很多）
   - 链接: https://tushare.pro/register?reg=408347

4. 本地缓存优先
   - 修改 config.py: use_cache = True ✓（已设置）
   - 只获取增量数据
    """)


def check_cache_freshness():
    """检查缓存新鲜度"""
    cache_dir = Path('data_cache')

    if not cache_dir.exists():
        return None

    cache_files = list(cache_dir.glob('daily_all_*.pkl'))

    if not cache_files:
        return None

    latest_cache = max(cache_files, key=lambda p: p.stat().st_mtime)

    # 解析日期
    import re
    match = re.search(r'_(\d{8})\.pkl', latest_cache.name)
    if match:
        cache_date = match.group(1)
        cache_dt = datetime.strptime(cache_date, '%Y%m%d')
        days_old = (datetime.now() - cache_dt).days

        return {
            'path': latest_cache,
            'date': cache_date,
            'days_old': days_old
        }

    return None


def run_with_cache():
    """使用缓存运行增强选股"""
    logger.info("\n" + "=" * 60)
    logger.info("🚀 使用缓存快速运行")
    logger.info("=" * 60)

    cache_info = check_cache_freshness()

    if not cache_info:
        logger.error("❌ 未找到缓存数据")
        return False

    logger.info(f"✓ 找到缓存: {cache_info['path'].name}")
    logger.info(f"  缓存日期: {cache_info['date']}")
    logger.info(f"  已缓存: {cache_info['days_old']} 天")

    if cache_info['days_old'] > 7:
        logger.warning(f"⚠️  缓存较旧（{cache_info['days_old']}天）")
        logger.info("建议获取最新数据（但会很慢）")

    # 直接运行选股
    logger.info("\n开始选股...")

    from config import Config
    from data_manager import DataManager
    from factor_engine import FactorEngine
    from ml_model import MLModel, WalkForwardTrainer
    from enhanced_live_strategy import EnhancedLiveStrategy
    import pandas as pd

    config = Config()

    # 加载缓存
    logger.info("1. 加载缓存数据...")
    df_history = pd.read_pickle(cache_info['path'])

    # 添加行业
    dm = DataManager(config)
    try:
        industry_df = dm.get_industry_data()
        df_history = df_history.merge(
            industry_df[['ts_code', 'industry']],
            on='ts_code',
            how='left'
        )
    except:
        pass

    logger.info(f"   ✓ {len(df_history)} 条记录")

    # 计算因子
    logger.info("2. 计算因子...")
    fe = FactorEngine(config)
    df_with_factors = fe.calculate_all_factors(df_history)

    # 加载模型
    logger.info("3. 加载模型...")
    model_path = check_model_exists()

    if not model_path:
        logger.error("❌ 未找到模型")
        return False

    model = MLModel(config)
    model.model = joblib.load(str(model_path))
    logger.info("   ✓ 模型加载完成")

    # 预测
    logger.info("4. 生成预测...")
    latest_date = df_with_factors['trade_date'].max()
    df_latest = df_with_factors[df_with_factors['trade_date'] == latest_date]

    trainer = WalkForwardTrainer(config)
    X, _ = trainer.prepare_data(df_latest)
    preds = model.predict(X)
    df_latest['ml_score'] = preds

    logger.info(f"   ✓ 评分完成: {len(df_latest)} 只股票")

    # 获取指数
    try:
        index_data = dm.get_index_data('000300.SH', cache_info['date'])
    except:
        index_data = None

    # 增强选股
    logger.info("5. 增强选股...")
    enhanced_strategy = EnhancedLiveStrategy(config)
    selected = enhanced_strategy.select_stocks_enhanced(
        df_date=df_latest,
        df_history=df_with_factors,
        index_data=index_data
    )

    # 输出结果
    if not selected.empty:
        print("\n" + "=" * 100)
        print(f"📋 {latest_date} 选股结果 (使用缓存)")
        print("=" * 100)

        for _, row in selected.head(15).iterrows():
            print(f"{row['ts_code']:10s} {row.get('name', 'N/A'):10s} "
                  f"评分:{row.get('composite_score', row['ml_score']):.3f} "
                  f"级别:{row.get('position_tier', 'B'):>4s} "
                  f"仓位:{row.get('weight', 0) * 100:5.2f}%")

        print("=" * 100)
        print(f"总计: {len(selected)} 只")

        # 保存
        output_dir = Path('output/enhanced_live_recommendations')
        output_dir.mkdir(parents=True, exist_ok=True)

        csv_path = output_dir / f'recommendations_{latest_date}_cached.csv'
        selected.to_csv(csv_path, index=False, encoding='utf-8-sig')
        logger.info(f"\n✓ 结果已保存: {csv_path}")

        return True
    else:
        logger.warning("未选出股票")
        return False


def main():
    print("\n" + "=" * 70)
    print(" " * 20 + "🚀 快速启动助手")
    print("=" * 70)

    # 1. 检查模型
    model_path = check_model_exists()

    if not model_path:
        logger.warning("\n⚠️  未找到模型文件")

        choice = input("\n是否训练模型? (y/n): ")
        if choice.lower() == 'y':
            success = train_model_quick()
            if not success:
                logger.error("模型训练失败")
                return
        else:
            logger.info("跳过模型训练")
            return
    else:
        logger.info(f"\n✓ 模型已存在: {model_path}")

    # 2. 检查缓存
    cache_info = check_cache_freshness()

    if cache_info:
        logger.info(f"✓ 缓存已存在: {cache_info['date']} ({cache_info['days_old']}天前)")

        if cache_info['days_old'] <= 7:
            logger.info("缓存较新，推荐直接使用")

            choice = input("\n使用缓存快速运行? (y/n): ")
            if choice.lower() == 'y':
                run_with_cache()
                return

    # 3. 数据获取建议
    optimize_data_fetching()

    print("\n" + "=" * 70)
    print("推荐操作:")
    print("=" * 70)
    print("""
方案A（推荐）：使用缓存
    python quick_start.py  # 然后选 y

方案B：获取新数据（慢）
    python run_live_strategy_enhanced.py --lookback 30

方案C：使用回测数据
    python main.py --mode backtest
    """)
    print("=" * 70)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n已取消")
    except Exception as e:
        logger.error(f"错误: {e}", exc_info=True)