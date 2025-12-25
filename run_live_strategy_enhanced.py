"""
增强版实盘选股推荐 v3.0
集成多信号验证、市场自适应、财务筛选
Date: 2025-12-24

使用方法:
    python run_live_strategy_enhanced.py
    python run_live_strategy_enhanced.py --date 20241220
    python run_live_strategy_enhanced.py --lookback 90 --debug
"""

import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import joblib

# 导入原有模块
from config import Config
from data_manager import DataManager
from factor_engine import FactorEngine
from ml_model import MLModel, WalkForwardTrainer

# 【关键】导入增强策略
from enhanced_live_strategy import EnhancedLiveStrategy

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('enhanced_live.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)


def get_latest_available_trading_date(cache_dir, lookback=5):
    """
    智能获取最新可用交易日期
    """
    from datetime import datetime, timedelta

    today = datetime.now()

    # 如果是周末，往前推
    if today.weekday() >= 5:  # 周六(5)或周日(6)
        days_back = today.weekday() - 4  # 推到周五
        today = today - timedelta(days=days_back)

    # 如果是交易日的早盘前，使用前一天
    if today.hour < 15:
        today = today - timedelta(days=1)

    latest_date = today.strftime('%Y%m%d')
    return latest_date, True, "实时"


def main():
    parser = argparse.ArgumentParser(description='增强版实盘选股推荐系统 v3.0')
    parser.add_argument('--date', type=str, help='指定日期 (格式: YYYYMMDD)')
    parser.add_argument('--lookback', type=int, default=150, help='回看天数 (建议150天以上)')
    parser.add_argument('--debug', action='store_true', help='调试模式')
    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("🚀 增强版实盘选股推荐系统 v3.0")
    print("=" * 80)
    print("核心功能:")
    print("  ✅ 多信号交叉验证 (4层金字塔)")
    print("  ✅ 最佳买入时点识别")
    print("  ✅ 财务质量深度筛查")
    print("  ✅ 市场环境自适应")
    print("  ✅ 动态仓位分配")
    print("  ✅ 行业轮动捕捉")
    print("=" * 80 + "\n")

    # ===== 1. 初始化系统 =====
    logger.info("初始化系统...")

    config = Config()
    config.data.use_cache = True

    dm = DataManager(config)
    fe = FactorEngine(config)
    trainer = WalkForwardTrainer(config)
    enhanced_strategy = EnhancedLiveStrategy(config)  # 【关键】增强策略

    # ===== 2. 确定日期 =====
    if args.date:
        latest_date = args.date
        is_real_time = False
        data_source = "用户指定"
        logger.info(f"✓ 使用指定日期: {latest_date}")
    else:
        latest_date, is_real_time, data_source = get_latest_available_trading_date(
            config.data.cache_dir
        )
        logger.info(f"✓ 自动选择日期: {latest_date} (来源: {data_source})")

    # ===== 3. 获取数据 =====
    # 【优先使用缓存】
    cache_dir = Path(config.data.cache_dir)
    cache_files = list(cache_dir.glob('daily_all_*.pkl'))

    if cache_files:
        latest_cache = max(cache_files, key=lambda p: p.stat().st_mtime)
        logger.info(f"使用缓存: {latest_cache.name}")
        
        df_history = pd.read_pickle(latest_cache)
        
        # 使用缓存中的最新日期
        latest_date = df_history['trade_date'].max()
        logger.info(f"✓ 缓存最新日期: {latest_date}")
        
        # 确保start_date是可用的
        start_date = (
            datetime.strptime(latest_date, '%Y%m%d') - timedelta(days=args.lookback)
        ).strftime('%Y%m%d')
    else:
        # 原来的获取逻辑
        try:
            start_date = (
                datetime.strptime(latest_date, '%Y%m%d') - timedelta(days=args.lookback)
            ).strftime('%Y%m%d')
        except ValueError:
            logger.error(f"❌ 日期格式错误: {latest_date}")
            return

        logger.info(f"\n{'=' * 60}")
        logger.info(f"📊 数据准备: {start_date} ~ {latest_date}")
        logger.info(f"{'=' * 60}\n")

        try:
            df_history = dm.get_daily_data(start_date=start_date, end_date=latest_date)
            logger.info(f"✓ 获取数据: {len(df_history)} 条")
        except Exception as e:
            logger.error(f"❌ 数据获取失败: {e}")
            return

    # 添加行业信息
    try:
        industry_df = dm.get_industry_data()
        df_history = df_history.merge(
            industry_df[['ts_code', 'industry']],
            on='ts_code',
            how='left'
        )
        logger.info(f"✓ 添加行业信息")
    except Exception as e:
        logger.warning(f"⚠️  无法获取行业信息: {e}")

    if latest_date not in df_history['trade_date'].values:
        logger.error(f"❌ 数据中不包含 {latest_date}")
        available_dates = sorted(df_history['trade_date'].unique())[-5:]
        logger.info(f"可用日期: {available_dates}")
        logger.info(f"尝试: python {sys.argv[0]} --date {available_dates[-1]}")
        return

    # ===== 4. 计算因子 =====
    logger.info(f"\n{'=' * 60}")
    logger.info("⚙️  计算因子特征...")
    logger.info(f"{'=' * 60}\n")

    try:
        df_with_factors = fe.calculate_all_factors(df_history)
        logger.info(f"✓ 因子计算完成")
    except Exception as e:
        logger.error(f"❌ 因子计算失败: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return

    df_latest = df_with_factors[df_with_factors['trade_date'] == latest_date].copy()

    if df_latest.empty:
        logger.error(f"❌ 因子计算后 {latest_date} 数据为空")
        return

    logger.info(f"✓ 当日候选股票: {len(df_latest)} 只")

    # ===== 5. 加载模型 =====
    logger.info(f"\n{'=' * 60}")
    logger.info("🤖 加载机器学习模型...")
    logger.info(f"{'=' * 60}\n")

    model_path = Path(config.data.cache_dir) / 'latest_model.pkl'
    if not model_path.exists():
        model_path = Path('latest_model.pkl')

    if not model_path.exists():
        logger.error("❌ 未找到模型文件")
        logger.error("请先运行: python main.py --mode train")
        return

    try:
        model = MLModel(config)
        model.model = joblib.load(str(model_path))
        logger.info(f"✓ 模型加载成功")

        # 【关键修复】获取模型训练时使用的特征列表
        if hasattr(model.model, 'feature_names_in_'):
            expected_features = list(model.model.feature_names_in_)
        elif hasattr(model.model, 'feature_name_'):
            expected_features = model.model.feature_name_
        else:
            logger.error("❌ 无法获取模型特征列表")
            return
        
        logger.info(f"   模型需要 {len(expected_features)} 个特征")
        
        # 检查当前数据有哪些特征
        current_features = set(df_latest.columns)
        missing_features = set(expected_features) - current_features
        
        if missing_features:
            logger.warning(f"   缺失 {len(missing_features)} 个特征，将自动生成")
            
            # 为缺失的滞后特征填充
            for feat in missing_features:
                if feat.endswith('_lag1'):
                    # 滞后特征：尝试从前一天获取
                    base_feat = feat[:-5]  # 去掉 '_lag1'
                    if base_feat in df_with_factors.columns:
                        # 从历史数据获取前一天的值
                        df_latest[feat] = df_with_factors.groupby('ts_code')[base_feat].shift(1)
                    else:
                        df_latest[feat] = 0
                else:
                    # 其他缺失特征填0
                    df_latest[feat] = 0
        
        # 确保所有特征都存在且按正确顺序
        X = pd.DataFrame(df_latest[expected_features].copy())
        
        # 填充NaN（滞后特征的第一天会是NaN）
        X = X.fillna(0)
        
        # 预测
        preds = model.predict(X)
        
        # 确保preds是numpy数组以支持后续操作
        import numpy as np
        if not isinstance(preds, np.ndarray):
            preds = np.array(preds)
        
        df_latest['ml_score'] = preds

        logger.info(f"✓ 评分完成")
        logger.info(f"   评分范围: {preds.min():.3f} ~ {preds.max():.3f}")
        logger.info(f"   平均评分: {preds.mean():.3f}")
        logger.info(f"   高分股票(>0.6): {(preds > 0.6).sum()} 只")

    except Exception as e:
        logger.error(f"❌ 模型预测失败: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return

    # ===== 6. 获取指数数据 =====
    logger.info(f"\n{'=' * 60}")
    logger.info("📈 获取指数数据...")
    logger.info(f"{'=' * 60}\n")

    try:
        index_data = dm.get_index_data(
            config.backtest.benchmark,
            start_date
        )
        logger.info(f"✓ 指数数据获取成功")
    except Exception as e:
        logger.warning(f"⚠️  指数数据获取失败，将使用默认策略: {e}")
        index_data = None

    # ===== 7. 增强选股（核心） =====
    logger.info(f"\n{'=' * 60}")
    logger.info("🎯 执行增强版选股策略...")
    logger.info(f"{'=' * 60}\n")

    try:
        selected = enhanced_strategy.select_stocks_enhanced(
            df_date=df_latest,
            df_history=df_with_factors,
            index_data=index_data
        )
    except Exception as e:
        logger.error(f"❌ 选股失败: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return

    # ===== 8. 输出结果 =====
    print("\n" + "=" * 100)
    print(f"📋 {latest_date} 增强版实盘选股推荐")
    print(f"   数据来源: {data_source}")
    print(f"   生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 100 + "\n")

    if selected.empty:
        print("⚠️  今日无推荐股票")
        print("\n可能原因:")
        print("  - 市场整体评分较低")
        print("  - 未找到符合多重验证的标的")
        print("  - 市场环境不佳，系统自动降低推荐数量")
        print("  - 风险控制触发限制\n")
    else:
        # 格式化显示
        print(f"{'代码':10s} {'名称':10s} {'价格':>8s} {'涨跌':>8s} "
              f"{'综合评分':>8s} {'级别':>4s} {'仓位':>6s} {'市场状态':>10s} {'推荐理由':40s}")
        print("-" * 100)

        for _, row in selected.iterrows():
            print(f"{row['ts_code']:10s} "
                  f"{row.get('name', 'N/A'):10s} "
                  f"{row['close']:8.2f} "
                  f"{row.get('pct_chg', 0):+7.2f}% "
                  f"{row.get('composite_score', row.get('ml_score', 0)):8.3f} "
                  f"{row.get('position_tier', 'B'):>4s} "
                  f"{row.get('weight', 0) * 100:5.2f}% "
                  f"{row.get('market_state', 'unknown'):>10s} "
                  f"{row.get('recommendation', '评分良好')[:40]:40s}")

        print("=" * 100)

        # 统计信息
        print(f"\n📊 统计信息:")
        print(f"   推荐数量: {len(selected)} 只")
        print(f"   市场状态: {selected.iloc[0].get('market_state', 'unknown')}")

        if 'composite_score' in selected.columns:
            print(f"   平均评分: {selected['composite_score'].mean():.3f}")

        if 'position_tier' in selected.columns:
            for tier in ['S', 'A', 'B', 'C']:
                count = (selected['position_tier'] == tier).sum()
                if count > 0:
                    print(f"   {tier}级股票: {count} 只")

        # 行业分布
        if 'industry' in selected.columns:
            industry_counts = selected['industry'].value_counts()
            print(f"\n📈 行业分布:")
            for ind, cnt in industry_counts.items():
                print(f"   {ind}: {cnt} 只")

        print()

    # ===== 9. 保存结果 =====
    output_dir = Path('output/enhanced_live_recommendations')
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%H%M%S')

    if not selected.empty:
        # CSV
        csv_path = output_dir / f'recommendations_{latest_date}_{timestamp}.csv'
        selected.to_csv(csv_path, index=False, encoding='utf-8-sig')
        logger.info(f"\n✓ CSV结果已保存: {csv_path}")

        # Excel
        try:
            excel_path = output_dir / f'recommendations_{latest_date}_{timestamp}.xlsx'
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                selected.to_excel(writer, sheet_name='推荐股票', index=False)
            logger.info(f"✓ Excel报告已保存: {excel_path}")
        except Exception as e:
            logger.warning(f"⚠️  Excel保存失败: {e}")

    print(f"\n{'=' * 100}")
    print("✅ 程序执行完成")
    print(f"{'=' * 100}\n")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n用户中断程序")
    except Exception as e:
        logger.error(f"程序异常: {e}", exc_info=True)
        raise