"""
优化数据获取脚本
解决Tushare限流问题

策略:
1. 只获取最新1天数据（增量更新）
2. 智能重试和错误恢复
3. 优先使用缓存

使用方法:
    python fetch_latest_data.py
"""

import logging
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import time

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_latest_cache():
    """获取最新缓存文件"""
    cache_dir = Path('data_cache')

    if not cache_dir.exists():
        return None, None

    cache_files = list(cache_dir.glob('daily_all_*.pkl'))

    if not cache_files:
        return None, None

    latest_cache = max(cache_files, key=lambda p: p.stat().st_mtime)

    # 解析日期
    import re
    match = re.search(r'_(\d{8})\.pkl', latest_cache.name)
    if match:
        cache_date = match.group(1)
        return latest_cache, cache_date

    return latest_cache, None


def fetch_single_day(ts_api, date):
    """获取单日数据（带重试）"""
    max_retries = 3
    retry_delay = 60  # 秒

    for attempt in range(max_retries):
        try:
            df = ts_api.daily(trade_date=date)

            if df is not None and not df.empty:
                logger.info(f"✓ 获取 {date}: {len(df)} 条记录")
                return df
            else:
                logger.warning(f"⚠️  {date} 无数据")
                return pd.DataFrame()

        except Exception as e:
            if 'exceeded' in str(e).lower() or 'limit' in str(e).lower():
                if attempt < max_retries - 1:
                    logger.warning(f"限流，等待 {retry_delay} 秒... (重试 {attempt + 1}/{max_retries})")
                    time.sleep(retry_delay)
                else:
                    logger.error(f"❌ {date} 获取失败: 超过重试次数")
                    return None
            else:
                logger.error(f"❌ {date} 获取失败: {e}")
                return None

    return None


def update_cache_incremental():
    """增量更新缓存"""
    logger.info("\n" + "=" * 60)
    logger.info("📊 增量更新数据")
    logger.info("=" * 60)

    # 1. 加载旧缓存
    cache_path, cache_date = get_latest_cache()

    if not cache_path:
        logger.error("❌ 未找到缓存文件")
        logger.info("请先运行完整获取:")
        logger.info("  python run_live_strategy_enhanced.py --lookback 30")
        return False

    logger.info(f"1. 找到缓存: {cache_path.name}")
    logger.info(f"   缓存日期: {cache_date}")

    df_old = pd.read_pickle(cache_path)
    logger.info(f"   旧数据: {len(df_old)} 条")

    # 2. 确定需要更新的日期
    cache_dt = datetime.strptime(cache_date, '%Y%m%d')
    today = datetime.now()

    days_to_fetch = []
    current_dt = cache_dt + timedelta(days=1)

    while current_dt <= today:
        # 跳过周末
        if current_dt.weekday() < 5:
            days_to_fetch.append(current_dt.strftime('%Y%m%d'))
        current_dt += timedelta(days=1)

    if not days_to_fetch:
        logger.info("✓ 缓存已是最新")
        return True

    logger.info(f"2. 需要更新 {len(days_to_fetch)} 天")

    # 3. 初始化Tushare
    try:
        import tushare as ts
        from config import Config

        config = Config()
        ts.set_token(config.data.tushare_token)
        pro = ts.pro_api()

        logger.info("✓ Tushare已初始化")
    except Exception as e:
        logger.error(f"❌ Tushare初始化失败: {e}")
        return False

    # 4. 逐日获取
    new_data = []

    for i, date in enumerate(days_to_fetch, 1):
        logger.info(f"[{i}/{len(days_to_fetch)}] 获取 {date}...")

        df_day = fetch_single_day(pro, date)

        if df_day is not None and not df_day.empty:
            new_data.append(df_day)

        # 避免限流
        if i < len(days_to_fetch):
            time.sleep(0.5)

    # 5. 合并数据
    if new_data:
        df_new = pd.concat(new_data, ignore_index=True)
        logger.info(f"✓ 新数据: {len(df_new)} 条")

        df_combined = pd.concat([df_old, df_new], ignore_index=True)
        df_combined = df_combined.drop_duplicates(
            subset=['ts_code', 'trade_date'],
            keep='last'
        )

        logger.info(f"✓ 合并后: {len(df_combined)} 条")

        # 6. 保存新缓存
        latest_date = df_combined['trade_date'].max()
        new_cache_path = cache_path.parent / f'daily_all_{cache_date}_{latest_date}.pkl'

        df_combined.to_pickle(new_cache_path)
        logger.info(f"✓ 已保存: {new_cache_path.name}")

        return True
    else:
        logger.warning("⚠️  未获取到新数据")
        return False


def quick_fetch_today():
    """快速获取今日数据"""
    logger.info("\n" + "=" * 60)
    logger.info("🚀 快速获取今日数据")
    logger.info("=" * 60)

    # 确定今日日期
    today = datetime.now()

    # 如果是周末或早盘前，使用上一交易日
    if today.weekday() >= 5:  # 周末
        days_back = today.weekday() - 4
        today = today - timedelta(days=days_back)
    elif today.hour < 15:  # 收盘前
        today = today - timedelta(days=1)

    today_str = today.strftime('%Y%m%d')

    logger.info(f"目标日期: {today_str}")

    # 初始化Tushare
    try:
        import tushare as ts
        from config import Config

        config = Config()
        ts.set_token(config.data.tushare_token)
        pro = ts.pro_api()

        logger.info("✓ Tushare已初始化")
    except Exception as e:
        logger.error(f"❌ Tushare初始化失败: {e}")
        return None

    # 获取数据
    df_today = fetch_single_day(pro, today_str)

    if df_today is not None and not df_today.empty:
        # 保存
        output_path = Path('data_cache') / f'daily_{today_str}.pkl'
        output_path.parent.mkdir(exist_ok=True)

        df_today.to_pickle(output_path)
        logger.info(f"✓ 已保存: {output_path}")

        return df_today
    else:
        logger.error("❌ 获取失败")
        return None


def main():
    print("\n" + "=" * 70)
    print(" " * 20 + "📊 数据获取工具")
    print("=" * 70)

    print("""
选择操作:

1. 增量更新（推荐）
   - 只获取缓存后的新数据
   - 速度快，不易限流

2. 快速获取今日
   - 只获取今天1天数据
   - 最快，但缓存不完整

3. 查看缓存状态
   - 查看现有缓存信息

0. 退出
    """)

    choice = input("请选择 (0-3): ").strip()

    if choice == '1':
        update_cache_incremental()

    elif choice == '2':
        df = quick_fetch_today()
        if df is not None:
            print(f"\n✓ 获取成功: {len(df)} 条记录")

    elif choice == '3':
        cache_path, cache_date = get_latest_cache()
        if cache_path:
            df = pd.read_pickle(cache_path)

            print("\n" + "=" * 60)
            print("缓存信息:")
            print("=" * 60)
            print(f"文件: {cache_path.name}")
            print(f"大小: {cache_path.stat().st_size / 1024 / 1024:.2f} MB")
            print(f"日期: {cache_date}")
            print(f"记录数: {len(df)}")
            print(f"股票数: {df['ts_code'].nunique()}")
            print(f"日期范围: {df['trade_date'].min()} ~ {df['trade_date'].max()}")

            days_old = (datetime.now() - datetime.strptime(cache_date, '%Y%m%d')).days
            print(f"距今: {days_old} 天")

            if days_old == 0:
                print("✓ 缓存是最新的")
            elif days_old <= 3:
                print("✓ 缓存较新")
            else:
                print("⚠️  缓存较旧，建议更新")

            print("=" * 60)
        else:
            print("\n❌ 未找到缓存文件")

    elif choice == '0':
        print("退出")

    else:
        print("无效选择")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n已取消")
    except Exception as e:
        logger.error(f"错误: {e}", exc_info=True)