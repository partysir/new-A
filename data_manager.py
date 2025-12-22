"""
数据处理模块
负责数据获取、清洗、缓存和增量更新
"""

import os
import pickle
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)


class DataManager:
    """数据管理器"""

    def __init__(self, config):
        self.config = config
        self.cache_dir = Path(config.data.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # 初始化tushare
        self._init_tushare()

        # 缓存
        self._cache = {}

    def _init_tushare(self):
        """初始化tushare"""
        try:
            import tushare as ts
            ts.set_token(self.config.data.tushare_token)
            self.ts_pro = ts.pro_api()
            logger.info("Tushare initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Tushare: {e}")
            raise

    def get_stock_list(self, date: str = None) -> pd.DataFrame:
        """
        获取股票列表

        Args:
            date: 交易日期，格式YYYYMMDD

        Returns:
            股票列表DataFrame
        """
        cache_key = f"stock_list_{date}"

        # 检查缓存
        if self.config.data.use_cache and cache_key in self._cache:
            return self._cache[cache_key].copy()

        try:
            # 获取所有A股
            df = self.ts_pro.stock_basic(
                exchange='',
                list_status='L',
                fields='ts_code,symbol,name,area,industry,market,list_date'
            )

            # 过滤条件
            if self.config.data.exclude_st:
                df = df[~df['name'].str.contains('ST|退', na=False)]

            # 过滤新股
            if date and self.config.data.exclude_new_stock_days > 0:
                date_dt = pd.to_datetime(date)
                df['list_date'] = pd.to_datetime(df['list_date'])
                min_list_date = date_dt - timedelta(days=self.config.data.exclude_new_stock_days)
                df = df[df['list_date'] <= min_list_date]

            # 缓存
            if self.config.data.use_cache:
                self._cache[cache_key] = df.copy()

            logger.info(f"Got {len(df)} stocks for date {date}")
            return df

        except Exception as e:
            logger.error(f"Failed to get stock list: {e}")
            raise

    def get_daily_data(
        self,
        ts_code: str = None,
        start_date: str = None,
        end_date: str = None,
        fields: List[str] = None
    ) -> pd.DataFrame:
        """
        获取日线数据

        Args:
            ts_code: 股票代码，None表示所有股票
            start_date: 开始日期
            end_date: 结束日期
            fields: 字段列表

        Returns:
            日线数据DataFrame
        """
        start_date = start_date or self.config.data.start_date
        end_date = end_date or datetime.now().strftime('%Y%m%d')

        cache_file = self.cache_dir / f"daily_{ts_code or 'all'}_{start_date}_{end_date}.pkl"

        # 检查缓存
        if self.config.data.use_cache and cache_file.exists():
            logger.info(f"Loading from cache: {cache_file}")
            return pd.read_pickle(cache_file)

        try:
            if ts_code:
                # 单只股票
                df = self.ts_pro.daily(
                    ts_code=ts_code,
                    start_date=start_date,
                    end_date=end_date,
                    fields=fields
                )
            else:
                # 所有股票 - 使用优化的批量获取策略
                logger.info("Using optimized batch fetching strategy...")
                df = self._get_daily_data_batch(start_date, end_date, fields)


            # 数据清洗
            df = self._clean_daily_data(df)

            # 保存缓存
            if self.config.data.use_cache:
                df.to_pickle(cache_file)
                logger.info(f"Saved to cache: {cache_file}")

            return df

        except Exception as e:
            logger.error(f"Failed to get daily data: {e}")
            raise

    def _get_daily_data_batch(
        self,
        start_date: str,
        end_date: str,
        fields: List[str] = None
    ) -> pd.DataFrame:
        """
        批量获取日线数据（优化版）

        策略：按日期分批获取，而非按股票逐个获取
        """
        import time

        # 获取交易日历
        trade_dates = self._get_trade_calendar(start_date, end_date)
        logger.info(f"Fetching data for {len(trade_dates)} trading days...")

        df_list = []
        failed_dates = []

        # 每分钟最多450次调用（留余量）
        delay = 60.0 / 450

        for i, trade_date in enumerate(trade_dates):
            retry_count = 0
            max_retries = 3

            while retry_count < max_retries:
                try:
                    # 获取某一天所有股票的数据
                    df_day = self.ts_pro.daily(
                        trade_date=trade_date,
                        fields=fields
                    )

                    if df_day is not None and not df_day.empty:
                        df_list.append(df_day)

                        # 每10天显示一次进度
                        if (i + 1) % 10 == 0 or i == len(trade_dates) - 1:
                            logger.info(f"Progress: {i+1}/{len(trade_dates)} days, " +
                                      f"got {len(df_day)} records for {trade_date}")
                        break
                    else:
                        logger.warning(f"No data for {trade_date}")
                        break

                except Exception as e:
                    if "每分钟最多访问" in str(e):
                        logger.warning(f"Rate limit hit, waiting 65 seconds... (retry {retry_count+1}/{max_retries})")
                        time.sleep(65)
                        retry_count += 1
                    else:
                        logger.warning(f"Failed to get data for {trade_date}: {e}")
                        failed_dates.append(trade_date)
                        break

            if retry_count >= max_retries:
                logger.error(f"Failed to get data for {trade_date} after {max_retries} retries")
                failed_dates.append(trade_date)

            # 添加延时
            time.sleep(delay)

        if failed_dates:
            logger.warning(f"Failed to get data for {len(failed_dates)} dates")

        if not df_list:
            raise ValueError("No data fetched successfully")

        df = pd.concat(df_list, ignore_index=True)
        logger.info(f"Successfully fetched {len(df)} records")

        return df

    def _get_trade_calendar(self, start_date: str, end_date: str) -> List[str]:
        """获取交易日历"""
        cache_key = f"trade_cal_{start_date}_{end_date}"

        if cache_key in self._cache:
            return self._cache[cache_key]

        try:
            df = self.ts_pro.trade_cal(
                exchange='SSE',
                start_date=start_date,
                end_date=end_date,
                is_open='1'
            )

            trade_dates = df['cal_date'].tolist()
            self._cache[cache_key] = trade_dates

            return trade_dates

        except Exception as e:
            logger.warning(f"Failed to get trade calendar: {e}, using date range")
            # 生成日期范围
            dates = pd.date_range(start_date, end_date, freq='D')
            return [d.strftime('%Y%m%d') for d in dates]

    def _clean_daily_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """清洗日线数据"""
        # 删除重复数据
        df = df.drop_duplicates(subset=['ts_code', 'trade_date'])

        # 排序
        df = df.sort_values(['ts_code', 'trade_date'])

        # 处理缺失值
        df = df.dropna(subset=['close', 'open', 'high', 'low', 'vol'])

        # 过滤异常数据
        df = df[df['close'] > 0]
        df = df[df['vol'] > 0]

        # 过滤停牌
        if self.config.data.exclude_suspended:
            df = df[df['vol'] > 100]  # 成交量>100手

        # 过滤流动性不足
        if self.config.data.min_liquidity > 0:
            df['amount'] = df['amount'] * 1000  # 转换为元
            df = df[df['amount'] >= self.config.data.min_liquidity]

        # 重置索引
        df = df.reset_index(drop=True)

        logger.info(f"Cleaned data: {len(df)} records")
        return df

    def incremental_update(self, last_date: str = None) -> pd.DataFrame:
        """
        增量更新数据

        Args:
            last_date: 上次更新日期

        Returns:
            新增数据DataFrame
        """
        if not self.config.data.incremental_update:
            return self.get_daily_data()

        # 确定更新起始日期
        if last_date is None:
            # 从缓存中查找最新日期
            cache_files = list(self.cache_dir.glob("daily_all_*.pkl"))
            if cache_files:
                latest_file = max(cache_files, key=os.path.getctime)
                df_cached = pd.read_pickle(latest_file)
                last_date = df_cached['trade_date'].max()
            else:
                last_date = self.config.data.start_date

        # 计算更新日期范围
        start_date = (pd.to_datetime(last_date) + timedelta(days=1)).strftime('%Y%m%d')
        end_date = datetime.now().strftime('%Y%m%d')

        if start_date >= end_date:
            logger.info("Data is already up to date")
            return pd.DataFrame()

        logger.info(f"Incremental update from {start_date} to {end_date}")

        # 获取增量数据
        df_new = self.get_daily_data(start_date=start_date, end_date=end_date)

        # 合并历史数据
        if last_date != self.config.data.start_date:
            df_old = self.get_daily_data(end_date=last_date)
            df = pd.concat([df_old, df_new], ignore_index=True)
        else:
            df = df_new

        return df

    def get_fundamental_data(
        self,
        ts_code: str = None,
        start_date: str = None,
        end_date: str = None
    ) -> pd.DataFrame:
        """
        获取基本面数据

        Args:
            ts_code: 股票代码
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            基本面数据DataFrame
        """
        start_date = start_date or self.config.data.start_date
        end_date = end_date or datetime.now().strftime('%Y%m%d')

        try:
            # 获取财务数据
            df_basic = self.ts_pro.daily_basic(
                ts_code=ts_code,
                start_date=start_date,
                end_date=end_date,
                fields='ts_code,trade_date,close,pe_ttm,pb,ps_ttm,pcf_ncf,total_mv,circ_mv'
            )

            # 获取财务指标
            df_indicator = self.ts_pro.fina_indicator(
                ts_code=ts_code,
                start_date=start_date[:4] + '0101',
                end_date=end_date[:4] + '1231',
                fields='ts_code,end_date,roe,roa,gross_margin,netprofit_margin,current_ratio,debt_to_assets'
            )

            # 合并数据
            df_basic['trade_date'] = pd.to_datetime(df_basic['trade_date'])
            df_indicator['end_date'] = pd.to_datetime(df_indicator['end_date'])

            # 将季度数据扩展到每日
            df_merged = pd.merge_asof(
                df_basic.sort_values('trade_date'),
                df_indicator.sort_values('end_date'),
                left_on='trade_date',
                right_on='end_date',
                by='ts_code',
                direction='backward'
            )

            return df_merged

        except Exception as e:
            logger.error(f"Failed to get fundamental data: {e}")
            raise

    def get_money_flow_data(
        self,
        ts_code: str = None,
        start_date: str = None,
        end_date: str = None
    ) -> pd.DataFrame:
        """
        获取资金流数据

        Args:
            ts_code: 股票代码
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            资金流数据DataFrame
        """
        start_date = start_date or self.config.data.start_date
        end_date = end_date or datetime.now().strftime('%Y%m%d')

        try:
            df = self.ts_pro.moneyflow(
                ts_code=ts_code,
                start_date=start_date,
                end_date=end_date,
                fields='ts_code,trade_date,buy_sm_amount,sell_sm_amount,buy_md_amount,sell_md_amount,buy_lg_amount,sell_lg_amount,buy_elg_amount,sell_elg_amount'
            )

            return df

        except Exception as e:
            logger.error(f"Failed to get money flow data: {e}")
            raise

    def get_index_data(
        self,
        ts_code: str = '000300.SH',
        start_date: str = None,
        end_date: str = None
    ) -> pd.DataFrame:
        """
        获取指数数据

        Args:
            ts_code: 指数代码
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            指数数据DataFrame
        """
        start_date = start_date or self.config.data.start_date
        end_date = end_date or datetime.now().strftime('%Y%m%d')

        try:
            df = self.ts_pro.index_daily(
                ts_code=ts_code,
                start_date=start_date,
                end_date=end_date,
                fields='ts_code,trade_date,close,open,high,low,vol,amount'
            )

            return df

        except Exception as e:
            logger.error(f"Failed to get index data: {e}")
            raise

    def get_industry_data(self, ts_code: str = None) -> pd.DataFrame:
        """
        获取行业分类数据

        Args:
            ts_code: 股票代码

        Returns:
            行业分类DataFrame
        """
        try:
            df = self.ts_pro.stock_basic(
                ts_code=ts_code,
                fields='ts_code,name,industry'
            )
            return df
        except Exception as e:
            logger.error(f"Failed to get industry data: {e}")
            raise

    def get_trade_calendar(
        self,
        start_date: str = None,
        end_date: str = None
    ) -> pd.DataFrame:
        """
        获取交易日历

        Args:
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            交易日历DataFrame
        """
        start_date = start_date or self.config.data.start_date
        end_date = end_date or datetime.now().strftime('%Y%m%d')

        try:
            df = self.ts_pro.trade_cal(
                exchange='SSE',
                start_date=start_date,
                end_date=end_date,
                fields='cal_date,is_open'
            )

            # 只保留交易日
            df = df[df['is_open'] == 1]

            return df

        except Exception as e:
            logger.error(f"Failed to get trade calendar: {e}")
            raise

    def clear_cache(self):
        """清除缓存"""
        self._cache.clear()

        if self.cache_dir.exists():
            for file in self.cache_dir.glob("*.pkl"):
                file.unlink()
            logger.info("Cache cleared")

    def get_memory_usage(self) -> float:
        """获取内存使用情况(GB)"""
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024 / 1024


class DataQuality:
    """数据质量检查"""

    @staticmethod
    def check_missing_values(df: pd.DataFrame) -> pd.DataFrame:
        """检查缺失值"""
        missing = df.isnull().sum()
        missing_pct = 100 * missing / len(df)

        result = pd.DataFrame({
            'missing_count': missing,
            'missing_percentage': missing_pct
        })

        return result[result['missing_count'] > 0]

    @staticmethod
    def check_duplicates(df: pd.DataFrame, subset: List[str] = None) -> int:
        """检查重复值"""
        return df.duplicated(subset=subset).sum()

    @staticmethod
    def check_outliers(df: pd.DataFrame, column: str, method: str = 'iqr') -> pd.Series:
        """
        检查异常值

        Args:
            df: 数据
            column: 列名
            method: 方法(iqr, zscore)

        Returns:
            异常值布尔Series
        """
        if method == 'iqr':
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            return (df[column] < lower) | (df[column] > upper)

        elif method == 'zscore':
            from scipy import stats
            z_scores = np.abs(stats.zscore(df[column].dropna()))
            return z_scores > 3

        else:
            raise ValueError(f"Unknown method: {method}")

    @staticmethod
    def generate_report(df: pd.DataFrame) -> Dict:
        """生成数据质量报告"""
        report = {
            'shape': df.shape,
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
            'missing_values': DataQuality.check_missing_values(df).to_dict(),
            'duplicates': DataQuality.check_duplicates(df),
            'dtypes': df.dtypes.value_counts().to_dict()
        }

        return report