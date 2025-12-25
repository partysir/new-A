"""
因子工程模块 - 改进版 v2.0
主要改进：增强市值中和机制
"""

import logging
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from scipy import stats
import warnings

try:
    from pandarallel import pandarallel
    HAS_PANDARALLEL = True
except ImportError:
    HAS_PANDARALLEL = False

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class FactorEngine:
    """因子计算引擎 - 改进版"""

    def __init__(self, config):
        self.config = config
        self._init_parallel()

    def _init_parallel(self):
        """初始化并行计算"""
        if self.config.system.use_multiprocessing and HAS_PANDARALLEL:
            try:
                n_workers = self.config.system.n_jobs
                if n_workers == -1:
                    import psutil
                    n_workers = psutil.cpu_count(logical=False)
                
                pandarallel.initialize(
                    nb_workers=n_workers, 
                    progress_bar=True,
                    verbose=1
                )
                self.use_parallel = True
                logger.info(f"Pandarallel initialized with {n_workers} workers")
            except Exception as e:
                logger.warning(f"Failed to initialize pandarallel: {e}")
                self.use_parallel = False
        else:
            self.use_parallel = False

    def calculate_all_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算所有因子"""
        logger.info("Starting factor calculation...")

        if hasattr(self, 'use_parallel') and self.use_parallel:
            logger.info("Using parallel processing for factor calculation...")
            df_with_factors = df.groupby('ts_code', group_keys=False).parallel_apply(
                self._calculate_stock_factors
            )
        else:
            df_with_factors = df.groupby('ts_code', group_keys=False).apply(
                self._calculate_stock_factors
            )

        # 因子后处理
        if self.config.factor.winsorize:
            df_with_factors = self._winsorize_factors(df_with_factors)

        if self.config.factor.standardize:
            df_with_factors = self._standardize_factors(df_with_factors)

        if self.config.factor.neutralize:
            df_with_factors = self._neutralize_factors(df_with_factors)

        if self.config.factor.orthogonalize:
            df_with_factors = self._orthogonalize_factors(df_with_factors)

        df_with_factors = self.calculate_realtime_indicators(df_with_factors)
        
        # 因子质量控制
        df_with_factors = self.factor_quality_control(df_with_factors)
        
        logger.info(f"Factor calculation completed: {len(df_with_factors)} records")
        return df_with_factors

    def _calculate_stock_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算单只股票的所有因子"""
        df = df.copy()
        df = df.sort_values('trade_date')

        df = self._calculate_technical_factors(df)
        df = self._calculate_money_flow_factors(df)
        df = self._calculate_alpha_factors(df)
        df = self._calculate_alpha101_factors(df)

        return df

    def _calculate_technical_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算技术因子（稳健版）"""
        
        # ✅ 新增：无条件计算基础均线（放在函数开头）
        df['ma5'] = df['close'].rolling(5).mean()
        df['ma10'] = df['close'].rolling(10).mean()
        df['ma20'] = df['close'].rolling(20).mean()
        df['ma60'] = df['close'].rolling(60).mean()

        # === 改进1: 多周期动量综合 ===
        # 不只看单一周期,综合多个周期减少噪音
        if 'momentum_composite' in self.config.factor.technical_factors:
            df['momentum_composite'] = (
                0.4 * df['close'].pct_change(5) +   # 短期
                0.3 * df['close'].pct_change(20) +  # 中期
                0.3 * df['close'].pct_change(60)    # 长期
            ).rolling(3).mean()  # 平滑处理

        # === 改进2: 波动率调整动量 ===
        # 高波动股票的动量信号可靠性低,需要惩罚
        if 'momentum_20' in self.config.factor.technical_factors:
            df['momentum_20'] = df['close'].pct_change(20)
            df['volatility_20'] = df['close'].pct_change().rolling(20).std()
            df['momentum_adjusted'] = df['momentum_20'] / (df['volatility_20'] + 0.01)
            # 平滑处理
            df['momentum_adjusted'] = df['momentum_adjusted'].rolling(3).mean()

        if 'momentum_60' in self.config.factor.technical_factors:
            df['momentum_60'] = df['close'].pct_change(60)
            df['volatility_60'] = df['close'].pct_change().rolling(60).std()
            df['momentum_60_adjusted'] = df['momentum_60'] / (df['volatility_60'] + 0.01)
            # 平滑处理
            df['momentum_60_adjusted'] = df['momentum_60_adjusted'].rolling(5).mean()

        # === 改进3: 量价一致性 ===
        # 价涨量增才是真突破,价涨量缩要警惕
        if 'volume_price_consistency' in self.config.factor.technical_factors:
            price_change_5 = df['close'].pct_change(5)
            volume_change_5 = df['vol'].pct_change(5)
            df['vp_consistency'] = np.sign(price_change_5) == np.sign(volume_change_5)
            df['vp_consistency_score'] = df['vp_consistency'].rolling(10).mean()

        # === 改进4: 支撑压力位 ===
        # 接近支撑位买入,接近压力位卖出
        if 'price_position' in self.config.factor.technical_factors:
            df['support_20'] = df['low'].rolling(20).min()
            df['resistance_20'] = df['high'].rolling(20).max()
            df['price_position'] = (
                (df['close'] - df['support_20']) / 
                (df['resistance_20'] - df['support_20'] + 0.01)
            )

        # === 改进5: 趋势一致性(多周期) ===
        # MA5 > MA10 > MA20 > MA60 = 多头排列
        if 'trend_alignment' in self.config.factor.technical_factors:
            df['trend_alignment'] = (
                (df['ma5'] > df['ma10']).astype(int) +
                (df['ma10'] > df['ma20']).astype(int) +
                (df['ma20'] > df['ma60']).astype(int)
            ) / 3.0  # 归一化到0-1

        # === 改进6: 反转因子(短期超跌反弹) ===
        if 'reversal_5' in self.config.factor.technical_factors:
            df['reversal_5'] = -df['close'].pct_change(5)  # 取反，超跌反弹
            df['reversal_5'] = df['reversal_5'].rolling(3).mean()  # 平滑

        # === 保留并优化传统指标 ===
        if 'rsi_14' in self.config.factor.technical_factors:
            df['rsi_14'] = self._calculate_rsi(df['close'], 14)
            # 平滑处理
            df['rsi_14'] = df['rsi_14'].rolling(5).mean()

        if 'macd' in self.config.factor.technical_factors:
            df['macd'] = self._calculate_macd(df['close'])
            # 平滑处理
            df['macd'] = df['macd'].rolling(3).mean()

        # 【优化】对波动性较大的因子进行平滑处理
        if 'bbands_width' in self.config.factor.technical_factors:
            df['bbands_width'] = self._calculate_bbands_width(df['close'], 20)
            # 布林带宽度波动较大，进行平滑处理
            df['bbands_width'] = df['bbands_width'].rolling(5).mean()

        # 【优化】对波动性较大的因子进行平滑处理
        if 'atr_14' in self.config.factor.technical_factors:
            df['atr_14'] = self._calculate_atr(df, 14)
            # ATR波动较大，进行平滑处理
            df['atr_14'] = df['atr_14'].rolling(5).mean()

        if 'adx_14' in self.config.factor.technical_factors:
            df['adx_14'] = self._calculate_adx(df, 14)
            # 平滑处理
            df['adx_14'] = df['adx_14'].rolling(3).mean()

        if 'cci_20' in self.config.factor.technical_factors:
            df['cci_20'] = self._calculate_cci(df, 20)
            # CCI波动较大，进行平滑处理
            df['cci_20'] = df['cci_20'].rolling(5).mean()

        if 'willr_14' in self.config.factor.technical_factors:
            df['willr_14'] = self._calculate_willr(df, 14)
            # Williams %R波动较大，进行平滑处理
            df['willr_14'] = df['willr_14'].rolling(5).mean()

        if 'stoch_k' in self.config.factor.technical_factors:
            df['stoch_k'] = self._calculate_stoch(df, 14)
            # 随机指标波动较大，进行平滑处理
            df['stoch_k'] = df['stoch_k'].rolling(5).mean()

        if 'volume_ratio' in self.config.factor.technical_factors:
            df['volume_ratio'] = df['vol'] / df['vol'].rolling(20).mean()
            # 量比波动较大，进行平滑处理
            df['volume_ratio'] = df['volume_ratio'].rolling(5).mean()

        return df

    @staticmethod
    def _calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    @staticmethod
    def _calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        macd_histogram = macd_line - signal_line
        return macd_histogram

    @staticmethod
    def _calculate_bbands_width(prices: pd.Series, period: int = 20, std: float = 2.0) -> pd.Series:
        ma = prices.rolling(window=period).mean()
        std_dev = prices.rolling(window=period).std()
        upper = ma + (std_dev * std)
        lower = ma - (std_dev * std)
        width = (upper - lower) / ma
        return width

    @staticmethod
    def _calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return atr

    @staticmethod
    def _calculate_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
        high_diff = df['high'].diff()
        low_diff = -df['low'].diff()

        pos_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
        neg_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)

        atr = FactorEngine._calculate_atr(df, period)

        pos_di = 100 * (pos_dm.rolling(window=period).mean() / atr)
        neg_di = 100 * (neg_dm.rolling(window=period).mean() / atr)

        dx = 100 * np.abs(pos_di - neg_di) / (pos_di + neg_di)
        adx = dx.rolling(window=period).mean()

        return adx

    @staticmethod
    def _calculate_cci(df: pd.DataFrame, period: int = 20) -> pd.Series:
        tp = (df['high'] + df['low'] + df['close']) / 3
        ma = tp.rolling(window=period).mean()
        mad = tp.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean())
        cci = (tp - ma) / (0.015 * mad)
        return cci

    @staticmethod
    def _calculate_willr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        highest_high = df['high'].rolling(window=period).max()
        lowest_low = df['low'].rolling(window=period).min()
        willr = -100 * (highest_high - df['close']) / (highest_high - lowest_low)
        return willr

    @staticmethod
    def _calculate_stoch(df: pd.DataFrame, period: int = 14) -> pd.Series:
        lowest_low = df['low'].rolling(window=period).min()
        highest_high = df['high'].rolling(window=period).max()
        stoch_k = 100 * (df['close'] - lowest_low) / (highest_high - lowest_low)
        return stoch_k

    def _calculate_money_flow_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算资金流因子（保持原逻辑）"""
        if 'turnover_rate' in self.config.factor.money_flow_factors:
            df['turnover_rate'] = df['vol'] / df.get('float_share', df['vol'].mean())

        if 'amount_change' in self.config.factor.money_flow_factors:
            df['amount_change'] = df['amount'].pct_change(5)

        if 'volume_price_corr' in self.config.factor.money_flow_factors:
            df['volume_price_corr'] = df['vol'].rolling(20).corr(df['close'])

        if 'vwap_ratio' in self.config.factor.money_flow_factors:
            df['vwap'] = (df['amount'] * 1000) / (df['vol'] * 100)
            df['vwap_ratio'] = df['close'] / df['vwap'] - 1

        if 'big_order_ratio' in self.config.factor.money_flow_factors:
            if 'buy_lg_amount' in df.columns:
                total_amount = df['amount']
                big_amount = df['buy_lg_amount'] + df['buy_elg_amount']
                df['big_order_ratio'] = big_amount / total_amount

        if 'main_force_inflow' in self.config.factor.money_flow_factors:
            if 'buy_lg_amount' in df.columns:
                df['main_force_inflow'] = (
                    df['buy_lg_amount'] + df['buy_elg_amount'] -
                    df['sell_lg_amount'] - df['sell_elg_amount']
                ) / df['amount']

        return df

    def _calculate_alpha_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算Alpha因子（保持原逻辑）"""
        if 'alpha_001' in self.config.factor.alpha_factors:
            df['alpha_001'] = (df['close'] - df['open']) / ((df['high'] - df['low']) + 0.001)

        if 'alpha_002' in self.config.factor.alpha_factors:
            df['alpha_002'] = df['close'].rolling(20).corr(df['vol'])

        if 'alpha_003' in self.config.factor.alpha_factors:
            df['alpha_003'] = df['close'].rolling(20).std() / df['close'].rolling(20).mean()

        return df

    def _calculate_alpha101_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算Alpha101因子（保持原逻辑）"""
        close = df['close']
        open_ = df['open']
        high = df['high']
        low = df['low']
        vol = df['vol']
        
        def correlation(x, y, d):
            return x.rolling(window=d).corr(y)

        df['alpha_006'] = -1 * correlation(open_, vol, 10)
        df['alpha_012'] = np.sign(vol.diff()) * (-1 * close.diff())
        
        high_20_max = high.rolling(20).max()
        df['alpha_023'] = np.where(high > high_20_max, -1 * high.diff(2), 0)
        
        numerator = (low - close) * (open_ ** 5)
        denominator = (low - high) * (close ** 5) + 0.0001
        df['alpha_054'] = -1 * (numerator / denominator)

        for col in df.columns:
            if col.startswith('alpha_'):
                df[col] = df[col].fillna(0)

        return df

    def calculate_realtime_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算实盘专用指标（保持原逻辑）"""
        if 'buy_lg_amount' in df.columns and 'amount' in df.columns:
            df['smart_flow_rate'] = (
                df['buy_lg_amount'] + df['buy_elg_amount'] - 
                df['sell_lg_amount'] - df['sell_elg_amount']
            ) / (df['amount'] + 1e-9)
            
            df['smart_money_score'] = (
                df['smart_flow_rate'] - df['smart_flow_rate'].rolling(20).mean()
            ) / (df['smart_flow_rate'].rolling(20).std() + 1e-9)
        else:
            df['smart_money_score'] = df['close'].pct_change() * df['vol'].pct_change()

        ma_20 = df['close'].rolling(20).mean()
        df['bias_20'] = (df['close'] - ma_20) / ma_20
        
        upper = df['close'].rolling(20).mean() + 2 * df['close'].rolling(20).std()
        lower = df['close'].rolling(20).mean() - 2 * df['close'].rolling(20).std()
        bandwidth = (upper - lower) / ma_20
        df['trend_energy'] = df['bias_20'] / (bandwidth + 0.01)

        support_level = df['low'].rolling(20).min()
        df['safety_margin'] = (df['close'] - support_level) / df['close']

        return df

    def factor_quality_control(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        因子质量控制 - 剔除低质量因子
        """
        factor_cols = [c for c in df.columns if c.startswith(('momentum_', 'alpha_', 'vp_', 'rsi_', 'macd', 'cci_', 'willr_', 'stoch_', 'volume_', 'price_', 'trend_', 'reversal_'))]
        
        for factor in factor_cols:
            if factor not in df.columns:
                continue
                
            # 1. 检查缺失率
            # 🟢 修改点 1: 提高缺失率容忍度 (从 0.3 改为 0.8 或更高)
            # 在实盘中，即使很多股票没有数据，我们也要保留列，为了那些有数据的股票
            missing_rate = df[factor].isna().sum() / len(df)
            if missing_rate > 0.8:  # 只有当80%以上都缺失时才剔除
                logger.warning(f"Factor {factor} has {missing_rate:.1%} missing, dropping")
                df = df.drop(columns=[factor])
                continue
            
            # 🟢 修改点 2: 仅在有 'forward_return' 时才检查 IC 和单调性
            # 实盘推理时 df 中没有 'forward_return'，这段逻辑会自动跳过，这是对的。
            if 'forward_return' in df.columns:
                # 2. 检查信息系数(IC) - 如果有forward_return列
                # 计算IC时去除NaN值
                mask = ~(df[factor].isna() | df['forward_return'].isna())
                if mask.sum() > 100:  # 确保有足够的数据点
                    ic = df.loc[mask, factor].corr(df.loc[mask, 'forward_return'])
                    if pd.isna(ic) or abs(ic) < 0.02:  # IC绝对值<2%,说明几乎无预测力
                        logger.warning(f"Factor {factor} has low IC={ic:.4f}, dropping")
                        df = df.drop(columns=[factor])
                        continue
                
                # 3. 检查单调性(防止过拟合) - 如果有forward_return列
                # 好的因子应该在分组后收益单调递增/递减
                # 创建因子分组
                mask = ~(df[factor].isna() | df['forward_return'].isna())
                if mask.sum() > 100:  # 确保有足够的数据点
                    df_temp = df.loc[mask].copy()
                    try:
                        # 使用qcut进行分组，处理重复值问题
                        df_temp['factor_quantile'] = pd.qcut(df_temp[factor], q=5, labels=False, duplicates='drop')
                        
                        if 'factor_quantile' in df_temp.columns and df_temp['factor_quantile'].notna().any():
                            # 计算各分组的平均收益
                            quantile_returns = df_temp.groupby('factor_quantile')['forward_return'].mean()
                            
                            if len(quantile_returns) >= 3:  # 至少需要3个分组来计算单调性
                                # 计算单调性: Spearman相关系数
                                from scipy.stats import spearmanr
                                monotonicity, _ = spearmanr(quantile_returns.index, quantile_returns.values)
                                
                                if pd.isna(monotonicity) or abs(monotonicity) < 0.5:
                                    logger.warning(f"Factor {factor} lacks monotonicity={monotonicity:.2f}, dropping")
                                    df = df.drop(columns=[factor])
                            else:
                                logger.warning(f"Factor {factor} has insufficient quantiles, dropping")
                                df = df.drop(columns=[factor])
                    except Exception as e:
                        logger.warning(f"Error in monotonicity test for {factor}: {e}, dropping")
                        df = df.drop(columns=[factor])
        
        logger.info(f"Factor quality control completed. Remaining factors: {[c for c in df.columns if c in factor_cols]}")
        return df

    def _winsorize_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """极值处理（保持原逻辑）"""
        logger.info("Winsorizing factors...")
        factor_cols = self._get_factor_columns(df)
        limits = self.config.factor.winsorize_limits

        for col in factor_cols:
            if col in df.columns:
                lower = df[col].quantile(limits[0])
                upper = df[col].quantile(limits[1])
                df[col] = df[col].clip(lower, upper)

        return df

    def _standardize_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """标准化因子（保持原逻辑）"""
        logger.info("Standardizing factors...")
        factor_cols = self._get_factor_columns(df)

        for col in factor_cols:
            if col in df.columns:
                mean = df[col].mean()
                std = df[col].std()
                if std > 0:
                    df[col] = (df[col] - mean) / std

        return df

    def _neutralize_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        【核心改进】增强的因子中和
        同时执行：
        1. 行业中和（原有）
        2. 市值中和（新增）
        """
        logger.info("Neutralizing factors (industry + market cap)...")
        
        factor_cols = self._get_factor_columns(df)
        
        # 第一步：行业中和
        if 'industry' in df.columns:
            for col in factor_cols:
                if col in df.columns:
                    industry_mean = df.groupby('industry')[col].transform('mean')
                    df[col] = df[col] - industry_mean
        else:
            logger.warning("Industry column not found, skipping industry neutralization")
        
        # 第二步：市值中和（新增）
        if 'circ_mv' in df.columns:
            logger.info("Applying market cap neutralization...")
            df = self._neutralize_by_market_cap(df, factor_cols)
        else:
            logger.warning("Market cap column (circ_mv) not found, skipping cap neutralization")
        
        return df

    def _neutralize_by_market_cap(self, df: pd.DataFrame, factor_cols: List[str]) -> pd.DataFrame:
        """
        【新增方法】市值中和
        
        方法：分组回归中和
        1. 对市值取对数（使其更正态）
        2. 将股票按市值分成10组
        3. 在每组内，减去该组因子均值
        
        优点：
        - 相比线性回归更稳健
        - 保留了组内差异
        - 计算效率高
        """
        # 计算对数市值
        df['log_mv'] = np.log(df['circ_mv'] + 1)
        
        # 按日期分组处理（避免未来信息泄露）
        def neutralize_group(group):
            # 按市值分10组
            group['mv_decile'] = pd.qcut(group['log_mv'], q=10, labels=False, duplicates='drop')
            
            # 对每个因子进行组内中和
            for col in factor_cols:
                if col in group.columns:
                    # 计算每组均值
                    group_mean = group.groupby('mv_decile')[col].transform('mean')
                    # 减去组均值
                    group[col] = group[col] - group_mean
            
            return group
        
        # 并行或串行处理
        if hasattr(self, 'use_parallel') and self.use_parallel:
            df = df.groupby('trade_date', group_keys=False).parallel_apply(neutralize_group)
        else:
            df = df.groupby('trade_date', group_keys=False).apply(neutralize_group)
        
        # 清理临时列
        df = df.drop(columns=['log_mv', 'mv_decile'], errors='ignore')
        
        logger.info("Market cap neutralization completed")
        return df

    def _orthogonalize_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """因子正交化（保持原逻辑）"""
        logger.info("Orthogonalizing factors...")
        factor_cols = self._get_factor_columns(df)

        if len(factor_cols) < 2:
            return df

        from numpy.linalg import qr

        def orthogonalize_group(group):
            factor_matrix = group[factor_cols].values
            mask = ~np.isnan(factor_matrix).any(axis=1)
            if mask.sum() < 2:
                return group

            Q, R = qr(factor_matrix[mask])
            group.loc[mask, factor_cols] = Q

            return group

        if hasattr(self, 'use_parallel') and self.use_parallel:
            df = df.groupby('trade_date', group_keys=False).parallel_apply(orthogonalize_group)
        else:
            df = df.groupby('trade_date', group_keys=False).apply(orthogonalize_group)

        return df

    def _get_factor_columns(self, df: pd.DataFrame) -> List[str]:
        """获取所有因子列名"""
        all_factors = (
            self.config.factor.technical_factors +
            self.config.factor.money_flow_factors +
            self.config.factor.fundamental_factors +
            self.config.factor.alpha_factors
        )
        return [col for col in all_factors if col in df.columns]

    def calculate_ic(self, df: pd.DataFrame, forward_return_col: str = 'forward_return') -> pd.DataFrame:
        """计算因子IC值（保持原逻辑）"""
        factor_cols = self._get_factor_columns(df)
        ic_results = []

        for date in df['trade_date'].unique():
            df_date = df[df['trade_date'] == date]

            for factor in factor_cols:
                if factor in df_date.columns and forward_return_col in df_date.columns:
                    ic = df_date[factor].corr(df_date[forward_return_col])
                    ic_results.append({
                        'date': date,
                        'factor': factor,
                        'ic': ic
                    })

        return pd.DataFrame(ic_results)

    def select_factors(self, ic_df: pd.DataFrame) -> List[str]:
        """基于IC值选择因子（保持原逻辑）"""
        factor_stats = ic_df.groupby('factor')['ic'].agg(['mean', 'std']).reset_index()
        factor_stats['ic_ir'] = factor_stats['mean'] / factor_stats['std']
        factor_stats = factor_stats.sort_values('ic_ir', ascending=False)
        selected = factor_stats.head(self.config.factor.max_factors)['factor'].tolist()
        logger.info(f"Selected {len(selected)} factors based on IC")
        return selected


class FeatureImportance:
    """特征重要性分析（保持原逻辑）"""

    @staticmethod
    def get_model_importance(model, feature_names: List[str]) -> pd.DataFrame:
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importance = np.abs(model.coef_)
        else:
            return pd.DataFrame()

        df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        })

        df = df.sort_values('importance', ascending=False)
        return df

    @staticmethod
    def plot_importance(importance_df: pd.DataFrame, top_n: int = 20, save_path: str = None):
        import matplotlib.pyplot as plt

        df = importance_df.head(top_n)

        plt.figure(figsize=(10, 8))
        plt.barh(df['feature'], df['importance'])
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title(f'Top {top_n} Feature Importance')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()

        plt.close()