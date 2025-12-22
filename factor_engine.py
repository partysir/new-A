"""
因子工程模块
计算技术、资金流、基本面和Alpha因子
"""

import logging
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class FactorEngine:
    """因子计算引擎"""

    def __init__(self, config):
        self.config = config

    def calculate_all_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算所有因子

        Args:
            df: 原始数据

        Returns:
            包含所有因子的DataFrame
        """
        logger.info("Starting factor calculation...")

        # 按股票分组计算
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

        logger.info(f"Factor calculation completed: {len(df_with_factors)} records")
        return df_with_factors

    def _calculate_stock_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算单只股票的所有因子"""
        df = df.copy()
        df = df.sort_values('trade_date')

        # 技术因子
        df = self._calculate_technical_factors(df)

        # 资金流因子
        df = self._calculate_money_flow_factors(df)

        # Alpha因子
        df = self._calculate_alpha_factors(df)

        return df

    def _calculate_technical_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算技术因子"""
        # 动量因子
        if 'momentum_20' in self.config.factor.technical_factors:
            df['momentum_20'] = df['close'].pct_change(20)

        if 'momentum_60' in self.config.factor.technical_factors:
            df['momentum_60'] = df['close'].pct_change(60)

        # RSI
        if 'rsi_14' in self.config.factor.technical_factors:
            df['rsi_14'] = self._calculate_rsi(df['close'], 14)

        # MACD
        if 'macd' in self.config.factor.technical_factors:
            df['macd'] = self._calculate_macd(df['close'])

        # 布林带
        if 'bbands_width' in self.config.factor.technical_factors:
            df['bbands_width'] = self._calculate_bbands_width(df['close'], 20)

        # ATR
        if 'atr_14' in self.config.factor.technical_factors:
            df['atr_14'] = self._calculate_atr(df, 14)

        # ADX
        if 'adx_14' in self.config.factor.technical_factors:
            df['adx_14'] = self._calculate_adx(df, 14)

        # CCI
        if 'cci_20' in self.config.factor.technical_factors:
            df['cci_20'] = self._calculate_cci(df, 20)

        # Williams %R
        if 'willr_14' in self.config.factor.technical_factors:
            df['willr_14'] = self._calculate_willr(df, 14)

        # 随机指标
        if 'stoch_k' in self.config.factor.technical_factors:
            df['stoch_k'] = self._calculate_stoch(df, 14)

        # 量比
        if 'volume_ratio' in self.config.factor.technical_factors:
            df['volume_ratio'] = df['vol'] / df['vol'].rolling(20).mean()

        return df

    @staticmethod
    def _calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """计算RSI指标"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    @staticmethod
    def _calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
        """计算MACD指标"""
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        macd_histogram = macd_line - signal_line
        return macd_histogram

    @staticmethod
    def _calculate_bbands_width(prices: pd.Series, period: int = 20, std: float = 2.0) -> pd.Series:
        """计算布林带宽度"""
        ma = prices.rolling(window=period).mean()
        std_dev = prices.rolling(window=period).std()
        upper = ma + (std_dev * std)
        lower = ma - (std_dev * std)
        width = (upper - lower) / ma
        return width

    @staticmethod
    def _calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """计算ATR指标"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return atr

    @staticmethod
    def _calculate_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """计算ADX指标"""
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
        """计算CCI指标"""
        tp = (df['high'] + df['low'] + df['close']) / 3
        ma = tp.rolling(window=period).mean()
        mad = tp.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean())
        cci = (tp - ma) / (0.015 * mad)
        return cci

    @staticmethod
    def _calculate_willr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """计算Williams %R指标"""
        highest_high = df['high'].rolling(window=period).max()
        lowest_low = df['low'].rolling(window=period).min()
        willr = -100 * (highest_high - df['close']) / (highest_high - lowest_low)
        return willr

    @staticmethod
    def _calculate_stoch(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """计算随机指标K"""
        lowest_low = df['low'].rolling(window=period).min()
        highest_high = df['high'].rolling(window=period).max()
        stoch_k = 100 * (df['close'] - lowest_low) / (highest_high - lowest_low)
        return stoch_k

    def _calculate_money_flow_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算资金流因子"""
        # 换手率
        if 'turnover_rate' in self.config.factor.money_flow_factors:
            df['turnover_rate'] = df['vol'] / df.get('float_share', df['vol'].mean())

        # 成交额变化
        if 'amount_change' in self.config.factor.money_flow_factors:
            df['amount_change'] = df['amount'].pct_change(5)

        # 量价相关性
        if 'volume_price_corr' in self.config.factor.money_flow_factors:
            df['volume_price_corr'] = df['vol'].rolling(20).corr(df['close'])

        # VWAP比率
        if 'vwap_ratio' in self.config.factor.money_flow_factors:
            df['vwap'] = (df['amount'] * 1000) / (df['vol'] * 100)  # 成交均价
            df['vwap_ratio'] = df['close'] / df['vwap'] - 1

        # 大单比例(如果有资金流数据)
        if 'big_order_ratio' in self.config.factor.money_flow_factors:
            if 'buy_lg_amount' in df.columns:
                total_amount = df['amount']
                big_amount = df['buy_lg_amount'] + df['buy_elg_amount']
                df['big_order_ratio'] = big_amount / total_amount

        # 主力资金流入
        if 'main_force_inflow' in self.config.factor.money_flow_factors:
            if 'buy_lg_amount' in df.columns:
                df['main_force_inflow'] = (
                                                  df['buy_lg_amount'] + df['buy_elg_amount'] -
                                                  df['sell_lg_amount'] - df['sell_elg_amount']
                                          ) / df['amount']

        return df

    def _calculate_alpha_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算Alpha因子"""
        # Alpha001: (close - open) / ((high - low) + 0.001)
        if 'alpha_001' in self.config.factor.alpha_factors:
            df['alpha_001'] = (df['close'] - df['open']) / ((df['high'] - df['low']) + 0.001)

        # Alpha002: 价格与成交量的相关性
        if 'alpha_002' in self.config.factor.alpha_factors:
            df['alpha_002'] = df['close'].rolling(20).corr(df['vol'])

        # Alpha003: 收盘价的20日标准差
        if 'alpha_003' in self.config.factor.alpha_factors:
            df['alpha_003'] = df['close'].rolling(20).std() / df['close'].rolling(20).mean()

        return df

    def _winsorize_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """极值处理"""
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
        """标准化因子"""
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
        """行业中性化"""
        logger.info("Neutralizing factors...")

        if 'industry' not in df.columns:
            logger.warning("Industry column not found, skipping neutralization")
            return df

        factor_cols = self._get_factor_columns(df)

        for col in factor_cols:
            if col in df.columns:
                # 按行业计算均值
                industry_mean = df.groupby('industry')[col].transform('mean')
                # 减去行业均值
                df[col] = df[col] - industry_mean

        return df

    def _orthogonalize_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """因子正交化"""
        logger.info("Orthogonalizing factors...")

        factor_cols = self._get_factor_columns(df)

        if len(factor_cols) < 2:
            return df

        # 使用QR分解进行正交化
        from numpy.linalg import qr

        # 按日期分组正交化
        def orthogonalize_group(group):
            factor_matrix = group[factor_cols].values

            # 去除NaN
            mask = ~np.isnan(factor_matrix).any(axis=1)
            if mask.sum() < 2:
                return group

            # QR分解
            Q, R = qr(factor_matrix[mask])

            # 用正交化后的因子替换
            group.loc[mask, factor_cols] = Q

            return group

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
        """
        计算因子IC值

        Args:
            df: 包含因子和未来收益的数据
            forward_return_col: 未来收益列名

        Returns:
            IC值DataFrame
        """
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
        """
        基于IC值选择因子

        Args:
            ic_df: IC值DataFrame

        Returns:
            选中的因子列表
        """
        # 计算平均IC和IC_IR
        factor_stats = ic_df.groupby('factor')['ic'].agg(['mean', 'std']).reset_index()
        factor_stats['ic_ir'] = factor_stats['mean'] / factor_stats['std']

        # 按IC_IR排序
        factor_stats = factor_stats.sort_values('ic_ir', ascending=False)

        # 选择Top因子
        selected = factor_stats.head(self.config.factor.max_factors)['factor'].tolist()

        logger.info(f"Selected {len(selected)} factors based on IC")
        return selected


class FeatureImportance:
    """特征重要性分析"""

    @staticmethod
    def get_model_importance(model, feature_names: List[str]) -> pd.DataFrame:
        """从模型获取特征重要性"""
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
        """绘制特征重要性图"""
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