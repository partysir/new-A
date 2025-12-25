"""
增强版实盘选股系统
集成10大胜率提升方案中的关键功能
Version: 2.0
Date: 2025-12-24
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class EnhancedLiveStrategy:
    """增强版实盘选股策略"""

    def __init__(self, config):
        self.config = config

    def select_stocks_enhanced(
            self,
            df_date: pd.DataFrame,
            df_history: pd.DataFrame,
            index_data = None
    ) -> pd.DataFrame:
        """
        增强版选股 - 集成多维度验证

        Args:
            df_date: 当日数据
            df_history: 历史数据(用于计算时机指标)
            index_data: 指数数据(用于判断市场状态)

        Returns:
            精选股票池
        """

        logger.info("\n" + "=" * 80)
        logger.info("🚀 增强版选股系统启动")
        logger.info("=" * 80)

        # 【关键修复】立即重置索引
        df_date = df_date.reset_index(drop=True).copy()
        df_history = df_history.reset_index(drop=True).copy()

        # ===== 阶段1: 基础过滤 =====
        logger.info("\n[阶段1] 基础过滤...")
        df_filtered = self._basic_filter(df_date)
        logger.info(f"  ✓ 基础过滤后: {len(df_filtered)} 只")

        if df_filtered.empty:
            logger.warning("基础过滤后无股票,返回空结果")
            return pd.DataFrame()

        # ===== 阶段2: 市场环境判断 =====
        logger.info("\n[阶段2] 市场环境判断...")
        market_state = self._classify_market_state(index_data) if index_data is not None else 'unknown'
        logger.info(f"  ✓ 市场状态: {market_state}")

        # ===== 阶段3: 多信号验证 =====
        logger.info("\n[阶段3] 四层信号金字塔验证...")
        df_validated = self._multi_signal_validation(df_filtered)
        logger.info(f"  ✓ 信号验证后: {len(df_validated)} 只")

        if df_validated.empty:
            logger.warning("信号验证后无股票,放宽条件重试...")
            df_validated = self._relaxed_validation(df_filtered)
            logger.info(f"  ✓ 放宽后: {len(df_validated)} 只")

        # ===== 阶段4: 买点时机识别 =====
        logger.info("\n[阶段4] 最佳买点时机识别...")
        df_with_timing = self._identify_entry_timing(df_validated, df_history)
        df_good_timing = df_with_timing[df_with_timing['timing_score'] > 0.2]  # 降低阈值
        logger.info(f"  ✓ 时机良好: {len(df_good_timing)} 只")

        if len(df_good_timing) < 5:  # 如果时机好的太少
            logger.info(f"  ⚠ 时机良好股票不足5只,保留Top 50候选")
            df_good_timing = df_with_timing.nlargest(min(50, len(df_with_timing)), 'timing_score')

        # ===== 阶段5: 财务质量筛查 =====
        logger.info("\n[阶段5] 财务质量筛查...")
        df_quality = self._financial_quality_filter(df_good_timing)
        logger.info(f"  ✓ 质量合格: {len(df_quality)} 只")

        # ===== 阶段6: 行业轮动优化 =====
        logger.info("\n[阶段6] 行业轮动优化...")
        hot_industries = self._detect_hot_industries(df_history)
        logger.info(f"  ✓ 热点行业: {hot_industries}")

        # ===== 阶段7: 根据市场状态调整策略 =====
        logger.info("\n[阶段7] 策略自适应...")
        df_final = self._adaptive_selection(df_quality, market_state, hot_industries)
        logger.info(f"  ✓ 最终候选: {len(df_final)} 只")

        # ===== 阶段8: 综合评分排序 =====
        logger.info("\n[阶段8] 综合评分...")
        df_scored = self._composite_scoring(df_final)

        # ===== 阶段9: 动态仓位分配 =====
        logger.info("\n[阶段9] 动态仓位分配...")
        df_positioned = self._dynamic_position_sizing(df_scored)

        # 选取Top N (根据市场状态动态调整)
        top_n = self._get_adaptive_top_n(market_state)
        df_selected = df_positioned.head(top_n)

        logger.info(f"\n{'=' * 80}")
        logger.info(f"✅ 选股完成: {len(df_selected)} 只股票")
        logger.info(f"{'=' * 80}\n")

        # 生成推荐理由
        df_selected = self._generate_recommendations(df_selected, market_state)

        return df_selected

    def _basic_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """基础过滤"""
        
        # 确保索引连续
        df = df.reset_index(drop=True)
        
        # 正确创建mask
        mask = pd.Series(True, index=df.index)

        # 市值过滤
        if 'circ_mv' in df.columns:
            mask &= df['circ_mv'] > df['circ_mv'].quantile(0.20)

        # 涨跌幅过滤
        if 'pct_chg' in df.columns:
            mask &= (df['pct_chg'] < 5.0) & (df['pct_chg'] > -8.0)

        # ST股票
        if 'is_st' in df.columns:
            mask &= df['is_st'] == 0
        # 双重检查：名字里带"ST"的股票
        if 'name' in df.columns:
            mask &= ~df['name'].str.contains('ST', na=False)

        # 流动性
        if 'amount' in df.columns:
            mask &= df['amount'] > 1e7

        # 短期暴涨过滤（可选）
        if 'momentum_5' in df.columns:
            mask &= df['momentum_5'] < 0.20
        elif 'pct_chg' in df.columns:
            # 如果没有momentum_5，使用当日涨跌幅替代
            pass  # 已经在上面过滤了涨跌幅

        result = df[mask].copy()
        return result

    def _multi_signal_validation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        四层信号金字塔验证
        """

        # 第1层: ML评分
        # 【修复】评分范围异常低时，使用相对阈值
        if df['ml_score'].max() < 0.1:
            # 评分异常低，使用前70%
            threshold = df['ml_score'].quantile(0.30)  # 降低到30%分位数
        else:
            threshold = 0.55
        mask_ml = df['ml_score'] > threshold

        # 第2层: 技术形态
        mask_tech = pd.Series([True] * len(df), index=df.index)
        if all(col in df.columns for col in ['rsi_14', 'macd', 'close', 'ma20']):
            mask_tech = (
                    (df['rsi_14'] > 35) & (df['rsi_14'] < 75) &  # RSI健康
                    (df['macd'] > -0.1) &  # MACD不太弱
                    (df['close'] > df['ma20'] * 0.95)  # 接近或突破均线
            )
        else:
            # ⚠ 缺少关键列，标记为 False 或仅给基础分
            logger.warning("缺少技术指标列，跳过技术面验证")
            mask_tech = pd.Series([False] * len(df), index=df.index)

        # 第3层: 资金流向
        mask_money = pd.Series([True] * len(df))
        if all(col in df.columns for col in ['turnover_rate']):
            mask_money = (
                    (df['turnover_rate'] > 0.02) &  # 有一定成交
                    (df['turnover_rate'] < 0.20)  # 不过度投机
            )

            if 'main_force_inflow' in df.columns:
                mask_money &= df['main_force_inflow'] > 0  # 主力流入

        # 第4层: 估值安全边际
        mask_value = pd.Series([True] * len(df))
        if 'pe_ttm' in df.columns:
            mask_value = (df['pe_ttm'] > 0) & (df['pe_ttm'] < df['pe_ttm'].quantile(0.80))

        # 至少通过2层验证（原来是3层）
        validation_score = (
                mask_ml.astype(int) +
                mask_tech.astype(int) +
                mask_money.astype(int) +
                mask_value.astype(int)
        )

        df['validation_score'] = validation_score

        return df[validation_score >= 2].copy()  # 降低门槛

    def _relaxed_validation(self, df: pd.DataFrame) -> pd.DataFrame:
        """放宽验证条件"""
        # 只要求ML评分+一个其他信号
        mask_ml = df['ml_score'] > 0.40  # 降低阈值

        mask_other = pd.Series([False] * len(df))
        if 'rsi_14' in df.columns:
            mask_other |= (df['rsi_14'] > 30) & (df['rsi_14'] < 80)
        if 'turnover_rate' in df.columns:
            mask_other |= df['turnover_rate'] > 0.01

        return df[mask_ml & mask_other].copy()

    def _identify_entry_timing(
            self,
            df_current: pd.DataFrame,
            df_history: pd.DataFrame
    ) -> pd.DataFrame:
        """
        识别最佳买入时点
        """

        df_current = df_current.copy()
        timing_scores = []

        # 🟢 优化：构建查找字典，将复杂度降为 O(1)
        # 只取需要的列和最近30天，减少内存消耗
        required_cols = ['ts_code', 'trade_date', 'close', 'ma20', 'volume_ratio', 'rsi_14', 'macd']
        # 确保列存在
        valid_cols = [c for c in required_cols if c in df_history.columns]
        
        # 预处理：按代码分组并取最后30天
        # 这一步可能稍微花点时间，但比循环内过滤快几百倍
        history_dict = {}
        for code, grp in df_history[valid_cols].groupby('ts_code'):
            history_dict[code] = grp.sort_values('trade_date')

        for _, row in df_current.iterrows():
            code = row['ts_code']

            # 直接从字典获取，毫秒级
            stock_history = history_dict.get(code)

            if stock_history is None or len(stock_history) < 10:
                timing_scores.append(0.0)
                continue

            score = 0.0

            # 信号1: 价格突破均线 (30%)
            if 'close' in stock_history.columns and 'ma20' in stock_history.columns:
                latest_close = stock_history['close'].iloc[-1]
                latest_ma20 = stock_history['ma20'].iloc[-1]
                prev_close = stock_history['close'].iloc[-2] if len(stock_history) > 1 else latest_close
                prev_ma20 = stock_history['ma20'].iloc[-2] if len(stock_history) > 1 else latest_ma20

                # 刚突破或在均线附近
                if latest_close > latest_ma20 and (prev_close <= prev_ma20 or latest_close < latest_ma20 * 1.02):
                    score += 0.3

            # 信号2: 量能模式 (30%)
            if 'volume_ratio' in stock_history.columns:
                latest_vol_ratio = stock_history['volume_ratio'].iloc[-1]
                if 1.2 < latest_vol_ratio < 2.5:  # 放量但不过度
                    score += 0.3

            # 信号3: RSI恢复 (20%)
            if 'rsi_14' in stock_history.columns:
                latest_rsi = stock_history['rsi_14'].iloc[-1]
                if 40 < latest_rsi < 65:  # 健康区间
                    score += 0.2

                # 检查是否从超卖恢复
                min_rsi_5d = stock_history['rsi_14'].tail(5).min()
                if min_rsi_5d < 35 and latest_rsi > 40:
                    score += 0.1  # 额外加分

            # 信号4: MACD金叉 (20%)
            if 'macd' in stock_history.columns:
                latest_macd = stock_history['macd'].iloc[-1]
                if 0 < latest_macd < 0.15:  # 刚金叉或即将金叉
                    score += 0.2

            timing_scores.append(min(score, 1.0))

        df_current['timing_score'] = timing_scores
        return df_current

    def _financial_quality_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        财务质量筛查 - 向量化版本
        """

        if len(df) == 0:
            return df
        
        # 【临时】如果没有财务数据，直接返回（避免全部过滤掉）
        has_financial = any(col in df.columns for col in ['roe', 'pe', 'pb', 'debt_to_asset'])
        if not has_financial:
            logger.warning("  ⚠ 无财务数据，跳过财务筛选")
            return df

        df = df.copy()
        
        # 初始化质量得分
        quality_scores = pd.Series(0, index=df.index)

        # 盈利能力
        if 'roe' in df.columns and 'net_margin' in df.columns:
            quality_scores += ((df['roe'] > 0.05) & (df['net_margin'] > 0.02)).astype(int)

        # 成长性
        if 'revenue_yoy' in df.columns and 'profit_yoy' in df.columns:
            quality_scores += ((df['revenue_yoy'] > 0.05) & (df['profit_yoy'] > 0.05)).astype(int)

        # 财务健康
        if 'debt_to_asset' in df.columns and 'current_ratio' in df.columns:
            quality_scores += ((df['debt_to_asset'] < 0.70) & (df['current_ratio'] > 1.0)).astype(int)

        # 现金流
        if 'ocf' in df.columns:
            quality_scores += (df['ocf'] > 0).astype(int)

        # 估值
        if 'pe_ttm' in df.columns and 'profit_yoy' in df.columns:
            peg = df['pe_ttm'] / (df['profit_yoy'] * 100 + 0.001)
            quality_scores += ((df['pe_ttm'] > 0) & (df['profit_yoy'] > 0) & (0 < peg) & (peg < 2.0)).astype(int)

        df['quality_score'] = quality_scores

        # 保留质量得分>=1的股票（原来是>=2）
        return df[df['quality_score'] >= 1].copy()  # 降低门槛

    def _classify_market_state(self, index_data: pd.DataFrame) -> str:
        """判断市场状态"""

        if index_data is None or len(index_data) < 60:
            return 'unknown'

        # 确保index_data是DataFrame
        if not isinstance(index_data, pd.DataFrame):
            return 'unknown'

        latest = index_data.iloc[-1]

        # 计算均线
        ma20 = index_data['close'].rolling(20).mean().iloc[-1]
        ma60 = index_data['close'].rolling(60).mean().iloc[-1]

        # 计算动量
        momentum_20 = (latest['close'] - index_data['close'].iloc[-21]) / index_data['close'].iloc[-21]

        # 分类
        if latest['close'] > ma20 > ma60 and momentum_20 > 0.05:
            return 'strong_bull'  # 强势上涨
        elif latest['close'] > ma20 and momentum_20 > 0:
            return 'weak_bull'  # 震荡上涨
        elif abs(momentum_20) < 0.03:
            return 'consolidation'  # 横盘震荡
        elif latest['close'] < ma20 and momentum_20 > -0.10:
            return 'weak_bear'  # 弱势下跌
        else:
            return 'strong_bear'  # 强势下跌

    def _detect_hot_industries(self, df_history: pd.DataFrame) -> List[str]:
        """检测热点行业"""

        if 'industry' not in df_history.columns:
            return []

        # 获取最近5天的数据
        recent_dates = sorted(df_history['trade_date'].unique())[-5:]
        df_recent = df_history[df_history['trade_date'].isin(recent_dates)]

        # 计算各行业表现
        agg_dict = {'pct_chg': 'mean'}
        
        if 'momentum_5' in df_recent.columns:
            agg_dict['momentum_5'] = 'mean'
        
        if 'turnover_rate' in df_recent.columns:
            agg_dict['turnover_rate'] = 'mean'
        
        try:
            industry_perf = df_recent.groupby('industry').agg(agg_dict)
            
            # 综合评分
            if len(industry_perf) > 0:
                industry_perf['strength'] = industry_perf['pct_chg']
                hot_industries = industry_perf.nlargest(3, 'strength').index.tolist()
                return hot_industries
        except Exception:
            # 如果聚合失败，返回空列表
            pass

        return []

    def _adaptive_selection(
            self,
            df: pd.DataFrame,
            market_state: str,
            hot_industries: List[str]
    ) -> pd.DataFrame:
        """根据市场状态自适应选股"""

        if len(df) == 0:
            return df

        # 根据市场状态调整过滤条件
        if market_state == 'strong_bull':
            # 强势市场: 选成长+动量
            if 'momentum_20' in df.columns:
                df = df[df['momentum_20'] > 0]

        elif market_state == 'weak_bear':
            # 弱势市场: 选防御性行业
            if 'industry' in df.columns:
                defensive_industries = ['医药', '食品饮料', '银行', '公用事业']
                df = df[df['industry'].isin(defensive_industries)]

        elif market_state == 'strong_bear':
            # 强烈下跌: 只选超跌
            if 'momentum_20' in df.columns and 'rsi_14' in df.columns:
                df = df[(df['momentum_20'] < -0.10) & (df['rsi_14'] < 30)]

        # 热点行业加权
        if len(hot_industries) > 0 and 'industry' in df.columns:
            df = df.copy()
            df['in_hot_industry'] = df['industry'].isin(hot_industries).astype(int)

        return df

    def _composite_scoring(self, df: pd.DataFrame) -> pd.DataFrame:
        """综合评分"""

        if len(df) == 0:
            return df

        df = df.copy()

        # 归一化各项得分
        df['ml_score_norm'] = df['ml_score'] / df['ml_score'].max()

        if 'validation_score' in df.columns:
            df['validation_norm'] = df['validation_score'] / 4.0
        else:
            df['validation_norm'] = 0.5

        if 'timing_score' in df.columns:
            df['timing_norm'] = df['timing_score']
        else:
            df['timing_norm'] = 0.5

        if 'quality_score' in df.columns:
            df['quality_norm'] = df['quality_score'] / 5.0
        else:
            df['quality_norm'] = 0.5

        # 综合得分
        df['composite_score'] = (
                df['ml_score_norm'] * 0.30 +
                df['validation_norm'] * 0.25 +
                df['timing_norm'] * 0.25 +
                df['quality_norm'] * 0.20
        )

        # 热点行业加成
        if 'in_hot_industry' in df.columns:
            df['composite_score'] = df['composite_score'] * (1 + df['in_hot_industry'] * 0.1)

        # 排序
        df = df.sort_values('composite_score', ascending=False)

        return df

    def _dynamic_position_sizing(self, df: pd.DataFrame) -> pd.DataFrame:
        """动态仓位分配"""

        if len(df) == 0:
            return df

        df = df.copy()

        # 根据综合得分分级
        if 'composite_score' in df.columns:
            df['position_tier'] = pd.cut(
                df['composite_score'],
                bins=[0, 0.55, 0.70, 0.82, 1.0],
                labels=['C', 'B', 'A', 'S'],
                include_lowest=True
            )
        else:
            df['position_tier'] = 'B'

        # 仓位映射
        position_map = {
            'S': 0.12,  # 超优: 12%
            'A': 0.08,  # 优秀: 8%
            'B': 0.05,  # 良好: 5%
            'C': 0.03  # 一般: 3%
        }

        df['weight'] = df['position_tier'].map(position_map)

        # 🟢 优化：不要强制归一化到1，而是设置单日最大总仓位
        total_weight = df['weight'].sum()
        
        # 比如：限制单日推荐总仓位不超过 100% (满仓)，如果不足 100% 就保持原比例
        if total_weight > 1.0:
            df['weight'] = df['weight'] / total_weight
        
        # 或者：即使选出的少，也不要加仓，保持原定仓位（如选出1只就是5%仓位）
        # 这种方式更适合组合管理，意味着大部分资金空仓

        return df

    def _get_adaptive_top_n(self, market_state: str) -> int:
        """根据市场状态动态调整持仓数量"""

        state_map = {
            'strong_bull': 20,  # 强势市场多持仓
            'weak_bull': 15,  # 震荡市场中等持仓
            'consolidation': 12,  # 横盘少持仓
            'weak_bear': 8,  # 弱势更少
            'strong_bear': 5,  # 熊市极少
            'unknown': 10  # 未知状态保守
        }

        return state_map.get(market_state, 10)

    def _generate_recommendations(
            self,
            df: pd.DataFrame,
            market_state: str
    ) -> pd.DataFrame:
        """生成推荐理由"""

        if len(df) == 0:
            return df

        df = df.copy()
        recommendations = []

        for _, row in df.iterrows():
            reasons = []

            # ML评分
            if row.get('ml_score', 0) > 0.7:
                reasons.append(f"AI评分优秀({row['ml_score']:.2f})")
            elif row.get('ml_score', 0) > 0.6:
                reasons.append(f"AI评分良好({row['ml_score']:.2f})")

            # 买点时机
            if row.get('timing_score', 0) > 0.6:
                reasons.append("买点时机优秀")
            elif row.get('timing_score', 0) > 0.4:
                reasons.append("买点时机良好")

            # 财务质量
            if row.get('quality_score', 0) >= 4:
                reasons.append("财务质量优秀")
            elif row.get('quality_score', 0) >= 3:
                reasons.append("财务质量良好")

            # 热点行业
            if row.get('in_hot_industry', 0) == 1:
                reasons.append("热点行业")

            # 技术形态
            if 'rsi_14' in row:
                if 45 < row['rsi_14'] < 60:
                    reasons.append("RSI健康")

            if 'macd' in row and row['macd'] > 0:
                reasons.append("MACD金叉")

            recommendation = " | ".join(reasons) if reasons else "综合评分良好"
            recommendations.append(recommendation)

        df['recommendation'] = recommendations
        df['market_state'] = market_state

        return df


def integrate_with_existing_system(config):
    """
    与现有系统集成的示例
    """

    from data_manager import DataManager
    from factor_engine import FactorEngine
    from ml_model import MLModel, WalkForwardTrainer
    import joblib

    logger.info("=" * 80)
    logger.info("增强版实盘选股系统")
    logger.info("=" * 80)

    # 初始化
    dm = DataManager(config)
    fe = FactorEngine(config)
    trainer = WalkForwardTrainer(config)
    enhanced_strategy = EnhancedLiveStrategy(config)

    # 获取数据
    end_date = datetime.now().strftime('%Y%m%d')
    start_date = (datetime.now() - timedelta(days=90)).strftime('%Y%m%d')

    logger.info("加载数据...")
    df_history = dm.get_daily_data(start_date=start_date, end_date=end_date)

    # 计算因子
    logger.info("计算因子...")
    df_with_factors = fe.calculate_all_factors(df_history)

    # 加载模型
    logger.info("加载模型...")
    model = MLModel(config)
    model.model = joblib.load('latest_model.pkl')

    # 预测
    latest_date = df_with_factors['trade_date'].max()
    df_latest = df_with_factors[df_with_factors['trade_date'] == latest_date]

    X, _ = trainer.prepare_data(df_latest)
    preds = model.predict(X)
    df_latest['ml_score'] = preds

    # 获取指数数据
    index_data = dm.get_index_data('000300.SH', start_date)

    # 增强选股
    logger.info("执行增强选股...")
    selected = enhanced_strategy.select_stocks_enhanced(
        df_date=df_latest,
        df_history=df_with_factors,
        index_data=index_data
    )

    # 输出结果
    if not selected.empty:
        logger.info(f"\n✅ 选出 {len(selected)} 只股票:")

        display_cols = [
            'ts_code', 'name', 'close', 'pct_chg',
            'composite_score', 'position_tier', 'weight',
            'recommendation', 'market_state'
        ]

        display_cols = [c for c in display_cols if c in selected.columns]
        print(selected[display_cols].to_string(index=False))

        # 保存结果
        selected.to_csv(f'enhanced_recommendations_{latest_date}.csv', index=False)
        logger.info(f"\n结果已保存至: enhanced_recommendations_{latest_date}.csv")
    else:
        logger.warning("未选出任何股票")

    return selected


if __name__ == '__main__':
    # 示例使用
    from config import Config

    config = Config()
    results = integrate_with_existing_system(config)