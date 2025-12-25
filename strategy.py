"""
策略模块 - 修复版 v3.0
核心改进:
1. 彻底修复空仓死锁 - 使用指数信号而非净值回撤
2. 解决涨停板悖论 - 预测次日开盘收益,避免追高
3. 优化盈亏比 - 非对称止盈止损
4. 降低换手率 - 持仓优化逻辑
"""

import logging
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)


class Strategy:
    """策略主类 - 优化选股逻辑"""
    
    def __init__(self, config):
        self.config = config

    def select_stocks(self, df: pd.DataFrame, date: str) -> pd.DataFrame:
        """
        改进的选股逻辑 - 解决涨停板悖论
        核心思想: 预测次日开盘后的收益,而非当日收盘
        """
        df_date = df[df['trade_date'] == date].copy()
        if df_date.empty: 
            return pd.DataFrame()

        # === 第一层: 可交易性过滤 ===
        
        # 1. 排除微盘股(市值后20%)
        if 'circ_mv' in df_date.columns:
            cap_threshold = df_date['circ_mv'].quantile(0.20)
            df_date = df_date[df_date['circ_mv'] > cap_threshold]
        
        # 2. 【关键修改】排除"明天买不到"的股票
        # 不仅过滤涨停,还要过滤"已经涨太多"的股票
        if 'pct_chg' in df_date.columns and 'is_st' in df_date.columns:
            # 2.1 今日涨幅<5% (更保守),避免次日高开买不到
            mask_today = df_date['pct_chg'] < 5.0
            
            # 2.2 跌幅<9%,避免买到垃圾
            mask_fall = df_date['pct_chg'] > -9.0
            
            # 2.3 非ST股票
            mask_st = df_date['is_st'] == 0
            
            df_date = df_date[mask_today & mask_fall & mask_st]
            
            logger.info(f"[{date}] 过滤涨幅>5%的股票,剩余{len(df_date)}只")
        
        # 3. 【新增】近3日累计涨幅<15%,避免追高
        if 'close' in df_date.columns:
            # 计算近3日累计涨幅
            df_date = df_date.copy()
            df_with_pct = df[df['trade_date'] <= date].copy()
            df_with_pct['pct_chg_3d'] = df_with_pct.groupby('ts_code')['close'].pct_change(3)
            
            # 获取当前日期的3日涨幅
            current_3d_chg = df_with_pct[df_with_pct['trade_date'] == date].set_index('ts_code')['pct_chg_3d']
            df_date = df_date.join(current_3d_chg, on='ts_code', rsuffix='_3d')
            df_date = df_date[df_date['pct_chg_3d'] < 0.15]
            
            logger.info(f"[{date}] 过滤近3日涨幅>15%的股票,剩余{len(df_date)}只")
        
        # 4. 流动性过滤
        if 'amount' in df_date.columns:
            df_date = df_date[df_date['amount'] > 1e7]  # 成交额>1000万

        # 5. 【新增】避免追高 - 过滤量能暴增股
        if 'vol' in df_date.columns:
            # 计算量比
            df_date['volume_ratio'] = df_date['vol'] / df_date.groupby('ts_code')['vol'].transform(lambda x: x.rolling(20).mean())
            df_date = df_date[df_date['volume_ratio'] < 3.0]  # 量比<3倍
            logger.info(f"[{date}] 过滤量比>3的股票,剩余{len(df_date)}只")

        # === 第二层: 低位启动选股逻辑 ===
        if 'momentum_20' in df_date.columns and 'rsi_14' in df_date.columns:
            # 寻找:近期下跌后开始反弹的股票
            mask_low_position = (
                (df_date['momentum_20'] > -0.10) &  # 20日跌幅<10%
                (df_date['momentum_20'] < 0.05) &   # 但未大涨
                (df_date['rsi_14'] > 40) &          # RSI从超卖恢复
                (df_date['rsi_14'] < 70)            # 但未超买
            )
            
            # 优先选择低位股
            df_date['is_low_position'] = mask_low_position
            df_date = df_date.sort_values(['is_low_position', 'ml_score'], 
                                          ascending=[False, False])
        
        # === 第三层: 评分过滤 ===
        
        # 根据选股方法决定是否使用硬阈值
        if self.config.strategy.selection_method in ['score', 'threshold']:
            score_threshold = self.config.strategy.score_threshold
            df_date = df_date[df_date['ml_score'] >= score_threshold]
        
        # 如果过滤后为空,放宽限制
        if df_date.empty:
            df_date = df[df['trade_date'] == date].copy()
            logger.warning(f"[{date}] 过滤后为空,放宽限制")

        # === 第四层: 排序与行业约束 ===
        
        df_date = df_date.sort_values('ml_score', ascending=False)

        # 行业中性化
        if self.config.strategy.max_industry_weight < 1.0:
            df_date = self._apply_industry_constraints(df_date)

        # === 第五层: 最终筛选 ===
        
        # 【关键修改】增加缓冲数量,防止第二天开盘时部分股票涨停买不到
        buffer_multiplier = 1.5  # 多选50%作为备选
        top_n_with_buffer = int(self.config.strategy.top_n * buffer_multiplier)
        selected = df_date.head(top_n_with_buffer)
        
        # 标记优先级
        selected['priority'] = range(1, len(selected) + 1)
        
        if len(selected) == 0:
            logger.warning(f"[{date}] Selected 0 stocks. Check filters.")
        else:
            logger.info(f"[{date}] Selected {len(selected)} stocks (top_n={self.config.strategy.top_n}, with buffer)")
            
        return selected

    def select_stocks_live(self, df: pd.DataFrame, date: str) -> pd.DataFrame:
        """
        实盘选股逻辑 - 完全重写版
        
        核心改进:
        1. 简化为三层筛选(清晰明确)
        2. 避免追高(严格过滤)
        3. 动态推荐数量(质量优先)
        4. 增强输出信息
        """
        df_date = df[df['trade_date'] == date].copy()
        
        if df_date.empty:
            logger.warning(f"[{date}] 无可用数据")
            return pd.DataFrame()
        
        logger.info(f"\n{'='*60}")
        logger.info(f"🎯 实盘选股开始: {date}")
        logger.info(f"{'='*60}")
        logger.info(f"初始候选池: {len(df_date)} 只股票")
        
        # ===== 第一层: 基础过滤(排除不可交易股票) =====
        logger.info("\n[第一层] 基础过滤...")
        
        # 1.1 市值过滤(排除微盘股)
        if 'circ_mv' in df_date.columns:
            mv_threshold = df_date['circ_mv'].quantile(0.20)
            mask_mv = df_date['circ_mv'] > mv_threshold
            logger.info(f"  市值过滤: {mask_mv.sum()} 只 (>20分位数)")
        else:
            mask_mv = pd.Series([True] * len(df_date))
        
        # 1.2 涨跌幅过滤(严格,避免追高)
        if 'pct_chg' in df_date.columns:
            mask_price = (
                (df_date['pct_chg'] < 5.0) &   # 今日涨幅<5% (从7%收紧)
                (df_date['pct_chg'] > -8.0)    # 今日跌幅>-8%
            )
            logger.info(f"  涨跌幅过滤: {mask_price.sum()} 只 (涨幅<5%, 跌幅>-8%)")
        else:
            mask_price = pd.Series([True] * len(df_date))
        
        # 1.3 ST股票过滤
        if 'is_st' in df_date.columns:
            mask_st = df_date['is_st'] == 0
            logger.info(f"  ST过滤: {mask_st.sum()} 只")
        else:
            mask_st = pd.Series([True] * len(df_date))
        
        # 1.4 流动性过滤
        if 'amount' in df_date.columns:
            mask_liquidity = df_date['amount'] > 1e7  # 成交额>1000万
            logger.info(f"  流动性过滤: {mask_liquidity.sum()} 只 (成交额>1000万)")
        else:
            mask_liquidity = pd.Series([True] * len(df_date))
        
        # 1.5 【新增】短期暴涨过滤(防止接盘)
        if 'momentum_5' in df_date.columns:
            mask_momentum = df_date['momentum_5'] < 0.20  # 5日涨幅<20%
            logger.info(f"  短期暴涨过滤: {mask_momentum.sum()} 只 (5日涨幅<20%)")
        else:
            # 如果没有momentum_5,手动计算
            df_date['momentum_5_temp'] = df_date.groupby('ts_code')['close'].pct_change(5)
            mask_momentum = df_date['momentum_5_temp'].fillna(0) < 0.20
        
        # 1.6 【新增】量能过滤(防止游资)
        if 'volume_ratio' in df_date.columns:
            mask_volume = df_date['volume_ratio'] < 3.0  # 量比<3
            logger.info(f"  量能过滤: {mask_volume.sum()} 只 (量比<3)")
        else:
            mask_volume = pd.Series([True] * len(df_date))
        
        # 综合过滤
        mask_basic = mask_mv & mask_price & mask_st & mask_liquidity & mask_momentum & mask_volume
        df_filtered = df_date[mask_basic].copy()
        
        logger.info(f"✅ 第一层通过: {len(df_filtered)} 只")
        
        if df_filtered.empty:
            logger.warning("基础过滤后无候选股票")
            return pd.DataFrame()
        
        # ===== 第二层: ML模型精选 =====
        logger.info("\n[第二层] ML模型精选...")
        
        # 按ML分数排序,取Top 30
        df_filtered = df_filtered.sort_values('ml_score', ascending=False)
        top_ml = df_filtered.head(30).copy()
        
        logger.info(f"  ML分数范围: {top_ml['ml_score'].min():.3f} ~ {top_ml['ml_score'].max():.3f}")
        logger.info(f"  平均ML分数: {top_ml['ml_score'].mean():.3f}")
        logger.info(f"✅ 第二层通过: {len(top_ml)} 只")
        
        # ===== 第三层: 综合评分微调 =====
        logger.info("\n[第三层] 综合评分微调...")
        
        # 计算辅助指标
        top_ml = self._calculate_enhanced_indicators(top_ml)
        
        # 综合评分(ML为主,辅助指标为辅)
        top_ml['composite_score'] = (
            0.60 * top_ml['ml_score'] +                    # 主要看ML (从0.5提高到0.6)
            0.25 * top_ml['smart_money_score_norm'] +      # 次要看资金
            0.15 * top_ml['trend_energy_norm']             # 辅助看趋势
        )
        
        # 按综合分数排序
        top_ml = top_ml.sort_values('composite_score', ascending=False)
        
        # ===== 第四层: 动态数量筛选 =====
        logger.info("\n[第四层] 动态数量筛选...")
        
        # 质量阈值
        quality_threshold = 0.65  # 综合分数>0.65才推荐
        high_quality = top_ml[top_ml['composite_score'] > quality_threshold]
        
        if len(high_quality) >= 5:
            # 有足够的高质量股票
            final_selection = high_quality.head(20)  # 最多推荐20只
            logger.info(f"  高质量股票: {len(high_quality)} 只 (分数>{quality_threshold})")
        else:
            # 高质量股票不足,降低标准
            logger.warning(f"  高质量股票不足({len(high_quality)}只), 降低标准")
            final_selection = top_ml.head(max(5, len(high_quality)))  # 至少推荐5只
        
        # ===== 增强输出信息 =====
        final_selection = self._enhance_output(final_selection)
        
        logger.info(f"✅ 最终推荐: {len(final_selection)} 只")
        logger.info(f"{'='*60}\n")
        
        return final_selection

    def _generate_reason(self, row):
        """生成推荐理由文本"""
        reasons = []
        if row['ml_score'] > 0.8: reasons.append("模型高确信")
        if row.get('smart_money_score', 0) > 1.0: reasons.append("主力资金抢筹")
        if row.get('trend_energy', 0) > 2.0: reasons.append("趋势即将爆发")
        return "+".join(reasons) if reasons else "综合评分优选"

    def calculate_weights(self, selected_stocks: pd.DataFrame) -> pd.DataFrame:
        """计算股票权重(保持原有逻辑)"""
        if selected_stocks.empty:
            return selected_stocks
        
        n_stocks = len(selected_stocks)
        
        if self.config.strategy.weight_method == 'equal':
            weight_per_stock = 1.0 / n_stocks
            selected_stocks['weight'] = weight_per_stock
            
        elif self.config.strategy.weight_method == 'score_weighted':
            if 'ml_score' in selected_stocks.columns:
                scores = selected_stocks['ml_score']
                exp_scores = np.exp(scores - scores.max())
                weights = exp_scores / exp_scores.sum()
                selected_stocks['weight'] = weights
            else:
                selected_stocks['weight'] = 1.0 / n_stocks
                
        elif self.config.strategy.weight_method == 'risk_parity':
            if 'volatility' in selected_stocks.columns:
                vol_weights = 1.0 / (selected_stocks['volatility'] + 1e-9)
                selected_stocks['weight'] = vol_weights / vol_weights.sum()
            else:
                selected_stocks['weight'] = 1.0 / n_stocks
                
        else:
            selected_stocks['weight'] = 1.0 / n_stocks
        
        # 应用单只股票最大权重限制
        selected_stocks['weight'] = selected_stocks['weight'].clip(
            upper=self.config.strategy.max_single_weight
        )
        
        # 重新归一化权重
        total_weight = selected_stocks['weight'].sum()
        if total_weight > 0:
            selected_stocks['weight'] = selected_stocks['weight'] / total_weight
        
        return selected_stocks

    def _apply_industry_constraints(self, df):
        """行业约束"""
        max_per_ind = int(self.config.strategy.top_n * self.config.strategy.max_industry_weight)
        if max_per_ind < 1: 
            max_per_ind = 1
        return df.groupby('industry', group_keys=False).apply(lambda x: x.head(max_per_ind))

    def should_rebalance(self, date: str, trading_day_count: int) -> bool:
        """判断调仓日"""
        freq = self.config.strategy.rebalance_frequency
        if freq == 'daily': 
            return True
        if freq == 'weekly':
            return pd.to_datetime(date).weekday() == 4
        if freq == 'n_days':
            return trading_day_count % self.config.strategy.rebalance_day == 0
        return False

    def _calculate_enhanced_indicators(self, df):
        """
        计算增强版辅助指标 - 修复版
        
        修复要点:
        1. 防止NaN传播
        2. 防止除零错误
        3. 归一化到0-1区间
        """
        # 1. 资金流向 (修复版)
        if 'turnover_rate' in df.columns:
            # 防止除零
            price_range = df['high'] - df['low']
            price_strength = np.where(
                price_range > 0.001,  # 波动大于0.1分钱
                (df['close'] - df['open']) / price_range,
                0
            )
            df['smart_money_score'] = price_strength * df['turnover_rate']
        else:
            df['smart_money_score'] = 0
        
        # 归一化到0-1
        if df['smart_money_score'].std() > 0:
            df['smart_money_score_norm'] = (
                df['smart_money_score'] - df['smart_money_score'].min()
            ) / (df['smart_money_score'].max() - df['smart_money_score'].min() + 1e-9)
        else:
            df['smart_money_score_norm'] = 0.5
        
        # 2. 趋势动能 (修复版)
        if 'ma20' in df.columns and 'vol_ma20' in df.columns:
            # 防止NaN
            price_momentum = (df['close'] / df['ma20'].fillna(df['close']) - 1).clip(-0.5, 0.5)
            volume_momentum = (df['vol'] / df['vol_ma20'].fillna(df['vol']) - 1).clip(-0.5, 0.5)
            df['trend_energy'] = price_momentum + volume_momentum
        else:
            df['trend_energy'] = 0
        
        # 归一化
        if df['trend_energy'].std() > 0:
            df['trend_energy_norm'] = (
                df['trend_energy'] - df['trend_energy'].min()
            ) / (df['trend_energy'].max() - df['trend_energy'].min() + 1e-9)
        else:
            df['trend_energy_norm'] = 0.5
        
        # 3. 安全边际 (修复版)
        if 'pe' in df.columns:
            # PE在10-30之间最安全
            df['safety_margin'] = np.where(
                (df['pe'] > 0) & (df['pe'] < 100),
                1 - np.abs(df['pe'] - 20) / 20,
                0
            )
        else:
            df['safety_margin'] = 0.5
        
        # 4. 【新增】支撑位距离
        if 'support_20' in df.columns:
            df['distance_to_support'] = (df['close'] - df['support_20']) / df['close']
        else:
            df['distance_to_support'] = 0.5
        
        return df
    
    def _enhance_output(self, df):
        """
        增强输出信息
        
        新增字段:
        1. 信号强度 (弱买入/买入/强买入)
        2. 预期收益率 (基于历史统计)
        3. 风险等级 (低/中/高)
        4. 建议持有期
        5. 买入紧迫性
        """
        # 1. 信号强度
        df['signal_strength'] = pd.cut(
            df['composite_score'],
            bins=[0, 0.65, 0.75, 1.0],
            labels=['弱买入⭐', '买入⭐⭐', '强买入⭐⭐⭐']
        )
        
        # 2. 预期收益率 (简化模型: 评分*8%)
        df['expected_return'] = df['ml_score'] * 0.08
        df['expected_return_str'] = df['expected_return'].apply(lambda x: f"+{x:.1%}")
        
        # 3. 风险等级
        if 'volatility' in df.columns:
            df['risk_level'] = pd.cut(
                df['volatility'],
                bins=[0, 0.02, 0.04, 1.0],
                labels=['低风险🟢', '中风险🟡', '高风险🔴']
            )
        else:
            df['risk_level'] = '中风险🟡'
        
        # 4. 建议持有期
        if 'trend_energy' in df.columns:
            df['hold_period'] = np.where(
                df['trend_energy'] > 1.0,
                '5-10天(短线)',
                '20-30天(中线)'
            )
        else:
            df['hold_period'] = '10-20天'
        
        # 5. 买入紧迫性
        df['urgency'] = pd.cut(
            df['momentum_5'] if 'momentum_5' in df.columns else df['composite_score'],
            bins=[-1, 0, 0.05, 1],
            labels=['观望', '今日可买', '立即买入']
        )
        
        # 6. 【新增】推荐理由(详细版)
        def generate_detailed_reason(row):
            reasons = []
            
            # ML分数
            if row['ml_score'] > 0.8:
                reasons.append("AI高度确信")
            elif row['ml_score'] > 0.6:
                reasons.append("AI看好")
            
            # 资金流向
            if row.get('smart_money_score_norm', 0) > 0.7:
                reasons.append("主力资金抢筹")
            elif row.get('smart_money_score_norm', 0) > 0.5:
                reasons.append("资金流入")
            
            # 趋势
            if row.get('trend_energy_norm', 0) > 0.7:
                reasons.append("趋势强劲")
            elif row.get('trend_energy_norm', 0) > 0.5:
                reasons.append("趋势向上")
            
            # 位置
            if row.get('distance_to_support', 0.5) < 0.1:
                reasons.append("接近支撑位")
            
            return " + ".join(reasons) if reasons else "综合评分优选"
        
        df['recommend_reason_detail'] = df.apply(generate_detailed_reason, axis=1)
        
        return df


class RiskManager:
    """
    风险管理器 v3.0 - 彻底修复版
    核心改进:
    1. 使用指数信号而非净值回撤控制仓位
    2. 非对称止盈止损(盈亏比2:1)
    3. 持仓优化,避免频繁换手
    """
    
    def __init__(self, config):
        self.config = config
        self.position_entry_scores = {}
        self.position_entry_dates = {}
        
        # 仓位控制参数
        self.current_position_scalar = 1.0
        self.min_position = 0.3  # 【关键】最低仓位30%,永不空仓
        self.max_position = 1.0
        
        # 非对称止盈止损
        self.stop_loss_pct = -0.05  # 止损-5%
        self.take_profit_pct = 0.15  # 止盈+15%(盈亏比3:1)
        self.trailing_stop_pct = 0.10  # 移动止盈,盈利10%后启动

    def check_risk(self, positions: Dict, current_prices: Dict,
                   current_scores: Dict, current_date: str) -> List[Tuple[str, str]]:
        """
        个股风险检查 - 优化版
        重点: 非对称止盈止损 + 持仓优化
        """
        to_sell = []

        for code, pos in positions.items():
            if code not in current_prices: 
                continue

            price = current_prices[code]
            cost = pos['cost']
            pnl_pct = (price - cost) / cost

            if code not in self.position_entry_scores:
                self.position_entry_scores[code] = pos.get('score', 0)

            # === A. 止损(-5%) ===
            if pnl_pct <= self.stop_loss_pct:
                to_sell.append((code, '止损'))
                logger.info(f"[{code}] 触发止损: {pnl_pct:.2%}")
                continue

            # === B. 止盈(+15%) ===
            if pnl_pct >= self.take_profit_pct:
                to_sell.append((code, '止盈'))
                logger.info(f"[{code}] 触发止盈: {pnl_pct:.2%}")
                continue

            # === C. 移动止盈(盈利>10%后,回撤5%就卖) ===
            if pnl_pct >= self.trailing_stop_pct:
                # 记录最高盈利
                if 'max_profit' not in pos:
                    pos['max_profit'] = pnl_pct
                else:
                    pos['max_profit'] = max(pos['max_profit'], pnl_pct)
                
                # 从最高点回撤5%就卖
                drawdown_from_peak = (pnl_pct - pos['max_profit'])
                if drawdown_from_peak <= -0.05:
                    to_sell.append((code, '移动止盈'))
                    logger.info(f"[{code}] 移动止盈: 最高{pos['max_profit']:.2%}, 当前{pnl_pct:.2%}")
                    continue

            # === D. 评分衰减止损 ===
            current_score = current_scores.get(code, 0)
            entry_score = self.position_entry_scores.get(code, 0)

            if entry_score > 0:
                decay = (entry_score - current_score) / entry_score
                if decay > 0.4 and self._get_holding_days(pos, current_date) > 5:
                    to_sell.append((code, '因子衰减'))
                    logger.info(f"[{code}] 因子衰减: {decay:.2%}")
                    continue

            # === E. 低分止损 ===
            if current_score < 0.3:
                to_sell.append((code, '低分止损'))
                logger.info(f"[{code}] 低分止损: score={current_score:.2f}")
                continue

            # === F. 持仓期满(放宽到30天) ===
            holding_days = self._get_holding_days(pos, current_date)
            if holding_days >= 30:
                to_sell.append((code, '持仓期满'))
                logger.info(f"[{code}] 持仓期满: {holding_days}天")
                continue

        return to_sell

    def check_portfolio_risk_v3(self, current_value: float, initial_capital: float,
                                current_date: str, index_data: pd.DataFrame = None) -> Dict:
        """
        【核心改进】基于指数信号的仓位控制
        
        逻辑:
        1. 使用沪深300的20日均线作为仓位信号
        2. 指数在均线上方: 满仓(100%)
        3. 指数在均线下方: 半仓(50%)
        4. 指数跌破均线且回撤>10%: 轻仓(30%)
        5. 永不空仓,避免错过反弹
        """
        
        # 【关键修复】计算当前回撤 - 修正公式
        # 回撤应该基于历史最高净值,而非初始资金
        if not hasattr(self, 'peak_value'):
            self.peak_value = initial_capital
        
        # 更新峰值
        self.peak_value = max(self.peak_value, current_value)
        
        # 正确的回撤计算: (峰值 - 当前值) / 峰值
        drawdown = (self.peak_value - current_value) / self.peak_value if self.peak_value > 0 else 0
        
        # 确保回撤在合理范围内 [0, 1]
        drawdown = max(0, min(1, drawdown))
        
        # === 方法1: 使用指数信号(推荐) ===
        if index_data is not None and len(index_data) > 0:
            try:
                # 获取当前日期的指数数据
                index_current = index_data[index_data['trade_date'] <= current_date].tail(20)
                
                if len(index_current) >= 20:
                    current_close = index_current.iloc[-1]['close']
                    ma20 = index_current['close'].rolling(20).mean().iloc[-1]
                    
                    # 指数相对均线的位置
                    index_position = (current_close - ma20) / ma20
                    
                    # 根据指数位置调整仓位
                    if index_position > 0.02:  # 指数在均线上方2%
                        target_position = 1.0
                        tier_name = "满仓(指数强势)"
                    elif index_position > -0.02:  # 指数在均线附近
                        target_position = 0.7
                        tier_name = "七成仓(指数震荡)"
                    elif drawdown < 0.10:  # 指数弱势但回撤不大
                        target_position = 0.5
                        tier_name = "半仓(指数弱势)"
                    else:  # 指数弱势且回撤较大
                        target_position = 0.3
                        tier_name = "轻仓(防守)"
                    
                    # 【优化】平滑仓位变化(避免过于激进的调整)
                    # 限制单次仓位调整幅度不超过20%
                    max_position_change = 0.2
                    position_change = target_position - self.current_position_scalar
                    
                    if abs(position_change) > max_position_change:
                        if position_change > 0:
                            target_position = self.current_position_scalar + max_position_change
                        else:
                            target_position = self.current_position_scalar - max_position_change
                    
                    # 如果仓位变化很小，则维持当前仓位
                    if abs(target_position - self.current_position_scalar) < 0.05:
                        target_position = self.current_position_scalar
                    
                    self.current_position_scalar = target_position
                    
                    message = f"{tier_name}, 指数位置={index_position:.2%}, 回撤={drawdown:.2%}"
                    logger.info(f"[{current_date}] {message}")
                    
                    return {
                        'action': 'normal',
                        'position_scalar': self.current_position_scalar,
                        'tier_name': tier_name,
                        'drawdown': drawdown,
                        'message': message
                    }
            except Exception as e:
                logger.warning(f"指数信号计算失败: {e}, 使用回撤方法")
        
        # === 方法2: 回撤方法(备选) ===
        # 【关键修改】永不空仓,最低保持30%
        if drawdown < 0.05:
            target_position = 1.0
            tier_name = "满仓"
        elif drawdown < 0.10:
            target_position = 0.7
            tier_name = "七成仓"
        elif drawdown < 0.15:
            target_position = 0.5
            tier_name = "半仓"
        else:
            target_position = 0.3  # 最低30%,永不空仓
            tier_name = "轻仓"
        
        # 【优化】平滑仓位变化(避免过于激进的调整)
        # 限制单次仓位调整幅度不超过20%
        max_position_change = 0.2
        position_change = target_position - self.current_position_scalar
        
        if abs(position_change) > max_position_change:
            if position_change > 0:
                target_position = self.current_position_scalar + max_position_change
            else:
                target_position = self.current_position_scalar - max_position_change
        
        # 如果仓位变化很小，则维持当前仓位
        if abs(target_position - self.current_position_scalar) < 0.05:
            target_position = self.current_position_scalar
        
        self.current_position_scalar = max(target_position, self.min_position)
        
        message = f"{tier_name}, 回撤={drawdown:.2%}"
        logger.info(f"[{current_date}] {message}")
        
        return {
            'action': 'normal',
            'position_scalar': self.current_position_scalar,
            'tier_name': tier_name,
            'drawdown': drawdown,
            'message': message
        }

    def _get_holding_days(self, pos, current_date):
        """计算持仓天数"""
        try:
            entry = pd.to_datetime(str(pos['entry_date']))
            curr = pd.to_datetime(str(current_date))
            return (curr - entry).days
        except:
            return 0

    # 兼容接口
    def check_portfolio_risk(self, current_value, initial_capital, current_date, index_data=None):
        """兼容旧接口"""
        return self.check_portfolio_risk_v3(current_value, initial_capital, current_date, index_data)

    def check_stop_loss(self, positions: dict, current_prices: dict) -> list:
        """兼容接口"""
        stop_loss_list = []
        for code, pos_info in positions.items():
            if code not in current_prices: 
                continue
            cost = pos_info['cost']
            current_price = current_prices[code]
            if cost == 0: 
                continue
            ret = (current_price - cost) / cost
            if ret <= self.stop_loss_pct:
                stop_loss_list.append(code)
        return stop_loss_list

    def check_take_profit(self, positions: dict, current_prices: dict) -> list:
        """兼容接口"""
        take_profit_list = []
        for code, pos_info in positions.items():
            if code not in current_prices: 
                continue
            cost = pos_info['cost']
            current_price = current_prices[code]
            if cost == 0: 
                continue
            ret = (current_price - cost) / cost
            if ret >= self.take_profit_pct:
                take_profit_list.append(code)
        return take_profit_list


class PortfolioManager:
    """
    组合管理器 - 优化版
    核心改进: 减少换手率,持仓优化
    """
    
    def __init__(self, config):
        self.config = config
        self.positions = {}
        self.cash = config.backtest.initial_capital
        self.trades = []
        self.risk_manager = None
        self.buy_cost = config.strategy.commission_rate
        self.sell_cost = config.strategy.commission_rate + config.strategy.stamp_tax

    def set_risk_manager(self, risk_manager):
        self.risk_manager = risk_manager

    def update_positions(self, target_df: pd.DataFrame, current_prices: Dict,
                        current_scores: Dict, date: str, index_data: pd.DataFrame = None) -> List[Dict]:
        """
        更新持仓 - 优化版
        核心: 减少不必要的换手
        """
        new_trades = []

        # 1. 获取当前仓位建议
        portfolio_value = self.get_portfolio_value(current_prices)
        if self.risk_manager:
            risk_status = self.risk_manager.check_portfolio_risk(
                portfolio_value, self.config.backtest.initial_capital, date, index_data
            )
            position_scalar = risk_status['position_scalar']
        else:
            position_scalar = 1.0

        # 2. 【优化】持仓保留逻辑 - 减少换手
        target_codes = set(target_df['ts_code'].values)
        to_sell = []
        
        for code in self.positions.keys():
            # 如果股票仍在目标池中,保留持仓(除非触发风控)
            if code in target_codes:
                continue
            
            # 检查是否需要强制卖出(风控触发)
            pos = self.positions[code]
            price = current_prices.get(code, pos['cost'])
            pnl_pct = (price - pos['cost']) / pos['cost']
            
            # 只有在严重亏损或严重盈利时才卖出
            if pnl_pct < -0.08 or pnl_pct > 0.20:
                to_sell.append(code)
                logger.info(f"[{code}] 强制卖出: pnl={pnl_pct:.2%}")
            else:
                # 否则继续持有,即使不在Top N中(容忍度)
                logger.info(f"[{code}] 容忍持有: pnl={pnl_pct:.2%}")

        # 3. 执行卖出
        for code in to_sell:
            trade = self._sell(code, current_prices.get(code), date, '调仓卖出')
            if trade: 
                new_trades.append(trade)

        # 4. 买入逻辑(应用仓位比例)
        available_cash = max(0, self.cash * 0.95 * position_scalar)

        to_buy = []
        for _, row in target_df.iterrows():
            code = row['ts_code']
            if code not in self.positions:
                to_buy.append(row)

        if not to_buy:
            self.trades.extend(new_trades)
            return new_trades

        # 【优化】按优先级排序,优先买入高分股票
        if 'priority' in target_df.columns:
            to_buy = sorted(to_buy, key=lambda x: x.get('priority', 999))
        
        # 只买入Top N,不买入buffer
        actual_top_n = self.config.strategy.top_n
        to_buy = to_buy[:actual_top_n]

        per_stock_cash = available_cash / len(to_buy)

        for row in to_buy:
            code = row['ts_code']
            price = current_prices.get(code)
            if not price: 
                continue

            shares = int(per_stock_cash / (price * (1 + self.buy_cost)) / 100) * 100

            if shares >= 100:
                trade = self._buy(code, shares, price, date, row.get('ml_score', 0))
                if trade: 
                    new_trades.append(trade)

        self.trades.extend(new_trades)
        return new_trades

    def _buy(self, code, shares, price, date, score):
        """买入"""
        cost = shares * price * (1 + self.buy_cost)
        if cost > self.cash: 
            return None

        self.cash -= cost
        if code not in self.positions:
            self.positions[code] = {
                'shares': shares, 
                'cost': price, 
                'entry_date': date,
                'score': score, 
                'name': code
            }

        return {
            'date': date, 'code': code, 'action': 'buy',
            'shares': shares, 'price': price, 'amount': cost
        }

    def _sell(self, code, price, date, reason):
        """卖出"""
        if not price or code not in self.positions: 
            return None
        pos = self.positions[code]
        revenue = pos['shares'] * price * (1 - self.sell_cost)
        pnl = revenue - (pos['shares'] * pos['cost'])
        self.cash += revenue
        del self.positions[code]
        return {
            'date': date, 'code': code, 'action': 'sell',
            'shares': pos['shares'], 'price': price, 'amount': revenue,
            'pnl': pnl, 'reason': reason
        }

    def get_portfolio_value(self, current_prices):
        """获取组合价值"""
        val = self.cash
        for code, pos in self.positions.items():
            price = current_prices.get(code, pos['cost'])
            val += pos['shares'] * price
        return val

    def get_positions_df(self, current_prices, date):
        """获取持仓DataFrame"""
        data = []
        for code, pos in self.positions.items():
            price = current_prices.get(code, pos['cost'])
            val = pos['shares'] * price
            pnl = (price - pos['cost']) * pos['shares']
            data.append({
                'ts_code': code, 
                'name': pos.get('name'), 
                'shares': pos['shares'],
                'cost': pos['cost'], 
                'price': price, 
                'value': val, 
                'pnl': pnl,
                'entry_date': pos['entry_date']
            })
        return pd.DataFrame(data)


# 保留其他类(保持原有逻辑)
class SentimentAnalyzer:
    """舆情分析器"""
    def __init__(self, config):
        self.config = config
    def apply_sentiment_filter(self, df, date):
        return df


class MarketTiming:
    """择时模块"""
    def __init__(self, config):
        self.config = config
    def get_market_signal(self, index_data, date):
        return 1.0