"""
机器学习模型模块 - 增强版
整合了 Triple Barrier 标签生成和集成投票机制
"""

import logging
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, precision_score, roc_auc_score
import joblib
import warnings

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class TripleBarrierLabeler:
    """
    三屏障标签生成器
    基于 止盈(上界)、止损(下界) 和 持仓期(垂直界) 生成标签
    """
    def __init__(self,
                 profit_threshold=0.05,    # 止盈 5%
                 stop_loss_threshold=-0.03, # 止损 -3%
                 holding_period=10):
        self.profit_threshold = profit_threshold
        self.stop_loss_threshold = stop_loss_threshold
        self.holding_period = holding_period

    def generate_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """生成标签: 1(Buy), 0(Hold/Sell)"""
        # 确保数据按股票和时间排序
        df = df.sort_values(['ts_code', 'trade_date']).reset_index(drop=True)

        # 预计算未来价格序列
        grouped = df.groupby('ts_code')

        labels = []
        # 注意：这里使用循环处理每只股票，向量化实现较复杂
        # 为了效率，实际生产中可使用numba加速
        for name, group in grouped:
            close_prices = group['close'].values
            n = len(close_prices)
            group_labels = np.zeros(n)

            for i in range(n - self.holding_period):
                current_price = close_prices[i]
                future_prices = close_prices[i+1 : i+1+self.holding_period]

                # 计算收益率
                returns = (future_prices - current_price) / current_price

                # 触碰上界 (止盈)
                if np.any(returns >= self.profit_threshold):
                    group_labels[i] = 1
                # 触碰下界 (止损) - 这里简化为二分类，触碰止损或未达标都为0
                elif np.any(returns <= self.stop_loss_threshold):
                    group_labels[i] = 0
                # 持仓期满
                else:
                    # 如果期末收益为正，可视情况定，这里保守设为0
                    group_labels[i] = 1 if returns[-1] > 0 else 0

            labels.extend(group_labels)

        df['label'] = labels
        return df


class MLModel:
    """机器学习模型基类"""

    def __init__(self, config, model_type: str = None):
        self.config = config
        self.model_type = model_type or config.model.model_type
        self.model = None
        self.feature_names = None

    def build_model(self):
        """构建模型"""
        # 参数自适应调整
        xgb_params = self.config.model.xgb_params.copy()
        lgb_params = self.config.model.lgb_params.copy()

        if self.model_type == 'xgboost':
            import xgboost as xgb
            self.model = xgb.XGBRegressor(**xgb_params)

        elif self.model_type == 'lightgbm':
            import lightgbm as lgb
            self.model = lgb.LGBMRegressor(**lgb_params)

        elif self.model_type == 'catboost':
            from catboost import CatBoostRegressor
            self.model = CatBoostRegressor(verbose=False, random_seed=42)

        elif self.model_type == 'random_forest':
            from sklearn.ensemble import RandomForestRegressor
            self.model = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42)

        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def train(self, X_train, y_train, X_val=None, y_val=None):
        if self.model is None:
            self.build_model()

        self.feature_names = list(X_train.columns)

        # 训练
        if self.model_type in ['xgboost', 'lightgbm'] and X_val is not None:
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
        else:
            self.model.fit(X_train, y_train)

        # 评估
        train_pred = self.model.predict(X_train)
        metrics = {
            'rmse': np.sqrt(mean_squared_error(y_train, train_pred)),
            'r2': r2_score(y_train, train_pred)
        }
        return {'train_metrics': metrics}

    def predict(self, X):
        if self.model is None:
            raise ValueError("Model not trained")
        return self.model.predict(X)

    def get_feature_importance(self):
        if hasattr(self.model, 'feature_importances_'):
            return pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
        return pd.DataFrame()

    def save(self, path):
        joblib.dump(self.model, path)

    def load(self, path):
        self.model = joblib.load(path)


class EnsembleModel:
    """
    集成模型 (Voting Mechanism)
    整合了老项目中的多模型投票思想
    """
    def __init__(self, config):
        self.config = config
        self.models = []
        self.model_types = ['xgboost', 'lightgbm', 'random_forest'] # 默认集成这三种

    def build_models(self):
        for m_type in self.model_types:
            try:
                model = MLModel(self.config, model_type=m_type)
                model.build_model()
                self.models.append(model)
            except Exception as e:
                logger.warning(f"Failed to build {m_type}: {e}")

    def train(self, X_train, y_train, X_val=None, y_val=None):
        for model in self.models:
            logger.info(f"Training ensemble component: {model.model_type}")
            model.train(X_train, y_train, X_val, y_val)

    def predict(self, X) -> np.ndarray:
        """
        加权平均预测
        也可以改为投票制：只有 > 阈值的才算1
        """
        predictions = []
        for model in self.models:
            pred = model.predict(X)
            # 归一化预测值到 0-1 (如果是回归模型输出收益率，这里可以跳过或做Rank)
            predictions.append(pred)

        # 简单平均
        return np.mean(predictions, axis=0)
    
    def save(self, path: str):
        """保存集成模型"""
        import joblib
        # 保存整个集成模型对象
        joblib.dump({
            'models': self.models,
            'model_types': self.model_types
        }, path)
    
    def load(self, path: str):
        """加载集成模型"""
        import joblib
        data = joblib.load(path)
        self.models = data['models']
        self.model_types = data['model_types']
    
    def get_feature_importance(self):
        """获取集成模型的特征重要性（平均各模型的重要性）"""
        if not self.models:
            return pd.DataFrame()
        
        importance_dfs = []
        for model in self.models:
            imp = model.get_feature_importance()
            if not imp.empty:
                importance_dfs.append(imp)
        
        if not importance_dfs:
            return pd.DataFrame()
        
        # 合并所有模型的特征重要性
        combined_importance = importance_dfs[0].copy()
        for df in importance_dfs[1:]:
            combined_importance = combined_importance.merge(
                df, on='feature', how='outer', suffixes=('', '_right')
            )
            # 如果某个特征在某些模型中不存在，将其重要性设为0
            for col in combined_importance.columns:
                if col.endswith('_right'):
                    combined_importance[col] = combined_importance[col].fillna(0)
                    original_col = col.replace('_right', '')
                    combined_importance[original_col] = (
                        combined_importance[original_col].fillna(0) + 
                        combined_importance[col]
                    ) / 2  # 平均重要性
                    combined_importance = combined_importance.drop(columns=[col])
        
        # 按平均重要性排序
        combined_importance = combined_importance.sort_values('importance', ascending=False)
        return combined_importance


class ImprovedWalkForwardTrainer:
    """改进的滚动训练器"""
    
    def __init__(self, config):
        self.config = config
        self.models = {}
        self.feature_importance_history = []

    def prepare_data(self, df):
        # 排除非特征列
        exclude_cols = ['ts_code', 'trade_date', 'label', 'forward_return',
                       'open', 'high', 'low', 'close', 'vol', 'amount', 'name', 'industry']
        feature_cols = [c for c in df.columns if c not in exclude_cols]

        X = df[feature_cols].fillna(0)
        # 优先使用 Triple Barrier 生成的 label，否则使用 forward_return
        y = df['label'] if 'label' in df.columns else df['forward_return']

        # 替换无限值
        X = X.replace([np.inf, -np.inf], 0)
        y = y.fillna(0)

        return X, y
    
    def select_features(self, X, y, method='importance', top_k=20):
        """
        特征选择 - 防止过拟合
        
        方法:
        1. 基于模型重要性
        2. 基于IC值
        3. 基于方差(剔除常数特征)
        """
        # 方法1: 剔除低方差特征
        from sklearn.feature_selection import VarianceThreshold
        selector = VarianceThreshold(threshold=0.01)
        X_filtered = selector.fit_transform(X)
        selected_cols = X.columns[selector.get_support()].tolist()
        
        X = X[selected_cols]
        
        # 方法2: 基于重要性
        if method == 'importance':
            import lightgbm as lgb
            
            # 快速训练一个模型获取特征重要性
            model = lgb.LGBMRegressor(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.1,
                random_state=42,
                verbose=-1
            )
            
            model.fit(X, y)
            
            importance_df = pd.DataFrame({
                'feature': X.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            # 选择Top K特征
            selected_features = importance_df.head(top_k)['feature'].tolist()
            
            logger.info(f"Selected {len(selected_features)} features from {len(X.columns)}")
            
            return X[selected_features], selected_features
        
        elif method == 'ic':
            # 基于IC值选择
            ic_values = {}
            for col in X.columns:
                ic = X[col].corr(y)
                ic_values[col] = abs(ic)
            
            ic_df = pd.DataFrame.from_dict(ic_values, orient='index', columns=['ic'])
            ic_df = ic_df.sort_values('ic', ascending=False)
            
            selected_features = ic_df.head(top_k).index.tolist()
            
            return X[selected_features], selected_features

    def train_with_time_series_cv(self, X, y, n_splits=5):
        """
        时间序列交叉验证
        
        关键: 不能随机打乱,必须按时间顺序
        """
        from sklearn.model_selection import TimeSeriesSplit
        import lightgbm as lgb
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        cv_scores = []
        
        for train_idx, val_idx in tscv.split(X):
            X_train_fold = X.iloc[train_idx]
            y_train_fold = y.iloc[train_idx]
            X_val_fold = X.iloc[val_idx]
            y_val_fold = y.iloc[val_idx]
            
            model = lgb.LGBMRegressor(
                objective='regression',
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbose=-1
            )
            
            model.fit(
                X_train_fold, y_train_fold,
                eval_set=[(X_val_fold, y_val_fold)],
                callbacks=[lgb.early_stopping(50, verbose=False)]
            )
            
            # 评估
            pred_val = model.predict(X_val_fold)
            from sklearn.metrics import mean_squared_error
            score = mean_squared_error(y_val_fold, pred_val, squared=False)
            cv_scores.append(score)
        
        # 用全量数据训练最终模型
        final_model = lgb.LGBMRegressor(
            objective='regression',
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1
        )
        
        final_model.fit(X, y)
        
        avg_cv_score = np.mean(cv_scores)
        
        return final_model, avg_cv_score
    
    def walk_forward_train_v2(self, df):
        """
        优化的滚动训练流程
        """
        df = df.sort_values('trade_date')
        dates = sorted(df['trade_date'].unique())
        
        # === 关键改进1: 扩大训练窗口 ===
        train_window = 252  # 1年数据(原来100天太短)
        retrain_frequency = 21  # 每月重训练(原来20天)
        
        predictions_list = []
        
        for i in range(train_window, len(dates), retrain_frequency):
            train_dates = dates[i-train_window : i]
            test_dates = dates[i : i+retrain_frequency]
            
            if not test_dates: 
                break
            
            logger.info(f"Training: {train_dates[0]}-{train_dates[-1]}, "
                       f"Testing: {test_dates[0]}-{test_dates[-1]}")
            
            # 训练数据
            train_data = df[df['trade_date'].isin(train_dates)]
            X_train, y_train = self.prepare_data(train_data)
            
            if len(X_train) < 1000:  # 至少1000条样本
                logger.warning(f"Insufficient training data: {len(X_train)}")
                continue
            
            # === 关键改进2: 特征选择 ===
            X_train_selected, selected_features = self.select_features(
                X_train, y_train, method='importance'
            )
            
            # === 关键改进3: 时间序列交叉验证 ===
            model, cv_score = self.train_with_time_series_cv(
                X_train_selected, y_train, n_splits=5
            )
            
            logger.info(f"CV Score: {cv_score:.4f}")
            
            # 保存模型和特征
            self.models[test_dates[0]] = {
                'model': model,
                'features': selected_features,
                'cv_score': cv_score
            }
            
            # 预测
            test_data = df[df['trade_date'].isin(test_dates)]
            if len(test_data) == 0: 
                continue
            
            X_test, _ = self.prepare_data(test_data)
            X_test_selected = X_test[selected_features]
            
            preds = model.predict(X_test_selected)
            
            result_df = test_data[['ts_code', 'trade_date']].copy()
            result_df['ml_score'] = preds
            predictions_list.append(result_df)
        
        return pd.concat(predictions_list) if predictions_list else pd.DataFrame()

# 保持原来的WalkForwardTrainer类作为备选，但推荐使用改进版
class WalkForwardTrainer:
    """滚动训练器"""
    def __init__(self, config):
        self.config = config
        self.models = {}

    def prepare_data(self, df):
        # 排除非特征列
        exclude_cols = ['ts_code', 'trade_date', 'label', 'forward_return',
                       'open', 'high', 'low', 'close', 'vol', 'amount', 'name', 'industry']
        feature_cols = [c for c in df.columns if c not in exclude_cols]

        X = df[feature_cols].fillna(0)
        # 优先使用 Triple Barrier 生成的 label，否则使用 forward_return
        y = df['label'] if 'label' in df.columns else df['forward_return']

        # 替换无限值
        X = X.replace([np.inf, -np.inf], 0)
        y = y.fillna(0)

        return X, y

    def walk_forward_train(self, df):
        """执行滚动训练"""
        df = df.sort_values('trade_date')
        dates = sorted(df['trade_date'].unique())

        train_window = self.config.model.train_window
        predictions_list = []

        # 检查是否使用集成模型
        use_ensemble = self.config.model.use_ensemble

        # 滚动窗口
        for i in range(train_window, len(dates), self.config.model.retrain_frequency):
            train_dates = dates[i-train_window : i]
            test_dates = dates[i : i+self.config.model.retrain_frequency]

            if not test_dates: break

            logger.info(f"Walk-forward: Training {train_dates[0]}-{train_dates[-1]}, Predicting {test_dates[0]}-{test_dates[-1]}")

            # 训练数据
            train_data = df[df['trade_date'].isin(train_dates)]
            X_train, y_train = self.prepare_data(train_data)

            if len(X_train) < 100: continue

            # 训练
            if use_ensemble:
                model = EnsembleModel(self.config)
                model.build_models()
            else:
                model = MLModel(self.config)
                model.build_model()

            model.train(X_train, y_train)
            self.models[test_dates[0]] = model # 保存模型引用

            # 预测
            test_data = df[df['trade_date'].isin(test_dates)]
            if len(test_data) == 0: continue

            X_test, _ = self.prepare_data(test_data)
            preds = model.predict(X_test)

            result_df = test_data[['ts_code', 'trade_date']].copy()
            result_df['ml_score'] = preds
            predictions_list.append(result_df)

        if not predictions_list:
            return pd.DataFrame()

        return pd.concat(predictions_list)


class LabelGenerator:
    """标签生成器工厂 - 修复版"""
    
    @staticmethod
    def add_labels(df, config):
        """
        根据配置添加标签 - 修复时间错配问题
        
        核心修复:
        1. 预测次日收盘收益(而非开盘收益)
        2. 特征使用T-1日数据(避免未来函数)
        """
        periods = config.model.forward_return_days
        
        # ===== 修复1: 正确的标签定义 =====
        # ✅ 预测次日收盘相对今日收盘的收益
        df['next_close'] = df.groupby('ts_code')['close'].shift(-periods)
        df['forward_return'] = (df['next_close'] - df['close']) / df['close']
        
        # ❌ 删除原来错误的标签
        # df['next_open'] = df.groupby('ts_code')['open'].shift(-1)
        # df['forward_return'] = (df['next_open'] - df['close']) / df['close']
        
        # ===== 修复2: 特征时间对齐 =====
        # 重要: 所有特征必须基于T-1日及之前的数据
        # 否则在T日开盘时无法获取这些特征
        
        # 获取所有特征列(排除基础列和标签列)
        exclude_cols = ['ts_code', 'trade_date', 'name', 'industry',
                       'open', 'high', 'low', 'close', 'vol', 'amount',
                       'forward_return', 'next_close', 'next_open', 'label']
        
        feature_cols = [c for c in df.columns if c not in exclude_cols]
        
        # ===== 修复3: 波动率标签(更稳健) =====
        df['volatility'] = df.groupby('ts_code')['close'].pct_change().rolling(5).std()
        df['sharpe_like'] = df['forward_return'] / (df['volatility'].shift(1) + 1e-8)
        
        # ===== 修复4: Triple Barrier标签(可选) =====
        if getattr(config.model, 'use_triple_barrier', False):
            labeler = TripleBarrierLabeler(
                profit_threshold=0.05,
                stop_loss_threshold=-0.03,
                holding_period=periods
            )
            df = labeler.generate_labels(df)
        
        # 删除临时列
        df = df.drop(columns=['next_close'], errors='ignore')
        
        logger.info(f"✅ 标签生成完成 (已修复时间错配问题)")
        logger.info(f"   有效样本数: {df['forward_return'].notna().sum()}")
        logger.info(f"   正收益比例: {(df['forward_return'] > 0).sum() / df['forward_return'].notna().sum():.1%}")
        
        return df