def main():

    parser = argparse.ArgumentParser(description='实盘选股推荐 - 修复增强版')

    parser.add_argument('--date', type=str, help='强制指定运行日期 (格式: YYYYMMDD)')

    parser.add_argument('--lookback', type=int, default=60, help='回看天数 (默认60天)')

    parser.add_argument('--debug', action='store_true', help='调试模式(显示详细信息)')

    args = parser.parse_args()



    print("\n" + "="*80)

    print("🚀 实盘选股推荐系统 v2.1 (修复增强版)")

    print("="*80)

    print("主要改进:")

    print("  ✅ 修复时间错配问题")

    print("  ✅ 优化选股逻辑")

    print("  ✅ 增强输出信息")

    print("  ✅ 动态推荐数量")

    print("="*80 + "\n")

    

    logger.info("初始化系统...")

    

    config = Config()

    config.data.use_cache = True

    

    dm = DataManager(config)

    fe = FactorEngine(config)

    trainer = WalkForwardTrainer(config)

    

    # 1. 智能日期选择

    if args.date:

        latest_date = args.date

        is_real_time = False

        data_source = "用户指定"

        logger.info(f"✓ 使用用户指定日期: {latest_date}")

    else:

        latest_date, is_real_time, data_source = get_latest_available_trading_date(

            config.data.cache_dir

        )

        logger.info(f"✓ 自动选择日期: {latest_date} (来源: {data_source})")

    

    # 2. 获取数据

    try:

        start_date = (

            datetime.strptime(latest_date, '%Y%m%d') - timedelta(days=args.lookback)

        ).strftime('%Y%m%d')

    except ValueError:

        logger.error(f"❌ 日期格式错误: {latest_date}")

        return



    logger.info(f"\n{'='*60}")

    logger.info(f"📊 数据准备: {start_date} ~ {latest_date}")

    logger.info(f"{'='*60}\n")



    df_history = try_load_from_large_cache(config.data.cache_dir, latest_date, start_date)



    if df_history is None:

        logger.info("本地大缓存未命中，尝试常规获取流程...")

        

        try:

            df_history = dm.get_daily_data(start_date=start_date, end_date=latest_date)

        except Exception as e:

            logger.error(f"❌ 数据获取失败: {e}")

            return



    if latest_date not in df_history['trade_date'].values:

        logger.error(f"❌ 数据中不包含 {latest_date}")

        logger.error(f"解决方案: python {sys.argv[0]} --date {df_history['trade_date'].max()}")

        return



    # 3. 计算因子

    logger.info(f"\n{'='*60}")

    logger.info("⚙️  计算因子特征...")

    logger.info(f"{'='*60}\n")

    

    df_with_factors = fe.calculate_all_factors(df_history)

    

    df_latest = df_with_factors[df_with_factors['trade_date'] == latest_date].copy()

    if df_latest.empty:

        logger.error(f"❌ 因子计算后 {latest_date} 数据为空")

        return



    logger.info(f"✅ 因子计算完成: {len(df_latest)} 只股票")



    # 4. 加载模型并预测

    logger.info(f"\n{'='*60}")

    logger.info("🤖 加载机器学习模型...")

    logger.info(f"{'='*60}\n")

    

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

        logger.info(f"✅ 模型加载成功")

        

        X, _ = trainer.prepare_data(df_latest)

        preds = model.predict(X)

        df_latest['ml_score'] = preds

        

        logger.info(f"✅ 评分完成")

        logger.info(f"   评分范围: {preds.min():.3f} ~ {preds.max():.3f}")

        logger.info(f"   平均评分: {preds.mean():.3f}")

        logger.info(f"   高分股票(>0.6): {(preds > 0.6).sum()} 只")

        

    except Exception as e:

        logger.error(f"❌ 模型预测失败: {e}")

        if args.debug:

            import traceback

            traceback.print_exc()

        return



    # 5. 选股

    logger.info(f"\n{'='*60}")

    logger.info("🎯 执行选股策略...")

    logger.info(f"{'='*60}\n")

    

    strategy = Strategy(config)

    selected = strategy.select_stocks_live(df_latest, latest_date)

    

    if not selected.empty:

        selected = strategy.calculate_weights(selected)



    # 6. 输出结果

    output_dir = Path('output/live_recommendations')

    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%H%M%S')



    print("\n" + "="*80)

    print(f"📋 {latest_date} 实盘选股推荐")

    print(f"   数据来源: {data_source}")

    print(f"   生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    print("="*80 + "\n")



    if selected.empty:

        print("⚠️  今日无推荐股票")

        print("\n可能原因:")

        print("  - 市场整体评分较低")

        print("  - 未找到符合策略条件的标的")

        print("  - 风险控制触发限制\n")

    else:

        # 准备显示列

        display_cols = [

            'ts_code', 'name', 'close', 'pct_chg',

            'signal_strength', 'expected_return_str', 'risk_level',

            'urgency', 'hold_period', 'weight_pct',

            'recommend_reason_detail'

        ]

        

        # 确保列存在

        display_cols = [c for c in display_cols if c in selected.columns]

        

        # 列名中文化

        col_rename = {

            'ts_code': '代码',

            'name': '名称',

            'close': '现价',

            'pct_chg': '今日涨跌',

            'signal_strength': '信号强度',

            'expected_return_str': '预期收益',

            'risk_level': '风险等级',

            'urgency': '紧迫性',

            'hold_period': '建议持有期',

            'weight_pct': '建议仓位',

            'recommend_reason_detail': '推荐理由'

        }

        

        display_df = selected[display_cols].copy()

        display_df = display_df.rename(columns=col_rename)

        

        # 格式化数值

        if '今日涨跌' in display_df.columns:

            display_df['今日涨跌'] = display_df['今日涨跌'].apply(lambda x: f"{x:+.2f}%")

        

        if '现价' in display_df.columns:

            display_df['现价'] = display_df['现价'].apply(lambda x: f"{x:.2f}")

        

        # 设置显示选项

        pd.set_option('display.max_rows', None)

        pd.set_option('display.max_columns', None)

        pd.set_option('display.width', 1000)

        pd.set_option('display.unicode.ambiguous_as_wide', True)

        pd.set_option('display.unicode.east_asian_width', True)

        pd.set_option('display.max_colwidth', 50)



        print(display_df.to_string(index=False))