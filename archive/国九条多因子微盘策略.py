"""
小市值轮动量化投资策略
策略类型：小市值轮动 + 趋势动态调仓
基准指数：399101.XSHE（深证综指）
调仓频率：每周二 10:00
空仓月份：1月、4月
"""

from datetime import datetime
import pandas as pd

# ===== 持仓参数 =====
INITIAL_STOCK_NUM = 4
HIGHEST_PRICE = 50.0
MAX_MARKET_VALUE = 1e10
MIN_MARKET_VALUE = 1e9

# ===== 指数与标的参数 =====
BENCHMARK_INDEX = '399101.XSHE'  # 深证综指
ETF_SAFE = '511880.SS'  # 银华日利ETF

# ===== 调仓时间参数 =====
REBALANCE_WEEKDAY = 1
REBALANCE_TIME = '10:00'
TAIL_CHECK_TIME = '14:00'
EMPTY_MONTH_CHECK_TIME = '14:50'
RISK_CHECK_TIME = '10:00'

# ===== 空仓月份 =====
EMPTY_MONTHS = [1, 4]

# ===== 交易成本参数 =====
COMMISSION_RATE_BUY = 0.00025
COMMISSION_RATE_SELL = 0.00025
STAMP_TAX_RATE = 0.001
MIN_COMMISSION = 5.0
SLIPPAGE_RATE = 0.0003

# ===== 风控参数 =====
STOP_PROFIT_RATIO = 2.0
STOP_LOSS_RATIO = 0.09  # 对齐聚宽版：跌幅 > 9% 止损
MARKET_STOP_LOSS_THRESHOLD = 0.05  # 对齐聚宽：成分股平均绝对涨跌幅阈值
MARKET_STOP_LOSS_BATCH_SIZE = 100  # 市场止损分批获取的批量大小（容错用）

# ===== 审计意见过滤 =====
FILTER_AUDIT = False

# ===== 动态持仓数量映射 =====
POSITION_MAPPING = [
    (float('inf'), 500, 3),
    (500, 200, 3),
    (200, -200, 4),
    (-200, -500, 5),
    (-500, float('-inf'), 6),
]

# ===== 次新股与财务参数 =====
MIN_LISTING_DAYS = 375
FINANCE_CANDIDATE_MULTIPLIER = 3


def initialize(context):
    g.stock_num = INITIAL_STOCK_NUM
    g.highest = HIGHEST_PRICE
    g.filter_audit = FILTER_AUDIT
    g.max_mv = MAX_MARKET_VALUE

    g.last_rebalance_date = None
    g.current_candidates = []
    g.target_positions = []
    g.limit_up_opened_stocks = []
    g.stopped_out = False
    g.stop_loss_etf_bought = False
    g.yesterday_limit_up_stocks = []

    # 设置交易成本
    set_commission(commission_ratio=COMMISSION_RATE_BUY, min_commission=MIN_COMMISSION, type="STOCK")
    set_slippage(slippage=SLIPPAGE_RATE)

    run_daily(context, weekly_rebalance, time=REBALANCE_TIME)
    run_daily(context, risk_check, time=RISK_CHECK_TIME)
    run_daily(context, handle_empty_month_clear, time=EMPTY_MONTH_CHECK_TIME)
    run_daily(context, tail_limit_up_check, time=TAIL_CHECK_TIME)
    run_daily(context, handle_empty_month_etf, time=REBALANCE_TIME)

    log.info("策略初始化完成 | stock_num=%d, highest=%.0f, max_mv=%.0e, filter_audit=%s" % (
        g.stock_num, g.highest, g.max_mv, g.filter_audit))


def before_trading_start(context, data):
    current_date = context.current_dt.strftime('%Y-%m-%d')
    log.info("=" * 60)
    log.info("策略运行日期：%s" % current_date)
    log.info("=" * 60)

    g.stopped_out = False
    g.stop_loss_etf_bought = False
    g.limit_up_opened_stocks = []

    g.yesterday_limit_up_stocks = detect_yesterday_limit_up_stocks(context)


def is_empty_month(context):
    month = context.current_dt.month
    return month in EMPTY_MONTHS


def get_index_constituents(context):
    current_date = context.current_dt.strftime('%Y%m%d')
    stocks = get_index_stocks(BENCHMARK_INDEX, current_date)
    log.info("获取 %s 成分股数量：%d" % (BENCHMARK_INDEX, len(stocks)))
    return stocks


def is_st_stock(stock_name):
    """判断是否ST股（加类型保护）"""
    if not isinstance(stock_name, str):
        return False
    return 'ST' in stock_name or '*ST' in stock_name


def is_delisting_stock(stock_name):
    """判断是否退市股（加类型保护）"""
    if not isinstance(stock_name, str):
        return False
    return '退' in stock_name


def is_excluded_board(security):
    """排除创业板(30)、科创板(68)、北交所(8/4)"""
    if not isinstance(security, str) or len(security) < 2:
        return False
    prefix = security[:2]
    return prefix in ('30', '68', '8', '4')


def get_pos_security(pos):
    return getattr(pos, 'sid', getattr(pos, 'security', ''))


def get_stock_name(security):
    """获取股票名称（依据 ptrade_API.json：get_stock_info，字段 stock_name）"""
    try:
        info = get_stock_info(security)
        if info and isinstance(info, dict):
            stock_info = info.get(security, {})
            if isinstance(stock_info, dict):
                name = stock_info.get('stock_name', '')
                if isinstance(name, str):
                    return name
    except Exception:
        pass
    return ''


def get_hold_positions(context):
    positions = get_positions()
    stock_positions = []
    for security, pos in positions.items():
        if security.startswith('511880'):
            continue
        if pos.amount > 0:
            stock_positions.append(pos)
    return stock_positions


def get_hold_securities(context):
    positions = get_hold_positions(context)
    return [get_pos_security(pos) for pos in positions]


def get_position_cost(security):
    try:
        pos = get_position(security)
        if pos and pos.amount > 0:
            return getattr(pos, 'cost_basis', 0)
    except Exception:
        pass
    return 0


def filter_basic_stock_pool(context):
    """基础股票池过滤（依据 ptrade_API.json 第102-104页 get_stock_info）"""
    stocks = get_index_constituents(context)
    valid_stocks = []
    excluded_info = {'st': 0, 'delisting': 0, 'board': 0, 'new': 0}

    try:
        all_info = get_stock_info(stocks, ['stock_name', 'listed_date', 'de_listed_date'])
    except Exception as e:
        log.warning("批量获取股票信息失败：%s，降级为逐只获取" % str(e))
        all_info = {}

    for stock in stocks:
        try:
            info = None
            if isinstance(all_info, dict) and stock in all_info:
                info = all_info[stock]

            if not isinstance(info, dict) or not info:
                excluded_info['new'] += 1
                continue

            stock_name = info.get('stock_name', '')
            if not isinstance(stock_name, str):
                stock_name = ''
            listing_date_str = info.get('listed_date', None)

            if is_st_stock(stock_name):
                excluded_info['st'] += 1
                continue
            if is_delisting_stock(stock_name):
                excluded_info['delisting'] += 1
                continue
            if is_excluded_board(stock):
                excluded_info['board'] += 1
                continue

            if listing_date_str is not None and isinstance(listing_date_str, str):
                try:
                    listing_date = datetime.strptime(listing_date_str, '%Y-%m-%d')
                    days_listed = (context.current_dt - listing_date).days
                    if days_listed < MIN_LISTING_DAYS:
                        excluded_info['new'] += 1
                        continue
                except Exception:
                    pass

            valid_stocks.append(stock)
        except Exception as e:
            log.warning("处理股票 %s 时异常：%s" % (stock, str(e)))
            continue

    log.info("基础过滤结果：有效 %d 只 | 剔除：ST=%d, 退市=%d, 板块=%d, 次新=%d" % (
        len(valid_stocks), excluded_info['st'], excluded_info['delisting'],
        excluded_info['board'], excluded_info['new']))
    return valid_stocks


def filter_finance_and_market_value(context, stocks):
    """财务与市值筛选（对齐聚宽：只传 date，取最新发布期）"""
    if not stocks:
        return []

    current_date = context.current_dt.strftime('%Y%m%d')
    stock_cap_list = []
    total = len(stocks)

    for i, stock in enumerate(stocks):
        try:
            mv_result = get_fundamentals(stock, 'valuation',
                                          fields=['total_value'],
                                          date=current_date)
            if mv_result is None or mv_result.empty:
                continue

            market_cap = float(mv_result['total_value'].iloc[0])

            if market_cap < MIN_MARKET_VALUE:
                continue
            if market_cap > MAX_MARKET_VALUE:
                continue

            income_result = get_fundamentals(
                stock, 'income_statement',
                fields=['operating_revenue', 'net_profit', 'np_parent_company_owners'],
                date=current_date
            )
            if income_result is None or income_result.empty:
                continue

            net_profit = float(income_result['net_profit'].iloc[0])
            net_profit_parent = float(income_result['np_parent_company_owners'].iloc[0])
            operating_revenue = float(income_result['operating_revenue'].iloc[0])

            if net_profit <= 0:
                continue
            if net_profit_parent <= 0:
                continue
            if operating_revenue <= 1e8:
                continue

            stock_cap_list.append((stock, market_cap))
        except Exception as e:
            log.warning("处理股票 %s 财务数据异常：%s" % (stock, str(e)))
            continue

        if (i + 1) % 20 == 0:
            log.info("财务筛选进度：%d/%d" % (i + 1, total))

    stock_cap_list.sort(key=lambda x: x[1])
    candidate_count = g.stock_num * FINANCE_CANDIDATE_MULTIPLIER
    selected = stock_cap_list[:candidate_count]

    log.info("财务筛选结果：合格 %d 只，取前 %d 只" % (len(stock_cap_list), len(selected)))
    log.info("财务筛选候选股票：%s" % [s[0] for s in selected])
    return [s[0] for s in selected]


def filter_price_and_audit(context, stocks):
    """价格、涨跌停过滤（对齐聚宽版逻辑）

    数据获取依据 ptrade_API.json 第69页：
    单支股票(字符串)+多字段 → 返回 DataFrame，行索引为日期，列索引为字段名
    日线返回字段含 high_limit(涨停价)/low_limit(跌停价)，见 API 文档第68页
    """
    if not stocks:
        return []

    hold_securities = get_hold_securities(context)
    valid_stocks = []
    excluded_info = {'price': 0, 'limit_up': 0, 'limit_down': 0}

    log.info("价格与涨跌停过滤开始，候选股票：%s" % stocks)

    for stock in stocks:
        try:
            hist = get_history(1, '1d', ['close', 'high_limit', 'low_limit'], stock)
            if hist is None or hist.empty:
                log.warning("股票 %s 无历史数据" % stock)
                excluded_info['price'] += 1
                continue

            last_close = float(hist['close'].iloc[-1])
            high_limit = float(hist['high_limit'].iloc[-1])
            low_limit = float(hist['low_limit'].iloc[-1])

            if last_close <= 0:
                log.warning("股票 %s 价格数据异常：%.2f" % (stock, last_close))
                excluded_info['price'] += 1
                continue

            is_holding = stock in hold_securities

            if not is_holding:
                if high_limit > 0 and last_close >= high_limit * 0.998:
                    log.info("股票 %s 涨停剔除：现价=%.2f，涨停价=%.2f" % (stock, last_close, high_limit))
                    excluded_info['limit_up'] += 1
                    continue

                if low_limit > 0 and last_close <= low_limit * 1.002:
                    log.info("股票 %s 跌停剔除：现价=%.2f，跌停价=%.2f" % (stock, last_close, low_limit))
                    excluded_info['limit_down'] += 1
                    continue

            if is_holding:
                log.info("股票 %s 为持仓股，跳过价格限制，现价=%.2f" % (stock, last_close))
                valid_stocks.append(stock)
            else:
                if last_close <= g.highest:
                    valid_stocks.append(stock)
                else:
                    log.info("股票 %s 价格超限：%.2f > %.2f" % (stock, last_close, g.highest))
                    excluded_info['price'] += 1
        except Exception as e:
            log.warning("价格过滤股票 %s 异常：%s" % (stock, str(e)))
            continue

    log.info("价格与涨跌停过滤结果：有效 %d 只 | 剔除：价格超限=%d, 涨停=%d, 跌停=%d" % (
        len(valid_stocks), excluded_info['price'], excluded_info['limit_up'], excluded_info['limit_down']))

    if g.filter_audit and valid_stocks:
        log.warning("审计意见过滤已开启，但 ptrade 平台不支持 audit_report 表，跳过审计过滤")

    if not valid_stocks:
        log.info("无股票入选，将转持银华日利ETF")
        return []

    return valid_stocks


def adjust_position_count(context):
    try:
        hist = get_history(10, '1d', ['close'], BENCHMARK_INDEX)
        if hist is None or hist.empty:
            log.warning("获取指数数据失败，使用默认持仓数量：%d" % g.stock_num)
            return g.stock_num

        close_series = hist['close'].get(BENCHMARK_INDEX, hist.iloc[:, 0])
        close = float(close_series.iloc[-1])
        ma10 = float(close_series.mean())

        if close == 0 or ma10 == 0:
            return g.stock_num

        diff = close - ma10
        log.info("指数 %s 当前价：%.2f，10日均线：%.2f，差值：%.2f" % (BENCHMARK_INDEX, close, ma10, diff))

        new_num = g.stock_num
        for upper, lower, num in POSITION_MAPPING:
            if diff < upper and diff >= lower:
                new_num = num
                break

        log.info("动态调整持仓数量：%d -> %d（差值：%.2f）" % (g.stock_num, new_num, diff))
        g.stock_num = new_num
        return new_num
    except Exception as e:
        log.error("动态调整持仓数量异常：%s" % str(e))
        return g.stock_num


def select_target_stocks(context):
    log.info("=" * 40)
    log.info("开始执行选股流程")
    log.info("=" * 40)

    basic_pool = filter_basic_stock_pool(context)
    if not basic_pool:
        log.info("基础股票池为空，无候选股票")
        return []

    finance_pool = filter_finance_and_market_value(context, basic_pool)
    if not finance_pool:
        log.info("财务筛选后无候选股票")
        return []

    price_pool = filter_price_and_audit(context, finance_pool)
    if not price_pool:
        log.info("价格/审计过滤后无候选股票")
        return []

    target_count = adjust_position_count(context)
    target_stocks = price_pool[:target_count]

    log.info("目标持仓：%s" % target_stocks)
    log.info("目标数量：%d (候选池 %d 只)" % (len(target_stocks), len(price_pool)))
    log.info("=" * 40)

    g.current_candidates = price_pool
    return target_stocks


def detect_yesterday_limit_up_stocks(context):
    """检测昨日涨停的持仓股（用精确涨停价 high_limit）"""
    limit_up_stocks = []
    positions = get_hold_positions(context)
    for pos in positions:
        security = get_pos_security(pos)
        try:
            hist = get_history(2, '1d', ['close', 'high_limit'], security)
            if hist is None or len(hist) < 2:
                continue
            yesterday_close = float(hist['close'].iloc[-2])
            yesterday_high_limit = float(hist['high_limit'].iloc[-2])
            if yesterday_high_limit > 0 and yesterday_close >= yesterday_high_limit * 0.998:
                limit_up_stocks.append(security)
        except Exception:
            continue
    if limit_up_stocks:
        log.info("检测到昨日涨停持仓：%s" % limit_up_stocks)
    return limit_up_stocks


def is_yesterday_limit_up(security):
    return security in g.yesterday_limit_up_stocks


def execute_rebalance_sell(context, target_securities):
    positions = get_hold_positions(context)
    stocks_to_sell = []

    for pos in positions:
        security = get_pos_security(pos)
        if security not in target_securities:
            if is_yesterday_limit_up(security):
                log.info("持仓股 %s 昨日涨停，暂不卖出，留待尾盘观察" % security)
            else:
                stocks_to_sell.append(security)

    for security in stocks_to_sell:
        try:
            pos = get_position(security)
            if pos and pos.amount > 0:
                order(security, -pos.amount)
                log.info("调仓卖出：%s，数量：%d" % (security, pos.amount))
        except Exception as e:
            log.error("卖出 %s 失败：%s" % (security, str(e)))


def execute_rebalance_buy(context, target_securities, exclude_stocks=None):
    if exclude_stocks is None:
        exclude_stocks = []

    positions = get_hold_positions(context)
    hold_securities = [get_pos_security(p) for p in positions]

    to_buy = [s for s in target_securities if s not in hold_securities and s not in exclude_stocks]
    if not to_buy:
        return

    current_hold_count = len(hold_securities)
    target_count = min(len(target_securities), g.stock_num)
    can_buy_count = target_count - current_hold_count
    if can_buy_count <= 0:
        return

    to_buy = to_buy[:can_buy_count]
    available_cash = context.portfolio.cash
    if available_cash <= 0:
        log.info("可用资金为0，无法买入")
        return

    cash_per_stock = available_cash / len(to_buy)
    log.info("调仓买入：目标 %d 只，每只分配资金 %.2f" % (len(to_buy), cash_per_stock))

    for security in to_buy:
        try:
            order_value(security, cash_per_stock)
            log.info("买入：%s，分配资金：%.2f" % (security, cash_per_stock))
        except Exception as e:
            log.error("买入 %s 失败：%s" % (security, str(e)))


def weekly_rebalance(context):
    weekday = context.current_dt.weekday()
    if weekday != REBALANCE_WEEKDAY:
        return

    if is_empty_month(context):
        log.info("当前为空仓月份（%d月），不执行股票调仓" % context.current_dt.month)
        handle_empty_month_etf(context)
        return

    log.info("开始执行每周调仓（周二 10:00）")

    etf_sold = sell_etf_safe(context)
    if etf_sold:
        log.info("已释放ETF持仓资金，继续执行股票调仓")

    g.current_candidates = []
    g.limit_up_opened_stocks = []

    target_stocks = select_target_stocks(context)

    if not target_stocks:
        log.info("无目标股票，全部资金买入银华日利ETF")
        buy_etf_safe(context)
        g.target_positions = []
        return

    g.target_positions = target_stocks

    execute_rebalance_sell(context, target_stocks)

    execute_rebalance_buy(context, target_stocks)

    g.last_rebalance_date = context.current_dt.strftime('%Y-%m-%d')
    log.info("调仓完成，目标持仓：%s" % target_stocks)


def tail_limit_up_check(context):
    if is_empty_month(context):
        return

    positions = get_hold_positions(context)
    if not positions:
        return

    stocks_to_sell = []

    for pos in positions:
        security = get_pos_security(pos)
        if not is_yesterday_limit_up(security):
            continue

        try:
            hist = get_history(1, '1d', ['close', 'high_limit'], security)
            if hist is None or hist.empty:
                continue
            current_price = float(hist['close'].iloc[-1])
            high_limit = float(hist['high_limit'].iloc[-1])

            if high_limit > 0 and current_price < high_limit * 0.998:
                log.info("涨停打开：%s（现价=%.2f，涨停价=%.2f），执行卖出" % (security, current_price, high_limit))
                if pos.amount > 0:
                    order(security, -pos.amount)
                    log.info("尾盘卖出（涨停打开）：%s，数量：%d" % (security, pos.amount))
                    stocks_to_sell.append(security)
            else:
                log.info("涨停继续持有：%s（现价=%.2f，涨停价=%.2f）" % (security, current_price, high_limit))
        except Exception as e:
            log.warning("检查 %s 涨停打开状态异常：%s" % (security, str(e)))
            continue

    if stocks_to_sell:
        g.limit_up_opened_stocks.extend(stocks_to_sell)
        try_replenish_positions(context)


def try_replenish_positions(context):
    positions = get_hold_positions(context)
    hold_securities = [get_pos_security(p) for p in positions]
    current_count = len(hold_securities)

    if current_count >= g.stock_num:
        return

    available_cash = context.portfolio.cash
    if available_cash <= 0:
        return

    if not g.current_candidates:
        log.info("无候选池数据，无法补仓")
        return

    buyable = [s for s in g.current_candidates if s not in hold_securities and s not in g.limit_up_opened_stocks]
    need_count = g.stock_num - current_count
    buyable = buyable[:need_count]

    if not buyable:
        return

    cash_per_stock = available_cash / len(buyable)
    log.info("补仓：目标 %d 只，每只分配资金 %.2f" % (len(buyable), cash_per_stock))

    for security in buyable:
        try:
            order_value(security, cash_per_stock)
            log.info("补仓买入：%s，分配资金：%.2f" % (security, cash_per_stock))
        except Exception as e:
            log.error("补仓买入 %s 失败：%s" % (security, str(e)))


def check_stop_profit_loss(context):
    """个股止盈止损检查

    价格获取依据 ptrade_API.json（Position 对象属性章节）：
    - cost_basis: 持仓成本价格
    - last_sale_price: 最新价格（等价于聚宽 position.price）
    日线 get_history(1,'1d') 默认 include=False 不含当前周期，回测中取到的是
    昨日收盘价，会导致止损滞后一天/在反弹日误触发。改用持仓对象实时价。
    """
    positions = get_hold_positions(context)
    if not positions:
        return

    log.info("执行个股止盈止损检查")
    stopped_out = False

    for pos in positions:
        security = get_pos_security(pos)
        try:
            avg_cost = get_position_cost(security)
            if avg_cost <= 0:
                continue
            current_amount = pos.amount
            if current_amount <= 0:
                continue

            current_price = float(getattr(pos, 'last_sale_price', 0) or 0)
            if current_price <= 0:
                log.warning("持仓 %s 取 last_sale_price 失败，降级用昨收价" % security)
                hist = get_history(1, '1d', ['close'], security)
                if hist is None or hist.empty:
                    continue
                current_price = float(hist['close'].iloc[-1])
            if current_price <= 0:
                continue

            profit_ratio = (current_price - avg_cost) / avg_cost

            if profit_ratio >= STOP_PROFIT_RATIO - 1:
                order(security, -current_amount)
                log.info("止盈卖出：%s，成本=%.2f，现价=%.2f，收益率=%.2f%%" % (
                    security, avg_cost, current_price, profit_ratio * 100))
                stopped_out = True
            elif profit_ratio <= -STOP_LOSS_RATIO:
                order(security, -current_amount)
                log.info("止损卖出：%s，成本=%.2f，现价=%.2f，收益率=%.2f%%，原因=stoploss" % (
                    security, avg_cost, current_price, profit_ratio * 100))
                stopped_out = True
        except Exception as e:
            log.error("检查 %s 止盈止损异常：%s" % (security, str(e)))

    if stopped_out:
        g.stopped_out = True


def check_market_stop_loss(context):
    """市场趋势止损（对齐聚宽：成分股平均日内涨跌幅 abs(mean) >= 5% 就清仓）

    逻辑对齐聚宽原版 sell_stocks 函数：
    1. 取深证综指全部成分股的昨日 close 和 open
    2. 计算每只股票的日内涨跌幅 (close/open - 1)
    3. 求平均涨跌幅（正负会抵消）
    4. abs(mean) >= 5% 则清仓（单边行情触发）

    实现方式：分批获取 + 累加（方案B）
    依据 ptrade_API.json 第70页：多股票(list)+单字段 → DataFrame
    避开多股票+多字段的 Panel 兼容问题
    """
    try:
        current_date = context.current_dt.strftime('%Y%m%d')
        stocks = get_index_stocks(BENCHMARK_INDEX, current_date)
        if not stocks or len(stocks) == 0:
            log.warning("获取成分股失败，跳过市场止损检查")
            return False

        total_sum = 0.0
        total_count = 0
        batch_fail_count = 0
        batch_total = (len(stocks) + MARKET_STOP_LOSS_BATCH_SIZE - 1) // MARKET_STOP_LOSS_BATCH_SIZE

        for batch_idx in range(batch_total):
            batch_start = batch_idx * MARKET_STOP_LOSS_BATCH_SIZE
            batch_stocks = stocks[batch_start:batch_start + MARKET_STOP_LOSS_BATCH_SIZE]

            try:
                close_df = get_history(1, '1d', 'close', batch_stocks)
                open_df = get_history(1, '1d', 'open', batch_stocks)

                if close_df is None or open_df is None:
                    batch_fail_count += 1
                    continue
                if close_df.empty or open_df.empty:
                    batch_fail_count += 1
                    continue

                # ptrade 长格式 → 以股票代码为索引的 Series
                if 'code' in close_df.columns and 'close' in close_df.columns:
                    last_close = close_df.set_index('code')['close']
                else:
                    last_close = close_df.iloc[-1]
                if 'code' in open_df.columns and 'open' in open_df.columns:
                    last_open = open_df.set_index('code')['open']
                else:
                    last_open = open_df.iloc[-1]

                last_close = pd.to_numeric(last_close, errors='coerce')
                last_open = pd.to_numeric(last_open, errors='coerce')

                valid_mask = (last_open > 0) & last_close.notna() & last_open.notna()
                batch_returns = last_close[valid_mask] / last_open[valid_mask] - 1

                total_sum += batch_returns.sum()
                total_count += len(batch_returns)

            except Exception as batch_e:
                batch_fail_count += 1
                log.warning("市场止损第 %d/%d 批获取失败：%s" % (batch_idx + 1, batch_total, str(batch_e)))
                continue

        if batch_fail_count > batch_total / 2:
            log.warning("市场止损获取失败批次过多（%d/%d），跳过本次检查" % (batch_fail_count, batch_total))
            return False

        if total_count == 0:
            log.warning("无有效成分股数据，跳过市场止损检查")
            return False

        avg_direction = abs(total_sum / total_count)

        log.info("市场趋势检查：%s 成分股平均涨跌幅 = %.4f%%，abs后 = %.4f%%（有效 %d 只，失败 %d/%d 批）" % (
            BENCHMARK_INDEX, (total_sum / total_count) * 100, avg_direction * 100,
            total_count, batch_fail_count, batch_total))

        if avg_direction >= MARKET_STOP_LOSS_THRESHOLD:
            log.info("市场趋势止损触发！成分股 abs(平均涨跌幅) %.4f%% >= %.2f%%，清空所有持仓" % (
                avg_direction * 100, MARKET_STOP_LOSS_THRESHOLD * 100))

            positions = get_hold_positions(context)
            for pos in positions:
                try:
                    security = get_pos_security(pos)
                    sell_amount = pos.amount
                    if sell_amount > 0:
                        order(security, -sell_amount)
                        log.info("市场止损卖出：%s，数量：%d，原因=stoploss" % (security, sell_amount))
                except Exception as e:
                    log.error("市场止损卖出 %s 失败：%s" % (security, str(e)))
            g.stopped_out = True
            return True

        return False
    except Exception as e:
        log.error("市场趋势止损检查异常：%s" % str(e))
        return False


def risk_check(context):
    if is_empty_month(context):
        log.info("当前为空仓月份，跳过风控检查")
        return

    log.info("执行每日风控检查（10:00）")

    check_stop_profit_loss(context)

    check_market_stop_loss(context)


def sell_etf_safe(context):
    try:
        pos = get_position(ETF_SAFE)
        if pos and pos.amount > 0:
            sell_amount = pos.amount
            order(ETF_SAFE, -sell_amount)
            log.info("调仓卖出ETF：%s，数量：%d" % (ETF_SAFE, sell_amount))
            return True
        return False
    except Exception as e:
        log.error("卖出ETF失败：%s" % str(e))
        return False


def buy_etf_safe(context):
    try:
        available_cash = context.portfolio.cash
        if available_cash > 0:
            order_value(ETF_SAFE, available_cash)
            log.info("买入银华日利ETF：%.2f元" % available_cash)
        else:
            log.info("可用资金为0，无需买入银华日利ETF")
    except Exception as e:
        log.error("买入银华日利ETF失败：%s" % str(e))


def handle_empty_month_clear(context):
    if not is_empty_month(context):
        return

    month = context.current_dt.month
    log.info("空仓月份检查（%d月 14:50）：清理股票持仓" % month)

    positions = get_positions()
    for security, pos in positions.items():
        if security.startswith('511880'):
            continue
        if pos.amount > 0:
            try:
                order(security, -pos.amount)
                log.info("空仓月份清仓：%s，数量：%d" % (security, pos.amount))
            except Exception as e:
                log.error("空仓月份清仓 %s 失败：%s" % (security, str(e)))


def handle_empty_month_etf(context):
    if not is_empty_month(context):
        return

    month = context.current_dt.month
    weekday = context.current_dt.weekday()
    if weekday != REBALANCE_WEEKDAY:
        return

    log.info("空仓月份处理（%d月 周二）：买入银华日利ETF" % month)
    buy_etf_safe(context)


def handle_stop_loss_funds(context):
    if not g.stopped_out:
        return

    if g.stop_loss_etf_bought:
        return

    current_hour = context.current_dt.hour
    if current_hour < 14:
        return

    log.info("止损资金处理（14:00）：将剩余资金买入银华日利ETF，下次调仓时将自动卖出")
    buy_etf_safe(context)
    g.stop_loss_etf_bought = True
    g.stopped_out = False


def handle_data(context, data):
    handle_stop_loss_funds(context)