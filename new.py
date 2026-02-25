import requests
import pandas as pd
import numpy as np
import time
import akshare as ak
import argparse
from datetime import datetime, timedelta
import re
import os

# ================== 配置区域 ==================
ITICK_TOKEN = "6e22921dceb0492ea60d21c43c4833a2c00794ec321e49f498340d728645ae2c"          # 请替换为你的 iTick Token
ITICK_BASE_URL = "https://api.itick.org"  # iTick API基础地址（以官方文档为准）
REQUEST_INTERVAL = 10                # 请求间隔（秒），建议调大以避免触发 iTick 频率限制（429）

# 筛选参数
MIN_PRICE = 5.0                      # 最低股价（人民币）
MIN_VOLUME_AMOUNT = 30_000_000       # 最小成交额（3000万人民币）
LOOKBACK_DAYS = 60                    # 获取历史K线的天数

# K线缓存（默认关闭；可用参数开启）
KLINE_CACHE_DIR = "kline_cache"
# =============================================


def _today_yyyymmdd() -> str:
    return datetime.now().strftime("%Y%m%d")


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _kline_cache_path(cache_dir: str, cache_date: str, symbol: str) -> str:
    safe_symbol = str(symbol).strip().upper().replace("/", "_")
    d = os.path.join(cache_dir, cache_date)
    return os.path.join(d, f"{safe_symbol}.csv")


def load_kline_cache(cache_dir: str, cache_date: str, symbol: str) -> pd.DataFrame:
    path = _kline_cache_path(cache_dir, cache_date, symbol)
    if not os.path.exists(path):
        return pd.DataFrame()

    try:
        df = pd.read_csv(path)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df.dropna(subset=["date"], inplace=True)
        df.sort_values("date", inplace=True)
        return df
    except Exception as e:
        print(f"读取K线缓存失败 {symbol} ({path}):", e)
        return pd.DataFrame()


def save_kline_cache(df: pd.DataFrame, cache_dir: str, cache_date: str, symbol: str):
    if df is None or df.empty:
        return

    path = _kline_cache_path(cache_dir, cache_date, symbol)
    _ensure_dir(os.path.dirname(path))

    try:
        df_to_save = df.copy()
        # 统一为 ISO 字符串，避免跨平台/版本读写差异
        if "date" in df_to_save.columns:
            df_to_save["date"] = pd.to_datetime(df_to_save["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        df_to_save.to_csv(path, index=False, encoding="utf-8")
    except Exception as e:
        print(f"保存K线缓存失败 {symbol} ({path}):", e)


def fetch_kline_with_cache(symbol, token, days, cache_enabled: bool, cache_dir: str, cache_date: str, offline: bool = False):
    """带缓存的K线获取。

    - cache_enabled=True：如果当天缓存存在就直接用；否则请求 iTick 并写入缓存
    - offline=True：只读缓存，不请求 iTick（用于专门调策略/回测当天缓存）

    返回：(df, from_cache: bool)
    """

    if cache_enabled:
        cached = load_kline_cache(cache_dir=cache_dir, cache_date=cache_date, symbol=symbol)
        if not cached.empty:
            return cached.tail(int(days)), True

        if offline:
            return pd.DataFrame(), True

    df = fetch_kline(symbol, token, days=days)
    if cache_enabled and df is not None and not df.empty:
        save_kline_cache(df, cache_dir=cache_dir, cache_date=cache_date, symbol=symbol)

    return df, False


def _parse_itick_symbol(symbol: str):
    """把你脚本里的 symbol（如 SH688041）拆成 iTick 接口需要的 region+code。

    iTick 文档：
    - /stock/quote: query 需要 region + code，token 放在 header
    - /stock/kline: query 需要 region + code + kType (+ limit)
    """
    if not symbol:
        raise ValueError("symbol 为空")

    s = str(symbol).strip().upper()

    m = re.match(r"^(?P<region>[A-Z]{2})(?P<code>\d+)$", s)
    if m:
        return m.group("region"), m.group("code")

    # 兼容 688041.SH / 700.HK 这种写法
    if "." in s:
        code, region = s.split(".", 1)
        if region and code:
            return region.strip().upper(), code.strip()

    raise ValueError(f"无法解析 symbol={symbol}，期望类似 SH688041 或 688041.SH")


# --- 备用股票池（科创50）：暂时不用，但保留代码方便随时切回 ---
# def get_kc50_components():
#     """
#     获取科创50成分股列表
#     返回：list of str, 例如 ['SH688981', 'SH688111', ...]
#     """
#     components = []
#     try:
#         # 科创50指数代码 000688
#         df = ak.index_stock_cons_csindex("000688")
#         codes = df['成分券代码'].astype(str).str.zfill(6).tolist()
#         for code in codes:
#             # 科创50均为沪市（688开头）
#             components.append(f"SH{code}")
#         print(f"获取到科创50成分股 {len(components)} 只")
#     except Exception as e:
#         print("获取科创50成分股失败:", e)
#         # 备用方案：如果akshare失败，可以手动指定一个列表（例如从文件读取）
#         # 这里简单返回空列表
#         return []
#     return components


def _infer_cn_region_by_code(code6: str) -> str:
    """根据 6 位股票代码推断交易所（用于拼 iTick symbol：SH/SZ/BJ）。

    说明：AkShare 的 `stock_zh_a_spot_em()` 覆盖“沪深京 A 股”。这里用经验规则做映射，
    足以覆盖绝大多数情况。
    """

    c = str(code6).strip().zfill(6)

    # 沪市：60/68/69
    if c.startswith(("60", "68", "69")):
        return "SH"

    # 深市：00/30/20（20 通常为B股，但 spot_em 通常是A股；这里仍归为 SZ 兜底）
    if c.startswith(("00", "30", "20")):
        return "SZ"

    # 北交所：常见 43/83/87/88/92 或首位为 4/8/9
    if c.startswith(("43", "83", "87", "88", "92")) or c[0] in {"4", "8", "9"}:
        return "BJ"

    # 兜底：深市
    return "SZ"


def get_a_share_universe_step1(min_price: float = MIN_PRICE, min_amount: float = MIN_VOLUME_AMOUNT):
    """STEP 1：用 AkShare 全市场快照做初筛（价格/成交额），避免对全市场逐只打 iTick quote。

    返回：list[dict]
    - symbol: iTick 兼容的 symbol（如 SH600000）
    - name: 股票名称
    - price: 最新价
    - amount: 成交额
    """

    df = ak.stock_zh_a_spot_em()

    # 兼容不同 AkShare 版本字段
    code_col = next((c for c in ["代码", "code", "证券代码"] if c in df.columns), None)
    name_col = next((c for c in ["名称", "name", "证券简称"] if c in df.columns), None)
    price_col = next((c for c in ["最新价", "最新", "price"] if c in df.columns), None)
    amount_col = next((c for c in ["成交额", "amount", "turnover"] if c in df.columns), None)

    if not all([code_col, name_col, price_col, amount_col]):
        raise RuntimeError(
            f"AkShare 行情字段缺失：code={code_col}, name={name_col}, price={price_col}, amount={amount_col}"
        )

    base = df[[code_col, name_col, price_col, amount_col]].copy()
    base[price_col] = pd.to_numeric(base[price_col], errors="coerce")
    base[amount_col] = pd.to_numeric(base[amount_col], errors="coerce")
    base.dropna(subset=[price_col, amount_col], inplace=True)

    before = len(base)
    base = base[(base[price_col] >= float(min_price)) & (base[amount_col] >= float(min_amount))]
    after = len(base)

    print(
        f"STEP 1 初筛（全市场）：{before} -> {after}（价格>= {min_price}, 成交额>= {min_amount:,}）"
    )

    universe = []
    for _, row in base.iterrows():
        code6 = str(row[code_col]).strip().zfill(6)
        region = _infer_cn_region_by_code(code6)
        symbol = f"{region}{code6}"
        universe.append(
            {
                "symbol": symbol,
                "name": str(row[name_col]).strip(),
                "price": float(row[price_col]),
                "amount": float(row[amount_col]),
            }
        )

    return universe


def filter_step1_by_quote(symbol, token):
    """根据实时行情进行初筛（单只股票）"""
    url = f"{ITICK_BASE_URL}/stock/quote"

    try:
        region, code = _parse_itick_symbol(symbol)

        # 按 iTick 文档：token 放在 header，query 用 region + code
        headers = {
            "accept": "application/json",
            "token": token,
        }
        params = {
            "region": region,
            "code": code,
        }

        resp = requests.get(url, params=params, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        if data.get("code") == 0 and data.get("data"):
            stock = data["data"]

            # 文档字段：ld=latest price, tu=trading amount
            latest_price = stock.get("ld", 0) or 0
            amount = stock.get("tu", 0) or 0

            # 股价 > MIN_PRICE
            if latest_price < MIN_PRICE:
                return None
            # 成交额 > MIN_VOLUME_AMOUNT
            if amount < MIN_VOLUME_AMOUNT:
                return None

            # /stock/quote 文档响应里没有 name，这里保持为空即可
            return {
                "symbol": symbol,
                "name": "",
                "price": latest_price,
                "amount": amount,
                "raw": stock,
            }

        print(f"获取{symbol}行情失败:", data.get("msg"))
        return None

    except Exception as e:
        print(f"请求{symbol}行情异常:", e)
        return None


def fetch_kline(symbol, token, days=LOOKBACK_DAYS):
    """获取单个股票的日线K线数据"""
    url = f"{ITICK_BASE_URL}/stock/kline"

    try:
        region, code = _parse_itick_symbol(symbol)

        headers = {
            "accept": "application/json",
            "token": token,
        }

        # 按 iTick 文档：kType=8 为日线；用 limit 控制条数
        params = {
            "region": region,
            "code": code,
            "kType": 8,
            "limit": int(days),
        }

        resp = requests.get(url, params=params, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        if data.get("code") == 0:
            klines = data.get("data") or []
            df = pd.DataFrame(klines)
            if df.empty:
                return df

            # 文档字段：t=timestamp(ms), c=close, o=open, h=high, l=low, v=volume, tu=amount
            df.rename(
                columns={
                    "t": "date",
                    "c": "close",
                    "o": "open",
                    "h": "high",
                    "l": "low",
                    "v": "volume",
                    "tu": "amount",
                },
                inplace=True,
            )
            df["date"] = pd.to_datetime(df["date"], unit="ms")
            df.sort_values("date", inplace=True)
            df = df.tail(LOOKBACK_DAYS)
            return df

        print(f"获取{symbol}K线失败:", data.get("msg"))
        return pd.DataFrame()

    except Exception as e:
        print(f"请求{symbol}K线异常:", e)
        return pd.DataFrame()


def calculate_indicators(df):
    """计算技术指标：均线、涨幅、成交量倍数等"""
    if df.empty or len(df) < 50:
        return None

    df['MA5'] = df['close'].rolling(window=5).mean()
    df['MA20'] = df['close'].rolling(window=20).mean()
    df['MA50'] = df['close'].rolling(window=50).mean()

    df['pct_change_5'] = df['close'].pct_change(periods=5) * 100
    df['pct_change_20'] = df['close'].pct_change(periods=20) * 100

    df['vol_ma20'] = df['volume'].rolling(window=20).mean()
    df['vol_ratio'] = df['volume'] / df['vol_ma20']

    return df


def check_conditions(df):
    """右侧趋势股评分模型（100分制）
    
    总分 = 趋势强度(40) + 启动质量(30) + 资金确认(20) + 安全边际(10)
    """
    if df is None or len(df) < 50:
        return None

    latest = df.iloc[-1]
    prev = df.iloc[-2]
    
    # ================== 一、趋势强度评分（40分） ==================
    trend_score = 0
    
    # ① 均线多头结构（15分）
    if latest['MA5'] > latest['MA20'] > latest['MA50']:
        trend_score += 15
    elif latest['MA20'] > latest['MA50']:
        trend_score += 10
    
    # ② MA50斜率（10分）
    if len(df) >= 2:
        ma50_yesterday = df['MA50'].iloc[-2]
        if latest['MA50'] > ma50_yesterday:
            trend_score += 10
    
    # ③ 股价位置（15分）
    if latest['close'] > latest['MA20']:
        trend_score += 5
    if latest['close'] > latest['MA50']:
        trend_score += 10
    
    # ================== 二、启动质量评分（30分） ==================
    launch_score = 0
    
    # ④ 首次启动强度（15分）
    # 检查昨日收盘是否<20MA且今日收盘>5MA（回踩20MA后启动）
    if len(df) >= 2:
        yesterday_close = df['close'].iloc[-2]
        yesterday_ma20 = df['MA20'].iloc[-2]
        
        # 调试信息：打印所有股票的启动条件检查结果
        cond1 = yesterday_close < yesterday_ma20
        cond2 = latest['close'] > latest['MA5']
        
        # 计算今日涨幅（在所有路径中都需要）
        today_pct_change = (latest['close'] - df['close'].iloc[-2]) / df['close'].iloc[-2] * 100
        
        if cond1 and cond2:
            launch_score += 15
            # 调试：打印满足条件的股票
            print(f"DEBUG 启动条件满足: 昨日收盘={yesterday_close:.2f} < 昨日MA20={yesterday_ma20:.2f}, 今日收盘={latest['close']:.2f} > 今日MA5={latest['MA5']:.2f}")
            
            # ⑤ 启动K线强度（15分）- 只有在满足回踩条件时才能获得
            if today_pct_change >= 5:
                launch_score += 15
            elif today_pct_change >= 3:
                launch_score += 10
            elif today_pct_change >= 1:
                launch_score += 5
        else:
            # 不满足回踩条件，启动强度为0分
            launch_score += 0
            # 调试：打印不满足条件的详细信息
            print(f"DEBUG 启动条件不满足: 昨日收盘={yesterday_close:.2f} < 昨日MA20={yesterday_ma20:.2f}? {cond1}, 今日收盘={latest['close']:.2f} > 今日MA5={latest['MA5']:.2f}? {cond2}")
            
            # 不满足回踩条件时，启动K线强度也为0分
            launch_score += 0
    
    # ================== 三、资金确认评分（20分） ==================
    capital_score = 0
    
    # ⑥ 成交量放大（10分）
    if 'vol_ma20' in latest and not pd.isna(latest['vol_ma20']):
        vol_ratio = latest['volume'] / latest['vol_ma20'] if latest['vol_ma20'] > 0 else 1.0
        if vol_ratio >= 1.5:
            capital_score += 10
        elif vol_ratio >= 1.2:
            capital_score += 7
        elif vol_ratio >= 1.0:
            capital_score += 5
    
    # ⑦ 量价配合（10分）
    if today_pct_change > 0 and capital_score > 0:  # 上涨且放量
        capital_score += 10
    elif today_pct_change > 0:  # 上涨但缩量
        capital_score += 3
    
    # ================== 四、安全边际评分（10分） ==================
    safety_score = 0
    
    # ⑧ 是否追高（10分）
    last_10 = df.tail(10)
    llv_low_10 = last_10['low'].min()
    if llv_low_10 > 0:
        price_ratio = latest['close'] / llv_low_10
        if price_ratio <= 1.15:
            safety_score += 10
        elif price_ratio <= 1.25:
            safety_score += 5
    
    # ================== 总分计算 ==================
    total_score = trend_score + launch_score + capital_score + safety_score
    
    # ================== 级别判定 ==================
    level = "垃圾信号"
    if total_score >= 80:
        level = "主升候选"
    elif total_score >= 70:
        level = "强势观察"
    elif total_score >= 60:
        level = "潜力股"
    
    # 基础条件检查（必须满足趋势底座）
    base_conditions_passed = (
        latest['MA20'] > latest['MA50'] and  # 条件1：MA20 > MA50
        latest['close'] > latest['MA50'] and  # 条件3：CLOSE > MA50
        latest['close'] > latest['MA5']  # 条件5简化：今日收盘>5MA
    )
    
    # 计算技术指标用于展示
    dist_to_20ma = (latest['close'] - latest['MA20']) / latest['MA20'] * 100
    recent_high = df['high'].tail(20).max()
    drawdown = (recent_high - latest['close']) / recent_high * 100
    
    return {
        "passed": base_conditions_passed and total_score >= 60 and launch_score > 0,  # 至少潜力股级别且启动质量分>0才通过
        "score": total_score,
        "level": level,
        "trend_score": trend_score,
        "launch_score": launch_score,
        "capital_score": capital_score,
        "safety_score": safety_score,
        "dist_to_20ma": dist_to_20ma,
        "drawdown": drawdown,
        "latest_price": latest['close'],
        "signal_desc": f"评分{total_score}分({level}) - 趋势{trend_score}/启动{launch_score}/资金{capital_score}/安全{safety_score}",
        "name": None
    }


def _find_last_cross_up(df: pd.DataFrame, ma_col: str, lookback: int = 30):
    """在最近 lookback 天内，寻找“收盘价上穿指定均线”的最后一次发生位置。"""
    if df is None or df.empty or len(df) < 3:
        return None

    sub = df.tail(lookback + 1).copy()
    prev_close = sub['close'].shift(1)
    prev_ma = sub[ma_col].shift(1)
    cross = (prev_close <= prev_ma) & (sub['close'] > sub[ma_col])
    idxs = sub.index[cross.fillna(False)].tolist()
    return idxs[-1] if idxs else None


def _check_pullback_to_ma_stable(
    df: pd.DataFrame,
    ma_col: str,
    touch_lookback: int = 10,
    touch_tolerance: float = 0.01,
    reclaim_lookback: int = 3,  # 新增：允许站上5日线发生在过去几天内
    reclaim_tolerance: float = 0.02,  # 新增：允许收盘价稍微低于5日线但很接近
):
    """回踩某条均线，并在回踩后的某一天重新站上 5 日线（放宽条件版本）。

    修改口径：
    - 回踩：过去 touch_lookback 天内出现 low 触及该均线附近
    - 重新站上 5MA：从回踩日的"第二天"开始，寻找第一天满足 close > 当天 MA5 * (1 - reclaim_tolerance)
    - 输出时机：允许"站上5日线"发生在过去 reclaim_lookback 天内（不只是当天）
    - 放宽要求：允许收盘价稍微低于5日线但很接近（容忍 reclaim_tolerance）
    """

    if df is None or df.empty or len(df) < 60:
        return None

    today_idx = df.index[-1]

    # 仅看最近 touch_lookback 天内的回踩（不含今天）
    recent = df.tail(touch_lookback + 1).copy()
    recent_ex_today = recent.iloc[:-1].copy()
    if recent_ex_today.empty:
        return None

    # 回踩定义：low 触到均线附近（<= 1% 以内）
    touch = recent_ex_today['low'] <= recent_ex_today[ma_col] * (1 + float(touch_tolerance))
    touch_idxs = recent_ex_today.index[touch.fillna(False)].tolist()
    if not touch_idxs:
        return None

    # 从"最近一次回踩"开始往前找
    for touch_idx in reversed(touch_idxs):
        start_pos = df.index.get_loc(touch_idx)

        # 至少需要"回踩日 + 次日"两根K线
        if start_pos + 1 >= len(df):
            continue

        window_after_touch = df.iloc[start_pos + 1:].copy()
        
        # 放宽条件：允许收盘价稍微低于5日线但很接近
        reclaim = window_after_touch['close'] > window_after_touch['MA5'] * (1 - float(reclaim_tolerance))
        reclaim_idxs = window_after_touch.index[reclaim.fillna(False)].tolist()
        if not reclaim_idxs:
            continue

        reclaim_idx = reclaim_idxs[0]
        
        # 放宽输出时机：允许站上5日线发生在过去 reclaim_lookback 天内
        reclaim_pos = df.index.get_loc(reclaim_idx)
        today_pos = df.index.get_loc(today_idx)
        
        if reclaim_pos > today_pos:
            continue
            
        # 检查是否在允许的时间范围内
        days_ago = today_pos - reclaim_pos
        if days_ago > reclaim_lookback:
            continue

        return {
            "touch_date": df.loc[touch_idx, 'date'] if 'date' in df.columns else None,
            "reclaim_5ma_date": df.loc[reclaim_idx, 'date'] if 'date' in df.columns else None,
            "days_since_reclaim": days_ago,
            "ma_col": ma_col,
            "reclaim_tolerance_used": reclaim_tolerance,
        }

    return None


def check_pullback_strategy(df):
    """新增策略：回踩20/50MA→收盘重新站上5MA。"""
    if df is None or len(df) < 60:
        return None

    latest = df.iloc[-1]

    # 趋势底座（更宽松）：均线多头 + 50MA 走平/上行
    # 注意：回踩阶段最新收盘可能低于 20MA，因此这里不强制 close > MA20
    cond2_2 = latest['MA20'] > latest['MA50']
    ma50_5d_ago = df['MA50'].iloc[-6] if len(df) >= 6 else np.nan
    cond2_3 = latest['MA50'] > ma50_5d_ago if not pd.isna(ma50_5d_ago) else False
    trend_ok = cond2_2 and cond2_3
    if not trend_ok:
        return {
            "passed": False,
            "score": 0.0,
            "pullback_desc": "未满足趋势底座（MA20>MA50 且 MA50 上行）",
            "pullback_line": "",
            "drawdown": 0.0,
            "latest_price": latest['close'],
        }

    # 回踩策略：回踩 MA20 / MA50 任意一条都算；且回踩发生在近 10 天
    # 重新站上 5MA：收盘 > 当天 5MA（不要求严格上穿）
    pb20 = _check_pullback_to_ma_stable(df, ma_col='MA20', touch_lookback=10)
    pb50 = _check_pullback_to_ma_stable(df, ma_col='MA50', touch_lookback=10)

    chosen = None
    pullback_line = ""
    if pb20:
        chosen = pb20
        pullback_line = "20MA"
    elif pb50:
        chosen = pb50
        pullback_line = "50MA"

    passed = chosen is not None

    # 仅用于展示：近 20 日高点回撤（不参与筛选）
    recent_high = df['high'].tail(20).max()
    drawdown = (recent_high - latest['close']) / recent_high * 100

    # 评分沿用原逻辑
    score = 0.0
    if passed:
        weight_5 = 0.4
        weight_20 = 0.3
        weight_vol = 0.3
        p5 = latest['pct_change_5'] if not pd.isna(latest['pct_change_5']) else 0
        p20 = latest['pct_change_20'] if not pd.isna(latest['pct_change_20']) else 0
        vol_ratio = latest['vol_ratio'] if not pd.isna(latest['vol_ratio']) else 1.0
        score = p5 * weight_5 + p20 * weight_20 + vol_ratio * weight_vol

    pullback_desc = ""
    if passed:
        pullback_desc = f"近10天回踩{pullback_line}，且今天为回踩后首次收盘>5MA（当日信号）"
    else:
        pullback_desc = "未满足回踩后首次收盘>5MA条件"

    dist_to_20ma = (latest['close'] - latest['MA20']) / latest['MA20'] * 100

    return {
        "passed": passed,
        "score": score,
        "dist_to_20ma": dist_to_20ma,
        "pullback_desc": pullback_desc,
        "pullback_line": pullback_line,
        "drawdown": drawdown,
        "latest_price": latest['close'],
        "name": None
    }


def main():
    parser = argparse.ArgumentParser(description="全市场右侧交易扫描（STEP1 行情初筛 + STEP2/3 形态）")
    parser.add_argument("--min-price", type=float, default=MIN_PRICE, help="STEP1 最低股价，默认 5")
    parser.add_argument("--min-amount", type=float, default=MIN_VOLUME_AMOUNT, help="STEP1 最低成交额，默认 3000万")
    parser.add_argument(
        "--max-stocks",
        type=int,
        default=600,
        help="最多扫描多少只股票（按成交额从高到低排序后截断）；0 表示不限制（不建议）",
    )
    parser.add_argument(
        "--request-interval",
        type=float,
        default=REQUEST_INTERVAL,
        help="iTick 请求间隔秒数（需遵守频率限制）",
    )
    parser.add_argument(
        "--cache-kline",
        action="store_true",
        help="开启K线缓存：同一天同一只票如果已请求过则直接复用（默认按今天日期分目录）",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="离线模式：只读缓存，不请求iTick（用于专门调策略）",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=KLINE_CACHE_DIR,
        help="K线缓存目录（默认 kline_cache）",
    )
    parser.add_argument(
        "--cache-date",
        type=str,
        default="",
        help="缓存日期目录（YYYYMMDD），默认空=今天；用于复用某天缓存",
    )
    parser.add_argument(
        "--asof-date",
        type=str,
        default="",
        help="按指定日期视角运行（YYYYMMDD）：会把K线截断到该日，使‘今天/最新K线’=该日",
    )
    args = parser.parse_args()

    if args.offline:
        args.cache_kline = True

    cache_date = args.cache_date.strip() if args.cache_date else _today_yyyymmdd()

    print(f"开始执行全市场右侧交易扫描 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if args.cache_kline:
        print(f"K线缓存：开启（dir={args.cache_dir}, date={cache_date}, offline={args.offline}）")

    # 1. STEP 1：全市场初筛（价格/成交额）
    try:
        universe = get_a_share_universe_step1(min_price=args.min_price, min_amount=args.min_amount)
    except Exception as e:
        print("获取全市场股票池失败:", e)
        return
    if not universe:
        print("STEP 1 初筛后股票池为空，程序终止")
        return

    # 默认按成交额降序，优先扫流动性更好的票
    universe.sort(key=lambda x: x.get("amount", 0), reverse=True)
    if args.max_stocks and args.max_stocks > 0:
        universe = universe[: args.max_stocks]
        print(f"实际进入扫描股票数：{len(universe)}（max_stocks={args.max_stocks}）")

    # 2. 对每只股票进行分析（STEP 4 仍停用，仅保留 STEP 2、STEP 3）
    candidates = []
    total = len(universe)
    for i, stock_info in enumerate(universe):
        symbol = stock_info["symbol"]
        print(f"处理 [{i+1}/{total}] {symbol} {stock_info.get('name', '')}")

        # 获取K线（带缓存）
        df, from_cache = fetch_kline_with_cache(
            symbol,
            ITICK_TOKEN,
            days=LOOKBACK_DAYS,
            cache_enabled=bool(args.cache_kline),
            cache_dir=args.cache_dir,
            cache_date=cache_date,
            offline=bool(args.offline),
        )

        # 只有真的请求了 iTick 才 sleep
        if not from_cache:
            time.sleep(args.request_interval)

        if df.empty:
            continue

        if args.asof_date:
            try:
                asof = pd.to_datetime(args.asof_date.strip(), format="%Y%m%d")
                df["date"] = pd.to_datetime(df["date"], errors="coerce")
                df = df[df["date"] <= asof].copy()
                df.sort_values("date", inplace=True)
                df = df.tail(LOOKBACK_DAYS)
            except Exception as e:
                print(f"asof-date 解析失败：{args.asof_date}，错误：{e}")
                continue

        if df.empty:
            continue

        df = calculate_indicators(df)

        # 使用新的7个条件策略
        res_new = check_conditions(df)

        if res_new and res_new.get("passed"):
            res_new["symbol"] = symbol
            res_new["name"] = stock_info.get("name") or ""
            res_new["signals"] = "7条件策略"
            res_new["signal_desc"] = res_new.get("signal_desc", "")
            
            candidates.append(res_new)

        if (i+1) % 20 == 0:
            print(f"已处理 {i+1} 只，当前发现 {len(candidates)} 只候选股")

    # 3. 排序所有候选股
    candidates.sort(key=lambda x: x["score"], reverse=True)

    # 4. 输出结果
    print("\n" + "="*60)
    print(f"今日全市场右侧趋势候选（共{len(candidates)}只）")
    print("="*60)
    for idx, cand in enumerate(candidates, 1):
        print(f"{idx}️⃣ {cand['symbol']} {cand['name']}")
        print(f"   信号：{cand.get('signals', '')}")
        print(f"   趋势评分：{cand['score']:.2f}")
        print(f"   距20MA：{cand['dist_to_20ma']:.2f}%")
        print(f"   形态描述：{cand.get('signal_desc', '')}")
        print(f"   最新价：{cand['latest_price']:.2f}  回撤：{cand['drawdown']:.2f}%")
        print()

    # 保存到文件
    output_file = f"a_right_side_candidates_{datetime.now().strftime('%Y%m%d')}.csv"
    if candidates:
        df_out = pd.DataFrame(candidates)
        df_out.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"结果已保存至 {output_file}，共{len(candidates)}只候选股")


if __name__ == "__main__":
    main()