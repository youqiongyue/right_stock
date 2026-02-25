"""
右侧趋势策略回测引擎 v3（行业ETF过滤版）
==========================================
依赖：pip install akshare pandas numpy

v3 新增：行业 ETF 强弱过滤
  - 每只股票映射到对应的行业 ETF
  - 买入信号触发时同时检查：ETF收盘 > ETF的MA20 且 ETF近5日涨幅 > 0
  - 两个条件都满足 → 行业强势，允许入场
  - 任一不满足 → 行业弱势，跳过该信号
  - 每笔交易记录行业ETF得分，方便事后分析哪些行业效果最好
  - 可用 --no-etf-filter 关闭，与 v2 对比

用法：
  python backtest_engine_v3.py                        # 默认跑法
  python backtest_engine_v3.py --start 2023-01-01     # 自定义起始（推荐至少1年）
  python backtest_engine_v3.py --compare              # 同时跑v2做对比
  python backtest_engine_v3.py --no-etf-filter        # 关闭ETF过滤（纯v2逻辑）
  python backtest_engine_v3.py --etf-strict           # 严格模式：ETF须MA5>MA20>MA50全多头
"""

import akshare as ak
import pandas as pd
import numpy as np
import json
import argparse
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")


# ==================== 默认参数 ====================
DEFAULT_START        = "2023-01-01"
DEFAULT_END          = datetime.now().strftime("%Y-%m-%d")
DEFAULT_ATR_MULT     = 1.5
DEFAULT_TRAILING_TP  = 0.10
DEFAULT_MAX_HOLD     = 20
COMMISSION           = 0.001
SLIPPAGE             = 0.002
CSI300_SYMBOL        = "000300"

# 新增常量：用于动态股票选择
ITICK_TOKEN          = "6e22921dceb0492ea60d21c43c4833a2c00794ec321e49f498340d728645ae2c"
LOOKBACK_DAYS        = 60

DEFAULT_SYMBOLS = [
    "SH688981", "SH688111", "SH688036", "SH688599", "SH688012",
    "SH688396", "SH688180", "SH688169", "SH688009", "SH688008",
    "SH600036", "SH600519", "SH601318", "SH600900", "SH601166",
    "SZ300750", "SZ002415", "SZ000333", "SZ002594", "SZ000858",
]

# ──────────────────────────────────────────────────────────
# 行业 ETF 映射表
# 股票代码前缀/规则 → ETF基金代码（AkShare可拉）
#
# 覆盖逻辑：
#   科创板 688xxx → 科创50 ETF (588000)
#   半导体相关    → 半导体 ETF (512480)
#   新能源/电池   → 新能源车 ETF (515030)
#   医药/生物     → 医药 ETF (512010)
#   银行/金融     → 银行 ETF (512800)
#   消费/白酒     → 消费 ETF (159928)
#   军工          → 军工 ETF (512660)
#   其余默认      → 沪深300 ETF (510300)
#
# 可按需扩展更细的映射，key 为6位股票代码
# ──────────────────────────────────────────────────────────
STOCK_TO_ETF = {
    # ── 科创板（688xxx）→ 科创50 ETF
    "688981": "588000", "688111": "588000", "688036": "588000",
    "688599": "588000", "688012": "588000", "688396": "588000",
    "688180": "588000", "688169": "588000", "688009": "588000",
    "688008": "588000",

    # ── 半导体 → 半导体 ETF
    # （688xxx中芯片类已含在科创50；这里可单独覆盖）

    # ── 新能源/电池 → 新能源车 ETF
    "300750": "515030",  # 宁德时代
    "002594": "515030",  # 比亚迪

    # ── 银行/金融 → 银行 ETF
    "600036": "512800",  # 招商银行
    "601318": "512800",  # 中国平安
    "601166": "512800",  # 兴业银行

    # ── 消费/白酒 → 消费 ETF
    "600519": "159928",  # 贵州茅台
    "000858": "159928",  # 五粮液
    "002415": "159928",  # 海康威视（可视安防，暂归消费）

    # ── 电力/能源 → 中证红利 ETF（电力类常见）
    "600900": "515070",  # 长江电力

    # ── 制造/家电 → 沪深300 ETF（兜底）
    "000333": "510300",  # 美的集团
}

# 未在上表中的股票，按代码段兜底映射
def _default_etf_by_code(code6: str) -> str:
    """未精确匹配时，按板块规则兜底"""
    c = code6.zfill(6)
    if c.startswith("688") or c.startswith("689"):
        return "588000"   # 科创50 ETF
    if c.startswith("300") or c.startswith("301"):
        return "159915"   # 创业板 ETF
    if c.startswith(("60", "00")):
        return "510300"   # 沪深300 ETF（兜底）
    return "510300"


def get_etf_code(symbol: str) -> str:
    """根据股票symbol获取对应行业ETF代码"""
    s = symbol.strip().upper()
    code6 = s[2:] if s[:2] in ("SH", "SZ", "BJ") else s
    return STOCK_TO_ETF.get(code6, _default_etf_by_code(code6))
# ==================================================


# ─────────────────────────────────────────────────
# 数据获取
# ─────────────────────────────────────────────────

def fetch_history(symbol: str, start: str, end: str) -> pd.DataFrame:
    s = symbol.strip().upper()
    prefix = s[:2].lower() if s[:2] in ("SH", "SZ", "BJ") else "sh"
    code   = s[2:] if s[:2] in ("SH", "SZ", "BJ") else s
    sina_symbol = f"{prefix}{code}"
    try:
        df = ak.stock_zh_a_daily(
            symbol=sina_symbol,
            adjust="qfq",
        )
    except Exception as e:
        print(f"  [{symbol}] 数据获取失败: {e}")
        return pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame()

    # 新浪接口列名已是英文，直接使用
    df["date"] = pd.to_datetime(df["date"])
    for c in ["open", "close", "high", "low", "volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.dropna(subset=["close"], inplace=True)

    # 按日期范围过滤
    start_dt = pd.to_datetime(start)
    end_dt   = pd.to_datetime(end)
    df = df[(df["date"] >= start_dt) & (df["date"] <= end_dt)].reset_index(drop=True)
    return df


def fetch_etf(etf_code: str, start: str, end: str) -> pd.DataFrame:
    """
    拉 ETF 日线数据（前复权）。
    使用新浪接口：fund_etf_hist_sina
    返回含 date / close / MA20 / pct5 列的 DataFrame，以 date 为索引方便查询。
    """
    # 判断交易所前缀（上交所 sh，深交所 sz）
    code = etf_code.strip()
    if code.startswith("5") or code.startswith("51") or code.startswith("58"):
        sina_etf = f"sh{code}"
    elif code.startswith("15") or code.startswith("16"):
        sina_etf = f"sz{code}"
    else:
        # 通用规则：6开头上交所，其余深交所
        sina_etf = f"sh{code}" if code.startswith(("5", "6")) else f"sz{code}"
    try:
        df = ak.fund_etf_hist_sina(symbol=sina_etf)
    except Exception as e:
        print(f"  [ETF {etf_code}] 数据获取失败: {e}")
        return pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame()

    # 新浪接口列名已是英文
    df["date"]  = pd.to_datetime(df["date"])
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df.dropna(subset=["close"], inplace=True)
    df.sort_values("date", inplace=True)

    # 计算指标
    df["MA5"]  = df["close"].rolling(5).mean()
    df["MA20"] = df["close"].rolling(20).mean()
    df["MA50"] = df["close"].rolling(50).mean()
    df["pct5"] = df["close"].pct_change(5) * 100   # 5日涨幅

    df.set_index("date", inplace=True)
    return df


def fetch_csi300(start: str, end: str) -> pd.DataFrame:
    try:
        df = ak.stock_zh_index_daily(symbol=f"sh{CSI300_SYMBOL}")
        df["date"] = pd.to_datetime(df["date"])
        df["pct"]  = df["close"].pct_change() * 100
        df = df[["date", "pct"]].dropna()
        mask = (df["date"] >= start) & (df["date"] <= end)
        df = df[mask].sort_values("date").reset_index(drop=True)
        return df
    except Exception as e:
        print(f"  [大盘数据] 获取失败: {e}")
        return pd.DataFrame()


# ─────────────────────────────────────────────────
# ETF 强弱判断
# ─────────────────────────────────────────────────

def check_etf_strength(etf_df: pd.DataFrame, date: pd.Timestamp, strict: bool = False) -> dict:
    """
    判断指定日期的行业 ETF 是否处于强势。

    标准模式（strict=False）：
      - ETF 收盘 > MA20（中期趋势向上）
      - ETF 近5日涨幅 > 0（短期动能向上）
      两个都满足 → 强势（passed=True）

    严格模式（strict=True）：
      - MA5 > MA20 > MA50（全多头排列）

    返回 dict：
      passed     : bool  是否强势
      etf_vs_ma20: float ETF收盘相对MA20的偏离度(%)
      etf_pct5   : float ETF近5日涨幅(%)
      etf_score  : int   0~2 分（标准模式：每满足一个条件+1）
    """
    if etf_df is None or etf_df.empty:
        # 拉不到数据时不过滤（宽松降级）
        return {"passed": True, "etf_vs_ma20": 0.0, "etf_pct5": 0.0, "etf_score": -1}

    # 找当天或往前最近一个有数据的交易日
    available = etf_df.index[etf_df.index <= date]
    if available.empty:
        return {"passed": True, "etf_vs_ma20": 0.0, "etf_pct5": 0.0, "etf_score": -1}

    row = etf_df.loc[available[-1]]

    close = row.get("close", np.nan)
    ma20  = row.get("MA20",  np.nan)
    ma5   = row.get("MA5",   np.nan)
    ma50  = row.get("MA50",  np.nan)
    pct5  = row.get("pct5",  np.nan)

    if pd.isna(close) or pd.isna(ma20):
        return {"passed": True, "etf_vs_ma20": 0.0, "etf_pct5": 0.0, "etf_score": -1}

    etf_vs_ma20 = (close - ma20) / ma20 * 100
    etf_pct5    = float(pct5) if not pd.isna(pct5) else 0.0

    if strict:
        passed = (not pd.isna(ma5) and not pd.isna(ma50)
                  and float(ma5) > float(ma20) > float(ma50))
        score  = 2 if passed else 0
    else:
        cond1  = close > ma20
        cond2  = etf_pct5 > 0
        score  = int(cond1) + int(cond2)
        passed = cond1 and cond2

    return {
        "passed":      passed,
        "etf_vs_ma20": round(etf_vs_ma20, 2),
        "etf_pct5":    round(etf_pct5, 2),
        "etf_score":   score,
    }


# ─────────────────────────────────────────────────
# 指标计算
# ─────────────────────────────────────────────────

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["MA5"]      = df["close"].rolling(5).mean()
    df["MA20"]     = df["close"].rolling(20).mean()
    df["MA50"]     = df["close"].rolling(50).mean()
    df["vol_ma20"] = df["volume"].rolling(20).mean()
    df["vol_ratio"] = df["volume"] / df["vol_ma20"]
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"]  - prev_close).abs(),
    ], axis=1).max(axis=1)
    df["atr14"] = tr.rolling(14).mean()
    return df


# ─────────────────────────────────────────────────
# 买入信号
# ─────────────────────────────────────────────────

def generate_buy_signal(
    df: pd.DataFrame,
    i: int,
    bad_market_dates: set,
    etf_df: pd.DataFrame,
    etf_filter: bool,
    etf_strict: bool,
) -> dict | None:

    if i < 56:
        return None

    cur  = df.iloc[i]
    prev = df.iloc[i - 1]

    # ── 大盘过滤 ──
    if cur["date"] in bad_market_dates:
        return None

    # ── 趋势底座 ──
    if not (cur["MA20"] > cur["MA50"]):       return None
    if not (cur["close"] > cur["MA50"]):      return None
    if cur["MA50"] <= df["MA50"].iloc[i - 5]: return None

    # ── 回踩 ──
    if not (prev["close"] < prev["MA20"]):    return None

    # ── 右侧启动 ──
    if not (cur["close"] > cur["MA5"]):       return None

    # ── 实体突破 ──
    if cur["close"] <= prev["high"]:          return None

    # ── 量比 ──
    vol_ratio = float(cur["vol_ratio"]) if not pd.isna(cur["vol_ratio"]) else 0.0
    if vol_ratio < 1.2:                       return None

    # ── 不追高 ──
    recent_low = df["low"].iloc[max(0, i-10):i+1].min()
    if recent_low > 0 and (cur["close"] / recent_low) > 1.25:
        return None

    today_pct = (cur["close"] - prev["close"]) / prev["close"] * 100 if prev["close"] > 0 else 0.0
    atr = float(cur["atr14"]) if not pd.isna(cur["atr14"]) else cur["close"] * 0.02

    # ── 行业 ETF 过滤（核心新增逻辑）──
    etf_result = check_etf_strength(etf_df, cur["date"], strict=etf_strict)
    if etf_filter and not etf_result["passed"]:
        return None

    score = _calc_score(cur, prev, today_pct, vol_ratio, recent_low)

    return {
        "signal_date":   cur["date"],
        "entry_price":   cur["close"],
        "score":         score,
        "vol_ratio":     round(vol_ratio, 2),
        "today_pct":     round(today_pct, 2),
        "atr":           round(atr, 3),
        "etf_vs_ma20":   etf_result["etf_vs_ma20"],
        "etf_pct5":      etf_result["etf_pct5"],
        "etf_score":     etf_result["etf_score"],
    }


def _calc_score(cur, prev, today_pct, vol_ratio, recent_low) -> int:
    score = 0
    if cur["MA5"] > cur["MA20"] > cur["MA50"]: score += 15
    elif cur["MA20"] > cur["MA50"]:            score += 10
    score += 10
    if cur["close"] > cur["MA20"]: score += 5
    score += 10
    score += 15
    if today_pct >= 5:   score += 15
    elif today_pct >= 3: score += 10
    elif today_pct >= 1: score += 5
    if vol_ratio >= 1.5:   score += 10
    elif vol_ratio >= 1.2: score += 7
    if today_pct > 0: score += 10
    if recent_low > 0:
        ratio = cur["close"] / recent_low
        if ratio <= 1.15:   score += 10
        elif ratio <= 1.25: score += 5
    return score


# ─────────────────────────────────────────────────
# 出场逻辑
# ─────────────────────────────────────────────────

def check_exit(df, entry_idx, signal, atr_mult, trailing_tp, max_hold) -> dict | None:
    buy_idx = entry_idx + 1
    if buy_idx >= len(df):
        return None

    actual_entry = df.iloc[buy_idx]["open"] * (1 + SLIPPAGE)
    atr_stop     = max(actual_entry - atr_mult * signal["atr"], actual_entry * 0.90)

    trailing_active = False
    exit_price      = actual_entry
    exit_reason     = "超时平仓"
    hold_days       = 0

    for j in range(buy_idx, min(buy_idx + max_hold, len(df))):
        row = df.iloc[j]
        hold_days = j - buy_idx + 1

        if row["low"] <= atr_stop:
            exit_price  = atr_stop
            exit_reason = "ATR止损"
            break

        cur_return = (row["close"] - actual_entry) / actual_entry
        if not trailing_active and cur_return >= trailing_tp:
            trailing_active = True

        if trailing_active:
            if row["close"] < row["MA5"] and j + 1 < len(df):
                exit_price  = df.iloc[j + 1]["open"] * (1 - SLIPPAGE)
                exit_reason = "移动止盈"
                hold_days   = j - buy_idx + 2
                break
        else:
            if row["close"] < row["MA20"] and j + 1 < len(df):
                exit_price  = df.iloc[j + 1]["open"] * (1 - SLIPPAGE)
                exit_reason = "跌破MA20"
                hold_days   = j - buy_idx + 2
                break

        exit_price = row["close"]
        if hold_days >= max_hold:
            exit_reason = "超时平仓"
            break

    net_return = (exit_price - actual_entry) / actual_entry - COMMISSION * 2
    exit_date  = df.iloc[min(buy_idx + hold_days - 1, len(df)-1)]["date"]

    return {
        "actual_entry":  round(actual_entry, 3),
        "atr_stop":      round(atr_stop, 3),
        "exit_price":    round(exit_price, 3),
        "exit_date":     exit_date,
        "hold_days":     hold_days,
        "exit_reason":   exit_reason,
        "gross_return":  round((exit_price - actual_entry) / actual_entry * 100, 2),
        "net_return":    round(net_return * 100, 2),
        "trailing_used": trailing_active,
    }


# ─────────────────────────────────────────────────
# 单股回测
# ─────────────────────────────────────────────────

def backtest_single(
    symbol: str,
    start: str,
    end: str,
    atr_mult: float,
    trailing_tp: float,
    max_hold: int,
    bad_market_dates: set,
    etf_cache: dict,
    etf_filter: bool,
    etf_strict: bool,
    version_label: str = "v3",
) -> list[dict]:

    print(f"  [{version_label}] {symbol} ...", end=" ", flush=True)

    df = fetch_history(symbol, start, end)
    if df.empty or len(df) < 60:
        print("数据不足，跳过")
        return []

    df = calculate_indicators(df)

    # 获取该股对应的行业 ETF 数据（有缓存就不重复拉）
    etf_code = get_etf_code(symbol)
    if etf_code not in etf_cache:
        print(f"(拉 ETF {etf_code})", end=" ", flush=True)
        etf_cache[etf_code] = fetch_etf(etf_code, start, end)
    etf_df = etf_cache[etf_code]

    trades      = []
    last_entry_i = -999

    for i in range(56, len(df)):
        sig = generate_buy_signal(
            df, i, bad_market_dates, etf_df, etf_filter, etf_strict
        )
        if sig is None:
            continue
        if i - last_entry_i < 5:
            continue

        trade_exit = check_exit(df, i, sig, atr_mult, trailing_tp, max_hold)
        if trade_exit is None:
            continue

        trades.append({
            "symbol":        symbol,
            "version":       version_label,
            "etf_code":      etf_code,
            "signal_date":   sig["signal_date"].strftime("%Y-%m-%d"),
            "entry_date":    df.iloc[min(i+1, len(df)-1)]["date"].strftime("%Y-%m-%d"),
            "exit_date":     trade_exit["exit_date"].strftime("%Y-%m-%d"),
            "entry_price":   trade_exit["actual_entry"],
            "atr_stop":      trade_exit["atr_stop"],
            "exit_price":    trade_exit["exit_price"],
            "hold_days":     trade_exit["hold_days"],
            "exit_reason":   trade_exit["exit_reason"],
            "gross_return":  trade_exit["gross_return"],
            "net_return":    trade_exit["net_return"],
            "signal_score":  sig["score"],
            "vol_ratio":     sig["vol_ratio"],
            "today_pct":     sig["today_pct"],
            "trailing_used": trade_exit["trailing_used"],
            # 行业ETF专属字段
            "etf_vs_ma20":   sig["etf_vs_ma20"],
            "etf_pct5":      sig["etf_pct5"],
            "etf_score":     sig["etf_score"],
        })
        last_entry_i = i

    print(f"完成，{len(trades)} 笔")
    return trades


# ─────────────────────────────────────────────────
# 统计
# ─────────────────────────────────────────────────

def compute_stats(trades: list[dict], label: str = "") -> dict:
    if not trades:
        return {"label": label, "total_trades": 0}

    returns = np.array([t["net_return"] for t in trades])
    wins    = returns[returns > 0]
    losses  = returns[returns <= 0]

    cumulative = np.cumsum(returns)
    peak   = np.maximum.accumulate(cumulative)
    max_dd = float((cumulative - peak).min())

    exit_counts = {}
    for t in trades:
        r = t["exit_reason"]
        exit_counts[r] = exit_counts.get(r, 0) + 1

    # 按 ETF 分组统计
    etf_stats = {}
    for t in trades:
        ec = t.get("etf_code", "unknown")
        if ec not in etf_stats:
            etf_stats[ec] = {"trades": 0, "total_return": 0.0, "wins": 0}
        etf_stats[ec]["trades"]       += 1
        etf_stats[ec]["total_return"] += t["net_return"]
        if t["net_return"] > 0:
            etf_stats[ec]["wins"] += 1
    for ec in etf_stats:
        s = etf_stats[ec]
        s["avg_return"] = round(s["total_return"] / s["trades"], 2)
        s["win_rate"]   = round(s["wins"] / s["trades"] * 100, 1)

    return {
        "label":           label,
        "total_trades":    len(trades),
        "win_rate":        round(len(wins) / len(returns) * 100, 1),
        "avg_return":      round(float(returns.mean()), 2),
        "avg_win":         round(float(wins.mean()) if len(wins) else 0, 2),
        "avg_loss":        round(float(losses.mean()) if len(losses) else 0, 2),
        "profit_factor":   round(float(abs(wins.sum() / losses.sum())) if losses.sum() != 0 else 999, 2),
        "total_return":    round(float(returns.sum()), 2),
        "max_dd":          round(max_dd, 2),
        "max_loss_single": round(float(returns.min()), 2),
        "avg_hold_days":   round(float(np.mean([t["hold_days"] for t in trades])), 1),
        "exit_reasons":    exit_counts,
        "trailing_count":  sum(1 for t in trades if t.get("trailing_used")),
        "equity_curve":    cumulative.tolist(),
        "etf_breakdown":   etf_stats,
    }


def print_stats(stats: dict):
    print(f"\n{'─'*55}")
    print(f"  {stats.get('label','')}")
    print(f"{'─'*55}")
    if stats.get("total_trades", 0) == 0:
        print("  无有效交易")
        return
    print(f"  总交易次数:    {stats['total_trades']}")
    print(f"  胜率:          {stats['win_rate']}%")
    print(f"  平均单笔净收益:{stats['avg_return']}%")
    print(f"  平均盈利:      {stats['avg_win']}%  平均亏损: {stats['avg_loss']}%")
    print(f"  盈亏比:        {stats['profit_factor']}")
    print(f"  累计收益:      {stats['total_return']}%")
    print(f"  最大回撤:      {stats['max_dd']}%")
    print(f"  平均持仓天数:  {stats['avg_hold_days']} 天")
    print(f"  移动止盈触发:  {stats.get('trailing_count',0)} 次")
    print(f"  出场原因:      {stats['exit_reasons']}")

    # 行业 ETF 分组绩效
    etf_bd = stats.get("etf_breakdown", {})
    if etf_bd:
        print(f"\n  ── 行业ETF分组绩效 ──")
        sorted_etfs = sorted(etf_bd.items(), key=lambda x: x[1]["avg_return"], reverse=True)
        for etf_code, s in sorted_etfs:
            print(f"  ETF {etf_code}  {s['trades']}笔  "
                  f"胜率{s['win_rate']}%  均收益{s['avg_return']}%")


# ─────────────────────────────────────────────────
# 主函数
# ─────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="右侧趋势策略回测 v3（行业ETF过滤版）")
    parser.add_argument("--start",          default=DEFAULT_START)
    parser.add_argument("--end",            default=DEFAULT_END)
    parser.add_argument("--symbols",        nargs="+", default=None)
    parser.add_argument("--atr-mult",       type=float, default=DEFAULT_ATR_MULT)
    parser.add_argument("--trailing-tp",    type=float, default=DEFAULT_TRAILING_TP)
    parser.add_argument("--max-hold",       type=int,   default=DEFAULT_MAX_HOLD)
    parser.add_argument("--no-etf-filter",  action="store_true", help="关闭行业ETF过滤（降级为v2）")
    parser.add_argument("--etf-strict",     action="store_true", help="严格ETF模式：需全多头排列")
    parser.add_argument("--no-market-filter", action="store_true", help="关闭大盘过滤")
    parser.add_argument("--compare",        action="store_true", help="同时跑无ETF过滤版做对比")
    parser.add_argument("--output",         default="backtest_result_v3.json")
    # 新增参数：动态股票选择
    parser.add_argument("--dynamic-stocks", action="store_true", help="使用new.py策略动态选择股票（默认关闭）")
    parser.add_argument("--min-price", type=float, default=5.0, help="动态选择时的最低股价")
    parser.add_argument("--min-amount", type=float, default=30000000, help="动态选择时的最低成交额")
    parser.add_argument("--max-stocks", type=int, default=600, help="动态选择时的最大股票数量")
    args = parser.parse_args()

    etf_filter    = not args.no_etf_filter
    market_filter = not args.no_market_filter
    symbols       = args.symbols or DEFAULT_SYMBOLS

    # 如果启用了动态股票选择，则调用new.py的策略逻辑
    if args.dynamic_stocks:
        print("使用动态股票选择策略（基于new.py逻辑）...")
        symbols = get_dynamic_stock_universe(args.min_price, args.min_amount, args.max_stocks)
        if not symbols:
            print("动态股票选择失败，回退到默认股票池")
            symbols = DEFAULT_SYMBOLS
    
    print(f"\n{'='*60}")
    print(f"  右侧趋势策略回测 v3（行业ETF过滤版）")
    print(f"{'='*60}")
    print(f"  回测区间:   {args.start} ~ {args.end}")
    print(f"  股票池:     {len(symbols)} 只{'（动态选择）' if args.dynamic_stocks else '（固定列表）'}")
    print(f"  ETF过滤:    {'开启' + ('【严格模式：全多头排列】' if args.etf_strict else '【标准：收盘>MA20 且 5日涨>0】') if etf_filter else '关闭'}")
    print(f"  大盘过滤:   {'开启' if market_filter else '关闭'}")
    print(f"  ATR止损:    {args.atr_mult}×ATR14  移动止盈: >{args.trailing_tp*100:.0f}%")
    print(f"{'='*60}\n")

    # 大盘过滤日期
    bad_market_dates = set()
    if market_filter:
        print("拉取沪深300大盘数据...")
        csi300 = fetch_csi300(args.start, args.end)
        if not csi300.empty:
            bad_market_dates = set(csi300[csi300["pct"] < -1.5]["date"].tolist())
            print(f"大盘过滤：{len(bad_market_dates)} 天\n")

    # ETF 数据缓存（多只股票可能映射同一个ETF，避免重复拉）
    etf_cache: dict = {}

    # ── v3 回测 ──
    print("── v3（行业ETF过滤）回测 ──")
    all_v3 = []
    for sym in symbols:
        trades = backtest_single(
            sym, args.start, args.end,
            args.atr_mult, args.trailing_tp, args.max_hold,
            bad_market_dates, etf_cache,
            etf_filter=etf_filter, etf_strict=args.etf_strict,
            version_label="v3（ETF过滤）",
        )
        all_v3.extend(trades)

    stats_v3 = compute_stats(all_v3, label="v3：ETF过滤 + ATR止损 + 移动止盈 + 大盘过滤")
    print_stats(stats_v3)

    # ── 对比：关闭ETF过滤的版本 ──
    stats_v2 = {}
    all_v2   = []
    if args.compare:
        print("\n── 对比：关闭ETF过滤（v2逻辑）──")
        for sym in symbols:
            trades = backtest_single(
                sym, args.start, args.end,
                args.atr_mult, args.trailing_tp, args.max_hold,
                bad_market_dates, etf_cache,
                etf_filter=False, etf_strict=False,
                version_label="v2（无ETF过滤）",
            )
            all_v2.extend(trades)

        stats_v2 = compute_stats(all_v2, label="v2：无ETF过滤")
        print_stats(stats_v2)

        # 对比摘要
        if stats_v2.get("total_trades", 0) > 0:
            print(f"\n{'='*60}")
            print(f"  v2（无ETF）vs v3（有ETF）对比")
            print(f"{'='*60}")
            for name, key, unit in [
                ("交易次数", "total_trades", "次"),
                ("胜率",     "win_rate",     "%"),
                ("平均收益", "avg_return",   "%"),
                ("盈亏比",   "profit_factor",""),
                ("最大回撤", "max_dd",       "%"),
            ]:
                v2v = stats_v2.get(key, "-")
                v3v = stats_v3.get(key, "-")
                try:
                    diff  = float(v3v) - float(v2v)
                    # 对最大回撤：v3更小（负数更大）才是改善
                    if key == "max_dd":
                        arrow = "✅" if diff > 0 else ("⚠️" if diff < 0 else "→")
                    else:
                        arrow = "✅" if diff > 0 else ("⚠️" if diff < 0 else "→")
                    print(f"  {name:8}  v2={v2v}{unit}  →  v3={v3v}{unit}  {arrow} ({diff:+.2f}{unit})")
                except:
                    print(f"  {name:8}  v2={v2v}  →  v3={v3v}")

    # 保存结果
    output = {
        "params": {
            "start": args.start, "end": args.end, "symbols": symbols,
            "atr_mult": args.atr_mult, "trailing_tp": args.trailing_tp,
            "max_hold": args.max_hold,
            "etf_filter": etf_filter, "etf_strict": args.etf_strict,
            "market_filter": market_filter,
            "commission": COMMISSION, "slippage": SLIPPAGE,
        },
        "stats":    stats_v3,
        "stats_v2": stats_v2,
        "trades":   all_v3 + all_v2,
        "etf_mapping": {sym: get_etf_code(sym) for sym in symbols},
        "optimizations_applied": [
            "✅ 行业ETF强弱过滤（收盘>MA20 且 5日涨>0）" if etf_filter else "⬜ ETF过滤（已关闭）",
            "✅ ATR动态止损",
            "✅ 移动止盈（浮盈>10%跟踪MA5）",
            "✅ 大盘过滤（沪深300跌>1.5%不开仓）" if market_filter else "⬜ 大盘过滤（已关闭）",
            "✅ 实体突破过滤",
            "✅ ETF数据缓存（同行业不重复拉取）",
        ],
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2, default=str)

    print(f"\n✅ 结果已保存至 {args.output}")
    print(f"📊 可视化：打开 backtest_dashboard.html 上传该文件")
    print(f"{'='*60}\n")


# 新增函数：动态获取股票池
def _infer_cn_region_by_code(code6: str) -> str:
    """根据6位股票代码推断交易所前缀（SH/SZ/BJ）"""
    c = str(code6).strip().zfill(6)
    if c.startswith(("60", "68", "69")):
        return "SH"
    if c.startswith(("00", "30", "20")):
        return "SZ"
    if c.startswith(("43", "83", "87", "88", "92")) or c[0] in {"4", "8", "9"}:
        return "BJ"
    return "SZ"


def get_dynamic_stock_universe(min_price: float = 5.0, min_amount: float = 30000000, max_stocks: int = 600) -> list[str]:
    """
    动态获取股票池：与 new.py 相同的全市场初筛方式。
    使用 AkShare 全市场快照（ak.stock_zh_a_spot_em），按成交额从高到低排序，
    过滤掉价格过低和成交额过小的股票，取前 max_stocks 只。

    返回：股票symbol列表，如['SH600000', 'SZ000001', ...]
    """
    print("动态股票选择：开始获取全市场股票数据（ak.stock_zh_a_spot_em）...")
    try:
        df = ak.stock_zh_a_spot_em()
    except Exception as e:
        print(f"动态股票选择：全市场快照获取失败: {e}")
        return []

    if df is None or df.empty:
        print("动态股票选择：全市场快照返回空")
        return []

    # 兼容不同 AkShare 版本字段
    code_col   = next((c for c in ["代码", "code", "证券代码"] if c in df.columns), None)
    name_col   = next((c for c in ["名称", "name", "证券简称"] if c in df.columns), None)
    price_col  = next((c for c in ["最新价", "最新", "price"] if c in df.columns), None)
    amount_col = next((c for c in ["成交额", "amount", "turnover"] if c in df.columns), None)

    if not all([code_col, price_col, amount_col]):
        print(f"动态股票选择：字段缺失 code={code_col}, price={price_col}, amount={amount_col}")
        return []

    base = df[[code_col, price_col, amount_col]].copy()
    base[price_col]  = pd.to_numeric(base[price_col],  errors="coerce")
    base[amount_col] = pd.to_numeric(base[amount_col], errors="coerce")
    base.dropna(subset=[price_col, amount_col], inplace=True)

    before = len(base)
    base = base[(base[price_col] >= float(min_price)) & (base[amount_col] >= float(min_amount))]
    after = len(base)
    print(f"动态股票选择：全市场初筛 {before} -> {after}（价格>={min_price}, 成交额>={min_amount:,.0f}）")

    # 按成交额从高到低排序，优先流动性好的票
    base = base.sort_values(amount_col, ascending=False)

    # 截断到 max_stocks
    if max_stocks and max_stocks > 0:
        base = base.head(max_stocks)

    # 转换为 symbol 格式（SH/SZ + 6位代码）
    selected_symbols = []
    for _, row in base.iterrows():
        code6  = str(row[code_col]).strip().zfill(6)
        region = _infer_cn_region_by_code(code6)
        selected_symbols.append(f"{region}{code6}")

    print(f"动态股票选择完成：共 {len(selected_symbols)} 只（全市场按成交额排序）")
    return selected_symbols

if __name__ == "__main__":
    main()
