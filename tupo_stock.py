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
LOOKBACK_DAYS = 100                  # 获取历史K线的天数（需要足够长以计算200MA）

# 突破策略参数
PRE_BREAKOUT_DAYS = 10               # 突破前观察天数（检查是否始终在均线下方）
POST_BREAKOUT_DAYS = 5               # 突破后观察天数（检查右侧确立）

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


def fetch_kline_with_cache(symbol, ITICK_TOKEN, days, cache_enabled: bool, cache_dir: str, cache_date: str, offline: bool = False):
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

    df = fetch_kline(symbol, ITICK_TOKEN, days=days)
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


def _infer_cn_region_by_code(code6: str) -> str:
    """根据 6 位股票代码推断交易所（用于拼 iTick symbol：SH/SZ/BJ）。

    说明：AkShare 的 `stock_zh_a_spot_em()` 覆盖"沪深京 A 股"。这里用经验规则做映射，
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


def fetch_kline(symbol, ITICK_TOKEN, days=LOOKBACK_DAYS):
    """获取单个股票的日线K线数据"""
    url = f"{ITICK_BASE_URL}/stock/kline"

    try:
        region, code = _parse_itick_symbol(symbol)

        headers = {
            "accept": "application/json",
            "token": ITICK_TOKEN,
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
    """计算技术指标：均线、涨幅等"""
    if df.empty or len(df) < 200:
        return None

    # 计算均线
    df['MA5'] = df['close'].rolling(window=5).mean()
    df['MA20'] = df['close'].rolling(window=20).mean()
    df['MA50'] = df['close'].rolling(window=50).mean()
    df['MA200'] = df['close'].rolling(window=200).mean()

    # 计算涨幅
    df['pct_change_1'] = df['close'].pct_change(periods=1) * 100
    df['pct_change_5'] = df['close'].pct_change(periods=5) * 100

    return df


def check_breakout_strategy(df):
    """突破策略核心逻辑
    
    条件1：前期压制 - 突破前N天内收盘价始终位于200MA、50MA、20MA下方
    条件2：突破信号 - 某日收盘价同时冲破200MA、50MA、20MA，且收盘价位于所有均线上方
    条件3：右侧确立 - 突破后5个交易日内，收盘价始终未跌破突破当日的开盘价
    """
    if df is None or len(df) < (PRE_BREAKOUT_DAYS + POST_BREAKOUT_DAYS + 10):
        return None

    # 寻找可能的突破日
    candidates = []
    
    # 从第PRE_BREAKOUT_DAYS天开始检查（确保有足够的历史数据）
    for i in range(PRE_BREAKOUT_DAYS, len(df) - POST_BREAKOUT_DAYS):
        # 条件1：前期压制检查
        pre_period = df.iloc[i-PRE_BREAKOUT_DAYS:i]
        
        # 检查前期是否所有收盘价都在均线下方
        pre_condition = (
            (pre_period['close'] < pre_period['MA200']).all() and
            (pre_period['close'] < pre_period['MA50']).all() and
            (pre_period['close'] < pre_period['MA20']).all()
        )
        
        if not pre_condition:
            continue
        
        # 条件2：突破日检查
        breakout_day = df.iloc[i]
        
        # 检查是否同时突破三条均线
        breakout_condition = (
            breakout_day['close'] > breakout_day['MA200'] and
            breakout_day['close'] > breakout_day['MA50'] and
            breakout_day['close'] > breakout_day['MA20']
        )
        
        if not breakout_condition:
            continue
        
        # 条件3：右侧确立检查
        post_period = df.iloc[i+1:i+POST_BREAKOUT_DAYS+1]
        
        # 检查突破后是否始终未跌破突破日开盘价
        post_condition = (post_period['close'] >= breakout_day['open']).all()
        
        if not post_condition:
            continue
        
        # 计算突破强度评分
        breakout_strength = (
            (breakout_day['close'] - breakout_day['MA200']) / breakout_day['MA200'] * 100 +
            (breakout_day['close'] - breakout_day['MA50']) / breakout_day['MA50'] * 100 +
            (breakout_day['close'] - breakout_day['MA20']) / breakout_day['MA20'] * 100
        )
        
        # 计算成交量放大倍数
        vol_ma20 = df['volume'].rolling(window=20).mean().iloc[i]
        vol_ratio = breakout_day['volume'] / vol_ma20 if vol_ma20 > 0 else 1.0
        
        candidates.append({
            'breakout_date': breakout_day['date'],
            'breakout_price': breakout_day['close'],
            'ma200': breakout_day['MA200'],
            'ma50': breakout_day['MA50'],
            'ma20': breakout_day['MA20'],
            'strength_score': breakout_strength,
            'volume_ratio': vol_ratio,
            'position': i
        })
    
    # 返回最近的突破信号（如果有）
    if candidates:
        # 按突破日期排序，取最近的
        candidates.sort(key=lambda x: x['breakout_date'], reverse=True)
        latest_breakout = candidates[0]
        
        # 检查是否是最新交易日（即当前信号）
        latest_day_idx = len(df) - 1
        days_since_breakout = latest_day_idx - latest_breakout['position']
        
        # 如果是最近5天内的突破信号，认为是有效信号
        if days_since_breakout <= POST_BREAKOUT_DAYS:
            return {
                'passed': True,
                'breakout_date': latest_breakout['breakout_date'],
                'breakout_price': latest_breakout['breakout_price'],
                'current_price': df.iloc[-1]['close'],
                'strength_score': latest_breakout['strength_score'],
                'volume_ratio': latest_breakout['volume_ratio'],
                'days_since_breakout': days_since_breakout,
                'signal_desc': f"突破信号：{latest_breakout['breakout_date'].strftime('%Y-%m-%d')} 突破三条均线，强度{latest_breakout['strength_score']:.1f}分，成交量放大{latest_breakout['volume_ratio']:.1f}倍"
            }
    
    return {'passed': False}


def main():
    parser = argparse.ArgumentParser(description="全市场右侧突破策略扫描")
    parser.add_argument("--min-price", type=float, default=MIN_PRICE, help="STEP1 最低股价，默认 5")
    parser.add_argument("--min-amount", type=float, default=MIN_VOLUME_AMOUNT, help="STEP1 最低成交额，默认 3000万")
    parser.add_argument(
        "--max-stocks",
        type=int,
        default=200,
        help="最多扫描多少只股票（按成交额从高到低排序后截断）",
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
        help="开启K线缓存",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="离线模式：只读缓存，不请求iTick",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=KLINE_CACHE_DIR,
        help="K线缓存目录",
    )
    parser.add_argument(
        "--cache-date",
        type=str,
        default="",
        help="缓存日期目录（YYYYMMDD），默认空=今天",
    )
    parser.add_argument(
        "--asof-date",
        type=str,
        default="",
        help="按指定日期视角运行（YYYYMMDD）",
    )
    args = parser.parse_args()

    cache_date = args.cache_date.strip() if args.cache_date else _today_yyyymmdd()

    print(f"开始执行全市场突破策略扫描 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"策略参数：突破前{PRE_BREAKOUT_DAYS}天压制，突破后{POST_BREAKOUT_DAYS}天确立")

    # 1. STEP 1：全市场初筛
    try:
        universe = get_a_share_universe_step1(min_price=args.min_price, min_amount=args.min_amount)
    except Exception as e:
        print("获取全市场股票池失败:", e)
        return
    
    if not universe:
        print("STEP 1 初筛后股票池为空，程序终止")
        return

    # 按成交额降序排序
    universe.sort(key=lambda x: x.get("amount", 0), reverse=True)
    if args.max_stocks and args.max_stocks > 0:
        universe = universe[: args.max_stocks]
        print(f"实际扫描股票数：{len(universe)}")

    # 2. 对每只股票进行突破策略分析
    candidates = []
    total = len(universe)
    
    for i, stock_info in enumerate(universe):
        symbol = stock_info["symbol"]
        print(f"分析 [{i+1}/{total}] {symbol} {stock_info.get('name', '')}")

        # 获取K线数据
        df = fetch_kline(symbol, ITICK_TOKEN, days=LOOKBACK_DAYS)
        
        if df.empty:
            continue

        # 按指定日期截断数据
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

        # 计算技术指标
        df = calculate_indicators(df)
        if df is None:
            continue

        # 应用突破策略
        result = check_breakout_strategy(df)
        
        if result and result.get("passed"):
            result["symbol"] = symbol
            result["name"] = stock_info.get("name", "")
            candidates.append(result)

        # 控制请求频率
        if (i+1) % 10 == 0:
            time.sleep(args.request_interval)
            print(f"已处理 {i+1} 只，当前发现 {len(candidates)} 只突破候选股")

    # 3. 按突破强度排序
    candidates.sort(key=lambda x: x.get("strength_score", 0), reverse=True)

    # 4. 输出结果
    print("\n" + "="*60)
    print(f"突破策略候选股（共{len(candidates)}只）")
    print("="*60)
    
    for idx, cand in enumerate(candidates, 1):
        print(f"{idx}️⃣ {cand['symbol']} {cand['name']}")
        print(f"   突破日期：{cand['breakout_date'].strftime('%Y-%m-%d')}")
        print(f"   突破价格：{cand['breakout_price']:.2f}，当前价格：{cand['current_price']:.2f}")
        print(f"   突破强度：{cand['strength_score']:.1f}分")
        print(f"   成交量放大：{cand['volume_ratio']:.1f}倍")
        print(f"   突破后天数：{cand['days_since_breakout']}天")
        print(f"   信号描述：{cand['signal_desc']}")
        print()

    # 保存到文件
    if candidates:
        output_file = f"breakout_candidates_{datetime.now().strftime('%Y%m%d')}.csv"
        df_out = pd.DataFrame(candidates)
        df_out.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"结果已保存至 {output_file}，共{len(candidates)}只突破候选股")


if __name__ == "__main__":
    main()