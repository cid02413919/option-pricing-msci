"""Quick utility to download and cache the 2023 SPX weekly option chains from WRDS.
Run from repo root:
    python source/download_spx_cache.py
"""
import os
import sys

sys.path.append("source")

from data_importing.data_import import multi_option_multi_day, return_multi_day, wednesdays

# Your WRDS username (ensure your WRDS password has been set/saved)
USERNAME = "AlexMatijasevic"
SECID = 108105  # 108105 = SPX
START = "2023-01-01"
END = "2023-12-31"

def main():
    print(f"1. Connecting to WRDS as '{USERNAME}' to find trading dates...")
    rtn = return_multi_day(USERNAME, START, END, lookback=10, secid=SECID, save=False)
    dates = wednesdays(rtn.index, START, END)
    print(f"   Found {len(dates)} weekly Wednesday pricing dates in 2023.")

    print(f"2. Downloading SPX option chains from WRDS (OptionMetrics)... this may take 1-3 minutes.")
    df = multi_option_multi_day(USERNAME, dates, secid=SECID, save=False, calc_q=True)

    out_dir = os.path.join("source", "data")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"cache_weekly_chains_{SECID}_{START}_{END}.pkl")

    print(f"3. Saving {len(df)} quotes to {out_file}...")
    df.to_pickle(out_file)
    print("\nDone! The cache file is ready. You can now run random_forest.py.")

if __name__ == "__main__":
    main()
