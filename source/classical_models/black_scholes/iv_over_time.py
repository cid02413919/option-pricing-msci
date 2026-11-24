"""Plot the implied volatility of one option over time, from a start day to its expiry.

Also prints how much the implied vol moves from one trading day to the next:
  - average absolute change  = mean(|IV_today - IV_yesterday|)
  - mean squared change      = mean((IV_today - IV_yesterday)^2)

Run from the repo root:  python source/classical_models/black_scholes/iv_over_time.py
"""
import sys
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append("source")  # tells python to look in source (run from repo root)

from data_importing.data_import import multi_option_single_day, single_option_multi_day

# WRDS username, and the day the option is chosen on (it is followed from here to expiry)
username = "kairogers"
start = "2025-06-23"

# ---- Settings: switch these on/off ----
SAVE_FIG = False   # True saves the figure to figures/


def pick(prompt, n):
    """Ask for a row number between 0 and n-1 until a valid one is given."""
    while True:
        choice = input(prompt).strip()
        if choice.isdigit() and int(choice) < n:
            return int(choice)
        print(f"Enter a number from 0 to {n - 1}")


# ---- Choose the option ----
secid = int(input("SECID (e.g. 108105 = SPX, 110007 = XLB, 110014 = XLK): "))
chain = multi_option_single_day(username, start, secid=secid, save=False)
S0 = chain["S"].iloc[0]
print(f"\n{len(chain)} options quoted on {start}, spot = {S0:g}")

# step 1: expiry
expiries = [pd.Timestamp(e) for e in sorted(chain["exdate"].unique())]
counts = chain["exdate"].value_counts()
print("\n  #  expiry       options")
for i, e in enumerate(expiries):
    print(f"{i:3d}  {e.date()}   {counts[e]}")
exdate = expiries[pick("Choose an expiry #: ", len(expiries))]

# step 2: strike and call/put (strikes within 20% of spot, unless there are none)
options = chain[chain["exdate"] == exdate]
near = options[options["strike"].between(0.8 * S0, 1.2 * S0)]
options = (near if not near.empty else options).sort_values(["cp_flag", "strike"]).reset_index(drop=True)
print("\n" + options[["optionid", "cp_flag", "strike", "mid", "impl_volatility", "volume"]].to_string())
choice = options.loc[pick("Choose an option #: ", len(options))]

optionid = int(choice["optionid"])
end = exdate.strftime("%Y-%m-%d")
print(f"\nFollowing option {optionid} from {start} to expiry {end}")

# ---- Load the option's daily quotes ----
opt = single_option_multi_day(username, optionid, start, end, secid=secid, save=False)
opt = opt[opt["impl_volatility"].notna()]      # drop days with no implied vol
opt = opt.sort_values("date")                  # oldest -> newest

# ---- Day-to-day change in implied vol ----
opt["iv_change"] = opt["impl_volatility"].diff()   # today's IV minus the previous day's IV
changes = opt["iv_change"].dropna()                # the first day has no previous day

avg_abs_change = changes.abs().mean()
mean_sq_change = (changes ** 2).mean()

print(f"\nDays with an implied vol: {len(opt)}  ({len(changes)} day-to-day changes)")
print(f"Average absolute change:  {avg_abs_change:.4f}  ({avg_abs_change * 100:.2f} vol points)")
print(f"Mean squared change:      {mean_sq_change:.6f}")

# ---- Plot ----
K = opt["strike"].iloc[0]
cp = opt["cp_flag"].iloc[0]

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(opt["date"], opt["impl_volatility"], "o-", ms=3, color="purple")

ax.set_xlabel("Date")
ax.set_ylabel("Implied volatility")
ax.set_title(f"secid {secid} {'call' if cp == 'C' else 'put'}, K = {K:g}, expiry {end}",
             loc="left", fontweight="bold")

# put the two numbers on the plot too
ax.text(0.02, 0.95,
        f"avg |change| = {avg_abs_change:.4f}\nmean squared change = {mean_sq_change:.6f}",
        transform=ax.transAxes, va="top")

for side in ("top", "right"):
    ax.spines[side].set_visible(False)
fig.autofmt_xdate()                            # tilt the dates so they don't overlap

fig.tight_layout()
if SAVE_FIG:
    fig.savefig(f"figures/iv_over_time_{secid}_{optionid}.png", dpi=300, bbox_inches="tight")
plt.show()
