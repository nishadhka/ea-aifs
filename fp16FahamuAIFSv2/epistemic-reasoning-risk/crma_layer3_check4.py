"""Check 4: AIFS 20261001 basin features vs IFS 46-day op rows for the same init.

    python crma_layer3_check4.py "<crma>/layer3/ifs46/evidence_d1/op_2026_10.parquet" \
        /tank/projects/crma_layer3/20261001/evidence.parquet
"""
import glob, sys
import numpy as np, pandas as pd

ifs_glob, aifs_path = sys.argv[1], sys.argv[2]
I = pd.concat([pd.read_parquet(p) for p in glob.glob(ifs_glob)])
I = I[(I.init_time == "2026-10-01") & (I.region_type == "hydrobasin")]
A = pd.read_parquet(aifs_path)
A = A[A.region_type == "hydrobasin"]
for d in (I, A):
    for c in ("window", "region_id", "feature"):
        d[c] = d[c].astype(str)
print(f"IFS rows {len(I):,} ({I.member.nunique()} members), AIFS rows {len(A):,} ({A.member.nunique()} members)")
ids_i, ids_a = set(I.region_id), set(A.region_id)
print(f"basins: IFS {len(ids_i)}, AIFS {len(ids_a)}, shared {len(ids_i & ids_a)}")
print(f"features only in AIFS: {sorted(set(A.feature) - set(I.feature))}; only in IFS: {sorted(set(I.feature) - set(A.feature))}")

# ensemble mean per basin; then across basins: bias, correlation, spread ratio
key = ["window", "region_id", "feature"]
mi = I.groupby(key).value.agg(["mean", "std"]).rename(columns=lambda c: "ifs_" + c)
ma = A.groupby(key).value.agg(["mean", "std"]).rename(columns=lambda c: "aifs_" + c)
M = mi.join(ma, how="inner").reset_index()
rows = []
for (w, fe), g in M.groupby(["window", "feature"]):
    g = g.dropna(subset=["ifs_mean", "aifs_mean"])
    if len(g) < 3:
        continue
    r = np.corrcoef(g.ifs_mean, g.aifs_mean)[0, 1] if g.ifs_mean.std() > 0 and g.aifs_mean.std() > 0 else np.nan
    rows.append({"window": w, "feature": fe, "n": len(g), "ifs": g.ifs_mean.mean(), "aifs": g.aifs_mean.mean(),
                 "bias": (g.aifs_mean - g.ifs_mean).mean(), "r_across_basins": r,
                 "spread_ratio": (g.aifs_std / g.ifs_std).median()})
R = pd.DataFrame(rows)
pd.set_option("display.width", 200)
for w in ("D1-10", "D11-20", "D21-30"):
    print(f"\n== {w} ==")
    print(R[R.window == w].drop(columns="window").round(3).to_string(index=False))
