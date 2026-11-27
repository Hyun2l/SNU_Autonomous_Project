"""지표 계산. 위반=positive, 항목 단위 채점(상담 N × 5항목).
  python evaluate.py --split test
산출: 난이도별 Recall(+상담 단위 부트스트랩 95% CI), 오탐률, Precision, 근거 정확도,
      인용 유효율, 반복 안정성, 합성/직접작성 비교, McNemar, 막대그래프."""
import argparse, json
import numpy as np, pandas as pd
from scipy.stats import binomtest
from codebook import ITEMS
from common import DATA, RUNS, RESULTS, read_jsonl


def gold_table(cs):
    return pd.DataFrame([dict(id=c["id"], source=c["source"], item=it, gold=int(c["labels"][it]["verdict"] == "위반"),
                              level=c["labels"][it]["level"], gturns=c["labels"][it]["turns"])
                         for c in cs for it in ITEMS])


def collect(files):
    recs = []
    for f in files:
        by = {}
        for r in read_jsonl(f):
            if r["pred"] is not None:
                by.setdefault((r["method"], r["id"]), []).append(r["pred"])
        for (m, cid), preds in by.items():
            for it in ITEMS:
                v = [p[it]["verdict"] == "위반" for p in preds]
                qs = [p[it].get("quote_ok") for p in preds if p[it]["verdict"] == "위반" and p[it].get("quote_ok") is not None]
                recs.append(dict(method=m, id=cid, item=it, pred=int(sum(v) * 2 > len(v)),
                                 agree=max(sum(v), len(v) - sum(v)) / len(v), n_reps=len(v),
                                 pturns=next((p[it]["turns"] for p in preds if p[it]["verdict"] == "위반"), []),
                                 qok=np.mean(qs) if qs else np.nan))
    return pd.DataFrame(recs)


def near(g, p):                        # 근거 발화 일치(±1턴 허용)
    return any(abs(a - b) <= 1 for a in g for b in p)


def boot_ci(df, mask, B=2000, seed=0):
    ids = df.id.unique(); ix = {i: k for k, i in enumerate(ids)}
    num, den = np.zeros(len(ids)), np.zeros(len(ids))
    for i, p in zip(df[mask].id, df[mask].pred):
        num[ix[i]] += p; den[ix[i]] += 1
    s = np.random.default_rng(seed).integers(0, len(ids), (B, len(ids)))
    d = den[s].sum(1)
    r = np.where(d > 0, num[s].sum(1) / np.maximum(d, 1), np.nan)
    return np.nanpercentile(r, [2.5, 97.5])


def summarize(d):
    row = {}
    for lv in (1, 2, 3):
        s = d[d.level == lv]
        row[f"recall_L{lv}"] = s.pred.mean() if len(s) else np.nan
        row[f"n_L{lv}"] = len(s)
    neg, pp = d[d.gold == 0], d[d.pred == 1]
    row["FPR"], row["precision"] = neg.pred.mean(), (pp.gold.mean() if len(pp) else np.nan)
    tp = d[(d.gold == 1) & (d.pred == 1) & (d.gturns.str.len() > 0)]
    row["evidence_hit"] = tp.apply(lambda r: near(r.gturns, r.pturns), axis=1).mean() if len(tp) else np.nan
    row["quote_valid"] = d[(d.pred == 1)].qok.mean()
    row["stability"] = d.agree.mean() if d.n_reps.max() > 1 else np.nan
    return row


def mcnemar(df, gold, a, b, level=None):
    w = df.pivot(index=["id", "item"], columns="method", values="pred").join(
        gold.set_index(["id", "item"])[["gold", "level"]]).dropna()
    if level:
        w = w[w.level == level]
    ca, cb = (w[a] == w.gold), (w[b] == w.gold)
    x, y = int((ca & ~cb).sum()), int((~ca & cb).sum())
    return x, y, (binomtest(x, x + y, .5).pvalue if x + y else 1.0)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--split", default="test")
    ap.add_argument("--gold", default="test")
    a = ap.parse_args()
    gold = gold_table(read_jsonl(DATA / f"{a.gold}.jsonl"))
    pr = collect(sorted(RUNS.glob(f"{a.split}__*.jsonl")))
    df = pr.merge(gold, on=["id", "item"])
    print(f"채점 행 {len(df)} (상담 {df.id.nunique()}건) | 방식: {sorted(df.method.unique())}")
    summ = {}
    for m, d in df.groupby("method"):
        summ[m] = summarize(d)
        for lv in (1, 2, 3):
            lo, hi = boot_ci(d, d.level == lv)
            summ[m][f"CI_L{lv}"] = f"[{lo:.2f},{hi:.2f}]"
        lo, hi = boot_ci(d, d.gold == 0)
        summ[m]["CI_FPR"] = f"[{lo:.2f},{hi:.2f}]"
    S = pd.DataFrame(summ).T
    print("\n== 방식별 요약 =="); print(S.round(3).to_string())
    S.to_csv(RESULTS / f"summary_{a.split}.csv")
    print("\n== 합성 vs 직접작성 (Recall: 난이도 2·3 / FPR) ==")
    for (m, src), d in df.groupby(["method", "source"]):
        r = d[d.level >= 2].pred.mean(); f = d[d.gold == 0].pred.mean()
        print(f"{m:16s} {src:10s} recall(L2-3)={r:.2f} (n={int((d.level>=2).sum())})  FPR={f:.2f}")
    ms = sorted(df.method.unique())
    print("\n== McNemar(정답/오답 짝지은 비교; b=앞 방식만 정답, c=뒤 방식만 정답) ==")
    for i in range(len(ms)):
        for j in range(i + 1, len(ms)):
            for lv in (None, 2, 3):
                b, c, p = mcnemar(df, gold, ms[i], ms[j], lv)
                print(f"{ms[i]} vs {ms[j]} | {'전체' if lv is None else f'난이도{lv} 위반'}: b={b} c={c} p={p:.3f}")
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        plt.rcParams["font.family"] = ["Malgun Gothic", "AppleGothic", "NanumGothic", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
        fig, ax = plt.subplots(figsize=(7, 4)); w = 0.8 / max(len(ms), 1)
        for k, m in enumerate(ms):
            d = df[df.method == m]; ys, errs = [], [[], []]
            for lv in (1, 2, 3):
                y = d[d.level == lv].pred.mean(); lo, hi = boot_ci(d, d.level == lv)
                ys.append(y); errs[0].append(max(0, y - lo)); errs[1].append(max(0, hi - y))
            ax.bar(np.arange(3) + k * w, ys, w, yerr=errs, capsize=3, label=m)
        ax.set_xticks(np.arange(3) + w * (len(ms) - 1) / 2); ax.set_xticklabels(["L1 누락", "L2 형식적", "L3 오인유발"])
        ax.set_ylabel("Recall"); ax.set_ylim(0, 1.05); ax.legend(); fig.tight_layout()
        fig.savefig(RESULTS / f"recall_by_level_{a.split}.png", dpi=160)
        print("그래프 저장:", RESULTS / f"recall_by_level_{a.split}.png")
    except Exception as e:
        print("그래프 생략:", e)


if __name__ == "__main__":
    main()
