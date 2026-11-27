"""세 방식 실행: m1(정규식) / m2(LLM 기본) / m3(LLM 강화) + ablation(m3_noshot, m3_noevid).
예)  python judge.py --split test --method m1
     python judge.py --split test --method m2
     python judge.py --split test --method m3 --reps 3 --temp 0.3
     python judge.py --split test --method m3 --think          # thinking 켜기(느림)
결과: runs/<split>__<방식>.jsonl  (원문 호출 로그는 runs/calls.jsonl)"""
import argparse, json, re, time
from codebook import ITEMS, KO, SPEC, codebook_text
from common import LLM, pmap, extract_json, DATA, RUNS, read_jsonl, write_jsonl

# ---------- M1: 정규식 (dev로 다듬되, 부정 표현 처리까지 최선을 다한 버전) ----------
RX = {
 "suitability": [r"투자\s?성향", r"투자\s?(경험|목적|기간)", r"위험\s?(선호|감수|성향)", r"손실\s?(감내|감수)"],
 "loss_notice": [r"원금[^.?!]{0,12}(손실|손해|보장되지|보장이 안|깨질)",
                 r"(손실|손해)[^.?!]{0,10}(날 수|발생|가능성|있을 수)"],
 "fee": [r"수수료", r"선취|후취|운용\s?보수"],
 "understanding": [r"이해(하|되|가|셨|했)", r"궁금(한|하신)", r"질문", r"혹시\s?모르"],
}
NEG_LOSS = r"(손실|손해)[^.?!]{0,10}(없|안 ?나|거의|걱정)|원금[^.?!]{0,8}(보장돼|보장됩니다|지켜|안전)|(사실상|거의)\s?보장"
NEG_FEE = r"수수료[^.?!]{0,12}(없|거의|사실상|별로|미미)"
ASSERT = [r"무조건", r"100\s?%|백\s?프로", r"절대", r"확실(히|하게)", r"걱정\s?(마|하지\s?마|안\s?하)",
          r"(안|한\s?번도)\s?(잃|빠|떨어|깨)", r"(손실|손해)[^.?!]{0,8}(없|안 ?나)",
          r"원금[^.?!]{0,6}(지켜|보장(돼|됩니다|해))"]


def regex_predict(c):
    adv = [t for t in c["turns"] if t["spk"] == "상담사"]
    txt = {t["n"]: t["text"] for t in c["turns"]}
    cust = {t["n"] for t in c["turns"] if t["spk"] == "고객" and len(t["text"]) >= 6}
    hit = lambda pats: [t["n"] for t in adv if any(re.search(p, t["text"]) for p in pats)]
    out = {}
    for it in ["suitability", "loss_notice", "fee", "understanding"]:
        ns = hit(RX[it])
        if it == "loss_notice":
            ns = [n for n in ns if not re.search(NEG_LOSS, txt[n])]
        if it == "fee":
            ns = [n for n in ns if not re.search(NEG_FEE, txt[n])]
        if it == "understanding":   # 확인 질문 다음 고객 발화가 어느 정도 길어야 인정(정규식이 낼 수 있는 최선)
            ns = [n for n in ns if (n + 1) in cust]
        out[it] = {"verdict": "충족" if ns else "위반", "turns": ns[:2]}
    bad = hit(ASSERT)
    out["assertive"] = {"verdict": "위반" if bad else "충족", "turns": bad[:2]}
    return out


# ---------- M2/M3: LLM ----------
SYS = ("당신은 금융상품 판매 상담 녹취록을 점검하는 준법 검토자입니다. "
       "법적 결론을 내리는 것이 아니라, 사람이 다시 봐야 할 상담을 가려내는 보조 역할입니다.")
RULES = """1. 형식만 갖춘 경우(고객 답을 듣지 않음, 빠르게 읽고 서명 유도, 답을 강요하는 확인)는 해당 항목 위반이다.
2. 먼저 충족했어도 바로 뒤에서 뒤집으면(예: '사실상 보장') 위반이다.
3. 단정적 표현 금지(assertive)는 단정 표현이 '있으면' 위반이다. 과거 수익률을 미래 안전의 근거처럼 말하는 것도 해당한다.
4. 위반 판정일 때만 근거가 된 발화 번호(turns)와 원문 그대로의 인용(quote, 20자 이내)을 적는다. 해당 내용이 아예 없어서 위반이면 turns는 빈 배열. 충족이면 turns는 빈 배열.
5. 녹취록에 없는 말을 지어내지 않는다. 발화 번호는 녹취록의 [번호]를 그대로 쓴다."""
SCHEMA_EV = '{"items":[{"key":"항목키","verdict":"충족|위반","turns":[번호],"quote":"원문 인용","reason":"한 줄"}, ... 5개]}'
SCHEMA_NO = '{"items":[{"key":"항목키","verdict":"충족|위반","reason":"한 줄"}, ... 5개]}'
ITEM_LINE = "항목: " + ", ".join(f"{k}({KO[k]})" for k in ITEMS)

METHODS = {
 "m2":        dict(codebook=False, shots=0, evidence=False),
 "m3":        dict(codebook=True,  shots=3, evidence=True),
 "m3_noshot": dict(codebook=True,  shots=0, evidence=True),
 "m3_noevid": dict(codebook=True,  shots=3, evidence=False),
}


def render(c):
    return "\n".join(f"[{t['n']}] {t['spk']}: {t['text']}" for t in c["turns"])


def user_prompt(c, schema):
    return (f"[녹취록]\n{render(c)}\n\n{ITEM_LINE}\n각 항목을 '충족' 또는 '위반'으로 판정하고 "
            f"아래 형식의 JSON만 출력하세요.\n{schema}")


def shot_answer(c, evidence):
    by, items = {t["n"]: t for t in c["turns"]}, []
    for it in ITEMS:
        L = c["labels"][it]
        d = {"key": it, "verdict": L["verdict"]}
        if evidence:
            d["turns"] = L["turns"]
            d["quote"] = by[L["turns"][0]]["text"][:20] if L["turns"] else ""
        d["reason"] = SPEC[it][L["level"]] if L["verdict"] == "위반" else "기준 충족"
        items.append(d)
    return json.dumps({"items": items}, ensure_ascii=False)


def pick_shots(dev, k, ids=""):
    if k == 0:
        return []
    if ids:
        m = {c["id"]: c for c in dev}
        return [m[i] for i in ids.split(",")][:k]
    lv = lambda c: {L["level"] for L in c["labels"].values()}
    normal = next(c for c in dev if lv(c) == {0})
    l2 = next(c for c in dev if 2 in lv(c))
    l3 = next(c for c in dev if 3 in lv(c) and c is not l2)
    return [l2, l3, normal][:k]


def build_messages(c, cfg, shots):
    ev = cfg["evidence"]
    sys = SYS + (f"\n\n[점검 기준서]\n{codebook_text()}\n\n[판정 원칙]\n{RULES}" if cfg["codebook"] else "")
    schema = SCHEMA_EV if ev else SCHEMA_NO
    msgs = [{"role": "system", "content": sys}]
    for s in shots:
        msgs += [{"role": "user", "content": user_prompt(s, schema)},
                 {"role": "assistant", "content": shot_answer(s, ev)}]
    msgs.append({"role": "user", "content": user_prompt(c, schema)})
    return msgs


def parse_pred(text, by):
    js = extract_json(text)
    if not js or not isinstance(js.get("items"), list):
        return None
    out = {}
    for r in js["items"]:
        k, v = r.get("key"), r.get("verdict")
        if k not in ITEMS or v not in ("충족", "위반"):
            continue
        ts = [x for x in r.get("turns", []) if isinstance(x, int) and x in by]
        q = (r.get("quote") or "").replace(" ", "")
        qok = None
        if v == "위반" and ts and q:
            qok = any(q in by[x]["text"].replace(" ", "") for x in ts)   # 인용이 실제 원문에 있는가
        out[k] = {"verdict": v, "turns": ts, "quote_ok": qok}
    return out if set(out) == set(ITEMS) else None


def run_one(llm, c, rep, cfg, shots, a, name):
    t0, by = time.time(), {t["n"]: t for t in c["turns"]}
    msgs, pred, text = build_messages(c, cfg, shots), None, ""
    for k in range(2):
        m = msgs if k == 0 else msgs[:-1] + [{**msgs[-1], "content": msgs[-1]["content"] + "\n(JSON 객체 하나만 출력)"}]
        text = llm.chat(m, thinking=a.think, temperature=a.temp,
                        max_tokens=6000 if a.think else 1500, tag=f"rep{rep}-{k}")
        pred = parse_pred(text, by)
        if pred:
            break
    return {"id": c["id"], "method": name, "rep": rep, "pred": pred,
            "raw": None if pred else text, "sec": round(time.time() - t0, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test"); ap.add_argument("--method", default="m3")
    ap.add_argument("--think", action="store_true"); ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--reps", type=int, default=1); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shots", default="", help="dev id를 쉼표로(미지정 시 자동 선택)")
    a = ap.parse_args()
    cs = read_jsonl(DATA / f"{a.split}.jsonl")
    cs = cs[:a.limit] if a.limit else cs
    name = a.method + ("+think" if a.think else "") + (f"@T{a.temp}" if a.temp else "")
    if a.method == "m1":
        rows = [{"id": c["id"], "method": name, "rep": 0, "pred": regex_predict(c)} for c in cs]
    else:
        llm, cfg = LLM(), METHODS[a.method]
        shots = pick_shots(read_jsonl(DATA / "dev.jsonl"), cfg["shots"], a.shots)
        print("few-shot 예시(dev):", [s["id"] for s in shots])
        rows = pmap(lambda j: run_one(llm, j[0], j[1], cfg, shots, a, name),
                    [(c, r) for c in cs for r in range(a.reps)], desc=name)
    write_jsonl(RUNS / f"{a.split}__{name}.jsonl", rows)
    print("저장:", RUNS / f"{a.split}__{name}.jsonl", "| 파싱 실패:", sum(r["pred"] is None for r in rows))


if __name__ == "__main__":
    main()
