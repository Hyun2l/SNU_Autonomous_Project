"""데이터 계획·생성·검수. 사용 순서:
  python gen_data.py plan          # 어떤 상담에 어떤 위반을 심을지 확정 (+사람 작성 과제표)
  python gen_data.py generate      # dev/test_syn 합성 (LLM)
  python gen_data.py ingest_human  # data/human/test_hNN.txt 읽어 test.jsonl 완성
  python gen_data.py review        # 블라인드 검수 시트 내보내기
  python gen_data.py kappa a.csv b.csv   # 라벨러 간 일치도 + 심은 라벨과의 일치도
  python gen_data.py noisy         # STT 잡음 버전 test_noisy.jsonl
형식 규칙: 한 줄에 '상담사: ...' 또는 '고객: ...'. 위반 발화(난이도 2~3) 줄 끝에 [[V:항목키]].
난이도1(누락)은 '없는 것'이므로 태그를 달지 않는다."""
import argparse, itertools, json, random, re, csv, sys
from pathlib import Path
from codebook import ITEMS, KO, SPEC
from common import LLM, pmap, DATA, read_jsonl, write_jsonl

PRODUCTS = ["원금비보장 ELS (주가지수 연계, 3년 만기)", "해외 주식형 펀드",
            "채권혼합형 펀드", "금 연계 ETN"]
CUSTOMERS = ["60대 은퇴자, 예금 위주로만 해봄, 노후자금 일부 투자 희망",
             "30대 직장인, 주식 경험 약간, 목돈 굴리고 싶음",
             "40대 자영업자, 바빠서 서두름, 투자 경험 거의 없음",
             "50대 주부, 보수적, 만기 된 예금 재투자 고민",
             "20대 사회초년생, 유튜브 보고 방문"]
LINE = re.compile(r"^\s*(상담사|고객)\s*[:：]\s*(.+?)\s*$")
TAG = re.compile(r"\s*\[\[V:(\w+)\]\]")


def make_plan(prefix, n, n_normal, seed):
    rng, lv = random.Random(seed), itertools.cycle([1, 2, 3])
    flags = [False] * n_normal + [True] * (n - n_normal)
    rng.shuffle(flags)
    specs = []
    for i, has in enumerate(flags, 1):
        viol = {}
        if has:
            for it in rng.sample(ITEMS, rng.choice([1, 2])):
                viol[it] = next(lv)          # 난이도를 돌려가며 배정 → 균형
        specs.append({"id": f"{prefix}{i:02d}", "product": rng.choice(PRODUCTS),
                      "customer": rng.choice(CUSTOMERS), "viol": viol})
    return specs


def parse_dialogue(text):
    turns, tags = [], {}
    for raw in text.replace("*", "").splitlines():
        m = LINE.match(raw)
        if not m:
            continue
        spk, body = m.groups()
        n = len(turns) + 1
        for t in TAG.findall(body):
            tags.setdefault(t, []).append(n)
        turns.append({"n": n, "spk": spk, "text": TAG.sub("", body).strip()})
    return turns, tags


def validate(spec, turns, tags):
    want = {k for k, lv in spec["viol"].items() if lv >= 2}
    if not (12 <= len(turns) <= 45) or set(tags) != want:
        return False
    return all(turns[n - 1]["spk"] == "상담사" for ns in tags.values() for n in ns)


def labels_from(spec, tags):
    out = {}
    for it in ITEMS:
        if it in spec["viol"]:
            out[it] = {"verdict": "위반", "level": spec["viol"][it], "turns": tags.get(it, [])}
        else:
            out[it] = {"verdict": "충족", "level": 0, "turns": []}
    return out


def build_prompt(spec):
    L = []
    for it in ITEMS:
        if it in spec["viol"]:
            lv = spec["viol"][it]
            tail = (f" 위반이 드러나는 상담사 발화 줄 맨 끝에 [[V:{it}]] 를 붙인다."
                    if lv >= 2 else " (아예 다루지 않는 것이 위반이므로 표시 없음)")
            L.append(f"- {KO[it]} [위반, 난이도{lv}]: {SPEC[it][lv]}{tail}")
        else:
            L.append(f"- {KO[it]} [충족]: {SPEC[it]['ok']} 짧고 자연스럽게 충족시킨다.")
    return f"""금융회사 창구/전화 상담 녹취록(연습용 가상 데이터)을 작성하세요.
상품: {spec['product']}
고객: {spec['customer']}
항목별 지시:
{chr(10).join(L)}

규칙:
- 15~30턴, 실제 구어체(맞장구, 끊김, 짧은 대답 포함). 이름·연락처는 가상으로.
- 한 줄에 한 발화, 형식은 정확히 '상담사: ...' 또는 '고객: ...'. 그 외 설명·제목·마크다운 금지.
- 상담사가 '규정상 해야 하지만'처럼 스스로 위반임을 드러내는 말, '위반입니다' 같은 광고성 문장 금지. 위반은 평범한 영업 멘트처럼 보여야 한다.
- 난이도3은 겉보기에 문제없어 보이되 의미상 고객을 오인시키도록 쓴다.
- [위반] 항목이 아닌 항목은 위반으로 읽힐 여지가 없게 한다."""


def gen_one(llm, spec, tries=3):
    for k in range(tries):
        text = llm.chat([{"role": "user", "content": build_prompt(spec)}],
                        temperature=0.8, max_tokens=4000, tag=f"gen{k}")
        turns, tags = parse_dialogue(text)
        if validate(spec, turns, tags):
            return {"id": spec["id"], "source": "synthetic", "product": spec["product"],
                    "customer": spec["customer"], "turns": turns,
                    "labels": labels_from(spec, tags)}
    return None


def cmd_plan(_):
    plan = {"dev": make_plan("dev_s", 20, 4, 1),
            "test_syn": make_plan("test_s", 40, 8, 2),
            "human": make_plan("test_h", 10, 2, 3)}
    json.dump(plan, open(DATA / "plan.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    cnt = {1: 0, 2: 0, 3: 0}
    for k in ("test_syn", "human"):
        for s in plan[k]:
            for lv in s["viol"].values():
                cnt[lv] += 1
    print("test 위반 개수(난이도별):", cnt)
    md = ["# 사람 작성 과제표 (각자 data/human/<id>.txt 로 저장)\n"]
    for s in plan["human"]:
        v = ", ".join(f"{KO[k]}(난이도{lv})" for k, lv in s["viol"].items()) or "위반 없음(정상 상담)"
        md.append(f"- **{s['id']}** | {s['product']} | {s['customer']} | {v}")
    (DATA / "human_assignments.md").write_text("\n".join(md), encoding="utf-8")


def cmd_generate(_):
    llm, plan = LLM(), json.load(open(DATA / "plan.json", encoding="utf-8"))
    for split in ("dev", "test_syn"):
        res = pmap(lambda s: gen_one(llm, s), plan[split], desc=split)
        ok = [r for r in res if r]
        print(f"{split}: {len(ok)}/{len(plan[split])} 성공, 실패 id:",
              [s["id"] for s, r in zip(plan[split], res) if not r])
        write_jsonl(DATA / f"{split}.jsonl", ok)


def cmd_ingest_human(_):
    plan, out = json.load(open(DATA / "plan.json", encoding="utf-8")), []
    for spec in plan["human"]:
        p = DATA / "human" / f"{spec['id']}.txt"
        if not p.exists():
            print("없음:", p); continue
        turns, tags = parse_dialogue(p.read_text(encoding="utf-8"))
        if not validate(spec, turns, tags):
            print("검증 실패(턴 수/태그 확인):", spec["id"]); continue
        out.append({"id": spec["id"], "source": "human", "product": spec["product"],
                    "customer": spec["customer"], "turns": turns, "labels": labels_from(spec, tags)})
    write_jsonl(DATA / "test_human.jsonl", out)
    syn = read_jsonl(DATA / "test_syn.jsonl")
    write_jsonl(DATA / "test.jsonl", syn + out)
    print(f"test.jsonl = 합성 {len(syn)} + 직접작성 {len(out)}")


def cmd_review(_):
    d = DATA / "review"; d.mkdir(exist_ok=True)
    rows = []
    for c in read_jsonl(DATA / "test.jsonl"):
        (d / f"{c['id']}.txt").write_text(
            "\n".join(f"[{t['n']}] {t['spk']}: {t['text']}" for t in c["turns"]), encoding="utf-8")
        rows += [[c["id"], it, "", ""] for it in ITEMS]
    with open(d / "blind_sheet.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(["id", "item", "annotator_a", "annotator_b"]); w.writerows(rows)
    print("라벨러는 '충족' 또는 '위반'만 기입. 심은 라벨을 보지 않는 블라인드 방식.")


def kappa(x, y):
    n = len(x); po = sum(a == b for a, b in zip(x, y)) / n
    pe = sum((x.count(c) / n) * (y.count(c) / n) for c in set(x) | set(y))
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def cmd_kappa(a):
    ra = {(r["id"], r["item"]): r["annotator_a"] for r in csv.DictReader(open(a.files[0], encoding="utf-8-sig"))}
    rb = {(r["id"], r["item"]): r["annotator_b"] for r in csv.DictReader(open(a.files[1], encoding="utf-8-sig"))}
    gold = {(c["id"], it): c["labels"][it]["verdict"]
            for c in read_jsonl(DATA / "test.jsonl") for it in ITEMS}
    keys = [k for k in ra if ra[k] and rb.get(k)]
    print("A-B kappa:", round(kappa([ra[k] for k in keys], [rb[k] for k in keys]), 3))
    print("A-심은라벨 kappa:", round(kappa([ra[k] for k in keys], [gold[k] for k in keys]), 3))
    print("B-심은라벨 kappa:", round(kappa([rb[k] for k in keys], [gold[k] for k in keys]), 3))
    bad = [k for k in keys if ra[k] != gold[k] or rb[k] != gold[k]]
    print("라벨 재검토 필요(사람≠심은라벨):", bad[:30])


def stt_noise(text, rng, allow_drop):
    text = re.sub(r"[.,?!~]", "", text)
    out = []
    for w in text.split(" "):
        if allow_drop and rng.random() < 0.03:
            continue
        out.append(w)
        if rng.random() < 0.04:
            out.append(rng.choice(["어", "음", "그"]))
    s = out[0] if out else ""
    for w in out[1:]:
        s += ("" if rng.random() < 0.03 else " ") + w
    return s


def cmd_noisy(_):
    rng, out = random.Random(7), []
    for c in read_jsonl(DATA / "test.jsonl"):
        ev = {n for L in c["labels"].values() for n in L["turns"]}
        t2 = [{**t, "text": stt_noise(t["text"], rng, t["n"] not in ev)} for t in c["turns"]]
        out.append({**c, "id": c["id"], "turns": t2, "source": c["source"]})
    write_jsonl(DATA / "test_noisy.jsonl", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd"); ap.add_argument("files", nargs="*")
    a = ap.parse_args()
    {"plan": cmd_plan, "generate": cmd_generate, "ingest_human": cmd_ingest_human,
     "review": cmd_review, "kappa": cmd_kappa, "noisy": cmd_noisy}[a.cmd](a)
