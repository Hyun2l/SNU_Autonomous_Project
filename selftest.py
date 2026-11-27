"""오프라인 자체 점검(LLM 없이 가짜 응답): 파서·검증·판정·채점 코드가 돌아가는지 확인."""
import os, sys, json, random, tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(tempfile.mkdtemp())   # 실제 data/ runs/ 를 건드리지 않도록 임시 폴더에서 실행
os.environ.setdefault("LLM_BASE_URL", "http://localhost:1/v1"); os.environ.setdefault("LLM_API_KEY", "x")
os.environ.setdefault("LLM_MODEL", "fake")
import common, gen_data, judge, evaluate
from codebook import ITEMS, SPEC
from common import LLM, DATA, RUNS, read_jsonl, write_jsonl
import re, sys

def fake_dialogue(prompt):
    tags = re.findall(r"\[\[V:(\w+)\]\]", prompt)
    L = [f"{'상담사' if i % 2 == 0 else '고객'}: 발화 {i}번입니다 안녕하세요 네" for i in range(16)]
    for k, t in enumerate(tags):
        L[2 + 2 * k] += f" [[V:{t}]]"
    return "\n".join(L)

def fake_chat(self, messages, **kw):
    last = messages[-1]["content"]
    if "녹취록(연습용" in last:
        return fake_dialogue(last)
    random.seed(hash(last) % 1000)
    items = [{"key": k, "verdict": random.choice(["충족", "위반"]), "turns": [3], "quote": "발화 3번", "reason": "x"} for k in ITEMS]
    return "```json\n" + json.dumps({"items": items}, ensure_ascii=False) + "\n```"

LLM.chat = fake_chat
LLM.__init__ = lambda self, *a, **k: None
plan = {"dev": gen_data.make_plan("dev_s", 20, 4, 1), "test_syn": gen_data.make_plan("test_s", 40, 8, 2)}
llm = LLM()
for split in plan:
    rows = [gen_data.gen_one(llm, s) for s in plan[split]]
    assert all(rows), f"{split} 생성/검증 실패"
    write_jsonl(DATA / f"{split}.jsonl", rows)
write_jsonl(DATA / "test.jsonl", read_jsonl(DATA / "test_syn.jsonl"))
gen_data.cmd_noisy(None)
cnt = {1: 0, 2: 0, 3: 0}
for c in read_jsonl(DATA / "test.jsonl"):
    for L in c["labels"].values():
        if L["level"]: cnt[L["level"]] += 1
print("합성 test 난이도별 위반 수:", cnt)
sys.argv = ["judge"]
for m in ("m1", "m2", "m3"):
    sys.argv = ["judge", "--split", "test", "--method", m] + (["--reps", "3", "--temp", "0.3"] if m == "m3" else [])
    judge.main()
sys.argv = ["evaluate"]; evaluate.main()
print("SELFTEST OK")
