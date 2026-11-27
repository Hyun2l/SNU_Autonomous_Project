"""공통: 설정, Local LLM 클라이언트(레이트리밋·재시도·캐시·전량 로그), 유틸.
환경변수: LLM_BASE_URL, LLM_API_KEY (필수) / LLM_MODEL, LLM_RPM(기본 50), LLM_CONC(기본 2)
키는 코드·깃에 넣지 말고 .env 또는 셸 환경변수로만 관리할 것.
"""
import os, re, json, time, hashlib, threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from openai import (OpenAI, RateLimitError, APIConnectionError,
                    APITimeoutError, APIStatusError)

DATA, RUNS, RESULTS = Path("data"), Path("runs"), Path("results")
for _p in (DATA, RUNS, RESULTS):
    _p.mkdir(exist_ok=True)

RPM = int(os.environ.get("LLM_RPM", "50"))     # 한도(분당 60)보다 여유 있게
CONC = int(os.environ.get("LLM_CONC", "2"))    # 동시 2건 제한


class LLM:
    def __init__(self, log="runs/calls.jsonl"):
        self.client = OpenAI(base_url=os.environ["LLM_BASE_URL"],
                             api_key=os.environ["LLM_API_KEY"],
                             timeout=600, max_retries=0)
        self.model = os.environ.get("LLM_MODEL") or self.client.models.list().data[0].id
        self.log = Path(log)
        self.lock, self.next_t, self.interval = threading.Lock(), 0.0, 60.0 / RPM
        self.cache, self.last = {}, None
        if self.log.exists():
            for line in open(self.log, encoding="utf-8"):
                r = json.loads(line)
                self.cache[r["key"]] = r

    def _wait(self):                      # 요청 시작 간격을 RPM에 맞춰 직렬화
        with self.lock:
            now = time.monotonic()
            wait = max(0.0, self.next_t - now)
            self.next_t = max(now, self.next_t) + self.interval
        if wait:
            time.sleep(wait)

    def chat(self, messages, *, thinking=False, temperature=0.0,
             max_tokens=2048, tag=""):
        """tag: 같은 입력의 '반복 실험'을 캐시와 구분하기 위한 꼬리표(rep0, rep1...)."""
        key = hashlib.sha256(json.dumps(
            [self.model, messages, thinking, temperature, max_tokens, tag],
            ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        if key in self.cache:
            self.last = self.cache[key]
            return self.cache[key]["text"]
        r = None
        for attempt in range(6):
            self._wait()
            t0 = time.time()
            try:
                r = self.client.chat.completions.create(
                    model=self.model, messages=messages, temperature=temperature,
                    max_tokens=max_tokens,
                    extra_body={"chat_template_kwargs": {"enable_thinking": thinking}})
                break
            except (RateLimitError, APIConnectionError, APITimeoutError):
                time.sleep(min(60, 3 * 2 ** attempt))
            except APIStatusError as e:
                if e.status_code < 500:
                    raise
                time.sleep(min(60, 3 * 2 ** attempt))
        if r is None:
            raise RuntimeError("LLM 호출 6회 연속 실패")
        text = re.sub(r"<think>.*?</think>", "", r.choices[0].message.content or "",
                      flags=re.S).strip()
        rec = {"key": key, "model": self.model, "tag": tag, "thinking": thinking,
               "temperature": temperature, "messages": messages, "text": text,
               "finish": r.choices[0].finish_reason, "sec": round(time.time() - t0, 2),
               "usage": r.usage.model_dump() if r.usage else None}
        with self.lock:
            self.cache[key] = rec
            self.last = rec
            with open(self.log, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return text


def pmap(fn, items, desc=""):
    with ThreadPoolExecutor(CONC) as ex:
        return list(tqdm(ex.map(fn, items), total=len(items), desc=desc))


def extract_json(text):
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def read_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def write_jsonl(p, rows):
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
