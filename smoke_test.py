"""본 실험 전 5분 점검: 모델 ID, thinking 토글, 속도(tok/s), JSON 준수, 동시 2건 지연.
이 결과로 '호출 예산'을 정한다(특히 thinking on은 비쌀 수 있음)."""
import time, json
from concurrent.futures import ThreadPoolExecutor
from common import LLM, extract_json

llm = LLM()
print("모델:", llm.model)
Q = [{"role": "user", "content": '원금비보장 ELS를 한 문장으로 설명하고, 마지막 줄에 JSON {"ok": true} 만 출력하세요.'}]
for think in (False, True):
    t = time.time()
    text = llm.chat(Q, thinking=think, max_tokens=1500, tag=f"smoke-{think}-{t}")
    u = (llm.last or {}).get("usage") or {}
    ct = u.get("completion_tokens", 0); dt = time.time() - t
    print(f"thinking={think}: {dt:.1f}s, 출력 {ct}토큰 ≈ {ct/dt:.1f} tok/s, JSON 추출={extract_json(text)}, 종료={llm.last.get('finish')}")
    print("  응답 앞부분:", text[:120].replace("\n", " "))

def one(i):
    t = time.time(); llm.chat(Q, max_tokens=300, tag=f"conc-{i}-{t}"); return time.time() - t
t = time.time()
with ThreadPoolExecutor(2) as ex:
    lat = list(ex.map(one, range(4)))
print(f"동시 2건 x4: 개별 {['%.1f' % x for x in lat]}s, 총 {time.time()-t:.1f}s")
print("→ 총 호출 수 × (개별 지연 / 2) 로 전체 소요를 어림하세요.")
