// 상담 길잡이 AI 시연용 서버
//  - 공공데이터포털 복지서비스 API(중앙부처/지자체)를 호출해 복지서비스 정보를 가져온다.
//  - LLM으로 (1) 상담 내용에서 사건·가구 정보를 추출하고 (2) 후보 제도를 선별·우선순위화한다.
//  - 프런트(index.html)의 "서버 연결" 모드가 POST /api/analyze 를 호출한다.
//
// 확인 상태
//  - 중앙부처 목록조회(NationalWelfarelistV001): 실제 응답으로 필드명 확인 완료
//      servId, servNm, jurMnofNm, jurOrgNm, servDgst, lifeArray, trgterIndvdlArray,
//      intrsThemaArray, srvPvsnNm, sprtCycNm, rprsCtadr, servDtlLink, onapPsbltYn
//  - 중앙부처 상세조회(NationalWelfaredetailedV001): 필드명 미확인 → normalize 의 후보 필드로 추정.
//      /api/debug/central-detail?id=WLF00000024 로 실제 응답을 확인해 맞출 것.
//  - 지자체 API 주소: 활용가이드 확인 필요.

import 'dotenv/config';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import express from 'express';
import { XMLParser } from 'fast-xml-parser';
import Anthropic from '@anthropic-ai/sdk';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PORT = process.env.PORT || 3000;
const DATA_KEY = process.env.DATA_GO_KR_KEY;
const MODEL = process.env.CLAUDE_MODEL || 'claude-sonnet-5-5';

// ── 활용 API (한국사회보장정보원) ─────────────────────────────
// 중앙부처복지서비스: https://www.data.go.kr/data/15090532/openapi.do
const CENTRAL = {
  list:   'https://apis.data.go.kr/B554287/NationalWelfareInformationsV001/NationalWelfarelistV001',
  detail: 'https://apis.data.go.kr/B554287/NationalWelfareInformationsV001/NationalWelfaredetailedV001'
};
// 지자체복지서비스 (활용신청 후 활용가이드에서 주소 확인 필요)
const LOCAL = {
  list:   'https://apis.data.go.kr/B554287/LocalGovernmentWelfareInformations/LcgvWelfarelist',
  detail: 'https://apis.data.go.kr/B554287/LocalGovernmentWelfareInformations/LcgvWelfaredetailed'
};

const EVENT_IDS = ['job_loss', 'birth', 'housing', 'income_drop', 'burden', 'medical', 'elder', 'single_parent', 'disability'];
// 사건별 복지서비스 검색어
const EVENT_QUERY = {
  job_loss: '실업', birth: '출산', housing: '주거', income_drop: '긴급복지', burden: '저소득',
  medical: '의료비', elder: '노인 돌봄', single_parent: '한부모', disability: '장애인'
};
const NEED_KEYS = ['income', 'household', 'insurance', 'housing', 'childAge', 'elderAge', 'ltc', 'disabled', 'discharge'];

// 숫자처럼 보이는 값(문의처 등)이 숫자로 바뀌지 않게 문자열로 유지한다.
const xml = new XMLParser({ ignoreAttributes: true, trimValues: true, parseTagValue: false });
const llm = process.env.ANTHROPIC_API_KEY ? new Anthropic() : null;

// ── 공통 유틸 ────────────────────────────────────────────────
function httpError(status, message) {
  const e = new Error(message); e.status = status; return e;
}

const asArray = v => (v == null ? [] : Array.isArray(v) ? v : [v]);

// 값이 문자열·배열·중첩 객체 어느 쪽이든 읽을 수 있는 문자열로 만든다.
function flat(v) {
  if (v == null) return '';
  if (Array.isArray(v)) return v.map(flat).filter(Boolean).join(', ');
  if (typeof v === 'object') return Object.values(v).map(flat).filter(Boolean).join(' ');
  return String(v).trim();
}
const pick = (o, ...keys) => {
  for (const k of keys) {
    const s = flat(o && o[k]);
    if (s) return s;
  }
  return '';
};

// 간단한 메모리 캐시: 시연 중 같은 호출을 반복하지 않아 빠르고 호출 한도도 아낀다.
const cache = new Map();
const TTL_MS = 10 * 60 * 1000;
async function cached(key, fn) {
  const hit = cache.get(key);
  if (hit && Date.now() - hit.t < TTL_MS) return hit.v;
  const v = await fn();
  cache.set(key, { t: Date.now(), v });
  return v;
}

async function callApi(url, params) {
  if (!DATA_KEY) throw httpError(500, 'DATA_GO_KR_KEY 가 설정되지 않았습니다.');
  const qs = new URLSearchParams({ serviceKey: DATA_KEY, ...params });
  const res = await fetch(`${url}?${qs}`, { signal: AbortSignal.timeout(15000) });
  if (!res.ok) throw httpError(502, `공공데이터 API 오류 (HTTP ${res.status})`);
  const text = await res.text();
  const parsed = text.trim().startsWith('<') ? xml.parse(text) : JSON.parse(text);

  // 게이트웨이 오류 응답(인증키 미등록, 호출 한도 초과 등)
  const gw = parsed?.OpenAPI_ServiceResponse?.cmmMsgHeader;
  if (gw) throw httpError(502, `공공데이터 게이트웨이 오류: ${flat(gw.returnAuthMsg) || flat(gw.errMsg)} (${flat(gw.returnReasonCode)})`);
  const rc = parsed?.wantedList?.resultCode ?? parsed?.wantedDtl?.resultCode;
  if (rc != null && String(rc) !== '0') throw httpError(502, `복지서비스 API 오류: ${flat(parsed?.wantedList?.resultMessage || parsed?.wantedDtl?.resultMessage)}`);
  return parsed;
}

// ── 복지서비스 조회 ──────────────────────────────────────────
// 목록 응답: wantedList > servList[] (항목이 1개면 배열이 아니라 객체로 오므로 asArray 로 감싼다)
function findItems(parsed) {
  return asArray(parsed?.wantedList?.servList);
}

async function searchServices(api, keyword, rows = 5) {
  return cached(`list|${api.list}|${keyword}|${rows}`, async () => {
    const parsed = await callApi(api.list, {
      callTp: 'L', pageNo: '1', numOfRows: String(rows), srchKeyCode: '003', searchWrd: keyword
    });
    return findItems(parsed).map(it => ({
      servId: pick(it, 'servId'),
      name: pick(it, 'servNm'),
      ministry: pick(it, 'jurMnofNm'),
      org: pick(it, 'jurOrgNm'),
      digest: pick(it, 'servDgst'),
      life: pick(it, 'lifeArray'),
      targets: pick(it, 'trgterIndvdlArray'),
      themes: pick(it, 'intrsThemaArray'),
      provision: pick(it, 'srvPvsnNm'),
      cycle: pick(it, 'sprtCycNm'),
      contact: pick(it, 'rprsCtadr'),
      link: pick(it, 'servDtlLink')
    })).filter(x => x.servId);
  });
}

async function getDetail(api, servId) {
  return cached(`detail|${api.detail}|${servId}`, async () => {
    const parsed = await callApi(api.detail, { callTp: 'D', servId });
    return parsed?.wantedDtl || parsed;
  });
}

// 목록 + 상세 응답 → 프런트가 쓰는 제도 객체.
// 상세 필드가 비어 있어도 목록 정보만으로 카드가 만들어지도록 폴백을 둔다.
function normalize(base, d, source) {
  const detail = d && typeof d === 'object' ? d : {};
  const tags = [base.life, base.targets].filter(Boolean).join(' · ');
  return {
    id: base.servId,
    name: base.name,
    ministry: [base.ministry, base.org].filter(Boolean).join(' '),
    source,
    target: pick(detail, 'tgtrDtlCn', 'sprtTrgtCn') || [base.digest, tags && `(${tags})`].filter(Boolean).join(' '),
    criteria: pick(detail, 'slctCritCn', 'selectCritCn') || '상세 화면에서 선정기준을 확인하세요.',
    benefit: pick(detail, 'alwServCn', 'wlfareInfoOutlCn') || [base.provision, base.cycle].filter(Boolean).join(' · ') || base.digest,
    how: pick(detail, 'applmetList', 'aplyMtdCn', 'applmetCn') || '관할 주민센터 또는 복지로에서 확인',
    contact: pick(detail, 'inqplCtadrList') || base.contact,
    url: base.link,
    docs: [],
    need: []
  };
}

// ── LLM 호출 ─────────────────────────────────────────────────
async function askJson(system, user, maxTokens = 1200) {
  if (!llm) throw httpError(500, 'ANTHROPIC_API_KEY 가 설정되지 않았습니다.');
  const msg = await llm.messages.create({
    model: MODEL, max_tokens: maxTokens, system,
    messages: [{ role: 'user', content: user }]
  });
  const text = msg.content.filter(b => b.type === 'text').map(b => b.text).join('');
  const json = text.replace(/^```(?:json)?\s*|\s*```$/g, '').trim();
  return JSON.parse(json);
}

// (1) 상담 내용 → 사건·가구 정보
async function understand(text) {
  const out = await askJson(
    `당신은 복지 상담을 돕는 분석기다. 상담 내용에서 사건과 가구 정보를 추출해 JSON만 출력한다.
사건 id는 다음 중에서만 고른다: ${EVENT_IDS.join(', ')}.
없는 정보는 추측하지 말고 생략한다.
형식: {"events":[{"id":"...","hits":["상담 내용에서 근거가 된 짧은 표현"]}],
"facts":{"age":숫자,"household":숫자,"childMonths":숫자,"housingType":"월세|전세|자가","incomeMentioned":true|false}}`,
    text
  );
  const events = asArray(out.events).filter(e => EVENT_IDS.includes(e.id))
    .map(e => ({ id: e.id, hits: asArray(e.hits).slice(0, 2).map(String) }));
  return { events, facts: out.facts && typeof out.facts === 'object' ? out.facts : {} };
}

// (2) 후보 제도 선별·우선순위·추가 확인 질문 키
async function triage(text, events, candidates) {
  const brief = candidates.map(c => ({
    id: c.id, name: c.name, target: c.target.slice(0, 220), criteria: c.criteria.slice(0, 220)
  }));
  const out = await askJson(
    `상담 내용과 후보 복지서비스 목록을 보고, 이 상황에 실제로 도움이 될 만한 제도만 골라 JSON 배열로 출력한다.
반드시 후보 목록의 id만 사용하고, 목록에 없는 제도를 만들지 않는다.
phase: 1=바로 해야 할 긴급 제도, 2=이어서 신청할 제도, 3=보완 지원.
need: 자격 확인에 아직 필요한 정보 키 (${NEED_KEYS.join(', ')} 중에서만).
형식: [{"id":"...","phase":1|2|3,"matchedEvents":["사건 id"],"need":["키"]}]`,
    `상담 내용: ${text}\n인식된 사건: ${events.map(e => e.id).join(', ')}\n후보: ${JSON.stringify(brief)}`,
    2000
  );
  const byId = new Map(candidates.map(c => [c.id, c]));
  return asArray(out).filter(x => byId.has(x.id)).map(x => ({
    ...byId.get(x.id),
    phase: [1, 2, 3].includes(x.phase) ? x.phase : 2,
    matchedEvents: asArray(x.matchedEvents).filter(e => EVENT_IDS.includes(e)),
    need: asArray(x.need).filter(k => NEED_KEYS.includes(k))
  }));
}

// ── 서버 ─────────────────────────────────────────────────────
const app = express();
app.use(express.json({ limit: '20kb' }));

// 프런트만 제공한다. (server 폴더의 .env 가 노출되지 않도록 정적 폴더 공개는 하지 않는다)
app.get('/', (_req, res) => res.sendFile(path.join(__dirname, '..', 'index.html')));

app.post('/api/analyze', async (req, res, next) => {
  try {
    const text = String(req.body?.text || '').trim();
    if (!text) throw httpError(400, '상담 내용이 비어 있습니다.');
    if (text.length > 2000) throw httpError(400, '상담 내용은 2000자 이내로 입력해 주세요.');

    const { events, facts } = await understand(text);
    if (!events.length) return res.json({ events, facts, services: [] });

    // 사건별로 중앙부처·지자체 복지서비스 검색 → 상세 조회
    const searches = events.flatMap(e => [
      searchServices(CENTRAL, EVENT_QUERY[e.id], 4).then(r => r.map(x => ({ ...x, api: 'central' }))).catch(() => []),
      searchServices(LOCAL, EVENT_QUERY[e.id], 2).then(r => r.map(x => ({ ...x, api: 'local' }))).catch(() => [])
    ]);
    const found = new Map();
    (await Promise.all(searches)).flat().forEach(x => { if (!found.has(x.servId)) found.set(x.servId, x); });

    const picked = [...found.values()].slice(0, 16);
    const candidates = (await Promise.all(picked.map(async b => {
      const source = b.api === 'central' ? '중앙부처' : '지자체';
      try {
        const d = await getDetail(b.api === 'central' ? CENTRAL : LOCAL, b.servId);
        return normalize(b, d, source);
      } catch {
        return normalize(b, null, source); // 상세 조회가 실패해도 목록 정보로 카드는 만든다
      }
    })));

    const services = await triage(text, events, candidates);
    res.json({ events, facts, services });
  } catch (e) { next(e); }
});

// 응답 구조 확인용 디버그 엔드포인트
app.get('/api/debug/central-list', async (req, res, next) => {
  try {
    const q = String(req.query.q || '출산');
    res.json(await callApi(CENTRAL.list, { callTp: 'L', pageNo: '1', numOfRows: '2', srchKeyCode: '003', searchWrd: q }));
  } catch (e) { next(e); }
});
app.get('/api/debug/central-detail', async (req, res, next) => {
  try {
    const id = String(req.query.id || 'WLF00000024');
    res.json(await callApi(CENTRAL.detail, { callTp: 'D', servId: id }));
  } catch (e) { next(e); }
});

app.use((err, _req, res, _next) => {
  const status = err.status || 500;
  if (status >= 500) console.error('[error]', err.message);
  res.status(status).json({ error: err.message || '서버 오류' });
});

app.listen(PORT, () => console.log(`상담 길잡이 AI 서버: http://localhost:${PORT}`));
