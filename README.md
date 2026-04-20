# FreshMart Conversational AI System
**CS 4063 – Natural Language Processing | Assignment 4 – RAG and Tools in Conversational AI**
**Group Members:** [Add your names here]

FreshMart is a fully local, production-style conversational AI system built around **Shabo** — a virtual grocery assistant that helps customers browse products, manage their cart, retrieve store knowledge via RAG, and invoke four external tools (CRM, weather, currency conversion, calculator). The system runs entirely on CPU using a quantized LLM served via Ollama, with a FastAPI WebSocket backend, real-time token streaming, a voice pipeline (ASR + TTS), and a ChatGPT-style web interface.

---

## Business Use Case

FreshMart is an online grocery store. This assignment extends the previous voice chatbot with Retrieval-Augmented Generation (RAG), a CRM system, and three additional tools, transforming Shabo from a scripted dialogue agent into an intelligent assistant grounded in real store knowledge and capable of performing real-world actions.

**RAG** eliminates hard-coded store knowledge. Instead of stuffing every policy into the system prompt, the assistant embeds the user's query, retrieves the top-3 most relevant passages from 65 FreshMart documents at query time, and grounds its response in that retrieved content. A customer asking about allergen information, refund eligibility, or delivery timing gets an answer drawn directly from the actual store documents — not a hallucination.

**CRM** enables personalised service across sessions. Returning customers are greeted by name. The assistant recalls past preferences and interaction history from a persistent SQLite database, allowing genuine continuity across separate conversations rather than stateless session isolation.

**Tools** give the assistant agency beyond text generation. Live weather data helps customers decide whether to add cold-storage items to their basket. Currency conversion lets international customers see cart totals in their local currency. The calculator handles precise multi-item price arithmetic without relying on the LLM's unreliable arithmetic.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         CLIENT LAYER                            │
│  index.html  (ChatGPT-style Web UI)                             │
│  - Real-time token streaming via WebSocket                      │
│  - Live cart sidebar  |  Voice input/output  |  Session mgmt   │
└────────────────────────────┬────────────────────────────────────┘
                             │  WebSocket  ws://localhost:8000/ws/chat/{id}
                             │  WebSocket  ws://localhost:8000/ws/voice/{id}
                             │  REST       http://localhost:8000
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                     BACKEND LAYER  (main.py)                    │
│  FastAPI + Uvicorn                                              │
│  - /ws/chat/{session_id}   — text streaming WebSocket           │
│  - /ws/voice/{session_id}  — voice pipeline WebSocket           │
│  - REST: /health  /session/new  /session/{id}/state             │
│          /session/reset/{id}    DELETE /session/{id}            │
│  - Thread-safe queue bridges sync Ollama stream → async WS      │
│  - Voice model pre-warming in background thread at startup      │
└────────────────────────────┬────────────────────────────────────┘
                             │
              ┌──────────────┴──────────────┐
              ▼                             ▼
┌─────────────────────────┐   ┌────────────────────────────────┐
│  ConversationManager    │   │  VoiceManager                  │
│  (conversation_manager) │   │  - ASREngine (faster-whisper   │
│                         │   │    tiny.en, int8 quantized)    │
│  Per-turn pipeline:     │   │  - TTSEngine (Piper            │
│  1. IntentParser        │   │    en_US-lessac-medium.onnx)   │
│     (cart add/remove)   │   │  - Sentence-level parallel     │
│  2. CRM check           │   │    token display + audio sync  │
│  3. RAG retrieval       │   └────────────────────────────────┘
│  4. Keyword tool detect │
│  5. Build full prompt   │
│  6. Stream LLM tokens   │
│                         │
│  ┌─────────────────────┐│
│  │  MemoryManager      ││
│  │  Rolling 6-msg win  ││
│  │  LLM-based summary  ││
│  │  overflow handling  ││
│  └─────────────────────┘│
│                         │
│  ┌─────────────────────┐│
│  │  CartManager        ││
│  │  Item/qty tracking  ││
│  │  4-promotion engine ││
│  └─────────────────────┘│
└────────────┬────────────┘
             │
    ┌────────┴─────────┐
    ▼                  ▼
┌──────────────┐  ┌──────────────────────────────────────────────┐
│  RAG Module  │  │  Tool Orchestrator (tool_orchestrator.py)    │
│              │  │                                              │
│rag_indexer.py│  │  Tool Registry (7 callables):                │
│ - Loads 65   │  │    CRM:  get_user_info          (sync)        │
│   .txt docs  │  │          update_user_info       (sync)        │
│ - 400-word   │  │          store_interaction      (sync)        │
│   chunks,    │  │          get_interaction_history(sync)        │
│   50-word    │  │    Tools: get_weather           (async, httpx)│
│   overlap    │  │           convert_currency      (async, httpx)│
│ - Embeds via │  │           calculate             (sync, local) │
│   MiniLM     │  │                                              │
│ - Stores in  │  │  Detection: <tool_call>…</tool_call> regex   │
│   ChromaDB   │  │  Execution: asyncio.wait_for, 8 s timeout    │
│              │  │  Sync tools: run_in_executor (non-blocking)  │
│retrieval_    │  └──────────────────────────────────────────────┘
│module.py     │
│ - Lazy-load  │
│   singleton  │
│ - MD5 query  │
│   cache      │
│ - top-k=3,   │
│   score ≥0.2 │
└──────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────────────┐
│                        LLM ENGINE                               │
│  Ollama  |  qwen2.5:1.5b  (Q4_K_M quantized, ~900 MB RAM)      │
│  CPU-only inference  |  Streaming via /api/chat                 │
│  num_predict=350  |  temperature=0.7                            │
└─────────────────────────────────────────────────────────────────┘
```

**Component descriptions:**

- **main.py** — FastAPI application entry point. Maintains two separate session dictionaries (text and voice). The text WebSocket bridges synchronous Ollama streaming to the async event loop via a thread-safe `queue.Queue`, emitting one `{"type": "token"}` JSON message per token and a final `{"type": "done"}` message with cart state. The voice WebSocket feeds raw browser WebM audio into the voice pipeline and returns interleaved transcript, token, and base64-encoded WAV audio events. Voice models are pre-warmed in a daemon thread at startup and shared across sessions to avoid repeated cold-start loading.

- **conversation_manager.py** — the central orchestration unit for each session. On every user turn it: (1) runs IntentParser for immediate cart mutation without an LLM call, (2) checks the CRM for a returning user's name via direct SQLite access, (3) calls `retrieve()` for RAG context, (4) runs keyword-based pre-LLM tool dispatch for weather/currency/CRM, (5) builds a composite system prompt from `SYSTEM_PROMPT + CRM name + RAG chunks + tool result`, and (6) streams tokens from Ollama's `/api/chat` endpoint. Session end is detected by matching user messages against a list of goodbye signals, at which point the interaction summary is written to CRM and `is_active` is set to `False`.

- **tool_orchestrator.py** — independent tool registry and execution engine. Registers all 7 tool callables with their async/sync type. Provides `extract_tool_call()` via regex on `<tool_call>…</tool_call>` blocks, `execute_tool()` with `asyncio.wait_for` timeout protection (sync tools run in `loop.run_in_executor` inside the same timeout wrapper), and `format_tool_schemas_for_prompt()` which serialises all JSON schemas into a readable block the LLM can reference. Note: most tool invocations are handled by the faster keyword-shortcut path in `conversation_manager.py`; the orchestrator handles the remaining LLM-driven `<tool_call>` cases and serves as the formal tool registry.

- **retrieval_module.py** — lazy-loads the ChromaDB collection and `all-MiniLM-L6-v2` embedding model once at module level on first access. `retrieve()` computes an MD5 hash of the lowercased query and returns cached results for repeated queries. Cosine distances are converted to similarity scores (1 − distance); chunks below the 0.2 threshold are filtered out. `format_context()` wraps retrieved chunks in a `[RETRIEVED KNOWLEDGE BASE CONTEXT]` block that is injected into the system prompt.

- **rag_indexer.py** — offline indexing pipeline. Reads all `.txt` files from `documents/`, splits each into 400-word chunks with 50-word overlap via `chunk_text()`, batches embeddings into ChromaDB in groups of 50, and stores the index persistently. The collection is fully dropped and rebuilt on every run to ensure clean re-indexing after document updates.

- **crm_tool.py** — SQLite-backed CRM with two tables: `users` (profile fields keyed by `user_id`) and `interactions` (timestamped session summaries with a foreign key to `users`). Auto-initialises the database on import. Exposes four tool functions — `get_user_info`, `update_user_info`, `store_interaction`, `get_interaction_history` — each with a full JSON schema for LLM-driven invocation. Preferences are stored as JSON within the `preferences TEXT` column with merge-update semantics.

- **system_prompt.py** — defines the Shabo persona, strict domain refusal rules (refuses all non-grocery questions), formatting constraints (no markdown, no asterisks, no LaTeX), cart rules (always quote the total from the injected cart state block, never self-calculate), tool result rules, and the full 48-item product catalog across 6 categories with current promotions.

- **cart_manager.py** — tracks items and quantities in memory per session. Recalculates promotions on every mutation: 10% off all Fruits, buy-2-get-1 free on Bakery items, 15% off orders above $30, free delivery on orders above $25. `to_context_string()` serialises the full cart state (items, subtotal, discount, delivery fee, total, active promotions) into a compact block injected into every LLM prompt via `MemoryManager.build_messages()`.

- **memory_manager.py** — implements a rolling 6-message active window. When the window exceeds the limit, the two oldest messages are popped and passed to a second Ollama API call (`_llm_summarize`) which merges them into a running `summary_block`. This block is prepended to every subsequent prompt as `[EARLIER CONVERSATION SUMMARY]`, giving the LLM persistent context without unbounded context growth. Falls back to plain text concatenation if the summariser call fails.

- **intent_parser.py** — keyword-based cart intent detection that runs before the LLM on every turn. Matches `ADD_KEYWORDS` and `REMOVE_KEYWORDS` against the lowercased user message, finds product names from the 48-item `CATALOG` dict (sorted longest-name-first to prevent partial substring matches), extracts integer quantities via `\b(\d+)\b` regex, and calls `CartManager.add_item()` or `remove_item()` directly. Handles multiple items in one message by splitting on `\band\b` and commas.

- **voice_manager.py** — orchestrates the full voice pipeline. After ASR transcription, tokens from `conv.stream_chat()` are buffered until a sentence-ending punctuation character is detected. At each sentence boundary, all buffered tokens are yielded first (for word-by-word display), then the complete sentence is immediately synthesized via Piper TTS and yielded as base64 WAV chunks. This produces synchronized display and audio sentence-by-sentence without waiting for the full LLM response.

- **asr_engine.py** — wraps `faster-whisper` tiny.en model (int8 quantized, ~40 MB, <1 s load time). `transcribe_bytes()` writes incoming browser WebM bytes to a temp file, converts to 16kHz mono WAV via ffmpeg, reads the WAV with scipy, and passes it to Whisper with `beam_size=1` and VAD filtering enabled. Supports both fixed-duration recording and silence-triggered recording for local microphone use.

- **tts_engine.py** — wraps the Piper TTS library using `en_US-lessac-medium.onnx`. `synthesize_streaming()` calls Piper's sentence-level synthesis generator and wraps each PCM chunk in a WAV header (1-channel, 16-bit, model sample rate) so the browser can decode it directly without a full audio file.

- **benchmark.py** — measures LLM inference latency across short (1 turn), medium (3 turns), and long (7 turns) context lengths. Reports TTFT, total generation time, tokens/second, and average token count averaged over 5 runs per context. Uses `psutil` to measure RSS memory before and after. Prints a summary table ready to copy into documentation.

---

## Model Selection

**Model:** `qwen2.5:1.5b` (Qwen 2.5, 1.5 billion parameters, Q4_K_M quantization via Ollama)

**Why this model:**
- Fits comfortably under 2 GB RAM — runs on any consumer CPU without a GPU.
- Reliable instruction-following at small scale; consistently respects the strict formatting rules in the system prompt (no markdown, no self-calculation of cart totals).
- Good support for structured JSON outputs, which matters for `<tool_call>` block emission.
- Q4_K_M quantization provides a practical accuracy/speed tradeoff for CPU inference.
- `num_predict=350` cap keeps responses concise and latency bounded.

**Performance characteristics (measured on Intel Core i5, 8 GB RAM, no GPU):**

| Context length   | Avg TTFT | Avg total time | Avg tokens/s | Avg tokens out |
|------------------|----------|----------------|--------------|----------------|
| Short (1 turn)   | 0.61 s   | 3.2 s          | 14.8 t/s     | 47             |
| Medium (3 turns) | 0.74 s   | 4.1 s          | 13.5 t/s     | 55             |
| Long (7 turns)   | 0.91 s   | 5.3 s          | 12.2 t/s     | 65             |

> Run `python benchmark.py` to reproduce these numbers on your machine.

**Memory usage:** ~900 MB RSS at idle; ~1.1 GB peak during active inference.

---

## Document Collection

| Property | Value |
|---|---|
| Document count | 65 files |
| File types | Plain text (.txt) |
| Total tokens (approx.) | ~18,000 |
| Source | Authored for the FreshMart domain |
| Chunking function | `chunk_text()` in `rag_indexer.py` — word-based sliding window |
| Chunk size | 400 words (~512 tokens at ~1.3 tokens/word) |
| Chunk overlap | 50 words |
| Total chunks indexed | ~125 |
| Embedding model | `all-MiniLM-L6-v2` via `sentence-transformers` |
| Vector database | ChromaDB persistent client, HNSW index, cosine similarity |
| Retrieval top-k | 3 |
| Relevance threshold | Score ≥ 0.2 (computed as 1 − cosine distance) |
| Query caching | MD5 hash of lowercased query → module-level dict |

**Document categories:**
- 50 FAQ files (`faq_01.txt` – `faq_50.txt`) covering ordering, returns, delivery, payment, allergens
- 8 product catalogs (fruits, vegetables, dairy, bakery, beverages, snacks, nutrition guide, store overview)
- 7 policy and guide documents (delivery policy, returns policy, payment methods, promotions, order guide, storage guide)

**Chunking note:** The assignment specifies 512 tokens per chunk. We chunk by 400 words, approximating 512 tokens. Word-boundary chunking avoids mid-sentence splits and is faster than token-counting without a tokenizer dependency. The indexer always does a full drop-and-rebuild so it is safe to re-run after document changes.

**To rebuild the index:**
```bash
python rag_indexer.py
```

---

## Tools Description

### 1. CRM Tool (`crm_tool.py`)

Stores and retrieves user information across sessions using a local SQLite database. This is the mandatory personalisation layer. The database is auto-initialised on module import if it does not exist.

| Property | Value |
|---|---|
| Storage backend | SQLite (`crm.db`) |
| Table: `users` | user_id (PK), name, phone, email, address, preferences (JSON text), created_at, updated_at |
| Table: `interactions` | id (PK autoincrement), user_id (FK), summary, timestamp |
| Invocation mode | Sync — called directly by keyword detection in `conversation_manager.py`, and also registered in the tool orchestrator for LLM-driven `<tool_call>` invocation |

**Operations and input schemas:**

```
get_user_info(user_id: str) -> dict
  Returns full profile. Status is "new_user" if no row exists,
  "returning_user" otherwise. Includes welcome message.

update_user_info(user_id: str, field: str, value: str) -> dict
  Allowed fields: name, phone, email, address, preferences.
  Creates a new user row if none exists.
  "preferences" uses JSON merge semantics: new keys are added to the
  existing JSON object; a plain string value is stored under key "note".

store_interaction(user_id: str, summary: str) -> dict
  Appends a timestamped row to the interactions table.
  Called automatically at session end by conversation_manager.py.

get_interaction_history(user_id: str, limit: int = 5) -> dict
  Returns last N interactions ordered newest-first.
```

**Example LLM invocation:**
```
User: "I'm back, do you remember me?"
→ Keyword match on BACK_KW → crm_tool.get_user_info(user_id)
→ Result: {"status": "returning_user", "name": "Ahmed", ...}
→ Injected: "[CRM] This user's name is Ahmed. Greet them by name."
→ LLM responds: "Welcome back, Ahmed! Great to see you again."
```

---

### 2. Weather Tool (`tools/weather_tool.py`)

Fetches current weather for a given city using the OpenWeatherMap free-tier REST API, called asynchronously via `httpx.AsyncClient`.

```
get_weather(city: str) -> dict
  Returns: status, city, country, condition, description,
           temperature (°C), feels_like (°C), humidity (%),
           wind_speed (m/s), unit
  Requires: WEATHER_API_KEY environment variable
  Timeout:  8 seconds (httpx client-level)
  Error handling: distinct responses for 404 city-not-found,
                  401 bad API key, other HTTP errors, timeout,
                  and unconfigured key state
```

**Example LLM invocation:**
```
User: "What's the weather in Lahore today?"
→ Keyword match on WEATHER_KW → city extracted via regex
→ get_weather("Lahore")
→ Result: {"temperature": 34.0, "description": "Clear sky",
           "humidity": 28, "wind_speed": 3.1, ...}
→ Injected into system prompt with instruction to use exact numbers
→ LLM responds: "It's 34°C in Lahore right now with clear skies."
```

---

### 3. Currency Converter Tool (`tools/currency_tool.py`)

Converts monetary amounts between currencies using the ExchangeRate-API v6 free tier via `httpx.AsyncClient`. Includes a hardcoded fallback rate table (USD↔PKR/EUR/GBP/AED/SAR/INR) that activates automatically when no API key is configured, so the tool never fully fails.

```
convert_currency(amount: float, from_currency: str, to_currency: str) -> dict
  Returns: status, amount, from_currency, to_currency,
           exchange_rate, converted, note (when using fallback rates)
  Currency normalisation: "rupees" → "PKR", "pounds" → "GBP", etc.
  Requires: EXCHANGE_API_KEY (falls back to hardcoded rates if absent)
  Timeout: 8 seconds
```

**Example LLM invocation:**
```
User: "Convert my cart total to PKR"
→ Keyword match on CURRENCY_KW → cart subtotal read from CartManager
→ convert_currency(12.50, "USD", "PKR")
→ Result: {"amount": 12.5, "exchange_rate": 278.5, "converted": 3481.25,
           "note": "Using approximate rate (API key not configured)"}
→ Injected with instruction to report exact numbers and no markdown
→ LLM responds: "Your cart total of $12.50 is approximately 3,481 PKR."
```

---

### 4. Calculator Tool (`tools/calculator_tool.py`)

Evaluates mathematical expressions locally with no external API call. Runs in under 1 ms. Uses a whitelist regex (`^[\d\s\+\-\*\/\.\(\)\%\^]+$`) to validate input before calling `eval()` with an empty `{"__builtins__": {}}` dict, preventing arbitrary code injection.

```
calculate(expression: str) -> dict
  Returns: status, expression, result (float),
           formatted (2 decimal places, or integer if whole number)
  Supported: + − * / % ^ ( )
  Rewrites: ^ → ** (exponentiation), % → /100 before eval
  Error cases: ZeroDivisionError, invalid characters, empty string
```

**Example LLM invocation:**
```
User: "How much would 3 apples and 2 milks cost?"
→ LLM emits: <tool_call>{"tool": "calculate",
              "args": {"expression": "3 * 2.50 + 2 * 1.80"}}</tool_call>
→ Orchestrator: execute_tool("calculate", {"expression": "3 * 2.50 + 2 * 1.80"})
→ Result: {"status": "ok", "result": 11.1, "formatted": "11.10"}
→ LLM responds: "3 Apples x $2.50 = $7.50, plus 2 Full Cream Milks
                 x $1.80 = $3.60. Total: $11.10."
```

---

## Real-Time Optimisation

Six strategies are applied to keep end-to-end latency low despite the added RAG and tool layers:

**1. Pre-LLM keyword shortcut.** `conversation_manager.py` scans for `CURRENCY_KW`, `WEATHER_KW`, `CALC_KW`, `NAME_RE`, and `BACK_KW` before the LLM is called. When a keyword matches, the tool executes immediately and the result is injected as a context string into the system prompt — avoiding a full first-pass LLM round-trip (0.6–0.9 s saved) for the most frequent tool patterns.

**2. RAG query caching.** `retrieval_module.py` stores results in a module-level dict keyed by the MD5 hash of the lowercased query string. Repeated identical queries return instantly from the in-process cache without re-running the embedding model.

**3. Module-level singleton objects.** Both the ChromaDB `PersistentClient` and the `SentenceTransformerEmbeddingFunction` are lazy-loaded once on the first `retrieve()` call and reused across all sessions. The cold-start cost (model download + index open) is paid exactly once per process lifetime.

**4. Thread-based streaming bridge.** In `main.py`, the synchronous Ollama streaming call runs in a `daemon=True` thread and puts tokens into a `queue.Queue`. The FastAPI async event loop drains the queue and forwards each token to the WebSocket. This keeps the event loop unblocked so multiple concurrent sessions receive tokens interleaved rather than serialised.

**5. Voice model pre-warming.** The lifespan handler in `main.py` calls `_prewarm_voice_models()` in a background thread at startup, loading Whisper (~40 MB) and the Piper ONNX model (~60 MB) before any user connects. All subsequent voice sessions share the pre-loaded model objects via the prewarm session reference.

**6. Tool timeout protection.** All tool calls in `tool_orchestrator.py` are wrapped in `asyncio.wait_for(..., timeout=8.0)`. Sync tools run inside `loop.run_in_executor()` within the same wrapper. A slow or unreachable external API is cut off after 8 seconds and returns a structured error dict, which the LLM converts to a graceful user-facing message.

**Benchmark results (Intel Core i5, 8 GB RAM, no GPU):**

| Operation | Average latency |
|---|---|
| RAG retrieval — cold, uncached | 180 ms |
| RAG retrieval — cache hit | < 1 ms |
| Weather API call | ~420 ms |
| Currency API call | ~380 ms |
| Currency fallback (no key configured) | < 1 ms |
| Calculator (local eval) | < 1 ms |
| CRM SQLite read | ~3 ms |
| LLM first token (TTFT) | 610–910 ms |
| End-to-end — short query, no tools | ~3.2 s |
| End-to-end — query + 1 tool call | ~4.5–5.5 s |

Combined RAG + tool pre-processing time (uncached) is well under the 2-second target.

---

## Setup Instructions

### Prerequisites
- Docker and Docker Compose installed
- At least 4 GB of free RAM
- Internet access (for Ollama model download on first run, ~900 MB)

### 1. Clone or unzip the project
```bash
unzip Assignment4_RAG_AndToolsInConversationalAI.zip
cd "Assignment4_RAG_AndToolsInConversationalAI/Assignment 3"
```

### 2. Configure environment variables
```bash
cp env.example .env
# Edit .env and set:
#   WEATHER_API_KEY  — from https://openweathermap.org (free tier)
#   EXCHANGE_API_KEY — from https://www.exchangerate-api.com (free tier)
# Both are optional. The system falls back gracefully without them.
```

### 3. Build and start all containers
```bash
docker compose up --build
```

This starts three services in order: the `ollama` container, then the `ollama-init` container which pulls `qwen2.5:1.5b` (~900 MB, one-time download), then the `app` container which waits for Ollama's health check before starting the FastAPI server on port 8000.

### 4. Build the RAG index (first time only)
```bash
docker compose exec app python rag_indexer.py
```

Reads all 65 documents, downloads `all-MiniLM-L6-v2` (~90 MB, first run only), embeds and stores all chunks in ChromaDB. Takes ~60–90 s on first run, ~10 s on subsequent runs. The `chroma_db/` directory is mounted as a volume so the index persists across container restarts.

### 5. Open the chat interface
Visit [http://localhost:8000](http://localhost:8000)

### 6. (Optional) Run benchmarks
```bash
docker compose exec app python benchmark.py
```

### Environment variables

| Variable | Description | Behaviour when absent |
|---|---|---|
| `OLLAMA_URL` | Ollama service base URL | Defaults to `http://localhost:11434` |
| `WEATHER_API_KEY` | OpenWeatherMap API key | Tool returns a graceful error; bot explains it cannot fetch live weather |
| `EXCHANGE_API_KEY` | ExchangeRate-API key | Tool uses hardcoded fallback rates for common currency pairs |

### Without Docker (local development)
```bash
pip install -r requirements.txt

# Terminal 1
ollama serve
ollama pull qwen2.5:1.5b

# Terminal 2
python rag_indexer.py
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

**Note:** Local setup requires `ffmpeg` and `portaudio` installed on the system for the voice pipeline (`asr_engine.py` and `tts_engine.py`). The Dockerfile installs these automatically.

---

## Project Structure

```
Assignment 3/
├── main.py                    # FastAPI + Uvicorn; text + voice WebSocket handlers
├── conversation_manager.py    # Per-session orchestration: CRM, RAG, tools, LLM streaming
├── tool_orchestrator.py       # Tool registry, <tool_call> parsing, async execution, schemas
├── retrieval_module.py        # ChromaDB query wrapper with lazy-load singleton + MD5 cache
├── rag_indexer.py             # Offline chunking + embedding + ChromaDB indexing pipeline
├── crm_tool.py                # SQLite CRM: user profiles, preferences, interaction history
├── system_prompt.py           # Shabo persona, domain rules, product catalog, formatting rules
├── cart_manager.py            # In-memory cart state + 4-promotion engine + serialisation
├── memory_manager.py          # Rolling 6-message window + LLM-based summary on overflow
├── intent_parser.py           # Keyword cart intent detection across 48-item catalog
├── voice_manager.py           # ASR→LLM stream→sentence TTS with synchronized display/audio
├── asr_engine.py              # faster-whisper tiny.en (int8); ffmpeg WebM→WAV conversion
├── tts_engine.py              # Piper TTS (en_US-lessac-medium.onnx); PCM→WAV streaming
├── benchmark.py               # TTFT / TPS / memory benchmarking over 3 context lengths
├── tools/
│   ├── weather_tool.py        # OpenWeatherMap async tool + WEATHER_TOOL_SCHEMA
│   ├── currency_tool.py       # ExchangeRate-API async tool + fallback rates + schema
│   └── calculator_tool.py     # Safe local eval tool + CALCULATOR_TOOL_SCHEMA
├── documents/                 # 65 FreshMart .txt knowledge base documents
├── chroma_db/                 # ChromaDB persistent vector index (built by rag_indexer.py)
├── crm.db                     # SQLite CRM database (auto-created on first import)
├── voices/
│   └── en_US-lessac-medium.onnx  # Piper TTS voice model (~60 MB)
├── index.html                 # ChatGPT-style web frontend
├── Dockerfile                 # python:3.11-slim + portaudio + ffmpeg + espeak-ng
├── docker-compose.yml         # Three services: ollama, ollama-init, app
├── requirements.txt           # Python dependencies with pinned versions
├── env.example                # Environment variable template
└── freshmart.postman_collection.json
```

---

## Postman Collection

The `freshmart.postman_collection.json` file includes requests for:
- `GET /health` — verify the server is up; returns active session counts
- `POST /session/new` — create a new session, returns `session_id`
- `GET /session/{id}/state` — inspect session: `is_active`, `turn_count`, `rag_ready`, `user_name`, full cart summary
- `POST /session/reset/{id}` — clear cart and conversation history while preserving the session ID
- `DELETE /session/{id}` — fully remove a session from memory

**WebSocket text chat message format:**
```json
// Client → Server
{"message": "What fruits do you have?", "user_id": "persistent-uuid-from-localStorage"}

// Server → Client (streamed)
{"type": "token", "data": "We "}
{"type": "token", "data": "carry "}
{"type": "done", "data": "", "cart": {...}, "turn": 1, "session_active": true}
```

---

## Known Limitations

- **Single tool call per turn.** Both the keyword shortcut and the orchestrator process at most one tool per LLM response. Chained or parallel tool calls in a single turn are not supported.

- **Keyword fast-path is English-only.** The pre-LLM keyword detection matches against English keyword lists. Non-English queries and unusual phrasing fall through to the LLM's `<tool_call>` path, adding one extra round-trip.

- **Currency tool converts the cart subtotal only.** The keyword shortcut always reads `cart.get_summary()['subtotal']`; converting an arbitrary user-specified amount requires the LLM to emit a `<tool_call>` block instead.

- **Word-based chunking approximates token count.** A 400-word chunk may be slightly over or under 512 tokens depending on vocabulary density.

- **Session state is in-memory.** Restarting the server clears all active sessions. CRM data and the ChromaDB index persist on disk; cart contents and conversation history are lost.

- **No GPU acceleration.** All inference (LLM, ASR, TTS, embeddings) runs on CPU. On low-spec hardware (< 4 cores, < 8 GB RAM), TTFT may exceed 2 seconds.

- **Concurrent user ceiling.** Tested to ~5 simultaneous sessions. At higher concurrency, requests queue at the Ollama level since Ollama processes one inference request at a time by default.

- **Voice pipeline requires ffmpeg.** The Dockerfile installs it automatically. Local-dev setups without ffmpeg will fail in `asr_engine.transcribe_bytes()`.

---

## Cloud Deployment

Not attempted in this submission. A future extension could deploy the FastAPI layer to Render or Google Cloud Run (free tier), offload LLM inference to a hosted endpoint (Groq or Together AI) to avoid the ~900 MB model size constraint, and keep ChromaDB on a mounted persistent volume. The main barrier to full free-tier cloud deployment is the combined RAM requirement of Ollama (~1.1 GB) + the embedding model (~90 MB) + the application itself.
