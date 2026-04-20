"""
conversation_manager.py - Assignment 4 Final
Key fixes:
- CRM uses sqlite3 directly (no asyncio loop issues in uvicorn)
- Tool results injected before streaming so streaming still works
- Cart state injected cleanly without leaking into LLM visible text
"""

import re
import requests
import json
import logging
from system_prompt import SYSTEM_PROMPT
from cart_manager import CartManager
from memory_manager import MemoryManager
from intent_parser import parse_intent
from retrieval_module import retrieve, format_context, is_index_ready
import crm_tool
from tools.currency_tool import convert_currency
from tools.calculator_tool import calculate
import asyncio
import os

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434") + "/api/chat"
MODEL = "qwen2.5:1.5b"
logger = logging.getLogger(__name__)

# ── Keyword detection ─────────────────────────────────────────────────────────

CURRENCY_KW = ["rupee","pkr","convert","in euros","in pounds","in pkr",
               "exchange","currency","usd to","dollar to"]
WEATHER_KW  = ["weather","temperature","rain","sunny","forecast","degrees",
               "hot today","cold today","climate"]
CALC_KW     = ["how much would","calculate","what is the cost","total cost",
               "price of","cost me"]
NAME_RE     = re.compile(r"my name is ([a-zA-Z]+)", re.IGNORECASE)
BACK_KW     = ["i'm back","im back","i am back","returning","remember me",
               "do you remember","what is my name","whats my name"]


def _run(coro):
    """Run a coroutine safely regardless of event loop state."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # We're inside uvicorn — use a new thread loop
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, coro)
                return future.result(timeout=10)
        return loop.run_until_complete(coro)
    except Exception:
        try:
            return asyncio.run(coro)
        except Exception as e:
            logger.error(f"Async run failed: {e}")
            return {"error": str(e)}


class ConversationManager:

    END_SIGNALS = ["goodbye","bye","thank you, goodbye","that's all",
                   "order confirmed","see you","thanks, bye"]

    def __init__(self, session_id: str):
        self.session_id   = session_id
        self.cart         = CartManager()
        self.memory       = MemoryManager(self.cart)
        self.is_active    = True
        self.turn_count   = 0
        self._crm_checked = False
        self._user_name   = None
        self._crm_user_id = session_id

    # ── CRM (direct sqlite, no async needed) ─────────────────────────────────

    def set_user_id(self, user_id: str):
        if user_id and user_id != self._crm_user_id:
            self._crm_user_id = user_id
            self._crm_checked = False

    def _crm_check_returning(self):
        if self._crm_checked:
            return
        self._crm_checked = True
        result = crm_tool.get_user_info(self._crm_user_id)
        if result.get("status") == "returning_user" and result.get("name"):
            self._user_name = result["name"]

    def _crm_save_name(self, name: str):
        crm_tool.update_user_info(self._crm_user_id, "name", name)
        self._user_name = name
        logger.info(f"[CRM] Saved name '{name}' for {self.session_id}")

    def _crm_store_session(self):
        cart = self.cart.get_summary()
        crm_tool.store_interaction(
            self._crm_user_id,
            f"Session ended. Turns: {self.turn_count}. Cart: {cart}."
        )

    # ── Tool detection & execution ────────────────────────────────────────────

    def _detect_and_run_tools(self, user_message: str) -> str:
        """
        Detect tool needs from message keywords, run the tool,
        return a context string to inject into the prompt.
        Returns "" if no tool needed.
        """
        msg = user_message.lower()

        # Save name
        m = NAME_RE.search(user_message)
        if m:
            self._crm_save_name(m.group(1).capitalize())
            return f"[CRM UPDATE] User's name saved as '{self._user_name}'. Acknowledge warmly."

        # Returning user asking about name
        if any(kw in msg for kw in BACK_KW):
            self._crm_check_returning()
            if self._user_name:
                return f"[CRM] This user's name is {self._user_name}. Greet them by name."
            return "[CRM] No name on file for this user. Greet warmly without a name."

        # Weather
        if any(kw in msg for kw in WEATHER_KW):
            city_m = re.search(r"in ([A-Za-z\s]+?)(?:\?|today|tomorrow|now|$)",
                                user_message, re.IGNORECASE)
            city = city_m.group(1).strip() if city_m else "Rawalpindi"
            try:
                result = _run(convert_currency.__module__ and __import__(
                    'tools.weather_tool', fromlist=['get_weather']
                ).get_weather(city))
                if result and "error" not in result:
                    return (f"[TOOL RESULT: weather]\n"
                            f"City: {result.get('city')}, {result.get('country')}\n"
                            f"Condition: {result.get('description')}\n"
                            f"Temperature: {result.get('temperature')}°C "
                            f"(feels like {result.get('feels_like')}°C)\n"
                            f"Humidity: {result.get('humidity')}%\n"
                            f"Use these exact numbers in your response.")
            except Exception as e:
                logger.warning(f"Weather tool failed: {e}")
            return ("[TOOL NOTE] Weather API key not configured. "
                    "Tell the user you cannot fetch live weather and suggest "
                    "they check a weather app.")

        # Currency conversion
        if any(kw in msg for kw in CURRENCY_KW):
            cart_summary = self.cart.get_summary()
            amount = 0
            if isinstance(cart_summary, dict):
                amount = cart_summary.get("subtotal", 0)

            # Detect target currency
            to_cur = "PKR"
            if "euro" in msg or " eur" in msg: to_cur = "EUR"
            elif "pound" in msg or "gbp" in msg: to_cur = "GBP"
            elif "dirham" in msg or "aed" in msg: to_cur = "AED"

            if amount > 0:
                result = _run(convert_currency(amount, "USD", to_cur))
                if result and "error" not in result:
                    return (f"[TOOL RESULT: currency]\n"
                            f"Cart total: ${result.get('amount')} USD\n"
                            f"Converted: {result.get('converted')} {to_cur}\n"
                            f"Rate: 1 USD = {result.get('exchange_rate')} {to_cur}\n"
                            f"Note: {result.get('note','')}\n"
                            f"Report these exact numbers. No markdown.")
            return "[TOOL NOTE] Cart is empty, nothing to convert."

        # Price calculation
        if any(kw in msg for kw in CALC_KW):
            return ("[TOOL HINT] User wants a price calculation. "
                    "Use exact catalog prices. Show arithmetic in plain text like: "
                    "3 x $2.50 = $7.50. Give the final total clearly. No markdown.")

        return ""

    # ── Prompt building ───────────────────────────────────────────────────────

    def _build_system(self, rag_ctx: str = "", tool_ctx: str = "") -> str:
        parts = [SYSTEM_PROMPT]
        if self._user_name:
            parts.append(f"[CRM] User's name is {self._user_name}. Use their name naturally.")
        if rag_ctx:
            parts.append(rag_ctx)
        if tool_ctx:
            parts.append(tool_ctx)
        return "\n\n".join(parts)

    # ── Ollama ────────────────────────────────────────────────────────────────

    def _call_ollama(self, messages: list) -> str:
        payload = {"model": MODEL, "messages": messages,
                   "stream": False, "options": {"num_predict": 350, "temperature": 0.7}}
        try:
            r = requests.post(OLLAMA_URL, json=payload, timeout=120)
            r.raise_for_status()
            return r.json().get("message", {}).get("content", "").strip()
        except Exception as e:
            return f"Sorry, I'm having trouble connecting. ({e})"

    # ── Public API ────────────────────────────────────────────────────────────

    def chat(self, user_message: str) -> str:
        if not self.is_active:
            return "This session has ended. Please start a new conversation."
        parse_intent(user_message, self.cart)
        self._crm_check_returning()
        self.memory.add_message("user", user_message)
        rag_ctx  = self._get_rag(user_message)
        tool_ctx = self._detect_and_run_tools(user_message)
        msgs     = self.memory.build_messages(self._build_system(rag_ctx, tool_ctx))
        response = self._call_ollama(msgs)
        self.memory.add_message("assistant", response)
        self.turn_count += 1
        if self._should_end(user_message):
            self._crm_store_session()
            self.is_active = False
        return response

    def stream_chat(self, user_message: str):
        if not self.is_active:
            yield "This session has ended. Please start a new conversation."
            return
        parse_intent(user_message, self.cart)
        self._crm_check_returning()
        self.memory.add_message("user", user_message)
        rag_ctx  = self._get_rag(user_message)
        tool_ctx = self._detect_and_run_tools(user_message)
        msgs     = self.memory.build_messages(self._build_system(rag_ctx, tool_ctx))

        full_response = ""
        payload = {"model": MODEL, "messages": msgs, "stream": True,
                   "options": {"num_predict": 350, "temperature": 0.7}}
        try:
            with requests.post(OLLAMA_URL, json=payload,
                               stream=True, timeout=120) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if line:
                        chunk = json.loads(line)
                        token = chunk.get("message", {}).get("content", "")
                        if token:
                            full_response += token
                            yield token
                        if chunk.get("done"):
                            break
        except Exception as e:
            msg = f"Sorry, I'm having trouble connecting. ({e})"
            yield msg
            full_response = msg

        self.memory.add_message("assistant", full_response)
        self.turn_count += 1
        if self._should_end(user_message):
            self._crm_store_session()
            self.is_active = False

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _get_rag(self, msg: str) -> str:
        if not is_index_ready():
            return ""
        try:
            return format_context(retrieve(msg))
        except Exception:
            return ""

    def _should_end(self, msg: str) -> bool:
        return any(s in msg.lower() for s in self.END_SIGNALS)

    def reset_session(self):
        self.cart.clear()
        self.memory.reset()
        self.is_active    = True
        self.turn_count   = 0
        self._crm_checked = False

    def get_session_state(self) -> dict:
        return {
            "session_id":  self.session_id,
            "is_active":   self.is_active,
            "turn_count":  self.turn_count,
            "rag_ready":   is_index_ready(),
            "user_name":   self._user_name,
            "cart":        self.cart.get_summary()
        }
