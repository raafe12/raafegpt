import os
import uuid
import json
import base64
import requests
from datetime import datetime
from flask import Flask, render_template, request, jsonify, session, g, Response, stream_with_context
from groq import Groq
from dotenv import load_dotenv
import psycopg2
import psycopg2.extras

load_dotenv()

app = Flask(__name__)
app.secret_key = os.urandom(24)
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

MODEL = "openai/gpt-oss-20b"
VISION_MODEL = "qwen/qwen3.8-27b"
DATABASE_URL = os.getenv("DATABASE_URL")
LANGSEARCH_API_KEY = os.getenv("LANGSEARCH_API_KEY")


def friendly_error(e: Exception) -> str:
    msg = str(e)
    low = msg.lower()
    if "tool choice is none" in low or "model called a tool" in low:
        return "⚠️ The AI tried to use a tool incorrectly. Try asking again."
    if "rate limit" in low or "429" in low:
        return "⚠️ Rate limit hit. Wait a moment and try again."
    if "401" in low or "unauthorized" in low or "invalid api key" in low:
        return "⚠️ API key problem. Check your .env file."
    if "model_not_found" in low or "does not exist" in low:
        return "⚠️ The AI model isn't available. Try switching models."
    if "timeout" in low or "timed out" in low:
        return "⚠️ Request timed out. Check your internet and try again."
    if "connection" in low or "network" in low:
        return "⚠️ Network problem. Check your internet."
    if "context_length" in low or "too long" in low:
        return "⚠️ This conversation is too long. Start a new chat."
    short = msg.split("\n")[0][:200]
    return f"⚠️ Something went wrong: {short}"


def get_db():
    db = getattr(g, "_database", None)
    if db is None:
        db = g._database = psycopg2.connect(DATABASE_URL)
    return db


@app.teardown_appcontext
def close_db(exc):
    db = getattr(g, "_database", None)
    if db is not None:
        db.close()


def init_db():
    with psycopg2.connect(DATABASE_URL) as db:
        with db.cursor() as cur:
            cur.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                pinned INTEGER DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS messages (
                id SERIAL PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                time TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS memory (
                id SERIAL PRIMARY KEY,
                content TEXT NOT NULL UNIQUE,
                created_at TEXT NOT NULL
            );
            """)
        db.commit()


init_db()


def query(sql, params=None, fetch=False, commit=False, one=False):
    db = get_db()
    with db.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(sql, params or ())
        result = None
        if fetch:
            result = cur.fetchone() if one else cur.fetchall()
        if commit:
            db.commit()
        return result


def build_system_prompt():
    mem = query("SELECT content FROM memory ORDER BY id ASC", fetch=True) or []
    base = (
        "You are RaafeGPT, a helpful, friendly AI assistant. "
        "Talk in a casual, funny, short tone. Use markdown when helpful. Don't over-explain. "
        "When writing math, use LaTeX with $$ for display equations and $ for inline math. "
        "You have a tool to save memories permanently when the user asks you to remember something. "
        "You also have a tool to search the web when you need current or factual information."
    )
    if mem:
        facts = "\n".join(f"- {m['content']}" for m in mem)
        base += f"\n\nPermanent facts about the user that you must always remember:\n{facts}"
    return base


def save_memory_fact(content: str) -> bool:
    content = content.strip()
    if not content:
        return False
    try:
        db = get_db()
        with db.cursor() as cur:
            cur.execute(
                "INSERT INTO memory (content, created_at) VALUES (%s, %s) ON CONFLICT (content) DO NOTHING",
                (content, datetime.now().isoformat())
            )
            added = cur.rowcount > 0
        db.commit()
        return added
    except Exception:
        return False


def google_search(query_str: str, num: int = 5) -> str:
    if not LANGSEARCH_API_KEY:
        return "Web search is not configured."
    try:
        resp = requests.post(
            "https://api.langsearch.com/v1/web-search",
            headers={
                "Authorization": f"Bearer {LANGSEARCH_API_KEY}",
                "Content-Type": "application/json"
            },
            json={"query": query_str, "count": num, "summary": True},
            timeout=15
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        return f"Search failed: {str(e)}"

    results = []
    pages = data.get("webPages", {}).get("value", [])
    for i, item in enumerate(pages, 1):
        title = item.get("name", "")
        link = item.get("url", "")
        snippet = item.get("snippet", "")
        results.append(f"[{i}] {title}\n    {link}\n    {snippet}")

    if not results:
        return "No search results found."
    return "\n".join(results)


@app.errorhandler(Exception)
def handle_any_error(e):
    from werkzeug.exceptions import HTTPException
    if isinstance(e, HTTPException):
        return e
    return jsonify({"error": friendly_error(e)}), 500


@app.route("/")
def index():
    if "sid" not in session:
        session["sid"] = str(uuid.uuid4())
    return render_template("index.html")


@app.route("/api/upload", methods=["POST"])
def upload_image():
    if "image" not in request.files:
        return jsonify({"error": "No image file"}), 400
    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400
    allowed = {"png", "jpg", "jpeg", "gif", "webp"}
    ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if ext not in allowed:
        return jsonify({"error": f"Invalid file type. Allowed: {', '.join(allowed)}"}), 400
    file_data = file.read()
    if len(file_data) > 20 * 1024 * 1024:
        return jsonify({"error": "Image too large (max 20MB)"}), 400
    mime = file.mimetype or f"image/{ext}"
    b64 = base64.b64encode(file_data).decode("utf-8")
    data_url = f"data:{mime};base64,{b64}"
    return jsonify({"url": data_url, "filename": file.filename})


@app.route("/api/conversations", methods=["GET"])
def list_conversations():
    rows = query("""
        SELECT id, title, pinned, updated_at
        FROM conversations
        ORDER BY pinned DESC, updated_at DESC
    """, fetch=True)
    return jsonify([dict(r) for r in rows or []])


@app.route("/api/conversations", methods=["POST"])
def new_conversation():
    cid = str(uuid.uuid4())
    now = datetime.now().isoformat()
    query(
        "INSERT INTO conversations (id, title, pinned, created_at, updated_at) VALUES (%s, %s, 0, %s, %s)",
        (cid, "New Chat", now, now), commit=True
    )
    return jsonify({"id": cid, "title": "New Chat", "pinned": 0, "updated_at": now})


@app.route("/api/conversations/<cid>", methods=["GET"])
def get_conversation(cid):
    conv = query("SELECT * FROM conversations WHERE id = %s", (cid,), fetch=True, one=True)
    if not conv:
        return jsonify({"error": "not found"}), 404
    msgs = query(
        "SELECT role, content, time FROM messages WHERE conversation_id = %s ORDER BY id ASC",
        (cid,), fetch=True
    )
    return jsonify({"conversation": dict(conv), "messages": [dict(m) for m in msgs or []]})


@app.route("/api/conversations/<cid>", methods=["DELETE"])
def delete_conversation(cid):
    query("DELETE FROM messages WHERE conversation_id = %s", (cid,), commit=True)
    query("DELETE FROM conversations WHERE id = %s", (cid,), commit=True)
    return jsonify({"ok": True})


@app.route("/api/conversations/<cid>/pin", methods=["POST"])
def pin_conversation(cid):
    conv = query("SELECT pinned FROM conversations WHERE id = %s", (cid,), fetch=True, one=True)
    if not conv:
        return jsonify({"error": "not found"}), 404
    new_pin = 0 if conv["pinned"] else 1
    query("UPDATE conversations SET pinned = %s WHERE id = %s", (new_pin, cid), commit=True)
    return jsonify({"pinned": new_pin})


@app.route("/api/conversations/<cid>/rename", methods=["POST"])
def rename_conversation(cid):
    title = (request.get_json() or {}).get("title", "").strip()
    if not title:
        return jsonify({"error": "empty title"}), 400
    query("UPDATE conversations SET title = %s WHERE id = %s", (title, cid), commit=True)
    return jsonify({"ok": True, "title": title})


@app.route("/api/conversations/<cid>/chat", methods=["POST"])
def chat(cid):
    data = request.get_json() or {}
    user_msg = data.get("message", "").strip()
    image_url = data.get("image_url", "").strip()

    if not user_msg and not image_url:
        return jsonify({"error": "empty message"}), 400

    conv = query("SELECT * FROM conversations WHERE id = %s", (cid,), fetch=True, one=True)
    if not conv:
        return jsonify({"error": "not found"}), 404

    now = datetime.now().isoformat()
    time_short = datetime.now().strftime("%H:%M")

    display_msg = user_msg
    if image_url:
        display_msg = (user_msg + "\n\n[Image attached]") if user_msg else "[Image attached]"

    query(
        "INSERT INTO messages (conversation_id, role, content, time) VALUES (%s, 'user', %s, %s)",
        (cid, display_msg, time_short), commit=True
    )
    count = query("SELECT COUNT(*) AS c FROM messages WHERE conversation_id = %s", (cid,), fetch=True, one=True)
    if count["c"] == 1:
        title = (user_msg or "Image chat")[:40] + ("…" if len(user_msg or "") > 40 else "")
        query("UPDATE conversations SET title = %s WHERE id = %s", (title, cid), commit=True)
    query("UPDATE conversations SET updated_at = %s WHERE id = %s", (now, cid), commit=True)

    history = query(
        "SELECT role, content FROM messages WHERE conversation_id = %s ORDER BY id ASC",
        (cid,), fetch=True
    ) or []

    use_vision = bool(image_url)
    active_model = VISION_MODEL if use_vision else MODEL

    messages = [{"role": "system", "content": build_system_prompt()}]
    for m in history[-20:]:
        content = m["content"]
        if content.endswith("\n\n[Image attached]"):
            content = content[:-18]
        elif content == "[Image attached]":
            content = ""
        if content or m["role"] == "assistant":
            messages.append({"role": m["role"], "content": content})

    if use_vision:
        if messages and messages[-1]["role"] == "user":
            messages.pop()
        content_parts = []
        if user_msg:
            content_parts.append({"type": "text", "text": user_msg})
        content_parts.append({"type": "image_url", "image_url": {"url": image_url}})
        messages.append({"role": "user", "content": content_parts})

    tools = None
    if not use_vision:
        tools = [
            {"type": "function", "function": {
                "name": "save_memory",
                "description": "Save a permanent fact about the user when they explicitly ask to remember something.",
                "parameters": {"type": "object", "properties": {
                    "content": {"type": "string"}
                }, "required": ["content"]}
            }},
            {"type": "function", "function": {
                "name": "web_search",
                "description": "Search the web for current, factual, or up-to-date information.",
                "parameters": {"type": "object", "properties": {
                    "query": {"type": "string"}
                }, "required": ["query"]}
            }}
        ]

    def generate():
        full = ""
        saved_facts = []
        search_used = False
        error_fired = False

        try:
            if use_vision:
                stream = client.chat.completions.create(
                    model=active_model, messages=messages,
                    temperature=0.7, max_tokens=2048, stream=True,
                )
                for chunk in stream:
                    delta = chunk.choices[0].delta.content or ""
                    if delta:
                        full += delta
                        yield f"data: {json.dumps({'delta': delta})}\n\n"
            else:
                try:
                    planner = client.chat.completions.create(
                        model=active_model, messages=messages, tools=tools,
                        tool_choice="auto", temperature=0.7, max_tokens=1024,
                    )
                    choice = planner.choices[0].message
                    tool_calls = choice.tool_calls or []
                except Exception as e_plan:
                    yield f"data: {json.dumps({'status': friendly_error(e_plan)})}\n\n"
                    stream = client.chat.completions.create(
                        model=active_model, messages=messages,
                        temperature=0.7, max_tokens=2048, stream=True,
                    )
                    for chunk in stream:
                        delta = chunk.choices[0].delta.content or ""
                        if delta:
                            full += delta
                            yield f"data: {json.dumps({'delta': delta})}\n\n"
                    tool_calls = []
                    choice = None

                if choice and tool_calls:
                    for tc in tool_calls:
                        fn = tc.function
                        try:
                            args = json.loads(fn.arguments or "{}")
                        except Exception:
                            args = {}
                        if fn.name == "web_search":
                            q = args.get("query", user_msg)
                            yield f"data: {json.dumps({'status': f'🔍 Searching: {q}'})}\n\n"
                            results = google_search(q)
                            messages.append({"role": "assistant", "content": choice.content or "", "tool_calls": [{
                                "id": tc.id, "type": "function",
                                "function": {"name": fn.name, "arguments": fn.arguments}
                            }]})
                            messages.append({"role": "tool", "tool_call_id": tc.id, "content": results})
                            search_used = True
                        elif fn.name == "save_memory":
                            c = args.get("content", "").strip()
                            if c and save_memory_fact(c):
                                saved_facts.append(c)

                if choice and not tool_calls and choice.content:
                    full = choice.content
                    yield f"data: {json.dumps({'delta': choice.content})}\n\n"
                elif choice and tool_calls:
                    try:
                        stream = client.chat.completions.create(
                            model=active_model, messages=messages, temperature=0.7,
                            max_tokens=2048, stream=True,
                        )
                        for chunk in stream:
                            delta = chunk.choices[0].delta.content or ""
                            if delta:
                                full += delta
                                yield f"data: {json.dumps({'delta': delta})}\n\n"
                    except Exception as e2:
                        msg = friendly_error(e2)
                        full += msg
                        yield f"data: {json.dumps({'error_msg': msg})}\n\n"
                        error_fired = True

        except Exception as e:
            msg = friendly_error(e)
            full += msg
            yield f"data: {json.dumps({'error_msg': msg})}\n\n"
            error_fired = True

        if not saved_facts and not use_vision and not error_fired:
            try:
                classify = client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content":
                            "Extract a single permanent fact about the user ONLY if they explicitly asked to remember it. Otherwise reply exactly: NONE"},
                        {"role": "user", "content": user_msg}
                    ],
                    temperature=0, max_tokens=80,
                )
                candidate = (classify.choices[0].message.content or "").strip()
                if candidate and candidate.upper() != "NONE" and len(candidate) < 400:
                    if save_memory_fact(candidate):
                        saved_facts.append(candidate)
            except Exception:
                pass

        if saved_facts:
            note = "\n\n_🧠 Saved to memory:_ " + "; ".join(saved_facts)
            full += note
            yield f"data: {json.dumps({'delta': note})}\n\n"
            yield f"data: {json.dumps({'memory_saved': saved_facts})}\n\n"

        if full.strip():
            query(
                "INSERT INTO messages (conversation_id, role, content, time) VALUES (%s, 'assistant', %s, %s)",
                (cid, full, datetime.now().strftime("%H:%M")), commit=True
            )

        yield f"data: {json.dumps({'done': True, 'search_used': search_used, 'error': error_fired})}\n\n"

    return Response(stream_with_context(generate()), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/api/memory", methods=["GET"])
def list_memory():
    rows = query("SELECT id, content, created_at FROM memory ORDER BY id ASC", fetch=True)
    return jsonify([dict(r) for r in rows or []])


@app.route("/api/memory", methods=["POST"])
def add_memory():
    data = request.get_json() or {}
    if "bulk" in data:
        lines = [l.strip().lstrip("0123456789. ").strip() for l in data["bulk"].splitlines()]
        lines = [l for l in lines if l]
        added, skipped = 0, 0
        for line in lines:
            if save_memory_fact(line):
                added += 1
            else:
                skipped += 1
        return jsonify({"ok": True, "added": added, "skipped": skipped})

    content = (data.get("content") or "").strip()
    if not content:
        return jsonify({"error": "empty"}), 400
    added = save_memory_fact(content)
    return jsonify({"ok": True, "added": added})


@app.route("/api/memory/<int:mid>", methods=["DELETE"])
def delete_memory(mid):
    query("DELETE FROM memory WHERE id = %s", (mid,), commit=True)
    return jsonify({"ok": True})


@app.route("/api/memory/all", methods=["DELETE"])
def clear_memory():
    query("DELETE FROM memory", commit=True)
    return jsonify({"ok": True})


if __name__ == "__main__":
    app.run(debug=True, port=5000, threaded=True)