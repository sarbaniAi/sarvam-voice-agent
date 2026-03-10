"""
Yatra — Sovereign Voice Travel Assistant
Sarvam AI + Databricks | 22 Indian Languages | 100% India-Hosted

Architecture: STT (Sarvam Saaras, India) -> LLM (Sarvam-M/30B, India) -> TTS (Sarvam Bulbul, India)
"""

import os, re, json, base64, logging, sys, requests, io
import gradio as gr

logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)
logger.info(f"Python: {sys.version}, Gradio: {gr.__version__}")

SARVAM_API_KEY = os.environ.get("SARVAM_API_KEY", "")
DATABRICKS_HOST = os.environ.get("DATABRICKS_HOST", "")
DATABRICKS_TOKEN = os.environ.get("DATABRICKS_TOKEN", "")
SARVAM_ENDPOINT = os.environ.get("SARVAM_ENDPOINT_NAME", "sarvam-30b-serving")

SYSTEM_PROMPT = """You are Yatra, a friendly travel assistant for Indian travelers.
Respond in the SAME language the user speaks (Hindi, English, Tamil, Telugu, Hinglish etc.)
Be concise (2-3 sentences max for voice-friendly responses).
Help with: travel planning, flights, trains, hotels, visa, local tips, food, budget.
For bookings, guide to IRCTC, MakeMyTrip, Cleartrip etc."""

LANG_MAP = {
    "hi": "hi-IN", "en": "en-IN", "ta": "ta-IN", "te": "te-IN",
    "kn": "kn-IN", "ml": "ml-IN", "bn": "bn-IN", "gu": "gu-IN",
    "mr": "mr-IN", "pa": "pa-IN", "od": "od-IN",
}

LANG_NAMES = {
    "hi": "Hindi", "en": "English", "ta": "Tamil", "te": "Telugu",
    "kn": "Kannada", "ml": "Malayalam", "bn": "Bengali", "gu": "Gujarati",
    "mr": "Marathi", "pa": "Punjabi", "od": "Odia",
}

conversation = []


def call_llm(user_message):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for u, b in conversation:
        messages.append({"role": "user", "content": u})
        if b:
            messages.append({"role": "assistant", "content": b})
    messages.append({"role": "user", "content": user_message})

    if DATABRICKS_HOST and DATABRICKS_TOKEN:
        try:
            resp = requests.post(
                f"{DATABRICKS_HOST}/serving-endpoints/{SARVAM_ENDPOINT}/invocations",
                headers={"Authorization": f"Bearer {DATABRICKS_TOKEN}", "Content-Type": "application/json"},
                json={"messages": messages, "max_tokens": 300, "temperature": 0.7}, timeout=30)
            if resp.status_code == 200:
                c = resp.json()["choices"][0]["message"]["content"]
                return re.sub(r"<think>.*?</think>", "", c, flags=re.DOTALL).strip()
        except Exception as e:
            logger.warning(f"Databricks endpoint fallback: {e}")

    resp = requests.post("https://api.sarvam.ai/v1/chat/completions", headers={
        "api-subscription-key": SARVAM_API_KEY, "Content-Type": "application/json",
    }, json={"model": "sarvam-m", "messages": messages, "max_tokens": 300, "temperature": 0.7}, timeout=30)
    if resp.status_code != 200:
        raise Exception(f"LLM Error {resp.status_code}: {resp.text}")
    c = resp.json()["choices"][0]["message"]["content"]
    return re.sub(r"<think>.*?</think>", "", c, flags=re.DOTALL).strip()


def clean_for_tts(text):
    """Strip markdown/HTML so TTS doesn't read formatting characters aloud."""
    t = re.sub(r'<[^>]+>', '', text)          # HTML tags
    t = re.sub(r'\*\*(.+?)\*\*', r'\1', t)   # **bold**
    t = re.sub(r'\*(.+?)\*', r'\1', t)        # *italic*
    t = re.sub(r'__(.+?)__', r'\1', t)        # __bold__
    t = re.sub(r'_(.+?)_', r'\1', t)          # _italic_
    t = re.sub(r'~~(.+?)~~', r'\1', t)        # ~~strikethrough~~
    t = re.sub(r'`(.+?)`', r'\1', t)          # `code`
    t = re.sub(r'^#{1,6}\s+', '', t, flags=re.MULTILINE)  # headings
    t = re.sub(r'^\s*[-*+]\s+', '', t, flags=re.MULTILINE)  # bullet points
    t = re.sub(r'^\s*\d+\.\s+', '', t, flags=re.MULTILINE)  # numbered lists
    t = re.sub(r'^\s*>\s*', '', t, flags=re.MULTILINE)  # blockquotes
    t = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', t)  # [link](url)
    t = re.sub(r'[|]', ' ', t)               # table pipes
    t = re.sub(r'-{3,}', '', t)              # horizontal rules
    t = re.sub(r'\s+', ' ', t).strip()       # collapse whitespace
    return t


def get_tts_html(text, lang="en"):
    lang_code = LANG_MAP.get(lang[:2] if lang else "en", "en-IN")
    clean_text = clean_for_tts(text)
    try:
        resp = requests.post("https://api.sarvam.ai/text-to-speech", headers={
            "api-subscription-key": SARVAM_API_KEY, "Content-Type": "application/json",
        }, json={
            "inputs": [clean_text[:500]], "target_language_code": lang_code,
            "speaker": "anushka", "model": "bulbul:v2",
            "pitch": 0, "pace": 1.0, "loudness": 1.5, "enable_preprocessing": True,
        })
        if resp.status_code == 200:
            ab = resp.json().get("audios", [None])[0]
            if ab:
                return f'<audio controls autoplay style="width:100%" src="data:audio/wav;base64,{ab}"></audio>'
    except Exception as e:
        logger.error(f"TTS error: {e}")
    return "<p style='color:#999'><em>Audio generation failed</em></p>"


def do_stt(audio_b64, filename="audio.webm"):
    headers = {"api-subscription-key": SARVAM_API_KEY}
    audio_bytes = base64.b64decode(audio_b64)
    files = {"file": (filename, io.BytesIO(audio_bytes), "audio/webm")}
    data = {"model": "saaras:v3", "language_code": "unknown", "with_timestamps": "false"}
    resp = requests.post("https://api.sarvam.ai/speech-to-text", headers=headers, files=files, data=data)
    if resp.status_code != 200:
        raise Exception(f"STT Error: {resp.text}")
    r = resp.json()
    return r.get("transcript", ""), r.get("language_code", "en")


def detect_lang(text):
    if any('\u0900' <= c <= '\u097F' for c in text):
        return "hi"
    if any('\u0B80' <= c <= '\u0BFF' for c in text):
        return "ta"
    if any('\u0C00' <= c <= '\u0C7F' for c in text):
        return "te"
    if any('\u0C80' <= c <= '\u0CFF' for c in text):
        return "kn"
    if any('\u0D00' <= c <= '\u0D7F' for c in text):
        return "ml"
    if any('\u0980' <= c <= '\u09FF' for c in text):
        return "bn"
    return "en"


def format_chat():
    if not conversation:
        return """*Start a conversation! Try asking in Hindi, English, Tamil, or any Indian language.*

**Example queries:**
- "Best time to visit Goa?"
- "5-day Kerala itinerary on budget"
- "Visa process for Thailand from India"
"""
    lines = []
    for u, b in conversation:
        lines.append(f'> **You:** {u}')
        lines.append(f'**Yatra:** {b}')
        lines.append('')
    return '\n\n'.join(lines)


def handle_text(user_text):
    global conversation
    if not user_text.strip():
        return "", format_chat(), "", ""
    try:
        reply = call_llm(user_text)
        lang = detect_lang(user_text)
        audio = get_tts_html(reply, lang)
        conversation.append((user_text, reply))
        lang_name = LANG_NAMES.get(lang, lang)
        return "", format_chat(), audio, f"Language: {lang_name}"
    except Exception as e:
        logger.error(f"Text error: {e}")
        conversation.append((user_text, f"Sorry, error occurred: {e}"))
        return "", format_chat(), "", f"Error: {e}"


def handle_voice_b64(audio_data_str):
    global conversation
    if not audio_data_str or not audio_data_str.strip():
        return format_chat(), "", "No audio data received. Please record first, then click Send Voice."
    try:
        if "base64," in audio_data_str:
            audio_data_str = audio_data_str.split("base64,")[1]
        user_text, lang = do_stt(audio_data_str)
        if not user_text.strip():
            return format_chat(), "", "Could not understand the audio. Please try again."
        reply = call_llm(user_text)
        audio = get_tts_html(reply, lang)
        conversation.append((user_text, reply))
        lang_name = LANG_NAMES.get(lang[:2], lang)
        return format_chat(), audio, f'Heard: "{user_text}" | Language: {lang_name}'
    except Exception as e:
        logger.error(f"Voice error: {e}")
        return format_chat(), "", f"Error: {e}"


def clear_all():
    global conversation
    conversation = []
    return format_chat(), "", "", "", "", ""


logger.info("Building UI...")

# JavaScript injected via head parameter — this runs in the page, not sanitized
HEAD_JS = """
<script>
let yatraRecorder = null;
let yatraChunks = [];
let yatraRecording = false;
let yatraTimer = null;
let yatraSecs = 0;

async function yatraStartRec() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        yatraRecorder = new MediaRecorder(stream);
        yatraChunks = [];
        yatraRecorder.ondataavailable = (e) => { if (e.data.size > 0) yatraChunks.push(e.data); };
        yatraRecorder.onstop = () => {
            const blob = new Blob(yatraChunks, { type: 'audio/webm' });
            const reader = new FileReader();
            reader.onloadend = () => {
                const b64 = reader.result;
                // Find the hidden textbox and set its value
                const el = document.querySelector('#voice-data-box textarea');
                if (el) {
                    const setter = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, 'value').set;
                    setter.call(el, b64);
                    el.dispatchEvent(new Event('input', { bubbles: true }));
                }
                document.getElementById('rec-status-text').textContent = 'Recording saved! Click "Send Voice" to process.';
            };
            reader.readAsDataURL(blob);
            stream.getTracks().forEach(t => t.stop());
        };
        yatraRecorder.start();
        yatraRecording = true;
        yatraSecs = 0;
        document.getElementById('rec-indicator').style.display = 'flex';
        document.getElementById('rec-status-text').textContent = 'Recording...';
        yatraTimer = setInterval(() => {
            yatraSecs++;
            const m = String(Math.floor(yatraSecs/60)).padStart(2,'0');
            const s = String(yatraSecs%60).padStart(2,'0');
            document.getElementById('rec-time').textContent = m + ':' + s;
        }, 1000);
    } catch (err) {
        document.getElementById('rec-status-text').textContent = 'Microphone access denied: ' + err.message;
    }
}

function yatraStopRec() {
    if (yatraRecorder && yatraRecorder.state !== 'inactive') {
        yatraRecorder.stop();
    }
    yatraRecording = false;
    clearInterval(yatraTimer);
    document.getElementById('rec-indicator').style.display = 'none';
}

function yatraToggleRec() {
    if (yatraRecording) {
        yatraStopRec();
    } else {
        yatraStartRec();
    }
}
</script>
"""

# JS for the record button — toggles recording and updates button text
RECORD_BTN_JS = """
() => {
    yatraToggleRec();
    return [];
}
"""

app = gr.Blocks(title="Yatra Voice Agent", head=HEAD_JS)
with app:
    gr.HTML("""<div style="text-align:center;margin-bottom:12px">
        <h1 style="color:#FF6B35;margin:0;font-size:2em">Yatra</h1>
        <p style="color:#555;margin:2px 0;font-size:1.1em">Sovereign Voice Travel Assistant</p>
        <p style="font-size:12px;color:#888;margin:2px 0">Powered by <b>Sarvam AI</b> + <b>Databricks</b> | 22 Indian Languages</p>
        <div style="background:linear-gradient(90deg,#FF9933 33%,#FFFFFF 33%,#FFFFFF 66%,#138808 66%);padding:4px 16px;border-radius:20px;display:inline-block;font-size:11px;font-weight:bold;color:#333;margin-top:4px">
            100% India-Hosted &mdash; Full Data Sovereignty
        </div></div>""")

    chat_display = gr.Markdown(value=format_chat())

    with gr.Tab("Text"):
        with gr.Row():
            text_input = gr.Textbox(
                label="Ask Yatra",
                placeholder="e.g. Best time to visit Goa? / Kerala 5-day budget / Thailand visa",
                scale=4, lines=1)
            text_btn = gr.Button("Send", variant="primary", scale=1)
        text_audio_html = gr.HTML()
        text_status = gr.Textbox(label="Status", interactive=False, max_lines=1)

    with gr.Tab("Voice"):
        gr.HTML("""<div style="text-align:center;padding:16px">
            <p style="color:#555;margin:0 0 8px 0">Use the buttons below to record your voice query in any Indian language</p>
            <div id="rec-indicator" style="display:none;align-items:center;justify-content:center;gap:10px;margin:8px 0">
                <span style="display:inline-block;width:12px;height:12px;background:#dc3545;border-radius:50%;animation:blink 1s infinite"></span>
                <span id="rec-time" style="font-size:22px;font-weight:bold;color:#dc3545">00:00</span>
            </div>
            <p id="rec-status-text" style="color:#888;margin:4px 0;font-size:13px">Click "Record" to start</p>
            <style>@keyframes blink{0%,100%{opacity:1}50%{opacity:0.3}}</style>
        </div>""")
        with gr.Row():
            rec_btn = gr.Button("🎙 Record", variant="secondary", scale=1)
            stop_btn = gr.Button("⏹ Stop", variant="stop", scale=1)
        voice_data = gr.Textbox(label="Audio Data", visible=False, elem_id="voice-data-box")
        voice_btn = gr.Button("Send Voice", variant="primary")
        voice_audio_html = gr.HTML()
        voice_status = gr.Textbox(label="Status", interactive=False, max_lines=1)

    clear_btn = gr.Button("Clear Conversation", variant="secondary")

    gr.Markdown("""---
| Component | Provider | Location |
|-----------|----------|----------|
| **Speech-to-Text** | Sarvam Saaras v3 | India |
| **LLM Brain** | Sarvam-M / Sarvam-30B on Databricks | India |
| **Text-to-Speech** | Sarvam Bulbul v2 | India |
| **Platform** | Databricks | Azure Central India |

*Zero data leaves India. Supports: Hindi, English, Tamil, Telugu, Kannada, Malayalam, Bengali, Gujarati, Marathi, Punjabi, Odia*""")

    # Text handlers
    text_btn.click(handle_text, [text_input], [text_input, chat_display, text_audio_html, text_status])
    text_input.submit(handle_text, [text_input], [text_input, chat_display, text_audio_html, text_status])

    # Voice handlers — Record/Stop use js parameter (client-side only, no server call)
    rec_btn.click(fn=None, inputs=None, outputs=None, js="() => { yatraStartRec(); }")
    stop_btn.click(fn=None, inputs=None, outputs=None, js="() => { yatraStopRec(); }")

    # Send Voice calls server with the base64 data from hidden textbox
    voice_btn.click(handle_voice_b64, [voice_data], [chat_display, voice_audio_html, voice_status])

    clear_btn.click(clear_all, outputs=[chat_display, text_audio_html, voice_audio_html, text_status, voice_status, voice_data])

logger.info("UI built successfully")

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=8000)
