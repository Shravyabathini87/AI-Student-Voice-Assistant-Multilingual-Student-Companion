import os
import time
from datetime import datetime

import altair as alt
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from gnani.stt import GnaniSTTClient
from gnani.tts import GnaniTTSClient, AudioConfig

# ==================================================
# 1. CONFIGURATION
# ==================================================

load_dotenv()

GNANI_API_KEY = os.getenv("GNANI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


st.set_page_config(
    page_title="AI Student Voice Assistant",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="collapsed"
)


LANGUAGES = {
    "Auto Detect": "indic-auto",
    "English": "en-IN",
    "Telugu": "te-IN",
    "Hindi": "hi-IN",
    "Tamil": "ta-IN",
    "Kannada": "kn-IN",
    "Malayalam": "ml-IN",
    "Marathi": "mr-IN",
    "Bengali": "bn-IN",
    "Gujarati": "gu-IN"
}

VOICES = ["Kaveri", "Nalini", "Asmita", "Suhana"]

PAGES = [
    "Dashboard",
    "Voice Assistant",
    "Text Assistant",
    "Chat History",
    "Settings"
]

# UI only
PAGE_ICONS = {
    "Dashboard": "🏠",
    "Voice Assistant": "🎤",
    "Text Assistant": "💬",
    "Chat History": "📚",
    "Settings": "⚙️"
}

EXAMPLE_QUESTIONS = [
    "Explain Artificial Intelligence in simple words",
    "What is the Pythagoras theorem?",
    "How does photosynthesis work?"
]

# ==================================================
# 2. SESSION STATE
# ==================================================

DEFAULTS = {
    "page": "Dashboard",
    "page_navigation": "Dashboard",
    "language": "English",
    "voice": "Kaveri",
    "transcript": "",
    "detected_language": "",
    "answer": "",
    "generated_audio": None,
    "history": [],
    "current_question": "",
    "current_source": ""
}

for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value

# ==================================================
# 3. NAVIGATION FUNCTIONS
# ==================================================

def navigate_to(page):
    st.session_state.page = page
    st.session_state.page_navigation = page


def sync_page():
    st.session_state.page = st.session_state.page_navigation


def use_example(text):
    # UI helper: fills the Text Assistant box with an example question
    st.session_state.dashboard_text_question = text


# ==================================================
# 3B. SMALL UI HELPERS (no backend logic)
# ==================================================

def h(markup):
    st.markdown(markup, unsafe_allow_html=True)


def empty_state(title, text):
    h(
        '<div class="empty-box">'
        f'<b>{title}</b><br>{text}'
        '</div>'
    )


def step_bar(labels, done_flags):
    active_index = next(
        (i for i, done in enumerate(done_flags) if not done),
        len(labels)
    )
    parts = ""
    for i, label in enumerate(labels):
        if done_flags[i]:
            css, mark = "done", "✓"
        elif i == active_index:
            css, mark = "active", str(i + 1)
        else:
            css, mark = "", str(i + 1)
        parts += (
            f'<div class="step {css}">'
            f'<div class="step-num">{mark}</div>{label}</div>'
        )
    h(f'<div class="steps">{parts}</div>')


def history_frame():
    rows = []
    for number, item in enumerate(reversed(st.session_state.history), start=1):
        rows.append({
            "No.": number,
            "Time": item["time"],
            "Source": item["source"],
            "Question": item["question"],
            "Words": len(item["answer"].split())
        })
    return pd.DataFrame(
        rows,
        columns=["No.", "Time", "Source", "Question", "Words"]
    )


# ==================================================
# 5. API KEY CHECK
# ==================================================

if not GNANI_API_KEY:
    st.error("Gnani API key is missing. Check your .env file.")
    st.stop()

if not GROQ_API_KEY:
    st.error("Groq API key is missing. Check your .env file.")
    st.stop()


# ==================================================
# 6. GENERATE AI ANSWER
# ==================================================

def generate_answer(question):

    client = Groq(api_key=GROQ_API_KEY)

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful multilingual AI assistant. "
                    "Answer questions about education, science, "
                    "technology, mathematics and general knowledge. "
                    "Give clear and accurate answers. "
                    "Reply in the same language as the user's question."
                )
            },
            {
                "role": "user",
                "content": question
            }
        ],
        max_completion_tokens=1024,
        reasoning_effort="low"
    )

    return response.choices[0].message.content


# ==================================================
# 7. GENERATE SPEECH
# ==================================================

def generate_speech(answer):

    client = GnaniTTSClient(api_key=GNANI_API_KEY)

    language = LANGUAGES[st.session_state.language]

    if language == "indic-auto":
        language = st.session_state.detected_language or "auto"

    last_error = None

    for attempt in range(2):
        try:

            audio_data = client.synthesize(
                answer[:800],
                voice=st.session_state.voice,
                model="timbre-v2.5",
                language=language,
                speed=1.0,
                audio_config=AudioConfig(
                    sample_rate=48000,
                    encoding="linear_pcm",
                    container="wav"
                )
            )

            if audio_data and len(audio_data) > 100:
                return bytes(audio_data)

            raise ValueError("No valid audio data returned.")

        except Exception as error:
            last_error = error

            if attempt == 0:
                time.sleep(2)

    raise last_error


# ==================================================
# 8. SAVE HISTORY
# ==================================================

def save_history(question, answer, source):

    st.session_state.history.insert(
        0,
        {
            "question": question,
            "answer": answer,
            "source": source,
            "time": datetime.now().strftime("%d-%m-%Y %I:%M %p")
        }
    )


# ==================================================
# 9. PROCESS QUESTION
# ==================================================

def process_question(question, source):

    if not question or not question.strip():
        st.warning("Please enter or record a question.")
        return

    try:

        with st.spinner("AI is generating your answer..."):
            answer = generate_answer(question.strip())

        if answer and answer.strip():

            st.session_state.answer = answer.strip()
            st.session_state.generated_audio = None
            st.session_state.current_question = question.strip()
            st.session_state.current_source = source

            save_history(
                question.strip(),
                answer.strip(),
                source
            )

            st.success("AI answer generated successfully!")

        else:
            st.error("AI did not return an answer.")

    except Exception as error:
        st.error("AI answer generation failed.")
        st.code(str(error))


# ==================================================
# 10. DISPLAY ANSWER (presentation only)
# ==================================================

def show_answer():

    if st.session_state.answer:

        with st.container(border=True):

            h('<div class="ans-label">💡 AI Generated Answer</div>')

            st.write(st.session_state.answer)

            st.caption(
                f"{len(st.session_state.answer.split())} words"
            )


# ==================================================
# 11. SPEAK ANSWER
# ==================================================

def speak_answer():

    if not st.session_state.answer:
        st.warning("Generate an AI answer first.")
        return

    try:

        with st.spinner("Gnani Timbre is generating speech..."):

            audio = generate_speech(
                st.session_state.answer
            )

        st.session_state.generated_audio = audio

        st.success("Voice response generated successfully!")

    except Exception as error:
        st.error("Voice generation failed.")
        st.code(str(error))


# ==================================================
# 12. DISPLAY AUDIO (presentation only)
# ==================================================

def show_audio():

    if st.session_state.generated_audio:

        with st.container(border=True):

            h('<div class="ans-label">🎧 Listen to AI Voice Response</div>')

            st.audio(
                st.session_state.generated_audio,
                format="audio/wav"
            )

            st.download_button(
                "⬇️ Download Voice Response",
                data=st.session_state.generated_audio,
                file_name="AI_Student_Voice_Response.wav",
                mime="audio/wav"
            )


# ==================================================
# VERSION 2 UI: MODERN CREATIVE DASHBOARD
# Purple and lavender, top pill navigation, gradient cards
# ==================================================

PURPLE = "#7c3aed"
LAVENDER = "#c4a8ff"

h("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700;800&display=swap');

html, body, [class*="st-"], .stApp, button, input, textarea {
    font-family: 'Outfit', 'Segoe UI', sans-serif;
}
.stApp {
    background:
        radial-gradient(900px 420px at 8% -8%, #e4d8ff 0%, transparent 60%),
        radial-gradient(700px 420px at 100% 4%, #f7e3ff 0%, transparent 55%),
        #f8f5ff;
}
header[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer { visibility: hidden; }
.block-container {
    max-width: 1200px; padding-top: 1.4rem !important;
    padding-bottom: 3rem !important; overflow: visible !important;
}
h1, h2, h3 { color: #3b1d8a; line-height: 1.4 !important; overflow: visible !important; }

/* Sidebar (secondary panel) */
[data-testid="stSidebar"], [data-testid="stSidebar"] > div:first-child {
    background: linear-gradient(180deg, #efe7ff, #e3d6ff);
}
[data-testid="stSidebar"] * { color: #3b1d8a; }
.side-chip {
    background: #fff; border-radius: 14px; padding: 10px 14px;
    font-size: 14px; font-weight: 600; margin: 7px 0;
    box-shadow: 0 4px 12px rgba(124,58,237,0.10);
}

/* Brand and pill navigation */
.brand2 { display: flex; align-items: center; gap: 12px; font-size: 22px; font-weight: 800; color: #4c1d95; }
.logo2 {
    width: 44px; height: 44px; border-radius: 15px;
    background: linear-gradient(135deg, #7c3aed, #c084fc);
    display: flex; align-items: center; justify-content: center; font-size: 22px;
    animation: float 4s ease-in-out infinite;
}
.brand-tag { font-size: 12px; font-weight: 500; color: #8b6fd0; margin-top: -2px; }

div[role="radiogroup"] { gap: 8px; flex-wrap: wrap; }
div[role="radiogroup"] label {
    background: #fff; border: 1px solid #e3d9fb; border-radius: 999px;
    padding: 8px 18px; cursor: pointer;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
div[role="radiogroup"] label > div:first-child { display: none; }
div[role="radiogroup"] label:hover { transform: translateY(-2px); box-shadow: 0 8px 18px rgba(124,58,237,0.18); }
div[role="radiogroup"] label:has(input:checked) {
    background: linear-gradient(90deg, #7c3aed, #a855f7); border-color: transparent;
    box-shadow: 0 8px 20px rgba(124,58,237,0.32);
}
div[role="radiogroup"] label p { font-weight: 600; font-size: 15px; color: #4c1d95 !important; }
div[role="radiogroup"] label:has(input:checked) p { color: #fff !important; }

/* Hero */
.hero2 {
    position: relative; overflow: hidden; border-radius: 28px; padding: 42px 44px; color: #fff;
    background: linear-gradient(120deg, #6d28d9 0%, #8b5cf6 50%, #c084fc 100%);
    box-shadow: 0 20px 44px rgba(124,58,237,0.30); margin: 18px 0 22px 0;
    animation: pop 0.7s ease both;
}
.hero2 h1 { color: #fff !important; font-size: 42px; font-weight: 800; margin: 0 0 8px 0; line-height: 1.2 !important; }
.hero2 p { color: #f1e8ff; font-size: 17px; max-width: 560px; margin: 0; line-height: 1.7; }
.blob { position: absolute; border-radius: 50%; background: rgba(255,255,255,0.14); animation: float 6s ease-in-out infinite; }
.b1 { width: 230px; height: 230px; right: -50px; top: -70px; }
.b2 { width: 120px; height: 120px; right: 190px; bottom: -45px; animation-delay: 1.2s; }
@keyframes float { 0%, 100% { transform: translateY(0); } 50% { transform: translateY(-12px); } }
@keyframes pop { from { opacity: 0; transform: scale(0.97); } to { opacity: 1; transform: none; } }

.page-head { text-align: center; margin: 22px 0 14px 0; }
.page-head h1 { font-size: 36px; font-weight: 800; margin: 0; }
.page-head p { color: #8b6fd0; font-size: 16px; margin: 4px 0 0 0; }
.sec2 { color: #3b1d8a; font-size: 24px; font-weight: 700; margin: 26px 0 12px 0; }

/* Gradient cards */
.gcard {
    border-radius: 22px; padding: 22px; color: #fff; min-height: 130px;
    box-shadow: 0 14px 30px rgba(124,58,237,0.22);
    transition: transform 0.25s ease;
}
.gcard:hover { transform: translateY(-6px) rotate(-0.6deg); }
.gl { font-size: 14px; opacity: 0.92; font-weight: 500; }
.gv { font-size: 36px; font-weight: 800; margin-top: 8px; }
.gn { font-size: 12px; opacity: 0.85; }
.g1 { background: linear-gradient(135deg, #7c3aed, #a78bfa); }
.g2 { background: linear-gradient(135deg, #6d28d9, #c084fc); }
.g3 { background: linear-gradient(135deg, #8b5cf6, #e0a8ff); }
.g4 { background: linear-gradient(135deg, #5b21b6, #8b5cf6); }

.tile {
    background: #fff; border-radius: 22px; border: 1px solid #eadffd;
    padding: 22px 16px; text-align: center; margin-bottom: 10px; min-height: 150px;
    transition: transform 0.25s ease, box-shadow 0.25s ease;
}
.tile:hover { transform: translateY(-6px); box-shadow: 0 16px 30px rgba(124,58,237,0.18); }
.tile .ico { font-size: 34px; }
.tile h4 { color: #4c1d95; margin: 8px 0 4px 0; font-size: 18px; }
.tile p { color: #7f6fb0; font-size: 13px; margin: 0; line-height: 1.6; }

.model-chip {
    display: inline-block; background: #fff; border: 1.5px solid #d6c4fb; border-radius: 999px;
    padding: 9px 20px; margin: 4px 8px 4px 0; color: #4c1d95; font-weight: 600; font-size: 14px;
}
.banner { background: linear-gradient(90deg, #7c3aed, #c084fc); color: #fff; border-radius: 14px; padding: 10px 16px; font-weight: 700; margin-bottom: 12px; }

div[data-testid="stVerticalBlockBorderWrapper"] {
    background: #fff; border-radius: 22px; border: 1px solid #eadffd !important;
    box-shadow: 0 10px 28px rgba(124,58,237,0.09);
}
div[data-testid="stChatMessage"] { background: #fff; border: 1px solid #eadffd; border-radius: 20px; }

/* Buttons */
div.stButton > button, div.stDownloadButton > button {
    background: linear-gradient(90deg, #7c3aed, #a855f7); color: #fff; border: none;
    border-radius: 999px; min-height: 46px; font-weight: 600;
    box-shadow: 0 8px 18px rgba(124,58,237,0.28);
    transition: transform 0.2s ease, box-shadow 0.2s ease, filter 0.2s ease;
}
div.stButton > button:hover, div.stDownloadButton > button:hover {
    color: #fff; border: none; transform: translateY(-3px) scale(1.01);
    filter: brightness(1.08); box-shadow: 0 14px 26px rgba(124,58,237,0.38);
}
div.stButton > button:active { color: #fff; transform: translateY(0); }
div.stButton > button:focus-visible { outline: 3px solid #d9c8ff; }

div[data-testid="stTextArea"] textarea, div[data-testid="stTextInput"] input {
    border-radius: 18px; background: #fff; border: 1.5px solid #e0d3fb; font-size: 16px;
}
div[data-testid="stTextArea"] textarea:focus { border-color: #8b5cf6; box-shadow: 0 0 0 4px rgba(139,92,246,0.18); }
div[data-baseweb="select"] > div { border-radius: 14px; }
div[data-testid="stAlert"] { border-radius: 16px; }

/* Steps */
.steps { display: flex; gap: 10px; margin: 6px 0 18px 0; flex-wrap: wrap; justify-content: center; }
.step {
    display: flex; align-items: center; gap: 10px; background: #fff; border: 1px solid #eadffd;
    border-radius: 999px; padding: 8px 18px 8px 10px; color: #8b6fd0; font-size: 14px; font-weight: 600;
}
.step-num { width: 28px; height: 28px; border-radius: 50%; background: #efe7ff; color: #6d28d9; display: flex; align-items: center; justify-content: center; font-size: 13px; font-weight: 800; }
.step.active { color: #4c1d95; border-color: #a855f7; box-shadow: 0 6px 16px rgba(168,85,247,0.25); }
.step.active .step-num { background: #7c3aed; color: #fff; }
.step.done .step-num { background: #22c55e; color: #fff; }

.ans-label { color: #6d28d9; font-weight: 700; font-size: 16px; margin-bottom: 6px; }
.empty-box { background: #fbf9ff; border: 2px dashed #d6c4fb; border-radius: 22px; padding: 34px 20px; text-align: center; color: #7f6fb0; font-size: 15px; line-height: 1.7; }
.empty-box b { color: #4c1d95; font-size: 17px; }
.pill { display: inline-block; padding: 3px 14px; border-radius: 999px; font-size: 12px; font-weight: 700; background: #efe7ff; color: #6d28d9; }
.pill.text { background: #fde7ff; color: #a21caf; }
.footer { text-align: center; color: #9a8bc4; font-size: 13px; padding: 20px; }

@media (max-width: 768px) { .hero2 { padding: 26px 22px; } .hero2 h1 { font-size: 30px; } }
@media (prefers-reduced-motion: reduce) { * { transition: none !important; animation: none !important; } }
</style>
""")


def style_chart(chart, height=260):
    return (
        chart.properties(height=height)
        .configure_view(strokeWidth=0)
        .configure_axis(
            labelColor="#7f6fb0", titleColor="#7f6fb0",
            gridColor="#efe7ff", domainColor="#e0d3fb"
        )
        .configure_legend(labelColor="#4c1d95", titleColor="#7f6fb0")
    )


def page_head(title, sub):
    h(f'<div class="page-head"><h1>{title}</h1><p>{sub}</p></div>')


def gcard(css, label, value, note=""):
    h(
        f'<div class="gcard {css}"><div class="gl">{label}</div>'
        f'<div class="gv">{value}</div><div class="gn">{note}</div></div>'
    )


# ==================================================
# 13. NAVIGATION (top pill bar) + SIDEBAR (status panel)
# ==================================================

with st.sidebar:

    h('<div class="brand2"><div class="logo2">🎙️</div>AI Assistant</div>')
    st.caption("Multilingual Student Companion")

    st.divider()

    h('<b>🔌 API Status</b>')
    h('<div class="side-chip">✅ Gnani API Connected</div>')
    h('<div class="side-chip">✅ Groq API Connected</div>')

    st.divider()

    h('<b>🤖 AI Models</b>')
    st.caption("🎤 Prisma - Speech to Text")
    st.caption("🧠 Groq - Answer Generation")
    st.caption("🔊 Timbre - Text to Speech")

    st.divider()

    st.caption("AI Student Voice Assistant")
    st.caption("Version 1.0")


nav_left, nav_right = st.columns([1.15, 3])

with nav_left:
    h('<div class="brand2"><div class="logo2">🎙️</div><div>AI Assistant'
      '<div class="brand-tag">Student Companion</div></div></div>')

with nav_right:
    st.radio(
        "Choose a page",
        PAGES,
        key="page_navigation",
        label_visibility="collapsed",
        horizontal=True,
        on_change=sync_page,
        format_func=lambda name: f"{PAGE_ICONS[name]}  {name}"
    )


# ==================================================
# 14. DASHBOARD PAGE
# ==================================================

if st.session_state.page == "Dashboard":

    h(
        '<div class="hero2"><div class="blob b1"></div><div class="blob b2"></div>'
        '<h1>Ask. Listen. Learn. ✨</h1>'
        '<p>Your multilingual study buddy. Speak or type a question, '
        'get a clear answer, and hear it read back to you.</p></div>'
    )

    history = st.session_state.history
    voice_count = sum(1 for item in history if item["source"] == "Voice")
    text_count = sum(1 for item in history if item["source"] == "Text")
    frame = history_frame()
    average_words = int(frame["Words"].mean()) if len(frame) else 0

    k1, k2, k3, k4 = st.columns(4)

    with k1:
        gcard("g1", "💬 Questions Asked", len(history), "This session")
    with k2:
        gcard("g2", "🎤 Voice Questions", voice_count, "Spoken to the mic")
    with k3:
        gcard("g3", "🌐 Language", st.session_state.language, f"Voice: {st.session_state.voice}")
    with k4:
        gcard("g4", "📝 Avg. Answer Length", average_words, "Words per answer")

    h('<div class="sec2">📈 Your learning pulse</div>')

    c1, c2 = st.columns([1.6, 1])

    with c1:
        with st.container(border=True):
            st.markdown("**Answer length over time (words)**")

            if history:
                chart = alt.Chart(frame).mark_area(
                    line={"color": PURPLE}, point={"color": PURPLE, "size": 70},
                    color=LAVENDER, opacity=0.45, interpolate="monotone"
                ).encode(
                    x=alt.X("No.:Q", title="Question", axis=alt.Axis(tickMinStep=1)),
                    y=alt.Y("Words:Q"),
                    tooltip=["No.:Q", "Question:N", "Source:N", "Words:Q"]
                ).interactive()

                st.altair_chart(style_chart(chart), use_container_width=True)
            else:
                empty_state("Nothing to chart yet", "Ask your first question to light this up.")

    with c2:
        with st.container(border=True):
            st.markdown("**Voice vs text**")

            if history:
                source_df = pd.DataFrame({
                    "Source": ["Voice", "Text"],
                    "Questions": [voice_count, text_count]
                })

                donut = alt.Chart(source_df).mark_arc(
                    innerRadius=58, outerRadius=100, cornerRadius=6
                ).encode(
                    theta=alt.Theta("Questions:Q"),
                    color=alt.Color(
                        "Source:N",
                        scale=alt.Scale(domain=["Voice", "Text"], range=[PURPLE, LAVENDER])
                    ),
                    tooltip=["Source:N", "Questions:Q"]
                )

                st.altair_chart(style_chart(donut), use_container_width=True)
            else:
                empty_state("No data yet", "Voice and text split shows here.")

    h('<div class="sec2">🚀 Where to next?</div>')

    t1, t2, t3, t4 = st.columns(4)

    with t1:
        h('<div class="tile"><div class="ico">🎤</div><h4>Voice Assistant</h4>'
          '<p>Speak your questions and receive spoken AI answers.</p></div>')
        st.button("Open Voice Assistant", key="open_voice", use_container_width=True,
                  on_click=navigate_to, args=("Voice Assistant",))

    with t2:
        h('<div class="tile"><div class="ico">💬</div><h4>Text Assistant</h4>'
          '<p>Type any question and get an intelligent AI answer.</p></div>')
        st.button("Open Text Assistant", key="open_text", use_container_width=True,
                  on_click=navigate_to, args=("Text Assistant",))

    with t3:
        h('<div class="tile"><div class="ico">📚</div><h4>Chat History</h4>'
          '<p>Review your previous questions and AI answers.</p></div>')
        st.button("View Chat History", key="open_history", use_container_width=True,
                  on_click=navigate_to, args=("Chat History",))

    with t4:
        h('<div class="tile"><div class="ico">⚙️</div><h4>Settings</h4>'
          '<p>Change your language and AI voice preferences.</p></div>')
        st.button("Open Settings", key="open_settings", use_container_width=True,
                  on_click=navigate_to, args=("Settings",))

    h('<div class="sec2">🧠 Powered by</div>')

    h(
        '<span class="model-chip">🎤 Gnani Prisma · Speech recognition</span>'
        '<span class="model-chip">🧠 Groq GPT-OSS 20B · Answer generation</span>'
        '<span class="model-chip">🔊 Gnani Timbre · Voice generation</span>'
    )


# ==================================================
# 15. VOICE ASSISTANT PAGE
# ==================================================

elif st.session_state.page == "Voice Assistant":

    page_head("🎤 Voice Assistant", "Ask your question using your microphone.")

    step_bar(
        ["Record", "Transcript", "AI answer", "Voice reply"],
        [
            bool(st.session_state.transcript),
            bool(st.session_state.transcript),
            bool(st.session_state.answer),
            st.session_state.generated_audio is not None
        ]
    )

    spacer_a, centre, spacer_b = st.columns([1, 2.4, 1])

    with centre:

        with st.container(border=True):

            st.markdown("### Step 1: Record Your Question")

            audio_input = st.audio_input("Record your voice")

            if st.button(
                "Convert Speech to Text",
                type="primary",
                key="convert_speech",
                use_container_width=True
            ):

                if audio_input is None:
                    st.warning("Please record your voice first.")

                else:

                    try:

                        client = GnaniSTTClient(
                            api_key=GNANI_API_KEY
                        )

                        with st.spinner("Prisma is recognizing your speech..."):

                            audio_input.seek(0)

                            result = client.transcribe(
                                audio_input,
                                language_code=LANGUAGES[
                                    st.session_state.language
                                ],
                                model="prisma-v3.0",
                                format="transcribe"
                            )

                        transcript = result.get("transcript", "")

                        if transcript:

                            st.session_state.transcript = transcript

                            st.session_state.detected_language = (
                                result.get("detected_language_code", "")
                            )

                            st.session_state.answer = ""
                            st.session_state.generated_audio = None

                            st.success("Speech converted successfully!")

                        else:
                            st.warning("No speech detected.")

                    except Exception as error:
                        st.error("Speech recognition failed.")
                        st.code(str(error))

        if st.session_state.transcript:

            with st.container(border=True):

                st.markdown("### Step 2: Your Transcript")

                st.info(st.session_state.transcript)

                if st.session_state.detected_language:
                    st.caption(
                        "Detected language: "
                        + st.session_state.detected_language
                    )

                if st.button(
                    "Generate AI Answer",
                    type="primary",
                    key="voice_generate",
                    use_container_width=True
                ):

                    process_question(
                        st.session_state.transcript,
                        "Voice"
                    )

        if st.session_state.answer:

            show_answer()

            if st.button(
                "Speak AI Answer",
                type="primary",
                key="voice_speak",
                use_container_width=True
            ):
                speak_answer()

            show_audio()


# ==================================================
# 16. TEXT ASSISTANT PAGE (chat-style conversation)
# ==================================================

elif st.session_state.page == "Text Assistant":

    page_head("💬 Text Assistant", "Ask questions about any topic.")

    conversation = st.container()

    with st.container(border=True):

        question = st.text_area(
            "Enter your question",
            key="dashboard_text_question",
            placeholder="Example: Explain Artificial Intelligence...",
            height=130
        )

        if st.button(
            "Generate AI Answer",
            type="primary",
            key="text_generate",
            use_container_width=True
        ):
            process_question(question, "Text")

        st.caption("Need an idea? Try one of these:")

        e1, e2, e3 = st.columns(3)

        for column, number in zip((e1, e2, e3), range(3)):
            with column:
                st.button(
                    EXAMPLE_QUESTIONS[number],
                    key=f"example_{number}",
                    use_container_width=True,
                    on_click=use_example,
                    args=(EXAMPLE_QUESTIONS[number],)
                )

    with conversation:

        if st.session_state.answer:

            with st.chat_message("user", avatar="🧑‍🎓"):
                st.write(st.session_state.current_question)

            with st.chat_message("assistant", avatar="✨"):
                st.write(st.session_state.answer)
                st.caption(
                    f"{len(st.session_state.answer.split())} words"
                )

            if st.button(
                "Speak AI Answer",
                type="primary",
                key="text_speak"
            ):
                speak_answer()

            show_audio()

        else:
            empty_state(
                "Start a conversation",
                "Type a question below or tap an idea, then generate your answer."
            )


# ==================================================
# 17. CHAT HISTORY PAGE (card grid)
# ==================================================

elif st.session_state.page == "Chat History":

    page_head("📚 Chat History", "Review your previous questions and answers.")

    if not st.session_state.history:

        empty_state("No chat history available yet", "Your questions and answers will show up here as cards.")

    else:

        f1, f2 = st.columns([2, 2])

        with f1:
            search_text = st.text_input(
                "Search questions and answers",
                key="history_search",
                placeholder="Type a keyword..."
            )

        with f2:
            source_filter = st.radio(
                "Input type",
                ["All", "Voice", "Text"],
                key="history_source_filter",
                horizontal=True
            )

        st.write(
            f"Total questions: {len(st.session_state.history)}"
        )

        matches = []

        for index, item in enumerate(st.session_state.history):

            if source_filter != "All" and item["source"] != source_filter:
                continue

            if search_text and search_text.lower() not in (
                item["question"] + " " + item["answer"]
            ).lower():
                continue

            matches.append((index, item))

        grid = st.columns(2, gap="large")

        for position, (index, item) in enumerate(matches):

            with grid[position % 2]:

                with st.container(border=True):

                    pill_class = "pill" if item["source"] == "Voice" else "pill text"
                    h(
                        f'<span class="{pill_class}">{item["source"]}</span> '
                        f'<span style="color:#9a8bc4;font-size:13px;">'
                        f'Question {index + 1} - {item["time"]}</span>'
                    )

                    st.markdown("**Question:**")
                    st.write(item["question"])

                    with st.expander("View AI Answer"):
                        st.markdown("**AI Answer:**")
                        st.write(item["answer"])

        if not matches:
            st.info("No conversations match your filters.")

        if st.button("Clear Chat History", key="clear_history"):

            st.session_state.history = []

            st.success("Chat history cleared.")

            st.rerun()


# ==================================================
# 18. SETTINGS PAGE
# ==================================================

elif st.session_state.page == "Settings":

    page_head("⚙️ Settings", "Customize your AI assistant.")

    left, right = st.columns(2, gap="large")

    with left:

        with st.container(border=True):

            h('<div class="banner">🌐 Language Preferences</div>')

            language_names = list(LANGUAGES.keys())

            new_language = st.selectbox(
                "Select application language",
                language_names,
                index=language_names.index(
                    st.session_state.language
                ),
                key="settings_language"
            )

    with right:

        with st.container(border=True):

            h('<div class="banner">🔊 Voice Preferences</div>')

            new_voice = st.selectbox(
                "Select AI voice",
                VOICES,
                index=VOICES.index(
                    st.session_state.voice
                ),
                key="settings_voice"
            )

    pad_a, middle, pad_b = st.columns([1, 1, 1])

    with middle:

        if st.button(
            "Save Settings",
            type="primary",
            key="save_settings",
            use_container_width=True
        ):

            st.session_state.language = new_language
            st.session_state.voice = new_voice

            st.success("Settings saved successfully!")

    h('<div class="sec2">🔌 Connected AI Services</div>')

    s1, s2, s3 = st.columns(3)

    with s1:
        st.success("Gnani Prisma V3.0")
    with s2:
        st.success("Groq GPT-OSS 20B")
    with s3:
        st.success("Gnani Timbre V2.5")

    with st.container(border=True):

        st.markdown("### ℹ️ Application Information")

        st.write("Application: AI Student Voice Assistant")
        st.write("Interface: Streamlit")
        st.write("Programming Language: Python")
        st.write("Version: 1.0")


# ==================================================
# 19. FOOTER
# ==================================================

st.divider()

h('<div class="footer">AI Student Voice Assistant | Powered by Gnani AI and Groq</div>')
