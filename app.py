"""
app.py — Streamlit chat UI for the BigBasket RAG assistant.

Presentation layer follows a Soft-UI (neumorphic) design system:
cool-grey single surface, dual RGB shadows, inset wells, violet accent.
Typography is Roboto with a compact commerce type scale.

Run: streamlit run app.py
"""

import html
import os
import random
from collections import Counter

import streamlit as st

from rag import (
    FALLBACK_MSG,
    GUARDRAIL_MSG,
    RAGPipeline,
    SIM_THRESHOLD,
    TOP_K,
    live_request_kwargs,
)
from stt import RealtimeTranscriber, TranscriptionError, wav_to_pcm16_24k

st.set_page_config(page_title="BigBasket Assistant", page_icon="🧺", layout="wide")

# ---------------------------------------------------------------------------
# ICONS (inline SVG so raw-HTML cards stay self-contained)
# ---------------------------------------------------------------------------
BASKET_SVG = (
    '<svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">'
    '<path d="M17.21 9l-4.38-6.56a.993.993 0 0 0-.83-.42c-.32 0-.64.14-.83.43L6.79 9H2c-.55 0-1 .45-1 1 '
    '0 .09.01.18.04.27l2.54 9.27c.23.84 1 1.46 1.92 1.46h13c.92 0 1.69-.62 1.93-1.46l2.54-9.27L23 10c0-.55-.45-1-1-1h-4.79z'
    'M9 9l3-4.4L15 9H9zm3 8c-1.1 0-2-.9-2-2s.9-2 2-2 2 .9 2 2-.9 2-2 2z"/></svg>'
)
TAG_SVG = (
    '<svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">'
    '<path d="M21.41 11.58l-9-9C12.05 2.22 11.55 2 11 2H4c-1.1 0-2 .9-2 2v7c0 .55.22 1.05.59 1.42l9 9c.36.36.86.58 '
    '1.41.58s1.05-.22 1.41-.59l7-7c.37-.36.59-.86.59-1.41s-.23-1.06-.59-1.42zM5.5 7C4.67 7 4 6.33 4 5.5S4.67 4 5.5 4 '
    '7 4.67 7 5.5 6.33 7 5.5 7z"/></svg>'
)

# ---------------------------------------------------------------------------
# DESIGN SYSTEM
# ---------------------------------------------------------------------------
st.html(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&display=swap');

:root{
  --surface:#E0E5EC;
  --ink:#3D4852;
  --muted:#6B7280;
  --accent:#6C63FF;
  --accent-soft:#8B84FF;
  --accent-ink:#4F46D6;
  --teal:#38B2AC;
  --teal-ink:#0F6C68;
  --danger-ink:#8A3125;
  --raised:9px 9px 16px rgb(163,177,198,0.6), -9px -9px 16px rgba(255,255,255,0.5);
  --raised-hover:12px 12px 20px rgb(163,177,198,0.7), -12px -12px 20px rgba(255,255,255,0.6);
  --raised-sm:5px 5px 10px rgb(163,177,198,0.6), -5px -5px 10px rgba(255,255,255,0.5);
  --pressed:inset 6px 6px 10px rgb(163,177,198,0.6), inset -6px -6px 10px rgba(255,255,255,0.5);
  --pressed-deep:inset 10px 10px 20px rgb(163,177,198,0.7), inset -10px -10px 20px rgba(255,255,255,0.6);
  --pressed-sm:inset 3px 3px 6px rgb(163,177,198,0.6), inset -3px -3px 6px rgba(255,255,255,0.5);
  --ease:300ms ease-out;
}

html{ scroll-behavior:smooth; }

html, body, .stApp, [data-testid="stAppViewContainer"]{
  background-color:#E0E5EC !important;
  background-image:none !important;
}

html, body, .stApp, button, input, textarea, select, [data-testid="stSidebar"]{
  font-family:'Roboto', Arial, sans-serif;
}

.stApp, .stApp p, .stApp li, .stApp label, [data-testid="stMarkdownContainer"]{
  color:#3D4852;
  line-height:1.5;
}

h1,h2,h3,h4,h5,h6,.font-display{
  font-family:'Roboto', Arial, sans-serif !important;
  letter-spacing:-0.025em;
  color:#3D4852;
}

/* Chrome --------------------------------------------------------------- */
#MainMenu, [data-testid="stDeployButton"], .stDeployButton, [data-testid="stDecoration"]{
  display:none !important;
}
header[data-testid="stHeader"]{ background:transparent !important; }
[data-testid="stToolbar"]{ right:0.6rem; }

.block-container{
  max-width:1360px;
  padding-top:0.9rem;
  padding-bottom:8rem;
}

/* Floating navigation bar ---------------------------------------------- */
.top-nav{
  display:flex;
  align-items:center;
  justify-content:space-between;
  gap:18px;
  flex-wrap:wrap;
  padding:13px 22px;
  border-radius:32px;
  background:rgba(224,229,236,0.97);
  backdrop-filter:blur(14px);
  box-shadow:var(--raised);
  margin-bottom:28px;
}
.stElementContainer:has(.top-nav){
  position:sticky;
  top:0;
  z-index:100;
}
.brand{ display:flex; align-items:center; gap:14px; }
.brand-mark{
  width:48px; height:48px; flex:0 0 48px;
  display:grid; place-items:center;
  border-radius:9999px;
  background:#E0E5EC;
  box-shadow:var(--pressed-deep);
  color:#6C63FF;
}
.brand-mark svg{ width:23px; height:23px; }
.brand-text{
  display:flex; flex-direction:column; gap:2px;
  font-family:'Roboto', Arial, sans-serif;
  font-weight:800; font-size:1.06rem; letter-spacing:-0.02em; color:#3D4852;
}
.brand-sub{
  font-family:'Roboto', Arial, sans-serif;
  font-size:0.66rem; font-weight:600; letter-spacing:0.16em;
  text-transform:uppercase; color:#6B7280;
}
.status-pill{
  display:inline-flex; align-items:center; gap:9px;
  padding:9px 16px; border-radius:9999px;
  background:#E0E5EC; box-shadow:var(--pressed-sm);
  font-size:0.75rem; font-weight:600; letter-spacing:0.02em; color:#6B7280;
}
.status-dot{
  width:8px; height:8px; border-radius:9999px; background:#38B2AC;
  box-shadow:0 0 0 4px rgba(56,178,172,0.16);
}

/* Typography blocks ----------------------------------------------------- */
.eyebrow{
  display:inline-flex; align-items:center; gap:8px;
  padding:8px 15px; border-radius:9999px;
  background:#E0E5EC; box-shadow:var(--pressed-sm);
  font-size:0.68rem; font-weight:700; letter-spacing:0.16em;
  text-transform:uppercase; color:#4F46D6;
}
.eyebrow::before{
  content:""; width:6px; height:6px; border-radius:9999px;
  background:#6C63FF; box-shadow:0 0 0 3px rgba(108,99,255,0.16);
}
.hero-title{
  font-family:'Roboto', Arial, sans-serif;
  font-weight:700; font-size:clamp(1.55rem, 2.35vw, 2rem);
  line-height:1.15; letter-spacing:-0.02em; color:#3D4852;
  margin:18px 0 12px;
  text-wrap:balance;
}
.hero-title .accent{ color:#6C63FF; }
.hero-subtitle{
  color:#6B7280; font-size:0.98rem; line-height:1.6; max-width:33em;
  margin-bottom:6px;
}
.section-label{
  display:flex; align-items:center; gap:11px;
  font-size:0.68rem; font-weight:700; letter-spacing:0.17em;
  text-transform:uppercase; color:#6B7280;
  margin:24px 0 12px;
}
.section-label::before{
  content:""; width:26px; height:6px; border-radius:9999px;
  background:#6C63FF; box-shadow:var(--pressed-sm);
}

/* Buttons --------------------------------------------------------------- */
.stButton > button{
  background:#E0E5EC !important;
  color:#3D4852 !important;
  border:none !important;
  border-radius:16px !important;
  padding:0.7rem 1.05rem !important;
  min-height:46px;
  font-family:'Roboto', Arial, sans-serif !important;
  font-weight:600 !important;
  font-size:0.9rem !important;
  box-shadow:var(--raised-sm) !important;
  transition:transform var(--ease), box-shadow var(--ease), color var(--ease) !important;
}
.stButton > button:hover{
  color:#4F46D6 !important;
  transform:translateY(-1px);
  box-shadow:var(--raised-hover) !important;
}
.stButton > button:active{
  transform:translateY(1px);
  box-shadow:var(--pressed) !important;
}
.stButton > button:focus-visible{
  outline:2px solid #6C63FF;
  outline-offset:3px;
}

/* Suggestion list ------------------------------------------------------- */
.st-key-suggestions .stButton > button{
  width:100%;
  justify-content:flex-start;
  text-align:left;
  min-height:62px;
  padding:14px 20px !important;
  border-radius:18px !important;
  font-weight:500 !important;
  font-size:0.94rem !important;
  line-height:1.35 !important;
}
.st-key-suggestions .stButton > button div[data-testid="stMarkdownContainer"]{
  width:100%;
  text-align:left;
}
.st-key-suggestions .stButton > button > div{
  width:100%;
  justify-content:flex-start !important;
}
.st-key-suggestions .stButton > button p{
  text-align:left;
  white-space:normal;
  line-height:1.35;
}
.st-key-suggestions [data-testid="stVerticalBlock"]{ gap:10px; }

/* Aisle tiles ----------------------------------------------------------- */
.st-key-aisles .stButton > button{
  width:100%;
  min-height:64px;
  padding:12px 12px !important;
  border-radius:18px !important;
  font-size:0.8rem !important;
  font-weight:600 !important;
  line-height:1.3 !important;
}
.st-key-aisles .stButton > button p{
  white-space:normal;
  line-height:1.3;
  text-align:center;
}
.st-key-aisles .stButton > button div[data-testid="stMarkdownContainer"]{
  justify-content:center;
  text-align:center;
}

/* Nested-depth hero visual --------------------------------------------- */
.depth-stage{
  position:relative;
  display:flex; flex-direction:column; align-items:center; gap:22px;
  padding:34px 26px 28px;
  border-radius:32px;
  background:#E0E5EC;
  box-shadow:var(--raised);
  overflow:hidden;
}
.depth-ring{
  width:232px; height:232px;
  display:grid; place-items:center;
  border-radius:9999px;
  background:#E0E5EC;
  box-shadow:var(--raised-hover);
  transition:transform 500ms ease-out;
}
.depth-ring:hover{ transform:scale(1.02); }
.depth-well{
  width:168px; height:168px;
  display:grid; place-items:center;
  border-radius:9999px;
  background:#E0E5EC;
  box-shadow:var(--pressed-deep);
}
.depth-core{
  width:112px; height:112px;
  display:grid; place-items:center;
  border-radius:9999px;
  background:#E0E5EC;
  box-shadow:var(--raised-sm);
  color:#6C63FF;
  transition:transform 500ms ease-out;
  animation:float 3s ease-in-out infinite;
}
.depth-ring:hover .depth-core{ transform:rotate(180deg); }
.depth-core svg{ width:44px; height:44px; }
@keyframes float{
  0%, 100%{ margin-top:0; }
  50%{ margin-top:-8px; }
}
.depth-dot{
  position:absolute; border-radius:9999px; background:#E0E5EC;
}
.depth-dot.a{ width:20px; height:20px; top:26px; left:32px; box-shadow:var(--raised-sm); }
.depth-dot.b{ width:13px; height:13px; bottom:40px; right:38px; box-shadow:var(--pressed-sm); }
.depth-dot.c{ width:9px; height:9px; top:52px; right:58px; box-shadow:var(--pressed-sm); }
.depth-caption{ text-align:center; }
.depth-title{
  display:block;
  font-family:'Roboto', Arial, sans-serif;
  font-size:0.98rem; font-weight:700; color:#3D4852;
}
.depth-sub{
  display:block; margin-top:5px;
  font-size:0.78rem; color:#6B7280;
}

/* Product cards --------------------------------------------------------- */
.card-grid{
  display:grid;
  grid-template-columns:repeat(auto-fit, minmax(158px, 1fr));
  gap:14px;
}
.shelf-grid{ grid-template-columns:repeat(3, minmax(0, 1fr)); }
.shelf-grid .product-card{ padding:13px 12px; border-radius:20px; }
.shelf-grid .product-icon{ width:32px; height:32px; flex:0 0 32px; border-radius:10px; }
.shelf-grid .product-icon svg{ width:15px; height:15px; }
.shelf-grid .product-name{ font-size:0.8rem; margin-top:10px; }
.shelf-grid .product-brand{ font-size:0.7rem; }
.shelf-grid .price-chip{ font-size:0.86rem; padding:4px 9px; }
.product-card{
  display:flex; flex-direction:column;
  padding:16px;
  border-radius:24px;
  background:#E0E5EC;
  box-shadow:var(--raised-sm);
  transition:transform var(--ease), box-shadow var(--ease);
}
.product-card:hover{
  transform:translateY(-2px);
  box-shadow:var(--raised-hover);
}
.product-head{
  display:flex; align-items:center; justify-content:space-between; gap:8px;
}
.product-icon{
  width:38px; height:38px; flex:0 0 38px;
  display:grid; place-items:center;
  border-radius:12px;
  background:#E0E5EC;
  box-shadow:var(--pressed-deep);
  color:#6C63FF;
}
.product-icon svg{ width:18px; height:18px; }
.product-index{
  padding:4px 9px; border-radius:9999px;
  background:#E0E5EC; box-shadow:var(--pressed-sm);
  font-size:0.68rem; font-weight:700; color:#6B7280;
}
.product-name{
  margin-top:12px;
  font-family:'Roboto', Arial, sans-serif;
  font-size:0.9rem; font-weight:500; line-height:1.35; color:#3D4852;
  display:-webkit-box; -webkit-line-clamp:2; -webkit-box-orient:vertical;
  overflow:hidden;
}
.product-brand{ margin-top:3px; font-size:0.75rem; color:#6B7280; }
.product-foot{
  display:flex; align-items:center; justify-content:space-between;
  gap:8px; flex-wrap:wrap; margin-top:12px;
}
.price-chip{
  padding:5px 11px; border-radius:12px;
  background:#E0E5EC; box-shadow:var(--raised-sm);
  font-family:'Roboto', Arial, sans-serif;
  font-size:0.98rem; font-weight:700; color:#3D4852;
  font-variant-numeric:tabular-nums;
}
.price-chip.policy{ font-size:0.7rem; color:#4F46D6; }
.match-chip{
  font-size:0.7rem; font-weight:700; letter-spacing:0.01em; color:#0F6C68;
}
.product-details{ margin-top:10px; }
.product-details summary{
  cursor:pointer; list-style:none;
  font-size:0.74rem; font-weight:700; color:#4F46D6;
}
.product-details summary::-webkit-details-marker{ display:none; }
.product-details summary::after{ content:"  +"; }
.product-details[open] summary::after{ content:"  –"; }
.product-details p{
  margin:8px 0 0; font-size:0.74rem; line-height:1.55; color:#6B7280;
}

/* Telemetry chips ------------------------------------------------------- */
.telemetry{ display:flex; flex-wrap:wrap; gap:8px; margin:0 0 14px; }
.chip{
  display:inline-flex; align-items:center; gap:7px;
  padding:5px 12px; border-radius:9999px;
  background:#E0E5EC; box-shadow:var(--pressed-sm);
  font-size:0.74rem; font-weight:600; color:#3D4852;
  font-variant-numeric:tabular-nums;
}
.chip-key{
  font-size:0.62rem; font-weight:700; letter-spacing:0.1em;
  text-transform:uppercase; color:#6B7280;
}
.chip.hit{ color:#0F6C68; }
.chip.accent{ color:#4F46D6; }

/* Chat ------------------------------------------------------------------ */
.st-key-chat{ max-width:860px; margin:0 auto; }
[data-testid="stChatMessage"]{
  background:transparent !important;
  padding:9px 0;
  gap:14px;
}
[data-testid="stChatMessageAvatarUser"],
[data-testid="stChatMessageAvatarAssistant"],
[data-testid="stChatMessageAvatarCustom"]{
  background:#E0E5EC !important;
  border-radius:9999px !important;
  box-shadow:var(--pressed-deep);
  width:44px !important; height:44px !important;
  display:grid; place-items:center;
  color:#6B7280 !important;
}
[data-testid="stChatMessageAvatarAssistant"]{ color:#4F46D6 !important; }
[data-testid="stChatMessageAvatarUser"] svg,
[data-testid="stChatMessageAvatarAssistant"] svg{
  fill:currentColor; color:inherit;
}
[data-testid="stChatMessageContent"]{
  background:#E0E5EC !important;
  border:none !important;
  border-radius:24px !important;
  padding:18px 22px !important;
  box-shadow:var(--raised) !important;
  font-size:0.97rem;
  line-height:1.65;
  color:#3D4852 !important;
}
[data-testid="stChatMessageContent"] p,
[data-testid="stChatMessageContent"] li,
[data-testid="stChatMessageContent"] span,
[data-testid="stChatMessageContent"] strong,
[data-testid="stChatMessageContent"] em{
  color:#3D4852 !important;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageContent"][aria-label*="user"]) [data-testid="stChatMessageContent"]{
  box-shadow:var(--pressed) !important;
  border-radius:24px 8px 24px 24px !important;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageContent"][aria-label*="assistant"]) [data-testid="stChatMessageContent"]{
  border-radius:8px 24px 24px 24px !important;
}
[data-testid="stChatMessage"]:has([aria-label="Chat message from assistant"]) [data-testid="stChatMessageAvatarCustom"]{
  background:#E0E5EC !important;
  box-shadow:var(--pressed-deep);
}
[data-testid="stChatMessage"]:has([aria-label="Chat message from assistant"]) [data-testid="stChatMessageAvatarCustom"] [data-testid="stIconMaterial"]{
  color:#4F46D6 !important;
}
[data-testid="stChatMessage"]:has([aria-label="Chat message from user"]) [data-testid="stChatMessageAvatarCustom"] [data-testid="stIconMaterial"]{
  color:#6B7280 !important;
}
[data-testid="stChatMessageContent"] [data-testid="stExpander"]{
  box-shadow:var(--pressed) !important;
  background:transparent !important;
  margin-top:14px;
}

/* Notices --------------------------------------------------------------- */
.notice{
  display:flex; gap:14px; align-items:flex-start;
  padding:18px 20px;
  border-radius:22px;
  background:#E0E5EC;
  box-shadow:var(--pressed);
}
.notice-icon{
  flex:0 0 34px; width:34px; height:34px;
  display:grid; place-items:center;
  border-radius:9999px;
  background:#E0E5EC; box-shadow:var(--pressed-deep);
  font-family:'Roboto', Arial, sans-serif;
  font-weight:800; font-size:0.95rem; color:#8A3125;
}
.notice strong{
  display:block; margin-bottom:4px;
  font-family:'Roboto', Arial, sans-serif;
  font-size:0.9rem; font-weight:700; color:#8A3125;
}
.notice p{ margin:0; font-size:0.9rem; line-height:1.6; color:#3D4852; }

/* Live delivery strip --------------------------------------------------- */
.live-strip{
  display:flex; align-items:center; gap:9px; flex-wrap:wrap;
  margin:0 0 14px; padding:11px 15px; border-radius:16px;
  background:#E0E5EC; box-shadow:var(--pressed-sm);
  font-size:0.82rem; color:#6B7280;
}
.live-strip b{ color:#0F6C68; font-weight:700; }
.live-dot{
  width:8px; height:8px; border-radius:50%; flex:0 0 auto;
  background:#0F6C68; box-shadow:0 0 0 3px rgba(15,108,104,0.18);
}

/* Expanders ------------------------------------------------------------- */
[data-testid="stExpander"]{
  background:#E0E5EC !important;
  border:none !important;
  border-radius:24px !important;
  box-shadow:var(--raised-sm) !important;
  overflow:hidden;
}
[data-testid="stExpander"] details{ background:transparent !important; border:none !important; }
[data-testid="stExpander"] summary{
  padding:14px 18px !important;
  border:none !important;
  font-weight:600;
  color:#3D4852 !important;
  transition:color var(--ease);
}
[data-testid="stExpander"] summary:hover{ color:#4F46D6 !important; }
[data-testid="stExpander"] summary:focus-visible{
  outline:2px solid #6C63FF; outline-offset:-2px;
}
[data-testid="stExpander"] summary svg{ color:#6B7280; }
[data-testid="stExpander"] [data-testid="stExpanderDetails"]{ padding:0 18px 16px; }

/* Sidebar --------------------------------------------------------------- */
[data-testid="stSidebar"]{
  background:#E0E5EC !important;
  border-right:none !important;
  box-shadow:10px 0 24px rgba(198,205,186,0.32);
}
[data-testid="stSidebar"] > div,
[data-testid="stSidebarContent"]{ background:transparent !important; }
[data-testid="stSidebarContent"]{
  padding-top:1.4rem;
  padding-left:1.4rem;
  padding-right:1.4rem;
}
[data-testid="stSidebar"] hr{
  border:none !important;
  height:6px;
  border-radius:9999px;
  background:#E0E5EC;
  box-shadow:var(--pressed-sm);
  margin:20px 0;
}
.side-title{
  font-family:'Roboto', Arial, sans-serif;
  font-size:1.12rem; font-weight:800; letter-spacing:-0.02em; color:#3D4852;
  margin-bottom:2px;
}
.side-sub{ font-size:0.78rem; line-height:1.5; color:#6B7280; margin-bottom:16px; }

[data-testid="stWidgetLabel"] p{
  font-size:0.78rem !important;
  font-weight:600 !important;
  color:#3D4852 !important;
}
[data-testid="stCaptionContainer"]{ color:#6B7280 !important; }

/* Inputs ---------------------------------------------------------------- */
[data-testid="stTextInput"] [data-baseweb="input"],
[data-testid="stTextInput"] [data-baseweb="base-input"]{
  background:#E0E5EC !important;
  border:none !important;
  border-radius:14px !important;
  box-shadow:var(--pressed) !important;
}
[data-testid="stTextInput"] input{
  background:transparent !important;
  color:#3D4852 !important;
  padding:0.55rem 0.9rem !important;
  font-size:0.88rem !important;
}
[data-testid="stTextInput"] input::placeholder{ color:#A0AEC0 !important; }
[data-testid="stTextInput"] [data-baseweb="input"]:focus-within{
  box-shadow:var(--pressed-deep), 0 0 0 3px #E0E5EC, 0 0 0 5px rgba(108,99,255,0.85) !important;
}

/* Segmented control ----------------------------------------------------- */
[data-testid="stButtonGroup"]{ background:transparent !important; }
[data-testid="stButtonGroup"] [role="radiogroup"]{
  gap:4px;
  padding:4px;
  border-radius:16px;
  box-shadow:var(--pressed-sm);
  flex-wrap:nowrap;
  width:100%;
}
[data-testid="stButtonGroup"] [role="radiogroup"] button{
  border:none !important;
  border-radius:12px !important;
  background:#E0E5EC !important;
  box-shadow:var(--raised-sm) !important;
  color:#3D4852 !important;
  font-size:0.72rem !important;
  font-weight:600 !important;
  flex:1 1 0 !important;
  min-width:0;
  padding:6px 4px !important;
  transition:transform var(--ease), box-shadow var(--ease), color var(--ease) !important;
}
[data-testid="stButtonGroup"] [role="radiogroup"] button p{
  font-size:0.72rem !important;
  white-space:nowrap;
}
[data-testid="stButtonGroup"] [role="radiogroup"] button:hover{
  color:#4F46D6 !important;
  transform:translateY(-1px);
}
[data-testid="stButtonGroup"] [role="radiogroup"] button[aria-checked="true"],
[data-testid="stButtonGroup"] [role="radiogroup"] button[data-selected="true"]{
  box-shadow:var(--pressed) !important;
  color:#4F46D6 !important;
}
[data-testid="stButtonGroup"] [role="radiogroup"] button:focus-visible{
  outline:2px solid #6C63FF; outline-offset:2px;
}

/* Slider / toggle ------------------------------------------------------- */
[data-testid="stSlider"] [role="slider"]{
  background:#6C63FF !important;
  box-shadow:3px 3px 8px rgb(163,177,198,0.8), -3px -3px 8px rgba(255,255,255,0.7) !important;
}
[data-testid="stSliderThumbValue"]{ color:#4F46D6 !important; font-weight:700; }
[data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"]:focus-visible{
  outline:2px solid #6C63FF; outline-offset:3px;
}
[data-baseweb="checkbox"] [role="switch"]{ box-shadow:var(--pressed-sm); }

/* Toggle switches */
[data-testid="stCheckbox"] label > div:first-of-type{
  background:#E0E5EC !important;
  box-shadow:var(--pressed-sm) !important;
}
[data-testid="stCheckbox"] label > div:first-of-type > div{
  background:#AEB8C6 !important;
  box-shadow:2px 2px 4px rgb(163,177,198,0.7), -2px -2px 4px rgba(255,255,255,0.8) !important;
}
[data-testid="stCheckbox"] label[data-selected="true"] > div:first-of-type > div{
  background:#6C63FF !important;
  box-shadow:2px 2px 5px rgba(108,99,255,0.45), -2px -2px 4px rgba(255,255,255,0.8) !important;
}

/* Info well + key status ------------------------------------------------ */
.info-well{
  padding:16px 18px;
  border-radius:20px;
  background:#E0E5EC;
  box-shadow:var(--pressed);
  display:flex; flex-direction:column; gap:10px;
}
.info-row{
  display:flex; align-items:center; justify-content:space-between;
  gap:10px; font-size:0.78rem; color:#6B7280;
}
.info-row b{
  font-family:'Roboto', Arial, sans-serif;
  font-size:0.78rem; font-weight:700; color:#3D4852; text-align:right;
}
.key-line{
  display:flex; align-items:center; gap:9px;
  padding:10px 14px; border-radius:9999px;
  background:#E0E5EC; box-shadow:var(--pressed-sm);
  font-size:0.76rem; font-weight:600;
  margin-bottom:6px;
}
.key-line.ok{ color:#0F6C68; }
.key-line.bad{ color:#8A3125; }
.key-dot{ width:7px; height:7px; flex:0 0 7px; border-radius:9999px; background:currentColor; }

/* Chat input ------------------------------------------------------------ */
[data-testid="stBottom"],
[data-testid="stBottomBlockContainer"],
footer[data-testid="stBottom"]{
  background:#E0E5EC !important;
  background-color:#E0E5EC !important;
}
[data-testid="stChatInput"]{
  background:#E0E5EC !important;
  border:none !important;
  border-radius:26px !important;
  box-shadow:var(--pressed-deep) !important;
  padding:4px 6px !important;
  transition:box-shadow var(--ease) !important;
}
[data-testid="stChatInputContainer"],
[data-testid="stChatInput"] > div,
[data-testid="stChatInput"] > div > div{
  background:transparent !important;
  border:none !important;
}
[data-testid="stChatInput"] textarea,
[data-testid="stChatInputContainer"] textarea,
div[data-baseweb="textarea"]{
  background:transparent !important;
  color:#3D4852 !important;
  caret-color:#6C63FF !important;
  font-size:0.96rem !important;
  line-height:1.5 !important;
}
[data-testid="stChatInput"] textarea::placeholder{ color:#A0AEC0 !important; }
[data-testid="stChatInput"]:focus-within{
  box-shadow:var(--pressed-deep), 0 0 0 3px #E0E5EC, 0 0 0 5px rgba(108,99,255,0.85) !important;
}
[data-testid="stChatInputSubmitButton"]{
  background:#6C63FF !important;
  border-radius:9999px !important;
  box-shadow:4px 4px 10px rgb(163,177,198,0.7), -4px -4px 10px rgba(255,255,255,0.7) !important;
  height:40px !important; width:40px !important;
  transition:transform var(--ease), background var(--ease) !important;
}
[data-testid="stChatInputSubmitButton"] svg{
  fill:#FFFFFF !important;
  color:#FFFFFF !important;
}
[data-testid="stChatInputSubmitButton"]:hover{
  background:#8B84FF !important;
  transform:translateY(-1px);
}
[data-testid="stChatInputSubmitButton"]:focus-visible{
  outline:2px solid #6C63FF; outline-offset:3px;
}
[data-testid="stChatInputSubmitButton"]:disabled{ opacity:0.45; }

/* Alerts / misc --------------------------------------------------------- */
[data-testid="stAlert"]{
  background:#E0E5EC !important;
  border:none !important;
  border-radius:20px !important;
  box-shadow:var(--pressed) !important;
  color:#3D4852 !important;
}
[data-testid="stAlert"] p{ color:#3D4852 !important; }
[data-testid="stSpinner"] p{ color:#6B7280 !important; }
[data-testid="stJson"]{
  background:transparent !important;
  border-radius:16px !important;
}

/* Fixed, non-scrolling app shell ---------------------------------------- */
@media (min-width: 1100px) and (min-height: 760px){
  html, body{ height:100%; overflow:hidden; }
  .stApp, [data-testid="stAppViewContainer"]{ height:100dvh; overflow:hidden !important; }

  main.stMain{
    height:100dvh;
    display:flex;
    flex-direction:column;
    overflow:hidden !important;
  }
  main.stMain > div:not([data-testid]){
    flex:0 0 0 !important;
    height:0 !important;
  }

  .block-container{
    flex:1 1 auto;
    min-height:0;
    height:auto;
    max-width:1360px;
    padding:clamp(10px, 1.3vh, 18px) clamp(20px, 2.4vw, 38px) 0;
    display:flex;
    flex-direction:column;
    overflow:hidden;
  }
  .block-container > [data-testid="stVerticalBlock"]{
    flex:1 1 auto;
    min-height:0;
    display:flex;
    flex-direction:column;
    justify-content:space-between;
  }

  /* Navigation sits inside the shell instead of floating over a scroll. */
  .stElementContainer:has(.top-nav){ position:static; }
  .top-nav{
    padding:11px 20px;
    border-radius:26px;
    margin-bottom:clamp(10px, 1.7vh, 22px);
  }
  .brand-mark{ width:42px; height:42px; flex:0 0 42px; }
  .brand-mark svg{ width:20px; height:20px; }
  .brand-text{ font-size:1rem; }

  /* Landing: columns and aisles share the free space evenly. */
  .block-container > [data-testid="stVerticalBlock"] > [data-testid="stLayoutWrapper"]{
    flex:0 0 auto;
  }
  .block-container > [data-testid="stVerticalBlock"] > [data-testid="stLayoutWrapper"]:has(.st-key-aisles){
    justify-content:flex-end;
  }

  .hero-title{ font-size:clamp(1.45rem, 2.5vw + 0.3rem, 1.85rem); margin:12px 0 9px; }
  .hero-subtitle{ font-size:0.92rem; line-height:1.55; }
  .section-label{ margin:clamp(10px, 1.7vh, 20px) 0 9px; }

  /* Example queries become a compact 2x2 grid. */
  .st-key-suggestions{
    display:grid;
    grid-template-columns:repeat(2, minmax(0, 1fr));
    gap:9px;
  }
  .st-key-suggestions [data-testid="stElementContainer"]{ width:100%; }
  .st-key-suggestions .stButton{ height:100%; }
  .st-key-suggestions .stButton > button{
    height:100%;
    min-height:clamp(46px, 5.4vh, 58px);
    padding:9px 14px !important;
    border-radius:14px !important;
    font-size:0.84rem !important;
    line-height:1.3 !important;
  }
  .st-key-aisles .stButton > button{
    min-height:clamp(46px, 5.2vh, 60px);
    padding:9px 12px !important;
    border-radius:16px !important;
    font-size:0.77rem !important;
  }

  /* Nested-depth visual scales with the window height. */
  .depth-stage{
    --ring:clamp(140px, 20vh, 220px);
    padding:clamp(16px, 2.3vh, 26px) clamp(16px, 1.8vw, 24px);
    gap:clamp(8px, 1.4vh, 16px);
    border-radius:28px;
  }
  .depth-ring{ width:var(--ring); height:var(--ring); }
  .depth-well{ width:calc(var(--ring) * 0.72); height:calc(var(--ring) * 0.72); }
  .depth-core{ width:calc(var(--ring) * 0.48); height:calc(var(--ring) * 0.48); }
  .depth-core svg{ width:calc(var(--ring) * 0.19); height:calc(var(--ring) * 0.19); }
  .shelf-grid .product-card{ padding:10px 11px; border-radius:16px; }
  .shelf-grid .product-name{ font-size:0.76rem; margin-top:7px; }
  .shelf-grid .product-brand{ font-size:0.66rem; }
  .shelf-grid .price-chip{ font-size:0.8rem; padding:3px 8px; }

  /* Chat: only the transcript scrolls, and only when there is one. */
  .block-container > [data-testid="stVerticalBlock"] > [data-testid="stLayoutWrapper"]:has(.st-key-chat){
    flex:1 1 auto;
    min-height:0;
  }
  .block-container > [data-testid="stVerticalBlock"] > [data-testid="stLayoutWrapper"]:has(.st-key-chat) > [data-testid="stVerticalBlock"]{
    height:100%;
    min-height:0;
  }
  .st-key-chat:not(:has([data-testid="stChatMessage"])){ display:none; }
  .block-container > [data-testid="stVerticalBlock"] > [data-testid="stLayoutWrapper"]:has(.st-key-chat):not(:has([data-testid="stChatMessage"])){
    display:none;
  }
  .st-key-chat:has([data-testid="stChatMessage"]){
    flex:1 1 auto;
    min-height:0;
    max-width:900px;
    overflow-y:auto;
    overscroll-behavior:contain;
    padding:2px 8px 10px 2px;
    scrollbar-width:thin;
    scrollbar-color:#C3CBD6 transparent;
  }
  .st-key-chat::-webkit-scrollbar{ width:8px; }
  .st-key-chat::-webkit-scrollbar-thumb{ background:#C3CBD6; border-radius:9999px; }
  .st-key-chat::-webkit-scrollbar-track{ background:transparent; }

  /* Composer docked to the bottom of the shell. */
  [data-testid="stBottom"]{
    position:static !important;
    flex:0 0 auto;
    padding:6px 0 clamp(10px, 1.6vh, 18px);
    background:transparent !important;
  }
  [data-testid="stBottom"] > div,
  [data-testid="stBottomBlockContainer"]{ background:transparent !important; }
  [data-testid="stBottomBlockContainer"]{
    max-width:900px;
    margin:0 auto;
    padding-top:0 !important;
    padding-bottom:0 !important;
  }
}

/* Short desktop windows: keep the page usable instead of clipping. */
@media (min-width: 1100px) and (max-height: 759px){
  .block-container{ padding-bottom:6.5rem; }
}

/* Narrow desktop heights: shave the landing so the shell still fits. */
@media (min-width: 1100px) and (max-height: 800px){
  .hero-title{ font-size:clamp(1.4rem, 2.2vw + 0.3rem, 1.75rem); margin:9px 0 8px; }
  .hero-subtitle{ font-size:0.88rem; line-height:1.5; }
  .section-label{ margin:8px 0 8px; }
  .st-key-suggestions .stButton > button{ min-height:44px; padding:7px 13px !important; }
  .st-key-aisles .stButton > button{ min-height:44px; padding:7px 12px !important; }
  .depth-stage{ --ring:clamp(128px, 16vh, 172px); padding:14px 18px; gap:8px; }
  .shelf-grid .product-card{ padding:8px 10px; }
  .shelf-grid .product-name{ font-size:0.72rem; margin-top:5px; }
  .shelf-grid .price-chip{ font-size:0.76rem; }
}

/* Responsive ------------------------------------------------------------ */
@media (max-width: 900px){
  .block-container{ padding-top:0.75rem; padding-bottom:7.5rem; }
  .top-nav{ padding:12px 16px 12px 46px; border-radius:26px; margin-bottom:26px; }
  .hero-title{ font-size:clamp(1.6rem, 7vw, 2rem); }
  .depth-stage{ padding:26px 20px 22px; }
  .depth-ring{ width:196px; height:196px; }
  .depth-well{ width:144px; height:144px; }
  .depth-core{ width:96px; height:96px; }
  .depth-core svg{ width:38px; height:38px; }
  .card-grid{ grid-template-columns:repeat(auto-fit, minmax(140px, 1fr)); }
  .shelf-grid{ grid-template-columns:repeat(auto-fit, minmax(140px, 1fr)); }
}
@media (max-width: 520px){
  .brand-sub{ display:none; }
  .top-nav{ flex-wrap:nowrap; padding:10px 14px 10px 46px; }
  .status-pill{ padding:9px; gap:0; margin-left:auto; }
  .status-text{ display:none; }
  .product-card{ padding:14px; }
  [data-testid="stChatMessageContent"]{ padding:16px 18px !important; }
}

@media (prefers-reduced-motion: reduce){
  html{ scroll-behavior:auto; }
  .depth-core{ animation:none; }
  *{ transition-duration:0.01ms !important; }
}
</style>
"""
)


# ---------------------------------------------------------------------------
# PIPELINE INITIALIZATION
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Indexing product catalogue (first run takes ~1-2 min)...")
def load_pipeline():
    return RAGPipeline()


def get_api_keys():
    groq_key = st.session_state.get("groq_key_input", "") or os.getenv("GROQ_API_KEY", "")
    try:
        if not groq_key:
            groq_key = st.secrets.get("GROQ_API_KEY", "")
    except Exception:
        pass

    gemini_key = st.session_state.get("gemini_key_input", "") or os.getenv("GEMINI_API_KEY", "")
    try:
        if not gemini_key:
            gemini_key = st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        pass

    return groq_key, gemini_key


pipeline = load_pipeline()

# Live QuickCommerce data: platforms that support both search and delivery ETA.
LIVE_PLATFORMS = [
    "BigBasket", "BlinkIt", "Zepto", "Swiggy", "JioMart", "DMart", "Minutes", "NoonMinutes", "TalabatMart",
]
# location -> (lat, lon, pincode); DMart/JioMart/Minutes need the pincode.
LIVE_LOCATIONS = {
    "Bengaluru · HSR Layout": (12.9021, 77.6639, "560102"),
    "Mumbai · Andheri": (19.1136, 72.8697, "400053"),
    "Delhi · Saket": (28.5245, 77.2066, "110017"),
    "Hyderabad · Gachibowli": (17.4401, 78.3489, "500032"),
    "Pune · Kothrud": (18.5074, 73.8077, "411038"),
    "Dubai · Downtown": (25.1972, 55.2744, ""),
}

PRODUCT_SOURCES = {"BigBasket Catalogue", "product"}
product_metas = [
    m for m in pipeline.metas if m and m.get("source") in PRODUCT_SOURCES and m.get("chunk") == 0
]
n_products = len(product_metas)
n_chunks = len(pipeline.ids)
aisles = Counter(m["category"] for m in product_metas).most_common(8)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "shelf" not in st.session_state:
    pool = [m for m in product_metas if 0 < m.get("price", 0) < 1500 and len(m.get("product", "")) < 48]
    st.session_state.shelf = random.sample(pool, 3) if len(pool) >= 3 else pool[:3]


# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="side-title">Retrieval settings</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="side-sub">Tune how the assistant searches, re-ranks and cites the catalogue.</div>',
        unsafe_allow_html=True,
    )

    groq_k, gemini_k = get_api_keys()

    st.text_input(
        "Groq API key",
        type="password",
        key="groq_key_input",
        value=groq_k if groq_k else "",
        placeholder="Paste Groq key",
        help="Uses openai/gpt-oss-20b for ultra-fast generation",
    )
    st.text_input(
        "Gemini API key",
        type="password",
        key="gemini_key_input",
        value=gemini_k if gemini_k else "",
        placeholder="Optional fallback key",
    )

    if groq_k:
        st.markdown(
            '<div class="key-line ok"><span class="key-dot"></span>Groq connected · gpt-oss-20b</div>',
            unsafe_allow_html=True,
        )
        if gemini_k:
            st.markdown(
                '<div class="key-line ok"><span class="key-dot"></span>Gemini fallback on standby</div>',
                unsafe_allow_html=True,
            )
    elif gemini_k:
        st.markdown(
            '<div class="key-line ok"><span class="key-dot"></span>Gemini engine connected</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="key-line bad"><span class="key-dot"></span>Add a key to start querying</div>',
            unsafe_allow_html=True,
        )

    st.divider()
    mode = st.segmented_control(
        "Retrieval mode",
        ["hybrid", "vector", "keyword"],
        default="hybrid",
        required=True,
        width="stretch",
        key="retrieval_mode",
        help="Hybrid merges vector semantics with BM25 keywords via Reciprocal Rank Fusion",
    )
    mode = mode or "hybrid"

    top_k = st.slider("Products to inspect", 2, 10, TOP_K)
    rerank = st.toggle("Cross-encoder re-ranking", value=True, help="Re-ranks with ms-marco-MiniLM-L-6-v2")
    decompose = st.toggle("Query decomposition", value=True, help="Splits multi-part compound queries")
    show_ctx = st.toggle("Show retrieved product tags", value=True)

    st.divider()
    st.markdown('<div class="side-sub">Live prices and delivery ETA for one location (QuickCommerce API).</div>', unsafe_allow_html=True)
    live_ready = pipeline.live_client is not None
    live_on = st.toggle(
        "Live prices & delivery ETA",
        value=live_ready,
        disabled=not live_ready,
        help="Adds live product prices, stock and delivery ETA to answers. Costs 1 credit per platform per lookup.",
    )
    live_platform = st.selectbox("Live platform", LIVE_PLATFORMS, index=0, disabled=not live_ready)
    live_city = st.selectbox("Deliver to", list(LIVE_LOCATIONS), index=0, disabled=not live_ready)
    live_lat, live_lon, live_pin = LIVE_LOCATIONS[live_city]

    if not live_ready:
        st.caption("Add `QuickCommerceAPI` to .env to turn live data on.")
    elif live_on:
        remaining = getattr(pipeline.live_client, "last_credits_remaining", None)
        st.caption(
            f"{live_lat}, {live_lon} · pincode {live_pin or '—'}"
            + (f" · {remaining} credits" if remaining is not None else "")
        )

    st.divider()
    st.markdown('<div class="side-sub">Ask by voice (OpenAI live transcription).</div>', unsafe_allow_html=True)
    voice_key = os.getenv("OPENAI_API", "") or os.getenv("OPENAI_API_KEY", "")
    if voice_key:
        clip = st.audio_input("Record a question", key="voice_clip")
        if clip is not None:
            blob = clip.getvalue()
            # Record once per clip: reruns must not re-open a billed session.
            if blob and blob != st.session_state.get("voice_last_blob"):
                st.session_state.voice_last_blob = blob
                try:
                    with st.spinner("Transcribing your question..."):
                        with RealtimeTranscriber(
                            api_key=voice_key,
                            prompt="A grocery shopping question for BigBasket",
                        ) as stt_client:
                            transcript = stt_client.transcribe_turn(wav_to_pcm16_24k(blob))
                except (TranscriptionError, ValueError) as e:
                    st.caption(f"Transcription failed: {e}")
                else:
                    if transcript.strip():
                        st.session_state.pending_question = transcript.strip()
                        st.rerun()
                    else:
                        st.caption("No speech detected — try recording again.")
    else:
        st.caption("Add `OPENAI_API` to .env to ask by voice.")

    # Reciprocal link to the adjacent React voice-orb frontend (same backend).
    st.link_button(
        "Open voice orb →",
        "http://127.0.0.1:5173",
        help="Opens the React WebGPU voice assistant in a new tab.",
        use_container_width=True,
    )

    st.divider()
    st.markdown(
        f"""<div class="info-well">
<div class="info-row"><span>Index status</span><b>{n_products:,} products</b></div>
<div class="info-row"><span>Chunks embedded</span><b>{n_chunks:,}</b></div>
<div class="info-row"><span>Embeddings</span><b>MiniLM-L6 · CPU</b></div>
<div class="info-row"><span>Vector store</span><b>ChromaDB · cosine</b></div>
</div>""",
        unsafe_allow_html=True,
    )

    with st.expander("Pipeline architecture", icon=":material/account_tree:"):
        st.markdown(
            "1. **Input Guardrail**: Blocks off-topic queries and prompt injections\n"
            "2. **Exact Hash Cache**: Returns repeated queries instantly in <2ms\n"
            "3. **Query Decomposition**: Resolves follow-ups and splits multi-intent queries\n"
            "4. **Hybrid Retrieval**: ChromaDB vector search + BM25Okapi via RRF ($k=60$)\n"
            "5. **Cross-Encoder Re-ranker**: ms-marco-MiniLM-L-6-v2 refines top candidates\n"
            "6. **Grounded Generation**: Groq `gpt-oss-20b` answers with numbered citations"
        )

    with st.expander("Cache statistics", icon=":material/bolt:"):
        stats = pipeline.cache_stats()
        st.markdown(f"**Hits:** {stats['hits']} | **Misses:** {stats['misses']} | **Size:** {stats['size']}")
        if st.button("Clear cache", width="stretch"):
            pipeline.clear_cache()
            st.rerun()

    if st.session_state.messages:
        st.space("small")
        if st.button("Start a new conversation", icon=":material/restart_alt:", width="stretch"):
            st.session_state.messages = []
            st.rerun()


# ---------------------------------------------------------------------------
# UI HELPERS
# ---------------------------------------------------------------------------
def product_card_html(m, n=None, match=None, desc=None):
    """One extruded card: inset icon well, price chip, optional evidence details."""
    has_price = m.get("price", -1) >= 0
    price = f"₹{m['price']:,.0f}" if has_price else "Policy"
    price_cls = "price-chip" if has_price else "price-chip policy"

    parts = ['<div class="product-card">', '<div class="product-head">']
    parts.append(f'<span class="product-icon">{TAG_SVG}</span>')
    if n is not None:
        parts.append(f'<span class="product-index">#{n}</span>')
    parts.append("</div>")
    parts.append(f'<div class="product-name">{html.escape(m.get("product", "Unknown"))}</div>')
    parts.append(f'<div class="product-brand">{html.escape(m.get("brand", "Unknown"))}</div>')
    parts.append(f'<div class="product-foot"><span class="{price_cls}">{price}</span>')
    if match:
        parts.append(f'<span class="match-chip">{html.escape(match)}</span>')
    parts.append("</div>")
    if desc:
        short = html.escape(desc[:280]) + ("…" if len(desc) > 280 else "")
        parts.append(f'<details class="product-details"><summary>Details</summary><p>{short}</p></details>')
    parts.append("</div>")
    return "".join(parts)


def render_tags(chunks, best_sim):
    cards = []
    for n, c in enumerate(chunks, 1):
        sim = c.get("similarity") or c.get("similarity_score") or c.get("score")
        match = f"{sim:.0%} match" if sim is not None else "word match"
        content = c.get("text") or c.get("content", "")
        desc = content.split("Description:", 1)[-1].strip()
        cards.append(product_card_html(c["metadata"], n=n, match=match, desc=desc))

    with st.expander(
        f"Retrieved catalogue evidence · {len(chunks)} items · top match {best_sim:.0%}",
        icon=":material/inventory_2:",
        expanded=False,
    ):
        st.markdown(f'<div class="card-grid">{"".join(cards)}</div>', unsafe_allow_html=True)


def render_notice(title, message):
    st.markdown(
        f"""<div class="notice">
<span class="notice-icon">!</span>
<div><strong>{html.escape(title)}</strong><p>{html.escape(message)}</p></div>
</div>""",
        unsafe_allow_html=True,
    )


def render_nodata():
    render_notice("Out of catalogue bounds", FALLBACK_MSG)


def render_live(result):
    """Show the live delivery ETA, or why live data is missing, for one answer."""
    live = result.get("live") or {}
    eta = live.get("eta")
    products = live.get("products") or []
    if eta:
        state = "open now" if eta.get("open") else "closed"
        extra = f" · {len(products)} live items" if products else ""
        st.markdown(
            f'<div class="live-strip"><span class="live-dot"></span>'
            f'<b>{html.escape(str(eta.get("platform")))}</b> delivers in '
            f'<b>{html.escape(str(eta.get("eta")))}</b> · {state}'
            f' · store {html.escape(str(eta.get("store_id")))}{extra}</div>',
            unsafe_allow_html=True,
        )
    elif result.get("live_error"):
        st.caption(f"Live data unavailable ({result['live_error']})")


def render_chips(chips):
    """Render the per-answer telemetry chip row."""
    if not chips:
        return
    chip_html = "".join(
        f'<span class="chip {cls}"><span class="chip-key">{key}</span>{value}</span>'
        for key, value, cls in chips
    )
    st.markdown(f'<div class="telemetry">{chip_html}</div>', unsafe_allow_html=True)


def ask(q):
    st.session_state.pending_question = q
    st.rerun()


AVATARS = {"user": ":material/person:", "assistant": ":material/shopping_basket:"}
AISLE_ICONS = {
    "Beauty & Hygiene": "spa",
    "Gourmet & World Food": "public",
    "Snacks & Branded Foods": "cookie",
    "Foodgrains, Oil & Masala": "grain",
    "Cleaning & Household": "cleaning_services",
    "Kitchen, Garden & Pets": "kitchen",
    "Bakery, Cakes & Dairy": "bakery_dining",
    "Beverages": "local_cafe",
    "Baby Care": "child_care",
    "Fruits & Vegetables": "nutrition",
    "Eggs, Meat & Fish": "egg",
}

# Top navigation bar
st.markdown(
    f"""<div class="top-nav">
  <div class="brand">
    <span class="brand-mark">{BASKET_SVG}</span>
    <span class="brand-text">BigBasket Assistant<span class="brand-sub">Grounded catalogue answers</span></span>
  </div>
  <span class="status-pill"><span class="status-dot"></span><span class="status-text">Hybrid retrieval · online</span></span>
</div>""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# OPENING SCREEN (Empty State)
# ---------------------------------------------------------------------------
SUGGESTIONS = [
    "Which face wash suits oily skin, and what does it cost?",
    "Compare Saffola Gold and Fortune Sunflower oil",
    "Suggest a healthy breakfast cereal under ₹300",
    "What is BigBasket's return policy for spoiled food items?",
]

pending = st.session_state.get("pending_question")
if not st.session_state.messages and not pending:
    left, right = st.columns([1.05, 0.95], gap="large")

    with left:
        st.markdown('<div class="eyebrow">Retrieval-augmented assistant</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="hero-title">What groceries are on <span class="accent">your list</span>?</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="hero-subtitle">Ask about prices, ingredients, brands, or BigBasket policies. '
            "Every recommendation is verified against the catalogue index — no guessing.</div>",
            unsafe_allow_html=True,
        )
        st.markdown('<div class="section-label">Try an example query</div>', unsafe_allow_html=True)
        with st.container(key="suggestions"):
            for i, q in enumerate(SUGGESTIONS):
                if st.button(q, key=f"sugg{i}", width="stretch"):
                    ask(q)

    with right:
        st.markdown(
            f"""<div class="depth-stage">
  <span class="depth-dot a"></span>
  <span class="depth-dot b"></span>
  <span class="depth-dot c"></span>
  <div class="depth-ring">
    <div class="depth-well">
      <div class="depth-core">{BASKET_SVG}</div>
    </div>
  </div>
  <div class="depth-caption">
    <span class="depth-title">{n_products:,} products indexed</span>
    <span class="depth-sub">{n_chunks:,} embedded chunks · cosine similarity</span>
  </div>
</div>""",
            unsafe_allow_html=True,
        )

        if st.session_state.shelf:
            st.markdown('<div class="section-label">On the shelf</div>', unsafe_allow_html=True)
            cards = "".join(product_card_html(m) for m in st.session_state.shelf)
            st.markdown(f'<div class="card-grid shelf-grid">{"".join(cards)}</div>', unsafe_allow_html=True)

    with st.container(key="aisles"):
        st.markdown('<div class="section-label">Browse by category aisle</div>', unsafe_allow_html=True)
        cols = st.columns(4)
        for i, (cat, count) in enumerate(aisles):
            icon = AISLE_ICONS.get(cat, "grocery")
            label = f":material/{icon}: {cat} ({count:,})"
            if cols[i % 4].button(label, key=f"aisle{i}", width="stretch"):
                ask(f"What are popular {cat} items under ₹250?")

    # In the scrolling (small-screen) layout the docked composer makes
    # Streamlit jump to the bottom on first paint; bring the hero back once.
    if not st.session_state.get("landing_scrolled"):
        st.html(
            """<script>
            (() => {
              if (window.matchMedia('(min-width: 1100px) and (min-height: 760px)').matches) return;
              let stop = false;
              ['wheel', 'touchstart', 'pointerdown', 'keydown'].forEach((evt) =>
                window.addEventListener(evt, () => { stop = true; }, { once: true, passive: true })
              );
              const started = Date.now();
              const tick = () => {
                if (stop) return;
                const main = document.querySelector('main.stMain');
                if (main && main.scrollTop > 0) main.scrollTop = 0;
                if (Date.now() - started < 4000) setTimeout(tick, 120);
              };
              tick();
            })();
            </script>""",
            unsafe_allow_javascript=True,
        )
        st.session_state["landing_scrolled"] = True

# ---------------------------------------------------------------------------
# CHAT CONVERSATION
# ---------------------------------------------------------------------------
question = st.chat_input(
    "Ask about products or policies...",
    submit_mode="disable",
    key="chat_input",
)
if not question:
    question = st.session_state.pop("pending_question", None)

with st.container(key="chat"):
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar=AVATARS[msg["role"]]):
            if msg.get("fallback"):
                render_nodata()
            elif msg.get("guardrail"):
                render_notice("Policy guardrail", msg["content"])
            else:
                st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("chips"):
                render_chips(msg["chips"])
            if msg["role"] == "assistant" and show_ctx and msg.get("chunks"):
                render_tags(msg["chunks"], msg.get("best_sim", 0.0))
            if msg["role"] == "assistant" and (msg.get("live") or msg.get("live_error")):
                render_live(msg)

    # The transcript is its own scroll region, so follow new answers here
    # instead of relying on the page-level auto-scroll.
    if st.session_state.messages:
        st.html(
            """<script>
            (() => {
              if (window.__bbChatObserver) return;
              const chat = document.querySelector('.st-key-chat');
              if (!chat) return;
              window.__bbChatObserver = true;
              const snap = () => { chat.scrollTop = chat.scrollHeight; };
              let stick = true;
              chat.addEventListener('scroll', () => {
                stick = chat.scrollHeight - chat.scrollTop - chat.clientHeight < 48;
              }, { passive: true });
              new MutationObserver(() => { if (stick) snap(); })
                .observe(chat, { childList: true, subtree: true });
              snap();
            })();
            </script>""",
            unsafe_allow_javascript=True,
        )

    if question:
        groq_k, gemini_k = get_api_keys()
        if not groq_k and not gemini_k:
            st.warning("Please provide a Groq or Gemini API key in the sidebar.", icon=":material/key:")
            st.stop()

        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user", avatar=AVATARS["user"]):
            st.markdown(question)

        with st.chat_message("assistant", avatar=AVATARS["assistant"]):
            with st.spinner("Searching product catalogue and grounding answer..."):
                try:
                    result = pipeline.answer(
                        question,
                        groq_api_key=groq_k,
                        gemini_api_key=gemini_k,
                        k=top_k,
                        mode=mode,
                        rerank=rerank,
                        decompose=decompose,
                        chat_history=st.session_state.messages[:-1],
                        **live_request_kwargs(
                            live_on, live_platform, live_lat, live_lon, live_pin or None
                        ),
                    )
                except Exception as e:
                    st.session_state.messages.pop()
                    st.error(f"Search retrieval error: {e}")
                    st.stop()

            content = result["answer"]
            latency_txt = f"{result.get('latency_ms', 0):.0f} ms"
            flow_txt = result.get("flow", "hybrid_rerank")
            cached_flag = result.get("cached", False)
            confidence = result.get("confidence_score", 0.0)

            chips = [
                ("Latency", latency_txt, ""),
                ("Flow", html.escape(str(flow_txt)), ""),
                ("Cache", "Hit" if cached_flag else "Miss", "hit" if cached_flag else ""),
                ("Confidence", f"{confidence:.0%}", "accent"),
            ]

            if result.get("guardrail"):
                render_notice("Policy guardrail", content)
            elif result.get("fallback"):
                render_nodata()
            else:
                if result.get("decomposed") and len(result.get("sub_queries", [])) > 1:
                    sq_tags = ", ".join(f'"{q}"' for q in result["sub_queries"])
                    st.caption(f"Decomposed sub-queries: {sq_tags}")
                st.markdown(content)

            render_chips(chips)
            render_live(result)

            if show_ctx and result.get("chunks"):
                render_tags(result["chunks"], result.get("best_sim", 0.0))

            with st.expander(
                "Backend audit log & chunk telemetry",
                icon=":material/receipt_long:",
                expanded=False,
            ):
                st.json(
                    result.get(
                        "audit_entry",
                        {
                            "latency_ms": result.get("latency_ms"),
                            "cached": cached_flag,
                            "flow": flow_txt,
                            "confidence_score": result.get("confidence_score"),
                            "chunks": [c.get("metadata", {}).get("product") for c in result.get("chunks", [])],
                        },
                    )
                )

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": content,
                "fallback": result.get("fallback", False),
                "guardrail": result.get("guardrail", False),
                "cached": result.get("cached", False),
                "chunks": result.get("chunks", []),
                "best_sim": result.get("best_sim", 0.0),
                "chips": chips,
                "live": result.get("live"),
                "live_error": result.get("live_error"),
            }
        )
        st.rerun()
