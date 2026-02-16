#!/usr/bin/env python3
"""Hive-Mind architecture block diagram using Pillow."""
from PIL import Image, ImageDraw, ImageFont
import os

W, H = 3400, 1480
BG = (18, 18, 28)

# Color palette
C_BLUE     = (55, 120, 220)
C_BLUE_LT  = (80, 150, 240)
C_TEAL     = (40, 180, 170)
C_GREEN    = (60, 190, 100)
C_ORANGE   = (230, 150, 40)
C_RED      = (210, 60, 80)
C_PURPLE   = (140, 80, 210)
C_PINK     = (210, 80, 160)
C_GRAY     = (70, 75, 90)
C_GRAY_LT  = (110, 115, 130)
C_WHITE    = (230, 230, 240)
C_DIM      = (160, 165, 180)
C_CYAN     = (60, 200, 220)
C_YELLOW   = (220, 200, 50)

def find_font(size):
    paths = [
        "/usr/share/fonts/google-noto-sans-fonts/NotoSans-Bold.ttf",
        "/usr/share/fonts/google-noto-sans-fonts/NotoSans-Regular.ttf",
        "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf",
        "/usr/share/fonts/liberation-sans/LiberationSans-Bold.ttf",
        "/usr/share/fonts/liberation-sans/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]
    for p in paths:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()

def find_font_regular(size):
    paths = [
        "/usr/share/fonts/google-noto-sans-fonts/NotoSans-Regular.ttf",
        "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf",
        "/usr/share/fonts/liberation-sans/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for p in paths:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()

font_title = find_font(52)
font_subtitle = find_font_regular(24)
font_heading = find_font(26)
font_body = find_font_regular(21)
font_small = find_font_regular(17)
font_label = find_font(17)
font_port = find_font(20)

img = Image.new('RGB', (W, H), BG)
draw = ImageDraw.Draw(img)


def rounded_rect(x, y, w, h, color, radius=16, border=None, border_w=2):
    r = radius
    draw.rounded_rectangle([x, y, x+w, y+h], radius=r, fill=color)
    if border:
        draw.rounded_rectangle([x, y, x+w, y+h], radius=r, outline=border, width=border_w)


def box(x, y, w, h, label, sublabel, color, text_color=C_WHITE, border=None):
    rounded_rect(x, y, w, h, color, border=border, border_w=3 if border else 2)
    bbox = draw.textbbox((0, 0), label, font=font_heading)
    tw = bbox[2] - bbox[0]
    if sublabel:
        draw.text((x + (w - tw) // 2, y + h // 2 - 20), label, fill=text_color, font=font_heading)
        bbox2 = draw.textbbox((0, 0), sublabel, font=font_small)
        tw2 = bbox2[2] - bbox2[0]
        draw.text((x + (w - tw2) // 2, y + h // 2 + 8), sublabel, fill=C_DIM, font=font_small)
    else:
        draw.text((x + (w - tw) // 2, y + (h - 26) // 2), label, fill=text_color, font=font_heading)


def small_box(x, y, w, h, label, sublabel, color, text_color=C_WHITE, border=None):
    rounded_rect(x, y, w, h, color, radius=10, border=border, border_w=2 if border else 1)
    bbox = draw.textbbox((0, 0), label, font=font_body)
    tw = bbox[2] - bbox[0]
    if sublabel:
        draw.text((x + (w - tw) // 2, y + h // 2 - 16), label, fill=text_color, font=font_body)
        bbox2 = draw.textbbox((0, 0), sublabel, font=font_small)
        tw2 = bbox2[2] - bbox2[0]
        draw.text((x + (w - tw2) // 2, y + h // 2 + 6), sublabel, fill=C_DIM, font=font_small)
    else:
        draw.text((x + (w - tw) // 2, y + (h - 20) // 2), label, fill=text_color, font=font_body)


def arrow(x1, y1, x2, y2, color=C_GRAY_LT, width=3, dashed=False):
    import math
    if dashed:
        dx = x2 - x1
        dy = y2 - y1
        length = math.sqrt(dx*dx + dy*dy)
        dash_len = 12
        gap_len = 8
        nx, ny = dx/length, dy/length
        pos = 0
        while pos < length - 15:
            end = min(pos + dash_len, length - 15)
            draw.line([(x1 + nx*pos, y1 + ny*pos), (x1 + nx*end, y1 + ny*end)], fill=color, width=width)
            pos = end + gap_len
    else:
        draw.line([(x1, y1), (x2, y2)], fill=color, width=width)

    angle = math.atan2(y2 - y1, x2 - x1)
    arr_len = 14
    arr_angle = 0.4
    ax1 = x2 - arr_len * math.cos(angle - arr_angle)
    ay1 = y2 - arr_len * math.sin(angle - arr_angle)
    ax2 = x2 - arr_len * math.cos(angle + arr_angle)
    ay2 = y2 - arr_len * math.sin(angle + arr_angle)
    draw.polygon([(x2, y2), (ax1, ay1), (ax2, ay2)], fill=color)


def arrow_label(x1, y1, x2, y2, label, color=C_GRAY_LT, width=3, dashed=False, offset=(0, -14)):
    arrow(x1, y1, x2, y2, color, width, dashed)
    mx = (x1 + x2) // 2 + offset[0]
    my = (y1 + y2) // 2 + offset[1]
    draw.text((mx, my), label, fill=color, font=font_small)


def section_bg(x, y, w, h, label, color):
    bg_color = (color[0] // 8, color[1] // 8, color[2] // 8)
    rounded_rect(x, y, w, h, bg_color, radius=20, border=color, border_w=2)
    draw.text((x + 16, y + 8), label, fill=color, font=font_label)


def port_badge(x, y, port, color=C_CYAN):
    text = f":{port}"
    bbox = draw.textbbox((0, 0), text, font=font_port)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    rounded_rect(x, y, tw + 16, th + 10, (30, 35, 50), radius=8, border=color, border_w=2)
    draw.text((x + 8, y + 4), text, fill=color, font=font_port)


# ============================================================
# Title
# ============================================================
draw.text((W // 2 - 300, 20), "Hive-Mind Architecture", fill=C_WHITE, font=font_title)
draw.text((W // 2 - 250, 78), "Distributed AI Memory + Semantic RAG + Dual LLM", fill=C_DIM, font=font_subtitle)

# ============================================================
# Section backgrounds
# ============================================================
# Top-left: Clients
section_bg(40, 120, 520, 340, "CLIENTS", C_WHITE)

# Middle-left: MCP Tools
section_bg(40, 500, 760, 520, "MCP TOOLS (13 endpoints)", C_BLUE)

# Center: Redis
section_bg(860, 120, 680, 900, "REDIS CLUSTER", C_RED)

# Right-top: LLM Inference
section_bg(1600, 120, 760, 520, "LLM INFERENCE (GPU)", C_TEAL)

# Right-bottom: Training
section_bg(1600, 690, 760, 330, "CONTINUOUS LEARNING", C_ORANGE)

# Bottom: Embedding + RAG
section_bg(40, 1080, 760, 340, "EMBEDDING ENGINE (CPU)", C_PURPLE)
section_bg(860, 1080, 680, 340, "RAG PIPELINE", C_GREEN)
section_bg(1600, 1080, 760, 340, "MULTI-NODE (planned)", C_GRAY_LT)

# ============================================================
# Client boxes
# ============================================================
box(80, 170, 220, 80, "Claude Code", "MCP client (stdio)", (50, 55, 70), border=C_WHITE)
box(320, 170, 220, 80, "HTTP Clients", "curl, apps, scripts", (50, 55, 70), border=C_GRAY_LT)

box(80, 290, 220, 80, "MCP Server", "stdio protocol", C_BLUE)
box(320, 290, 220, 80, "HTTP API", "FastAPI + Uvicorn", C_BLUE)
port_badge(340, 375, "8090", C_CYAN)

# Client arrows
arrow(190, 250, 190, 290, C_WHITE, 2)
arrow(430, 250, 430, 290, C_GRAY_LT, 2)

# Both servers use same backend
arrow(300, 330, 320, 330, C_BLUE, 2)

# ============================================================
# MCP Tool boxes (organized by category)
# ============================================================
# Memory tools
draw.text((60, 520), "Memory", fill=C_BLUE_LT, font=font_label)
small_box(60, 545, 170, 55, "memory_store", "ctx, files, task", C_BLUE)
small_box(240, 545, 170, 55, "memory_recall", "session retrieval", C_BLUE)
small_box(420, 545, 180, 55, "list_sessions", "recent sessions", C_BLUE)

# Fact tools
draw.text((60, 620), "Facts / RAG", fill=C_GREEN, font=font_label)
small_box(60, 645, 170, 55, "fact_store", "key + embedding", (40, 100, 55))
small_box(240, 645, 170, 55, "fact_get", "retrieve facts", (40, 100, 55))
small_box(420, 645, 180, 55, "fact_suggestions", "missed query analysis", (40, 100, 55))

# LLM tools
draw.text((60, 720), "LLM Inference", fill=C_TEAL, font=font_label)
small_box(60, 745, 170, 55, "llm_generate", "prompt + RAG", (30, 120, 110))
small_box(240, 745, 170, 55, "llm_code_assist", "review/fix/opt", (30, 120, 110))
small_box(420, 745, 180, 55, "llm_complete", "FIM completion", (30, 120, 110))

# Cache & Learning
draw.text((60, 820), "Cache & Learning", fill=C_ORANGE, font=font_label)
small_box(60, 845, 170, 55, "tool_cache", "get/set outputs", C_GRAY)
small_box(240, 845, 170, 55, "learning_add", "training samples", (160, 100, 20))
small_box(420, 845, 180, 55, "get_stats", "system health", C_GRAY)

# Connectors: MCP server -> tools region
arrow(190, 370, 190, 500, C_BLUE, 2, dashed=True)
arrow(430, 370, 400, 500, C_BLUE, 2, dashed=True)

# ============================================================
# Redis boxes
# ============================================================
# Data structures
box(900, 170, 280, 70, "session:{id}", "hash - ctx, files, task, node", (130, 40, 50))
box(1220, 170, 280, 70, "sessions:recent", "sorted set - by timestamp", (130, 40, 50))

box(900, 270, 280, 70, "facts:system", "hash - key -> value", (130, 40, 50))
box(1220, 270, 280, 70, "fact_embeddings:*", "base64 float32[384]", (100, 40, 80))

box(900, 370, 280, 70, "learning:queue", "stream - interactions", (150, 90, 30))
box(1220, 370, 280, 70, "llm:cache:{hash}", "inference cache (30m)", (130, 40, 50))

box(900, 470, 280, 70, "tool:{name}:{hash}", "output cache (1h)", (130, 40, 50))
box(1220, 470, 280, 70, "rag:retrieval_log", "stream - quality tracking", (100, 40, 80))

# Cluster info
box(940, 590, 560, 70, "3 Masters + 3 Replicas", "ports 7000-7005, 16384 hash slots, password auth", (60, 25, 35), border=C_RED)
port_badge(960, 665, "7000-7005", C_RED)

# TTL legend
draw.text((900, 710), "TTLs:", fill=C_DIM, font=font_label)
draw.text((950, 710), "sessions 7d  |  embeddings 30d  |  llm cache 30m  |  tool cache 1h", fill=(100, 105, 120), font=font_small)

# Throughput
draw.text((900, 740), "Perf:", fill=C_DIM, font=font_label)
draw.text((950, 740), "GET 14.7K/s  |  SET 10.6K/s  |  Pipeline 59.7K/s  |  <1ms latency", fill=(100, 105, 120), font=font_small)

# ============================================================
# LLM Inference boxes
# ============================================================
# HiveCoder (system service)
box(1640, 170, 320, 90, "HiveCoder-7B", "Q5_K_M  |  5.1 GB VRAM", C_TEAL)
port_badge(1660, 265, "8089", C_TEAL)
draw.text((1730, 270), "system service  |  127.0.0.1", fill=C_DIM, font=font_small)

# Qwen3-14B (user service)
box(1640, 310, 320, 90, "Qwen3-14B", "Q4_K_M  |  8.4 GB VRAM", C_CYAN)
port_badge(1660, 405, "8080", C_CYAN)
draw.text((1730, 410), "user service  |  0.0.0.0", fill=C_DIM, font=font_small)

# GPU
box(2000, 170, 320, 90, "AMD R9700", "32 GB VRAM  |  ROCm 7.12", (50, 55, 70), border=C_TEAL)
draw.text((2020, 270), "~7.7 GB used (HiveCoder)", fill=C_DIM, font=font_small)
draw.text((2020, 290), "~13 GB used (Qwen3-14B)", fill=C_DIM, font=font_small)
draw.text((2020, 310), "~11 GB free headroom", fill=C_GREEN, font=font_small)

# OpenAI-compatible API
box(1640, 460, 680, 70, "OpenAI-Compatible API (/v1/chat/completions)", "RAG fact injection into system prompt before inference", (30, 120, 110))
port_badge(2180, 540, "8090", C_CYAN)
draw.text((2250, 545), "proxied via HTTP API", fill=C_DIM, font=font_small)

# GPU arrows
arrow(1960, 215, 2000, 215, C_TEAL, 2)
arrow(1960, 355, 2000, 280, C_CYAN, 2)

# LLM -> API
arrow(1800, 260, 1800, 310, C_GRAY_LT, 1, dashed=True)
draw.text((1810, 275), "separate servers", fill=(80, 85, 100), font=font_small)

# ============================================================
# Training boxes
# ============================================================
box(1640, 740, 200, 80, "Drain Queue", "every 5 min", C_ORANGE)
box(1860, 740, 200, 80, "Quality Filter", "min len, success", (160, 100, 20))
box(1640, 850, 200, 80, "LoRA Training", "r=16, alpha=32", (160, 100, 20))
box(1860, 850, 200, 80, "GGUF Export", "Q5_K_M quantize", (160, 100, 20))
box(2100, 850, 240, 80, "Hot Swap", "symlink + restart", (160, 100, 20), border=C_TEAL)

# Training flow arrows
arrow(1840, 780, 1860, 780, C_ORANGE, 2)
arrow(1960, 820, 1740, 850, C_ORANGE, 2)
arrow(1840, 890, 1860, 890, C_ORANGE, 2)
arrow(2060, 890, 2100, 890, C_ORANGE, 2)
arrow(2220, 850, 2220, 540, C_TEAL, 2, dashed=True)  # hot swap -> llama-server
draw.text((2230, 700), "reload", fill=C_TEAL, font=font_small)

# Training threshold note
draw.text((1660, 940), "Triggers: 100+ samples  |  1 epoch  |  ~4 min cycle", fill=C_DIM, font=font_small)

# ============================================================
# Embedding boxes
# ============================================================
box(80, 1140, 300, 80, "bge-small-en-v1.5", "384-dim, SentenceTransformer", C_PURPLE)
draw.text((100, 1230), "Runs on CPU (keeps GPU free for LLM)", fill=C_DIM, font=font_small)

box(430, 1140, 160, 80, "Encode", "query/fact text", (100, 55, 160))
box(620, 1140, 150, 80, "Cosine Sim", "dot product", (100, 55, 160))

# Embedding arrows
arrow(380, 1180, 430, 1180, C_PURPLE, 2)
arrow(590, 1180, 620, 1180, C_PURPLE, 2)

# ============================================================
# RAG Pipeline boxes
# ============================================================
box(900, 1140, 280, 70, "Semantic Search", "cosine sim >= 0.45", C_GREEN)
box(900, 1240, 280, 70, "Keyword Fallback", "70+ keyword map", (40, 130, 70))
box(1220, 1140, 280, 70, "Top-K Selection", "rank + threshold", (40, 130, 70))
box(1220, 1240, 280, 70, "Retrieval Tracker", "hit rate, quality log", (40, 130, 70))

# RAG quality labels
draw.text((920, 1330), "Quality:  >= 0.6 good  |  0.45-0.6 weak  |  < 0.45 miss", fill=C_DIM, font=font_small)
draw.text((920, 1355), "Current:  84% hit rate  |  31 facts  |  semantic-only retrieval", fill=C_GREEN, font=font_small)

# RAG flow
arrow(770, 1180, 900, 1180, C_GREEN, 2)  # cosine -> semantic
arrow(1180, 1180, 1220, 1180, C_GREEN, 2)  # semantic -> top-k
arrow(1040, 1210, 1040, 1240, (40, 130, 70), 2, dashed=True)  # fallback
draw.text((1050, 1218), "fallback", fill=C_DIM, font=font_small)

# ============================================================
# Multi-node boxes
# ============================================================
box(1640, 1140, 320, 80, "aurora (BEAST)", "GPU inference + training", (50, 55, 70), border=C_TEAL)
box(1640, 1260, 320, 80, "r720xd (NAS)", "embeddings + storage", (50, 55, 70), border=C_GRAY_LT)

draw.text((2000, 1150), "AMD R9700 32GB", fill=C_DIM, font=font_small)
draw.text((2000, 1170), "Redis cluster (6 nodes)", fill=C_DIM, font=font_small)
draw.text((2000, 1190), "llama-server x2", fill=C_DIM, font=font_small)

draw.text((2000, 1270), "Dual Xeon E5-2660", fill=C_DIM, font=font_small)
draw.text((2000, 1290), "24x 2.5\" bays, 64GB RAM", fill=C_DIM, font=font_small)
draw.text((2000, 1310), "Embedding offload (future)", fill=C_DIM, font=font_small)

arrow(1800, 1220, 1800, 1260, C_GRAY_LT, 2, dashed=True)
draw.text((1810, 1230), "Tailscale VPN", fill=C_DIM, font=font_small)

# ============================================================
# Cross-section arrows (data flow)
# ============================================================

# MCP tools -> Redis (store/retrieve)
arrow_label(620, 572, 900, 205, "store/recall", C_BLUE_LT, 2, offset=(-20, -18))
arrow_label(620, 672, 900, 305, "facts r/w", C_GREEN, 2, offset=(-20, -18))
arrow_label(420, 872, 900, 405, "xadd samples", C_ORANGE, 2, offset=(-30, -18))
arrow_label(600, 872, 900, 505, "cache r/w", C_GRAY_LT, 2, offset=(-20, -18))

# MCP tools -> LLM (inference)
arrow_label(620, 772, 1640, 490, "prompt + facts", C_TEAL, 3, offset=(-60, -20))

# Redis -> Training (drain)
arrow_label(1040, 440, 1640, 780, "drain queue", C_ORANGE, 2, dashed=True, offset=(-50, -18))

# Redis -> Embedding (load cached embeddings)
arrow_label(900, 340, 700, 1140, "cached embeddings", C_PURPLE, 2, dashed=True, offset=(-70, 0))

# RAG -> LLM (inject facts into prompt)
arrow_label(1360, 1140, 1980, 530, "inject into prompt", C_GREEN, 3, offset=(-60, -18))

# Embedding -> Redis (store embeddings)
arrow_label(590, 1140, 900, 340, "store embeddings", C_PURPLE, 2, offset=(-60, 5))

# ============================================================
# Legend
# ============================================================
lx, ly = 2420, 1080
section_bg(lx - 20, ly - 10, 380, 340, "LEGEND", C_WHITE)

ly += 20
for label, color, dash in [
    ("Memory / Sessions", C_BLUE_LT, False),
    ("Fact Storage / RAG", C_GREEN, False),
    ("LLM Inference", C_TEAL, False),
    ("Embeddings", C_PURPLE, False),
    ("Training Pipeline", C_ORANGE, True),
    ("Cache / Internal", C_GRAY_LT, True),
]:
    arrow(lx, ly + 10, lx + 50, ly + 10, color, 3, dash)
    draw.text((lx + 60, ly), label, fill=C_DIM, font=font_small)
    ly += 28

ly += 15
draw.text((lx, ly), "Services", fill=C_WHITE, font=font_label)
ly += 22
for svc, port, status in [
    ("hivecoder-llm", "8089", "system"),
    ("llama-server", "8080", "user"),
    ("hive-mind-http", "8090", "system"),
    ("hivecoder-learning", "-", "system"),
]:
    draw.text((lx, ly), f"{svc}", fill=C_DIM, font=font_small)
    draw.text((lx + 210, ly), f":{port}", fill=C_CYAN, font=font_small)
    draw.text((lx + 270, ly), status, fill=(100, 105, 120), font=font_small)
    ly += 22

ly += 10
draw.text((lx, ly), "Updated: 2026-02-15", fill=(70, 75, 90), font=font_small)

# ============================================================
# Save
# ============================================================
out = "hivemind_architecture.png"
img.save(out, optimize=True)
print(f"Saved: {out} ({W}x{H})")
