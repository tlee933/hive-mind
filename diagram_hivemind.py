#!/usr/bin/env python3
"""Hive-Mind + AI Suricata full architecture block diagram using Pillow."""
from PIL import Image, ImageDraw, ImageFont
import os

W, H = 3800, 2200
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
C_CRIMSON  = (180, 40, 60)

def find_font(size):
    paths = [
        "/usr/share/fonts/google-noto-sans-fonts/NotoSans-Bold.ttf",
        "/usr/share/fonts/google-noto-sans-fonts/NotoSans-Regular.ttf",
        "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf",
        "/usr/share/fonts/liberation-sans/LiberationSans-Bold.ttf",
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

font_title = find_font(48)
font_subtitle = find_font_regular(22)
font_heading = find_font(24)
font_body = find_font_regular(19)
font_small = find_font_regular(16)
font_label = find_font(16)
font_port = find_font(18)
font_hw = find_font(20)

img = Image.new('RGB', (W, H), BG)
draw = ImageDraw.Draw(img)

import math

def rounded_rect(x, y, w, h, color, radius=16, border=None, border_w=2):
    draw.rounded_rectangle([x, y, x+w, y+h], radius=radius, fill=color)
    if border:
        draw.rounded_rectangle([x, y, x+w, y+h], radius=radius, outline=border, width=border_w)

def box(x, y, w, h, label, sublabel, color, text_color=C_WHITE, border=None):
    rounded_rect(x, y, w, h, color, border=border, border_w=3 if border else 2)
    bbox = draw.textbbox((0, 0), label, font=font_heading)
    tw = bbox[2] - bbox[0]
    if sublabel:
        draw.text((x + (w - tw) // 2, y + h // 2 - 18), label, fill=text_color, font=font_heading)
        bbox2 = draw.textbbox((0, 0), sublabel, font=font_small)
        tw2 = bbox2[2] - bbox2[0]
        draw.text((x + (w - tw2) // 2, y + h // 2 + 8), sublabel, fill=C_DIM, font=font_small)
    else:
        draw.text((x + (w - tw) // 2, y + (h - 24) // 2), label, fill=text_color, font=font_heading)

def small_box(x, y, w, h, label, sublabel, color, text_color=C_WHITE, border=None):
    rounded_rect(x, y, w, h, color, radius=10, border=border, border_w=2 if border else 1)
    bbox = draw.textbbox((0, 0), label, font=font_body)
    tw = bbox[2] - bbox[0]
    if sublabel:
        draw.text((x + (w - tw) // 2, y + h // 2 - 14), label, fill=text_color, font=font_body)
        bbox2 = draw.textbbox((0, 0), sublabel, font=font_small)
        tw2 = bbox2[2] - bbox2[0]
        draw.text((x + (w - tw2) // 2, y + h // 2 + 6), sublabel, fill=C_DIM, font=font_small)
    else:
        draw.text((x + (w - tw) // 2, y + (h - 18) // 2), label, fill=text_color, font=font_body)

def arrow(x1, y1, x2, y2, color=C_GRAY_LT, width=3, dashed=False):
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
    arr_len = 12
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
draw.text((W // 2 - 420, 18), "Hive-Mind + AI Suricata Architecture", fill=C_WHITE, font=font_title)
draw.text((W // 2 - 360, 72), "Distributed AI: Memory, RAG, Threat Analysis, Continuous Learning", fill=C_DIM, font=font_subtitle)

# ============================================================
# HARDWARE NODES (top strip)
# ============================================================
hw_y = 108
section_bg(40, hw_y, 1160, 100, "AURORA (192.168.1.100)", C_TEAL)
draw.text((60, hw_y + 32), "Ryzen 9 5900X  |  32GB RAM  |  R9700 32GB VRAM  |  ROCm 7.12  |  Fedora 43 Kinoite", fill=C_DIM, font=font_small)
draw.text((60, hw_y + 55), "HiveCoder LLM  |  Redis Cluster (7000-7005)  |  Training Pipeline  |  MCP Server", fill=C_TEAL, font=font_small)

section_bg(1240, hw_y, 1160, 100, "ALDERLAKE (192.168.1.10)", C_BLUE)
draw.text((1260, hw_y + 32), "i7-12700 (20T)  |  64GB RAM  |  6700XT 12GB (idle)  |  Fedora CoreOS", fill=C_DIM, font=font_small)
draw.text((1260, hw_y + 55), "AI Suricata (container)  |  Redis Replicas (7003-7005)  |  Embedding Server", fill=C_BLUE_LT, font=font_small)

section_bg(2440, hw_y, 620, 100, "NAS (192.168.1.7)", C_GRAY_LT)
draw.text((2460, hw_y + 32), "ReadyNAS  |  11TB  |  NFS", fill=C_DIM, font=font_small)
draw.text((2460, hw_y + 55), "EVE logs  |  Model archives  |  Git repos", fill=C_GRAY_LT, font=font_small)

section_bg(3100, hw_y, 660, 100, "OPNsense (192.168.1.1)", C_RED)
draw.text((3120, hw_y + 32), "Suricata 8.0.3 IPS (Netmap)  |  Unbound DNS + DNSBL", fill=C_DIM, font=font_small)
draw.text((3120, hw_y + 55), "Syslog EVE -> alderlake:5140  |  ai_blocklist alias  |  Firewall API", fill=C_RED, font=font_small)

# ============================================================
# AI SURICATA STACK (right side)
# ============================================================
sx = 2440
sy = 240
section_bg(sx, sy, 1320, 860, "AI SURICATA THREAT ANALYSIS", C_RED)

# OPNsense -> eve_receiver
box(sx + 40, sy + 40, 280, 70, "OPNsense Syslog", "EVE JSON + fast-log", (100, 30, 40), border=C_RED)
port_badge(sx + 330, sy + 50, "5140/TCP", C_RED)
arrow(sx + 320, sy + 75, sx + 430, sy + 75, C_RED, 3)

# eve_receiver
box(sx + 430, sy + 40, 280, 70, "eve_receiver.py", "Parse EVE, push to Redis", C_CRIMSON)

# ai_suricata daemon
box(sx + 40, sy + 150, 300, 80, "ai_suricata.py", "Poll Redis, analyze, auto-block", C_CRIMSON)
draw.text((sx + 60, sy + 240), "4-tier auto-block  |  conf >= 0.70", fill=C_DIM, font=font_small)

# HiveCoder analysis
box(sx + 380, sy + 150, 280, 80, "HiveCoder Analysis", "Qwen3-14B threat reasoning", C_TEAL, border=C_TEAL)
arrow(sx + 340, sy + 190, sx + 380, sy + 190, C_TEAL, 2)
draw.text((sx + 400, sy + 240), "severity, confidence, IOC, FP score", fill=C_DIM, font=font_small)

# Dashboard
box(sx + 700, sy + 40, 260, 70, "Dashboard", "Metrics, charts, blocks", C_BLUE)
port_badge(sx + 970, sy + 50, "8080", C_CYAN)

# Auto-block -> OPNsense
box(sx + 40, sy + 290, 300, 70, "Auto-Block Engine", "OPNsense API -> ai_blocklist", C_RED, border=C_ORANGE)
arrow(sx + 190, sy + 230, sx + 190, sy + 290, C_RED, 2)

# Correlation engine
box(sx + 380, sy + 290, 280, 70, "Correlation Engine", "IP patterns, kill chain stage", (130, 50, 50))
arrow(sx + 340, sy + 325, sx + 380, sy + 325, C_RED, 2, dashed=True)

# OPNsense firewall response
box(sx + 700, sy + 150, 260, 70, "OPNsense API", "Alias + rule management", (100, 30, 40), border=C_RED)
arrow(sx + 340, sy + 325, sx + 700, sy + 185, C_RED, 2, dashed=True)

# Blocked stats box
box(sx + 700, sy + 290, 260, 70, "Firewall Rules", "Banned ports, ICMP, RFC1918", (80, 30, 40))

# MCP Server for AI Suricata
box(sx + 40, sy + 400, 300, 70, "MCP Server", "13 tools: query, block, correlate", C_BLUE)
port_badge(sx + 350, sy + 410, "MCP", C_BLUE)

# Daily summary
box(sx + 380, sy + 400, 280, 70, "Daily Summary", "Threat level, noise ratio, trends", (130, 80, 30))

# DNSBL
box(sx + 700, sy + 400, 260, 70, "Unbound DNSBL", "7 blocklists (hagezi, ThreatFox)", C_PURPLE)

# NAS storage
box(sx + 40, sy + 510, 300, 70, "NAS Storage", "/var/mnt/ai/suricata/", C_GRAY, border=C_GRAY_LT)
draw.text((sx + 60, sy + 590), "EVE logs  |  AI results  |  analysis JSONL", fill=C_DIM, font=font_small)

# Learning feedback
box(sx + 380, sy + 510, 280, 70, "Learning Queue", "Training samples -> Redis", C_ORANGE)
arrow(sx + 380, sy + 545, sx + 340, sy + 545, C_ORANGE, 2, dashed=True)
draw.text((sx + 400, sy + 590), "Feeds back into HiveCoder training", fill=C_ORANGE, font=font_small)

# DNS redirect
box(sx + 700, sy + 510, 260, 70, "DNS Redirect", "Force all DNS through Unbound", (80, 50, 120))

# Suricata detection stats
draw.text((sx + 60, sy + 640), "Suricata: IPS Netmap mode  |  Detect: HIGH  |  HTTP extended logging", fill=C_DIM, font=font_small)
draw.text((sx + 60, sy + 665), "Auto-block tiers: critical/high + medium-block + critical-investigate + malicious-category", fill=C_RED, font=font_small)

# Internal flow arrows
arrow(sx + 570, sy + 75, sx + 570, sy + 150, C_CRIMSON, 2)  # eve -> analysis
arrow(sx + 710, sy + 75, sx + 830, sy + 40, C_BLUE, 2, dashed=True)  # eve -> dashboard

# ============================================================
# HIVE-MIND STACK (left-center)
# ============================================================
hx = 40
hy = 240
section_bg(hx, hy, 1160, 560, "HIVE-MIND AI MEMORY + RAG", C_BLUE)

# Clients
box(hx + 40, hy + 40, 200, 65, "Claude Code", "MCP client (stdio)", (50, 55, 70), border=C_WHITE)
box(hx + 260, hy + 40, 200, 65, "HTTP Clients", "curl, apps, scripts", (50, 55, 70), border=C_GRAY_LT)
box(hx + 480, hy + 40, 200, 65, "Chrome Ext", "Firefox bridge", (50, 55, 70), border=C_GRAY_LT)

# MCP/HTTP servers
box(hx + 40, hy + 130, 200, 65, "MCP Server", "stdio protocol", C_BLUE)
box(hx + 260, hy + 130, 200, 65, "HTTP API", "FastAPI + Uvicorn", C_BLUE)
port_badge(hx + 470, hy + 140, "8090", C_CYAN)

arrow(hx + 140, hy + 105, hx + 140, hy + 130, C_WHITE, 2)
arrow(hx + 360, hy + 105, hx + 360, hy + 130, C_GRAY_LT, 2)
arrow(hx + 580, hy + 105, hx + 460, hy + 130, C_GRAY_LT, 2)

# MCP Tools
draw.text((hx + 60, hy + 210), "MCP Tools (13 endpoints)", fill=C_BLUE_LT, font=font_label)

small_box(hx + 40, hy + 235, 150, 45, "memory_store", "ctx, files", C_BLUE)
small_box(hx + 200, hy + 235, 150, 45, "memory_recall", "sessions", C_BLUE)
small_box(hx + 360, hy + 235, 150, 45, "fact_store", "key + embed", (40, 100, 55))
small_box(hx + 520, hy + 235, 150, 45, "fact_get", "retrieve", (40, 100, 55))

small_box(hx + 40, hy + 290, 150, 45, "llm_generate", "prompt+RAG", (30, 120, 110))
small_box(hx + 200, hy + 290, 150, 45, "llm_code_assist", "review/fix", (30, 120, 110))
small_box(hx + 360, hy + 290, 150, 45, "llm_complete", "FIM", (30, 120, 110))
small_box(hx + 520, hy + 290, 150, 45, "learning_add", "samples", (160, 100, 20))

small_box(hx + 40, hy + 345, 150, 45, "tool_cache", "get/set", C_GRAY)
small_box(hx + 200, hy + 345, 150, 45, "web_fetch", "URL content", C_GRAY)
small_box(hx + 360, hy + 345, 150, 45, "web_search", "DDG search", C_GRAY)
small_box(hx + 520, hy + 345, 150, 45, "get_stats", "health", C_GRAY)

# RAG Pipeline
draw.text((hx + 720, hy + 210), "RAG Pipeline", fill=C_GREEN, font=font_label)
box(hx + 700, hy + 235, 200, 55, "Semantic Search", "cosine >= 0.45", C_GREEN)
box(hx + 920, hy + 235, 200, 55, "Keyword Fallback", "70+ keyword map", (40, 130, 70))
box(hx + 700, hy + 305, 200, 55, "Fact Injection", "into LLM prompt", (40, 130, 70))
box(hx + 920, hy + 305, 200, 55, "Quality Tracker", "hit rate, logs", (40, 130, 70))
draw.text((hx + 720, hy + 375), "84% hit rate  |  semantic-only retrieval", fill=C_GREEN, font=font_small)

# Embedding
box(hx + 700, hy + 410, 420, 55, "Embedding Engine (bge-small-en-v1.5)", "384-dim, CPU, SentenceTransformer", C_PURPLE)
port_badge(hx + 700, hy + 475, "8081 (alderlake)", C_PURPLE)

# ============================================================
# REDIS CLUSTER (center)
# ============================================================
rx = 1240
ry = 240
section_bg(rx, ry, 560, 560, "REDIS CLUSTER", C_RED)

box(rx + 30, ry + 40, 240, 55, "session:{id}", "hash - ctx, files, task", (130, 40, 50))
box(rx + 290, ry + 40, 240, 55, "facts:system", "hash - key -> value", (130, 40, 50))

box(rx + 30, ry + 110, 240, 55, "learning:queue", "stream - interactions", (150, 90, 30))
box(rx + 290, ry + 110, 240, 55, "fact_embeddings:*", "base64 float32[384]", (100, 40, 80))

box(rx + 30, ry + 180, 240, 55, "suricata:alerts", "EVE alerts queue", (130, 40, 50))
box(rx + 290, ry + 180, 240, 55, "suricata:ai_results", "enriched analysis", (130, 40, 50))

box(rx + 30, ry + 250, 240, 55, "suricata:ai_blocks", "auto-block audit log", (130, 40, 50))
box(rx + 290, ry + 250, 240, 55, "suricata:ai_correlations", "attack patterns", (130, 40, 50))

box(rx + 30, ry + 320, 500, 55, "llm:cache + tool:cache + rag:retrieval_log", "inference 30m, tool 1h, quality tracking", (60, 25, 35))

# Cluster topology
box(rx + 30, ry + 410, 500, 55, "Aurora: 3 Masters (7000-7002)  |  Alderlake: 3 Replicas (7003-7005)", "", (60, 25, 35), border=C_RED)
draw.text((rx + 50, ry + 480), "16384 hash slots  |  password auth  |  GET 14.7K/s  |  <1ms LAN", fill=C_DIM, font=font_small)

# ============================================================
# LLM INFERENCE (center-right)
# ============================================================
lx = 1840
ly = 240
section_bg(lx, ly, 560, 280, "LLM INFERENCE (GPU)", C_TEAL)

box(lx + 30, ly + 40, 240, 70, "Qwen3-14B", "Q4_K_M  |  ~8GB VRAM", C_TEAL)
port_badge(lx + 280, ly + 50, "8089", C_TEAL)
draw.text((lx + 360, ly + 55), "llama-server", fill=C_DIM, font=font_small)

box(lx + 30, ly + 130, 240, 70, "Hive-Mind HTTP", "OpenAI-compat proxy", C_BLUE)
port_badge(lx + 280, ly + 140, "8090", C_CYAN)
draw.text((lx + 360, ly + 145), "RAG injection", fill=C_DIM, font=font_small)

arrow(lx + 150, ly + 110, lx + 150, ly + 130, C_TEAL, 2)

# GPU box
box(lx + 30, ly + 220, 500, 40, "AMD R9700  |  32GB VRAM  |  ROCm 7.12  |  ~8GB used, ~24GB free", "", (40, 50, 60), border=C_TEAL)

# ============================================================
# CONTINUOUS LEARNING (center-right below)
# ============================================================
tx = 1840
ty = 560
section_bg(tx, ty, 560, 240, "CONTINUOUS LEARNING", C_ORANGE)

small_box(tx + 30, ty + 35, 160, 50, "Drain Queue", "every 5 min", C_ORANGE)
small_box(tx + 200, ty + 35, 160, 50, "Quality Filter", "min len, success", (160, 100, 20))
small_box(tx + 370, ty + 35, 160, 50, "Collect Data", "from Redis", (160, 100, 20))

small_box(tx + 30, ty + 100, 160, 50, "LoRA Training", "r=8, alpha=16", (160, 100, 20))
small_box(tx + 200, ty + 100, 160, 50, "GGUF Export", "Q5_K_M quant", (160, 100, 20))
small_box(tx + 370, ty + 100, 160, 50, "Hot Swap", "symlink + restart", (160, 100, 20), border=C_TEAL)

arrow(tx + 190, ty + 60, tx + 200, ty + 60, C_ORANGE, 2)
arrow(tx + 360, ty + 60, tx + 370, ty + 60, C_ORANGE, 2)
arrow(tx + 450, ty + 85, tx + 110, ty + 100, C_ORANGE, 2)
arrow(tx + 190, ty + 125, tx + 200, ty + 125, C_ORANGE, 2)
arrow(tx + 360, ty + 125, tx + 370, ty + 125, C_ORANGE, 2)

draw.text((tx + 50, ty + 170), "Timer: 2 AM daily  |  Daemon: 50-sample threshold  |  QLoRA 4-bit supported", fill=C_DIM, font=font_small)
draw.text((tx + 50, ty + 192), "Stops llama-server during training  |  Backup to NAS with 30-day cleanup", fill=C_DIM, font=font_small)

# ============================================================
# DATA FLOW ARROWS (cross-section)
# ============================================================

# Hive-Mind tools -> Redis
arrow_label(hx + 670, hy + 260, rx + 30, ry + 65, "store/recall", C_BLUE_LT, 2, offset=(-30, -16))
arrow_label(hx + 670, hy + 315, rx + 30, ry + 135, "samples", C_ORANGE, 2, offset=(-20, -16))

# Redis -> LLM (inference via HTTP)
arrow_label(rx + 530, ry + 65, lx + 30, ly + 160, "RAG facts", C_GREEN, 2, offset=(-30, -16))

# Redis -> AI Suricata (alerts)
arrow_label(rx + 530, ry + 205, sx + 40, sy + 190, "alerts queue", C_RED, 2, offset=(-30, -16))

# AI Suricata -> Redis (results)
arrow_label(sx + 40, sy + 350, rx + 530, ry + 275, "results + blocks", C_CRIMSON, 2, dashed=True, offset=(-40, 5))

# AI Suricata -> LLM (analysis)
arrow_label(sx + 380, sy + 160, lx + 530, ly + 75, "threat analysis", C_TEAL, 2, offset=(-50, -16))

# Training -> LLM (hot swap)
arrow(tx + 530, ty + 125, lx + 530, ly + 75, C_TEAL, 2, dashed=True)
draw.text((lx + 540, ly + 100), "hot swap", fill=C_TEAL, font=font_small)

# Learning -> Redis (drain)
arrow_label(tx + 30, ty + 60, rx + 530, ry + 135, "drain", C_ORANGE, 2, dashed=True, offset=(-20, -16))

# ============================================================
# NETWORK FLOW (bottom)
# ============================================================
ny_ = 1140
section_bg(40, ny_, 3720, 280, "NETWORK SECURITY FLOW", C_YELLOW)

# Internet
box(80, ny_ + 40, 200, 70, "Internet", "WAN traffic", (60, 60, 80), border=C_YELLOW)

# OPNsense Suricata
box(340, ny_ + 40, 280, 70, "Suricata IPS", "Netmap, detect HIGH", C_RED)
arrow(280, ny_ + 75, 340, ny_ + 75, C_YELLOW, 3)

# Firewall
box(680, ny_ + 40, 280, 70, "OPNsense Firewall", "ai_blocklist + banned ports", (100, 30, 40))
arrow(620, ny_ + 75, 680, ny_ + 75, C_RED, 3)

# Syslog
box(340, ny_ + 140, 280, 70, "Syslog (EVE JSON)", "TCP -> alderlake:5140", (80, 40, 50))
arrow(480, ny_ + 110, 480, ny_ + 140, C_RED, 2)

# LAN
box(1020, ny_ + 40, 200, 70, "LAN / WiFi", "Home network", (40, 80, 50), border=C_GREEN)
arrow(960, ny_ + 75, 1020, ny_ + 75, C_GREEN, 3)

# DNS flow
box(1020, ny_ + 140, 200, 70, "Unbound DNS", "DNSBL + redirect", C_PURPLE)
arrow(1120, ny_ + 110, 1120, ny_ + 140, C_PURPLE, 2)

# eve_receiver on alderlake
box(680, ny_ + 140, 280, 70, "eve_receiver", "alderlake container", C_CRIMSON)
arrow(620, ny_ + 175, 680, ny_ + 175, C_RED, 2)

# Redis
box(1020, ny_ + 140, 200, 70, "Unbound DNS", "DNSBL + redirect", C_PURPLE)

# AI analysis flow
box(1280, ny_ + 40, 260, 70, "AI Suricata", "Analyze + correlate", C_CRIMSON)
arrow(960, ny_ + 175, 1280, ny_ + 75, C_CRIMSON, 2)

# Auto-block feedback
box(1280, ny_ + 140, 260, 70, "Auto-Block", "API -> ai_blocklist", C_RED, border=C_ORANGE)
arrow(1410, ny_ + 110, 1410, ny_ + 140, C_RED, 2)
arrow(1280, ny_ + 175, 820, ny_ + 110, C_ORANGE, 2, dashed=True)
draw.text((1000, ny_ + 135), "feedback loop", fill=C_ORANGE, font=font_small)

# HiveCoder
box(1600, ny_ + 40, 240, 70, "HiveCoder LLM", "Threat reasoning", C_TEAL)
arrow(1540, ny_ + 75, 1600, ny_ + 75, C_TEAL, 2)

# Learning
box(1600, ny_ + 140, 240, 70, "Learning Queue", "Training feedback", C_ORANGE)
arrow(1720, ny_ + 110, 1720, ny_ + 140, C_ORANGE, 2, dashed=True)

# Dashboard
box(1900, ny_ + 40, 220, 70, "Dashboard", "Real-time metrics", C_BLUE)
port_badge(1900, ny_ + 115, "8080", C_CYAN)

# Blocked stats
draw.text((2180, ny_ + 50), "Active Defenses:", fill=C_WHITE, font=font_label)
draw.text((2180, ny_ + 75), "25+ CIDR blocks  |  24 banned ports", fill=C_RED, font=font_small)
draw.text((2180, ny_ + 95), "ICMP blocked  |  RFC1918 anti-spoof", fill=C_RED, font=font_small)
draw.text((2180, ny_ + 115), "7 DNS blocklists  |  DNS redirect", fill=C_PURPLE, font=font_small)
draw.text((2180, ny_ + 135), "4-tier auto-block  |  conf >= 0.70", fill=C_ORANGE, font=font_small)
draw.text((2180, ny_ + 155), "IPS Netmap  |  Detect HIGH", fill=C_RED, font=font_small)
draw.text((2180, ny_ + 175), "Continuous learning from alerts", fill=C_ORANGE, font=font_small)

# ============================================================
# LEGEND
# ============================================================
lgx, lgy = 2700, ny_ + 30
section_bg(lgx, lgy, 380, 240, "LEGEND", C_WHITE)

lgy += 25
for label, color, dash in [
    ("Hive-Mind / Memory", C_BLUE_LT, False),
    ("Threat Analysis", C_RED, False),
    ("LLM Inference", C_TEAL, False),
    ("RAG / Embeddings", C_GREEN, False),
    ("Training / Learning", C_ORANGE, True),
    ("DNS / Blocklists", C_PURPLE, False),
    ("Firewall Feedback", C_ORANGE, False),
]:
    arrow(lgx + 20, lgy + 10, lgx + 60, lgy + 10, color, 3, dash)
    draw.text((lgx + 70, lgy), label, fill=C_DIM, font=font_small)
    lgy += 24

lgy += 8
draw.text((lgx + 20, lgy), "Key Ports:", fill=C_WHITE, font=font_label)
lgy += 20
for svc, port in [
    ("llama-server (Qwen3-14B)", "8089"),
    ("hive-mind-http (proxy)", "8090"),
    ("eve_receiver (syslog)", "5140"),
    ("dashboard (metrics)", "8080"),
    ("Redis cluster", "7000-7005"),
]:
    draw.text((lgx + 20, lgy), svc, fill=C_DIM, font=font_small)
    draw.text((lgx + 290, lgy), f":{port}", fill=C_CYAN, font=font_small)
    lgy += 20

# ============================================================
# Timestamp
# ============================================================
draw.text((W - 260, H - 30), "Updated: 2026-03-01", fill=(70, 75, 90), font=font_small)

# ============================================================
# Save
# ============================================================
out = "hivemind_architecture.png"
img.save(out, optimize=True)
print(f"Saved: {out} ({W}x{H})")
