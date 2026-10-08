#!/usr/bin/env python3
"""
N.E.D.S rev C — complete system schematic generator (eight 16-bit channels).

Every component in the Rev C build:
- Eight gas sensor channels (MQ-2, MQ-3, MQ-5, MQ-137, MQ-138, MiCS-6814 OX/RED/NH3)
- 2 × ADS1115 16-bit I²C ADC converters (0x48 and 0x49) with 10k/10k level shifters
- Environmental compensation (BME688)
- Thermal-desorption heater (Q1 IRLB8721) and 370 pump (Q2 IRLB8721 + D1 1N5819 flyback)
- Switched sensor power rail (Q4 NPN + Q3 IRF9540N P-MOSFET)
- Battery monitor on ESP32 GPIO35 (33k/10k divider)
- K-type thermocouple readback (MAX6675 SPI)
- GPS module (NEO-6M UART)
- Officer UI (buzzer, LED1, SCAN button)
- Wired UART fallback (ESP32 GPIO33/32 to Pi 4 pins 10/8)
- Protected dual-buck power chain (LM2596 5V + 5A Sync Buck for Pi 4)

Writes neds-v3-schematic.svg and neds-full-schematic.svg, self-checking that no wire
crosses component boxes and no two different nets overlap.
"""
import os

OUT = os.path.dirname(os.path.abspath(__file__))
W, H = 2660, 1760

NET = dict(
    bat="#B0175A", v5="#E53935", mq5="#FF4F8B", v33="#FF8C1A", gnd="#3A4452",
    ana="#1E8C4E", i2c="#2F7DE1", scl="#D49B00", ctl="#00A3A3", spi="#7B5BD6",
    uart="#8E44AD", ui="#8D6E63", wifi="#1F6FEB",
)
RAIL = {
    "BAT+ 7.4 V": (1420, "bat"),
    "+5 V regulated": (1470, "v5"),
    "MQ_5V (switched)": (1520, "mq5"),
    "3V3": (1570, "v33"),
    "GND (common)": (1620, "gnd"),
}
RAIL_X0, RAIL_X1 = 60, 2600
GND_Y = RAIL["GND (common)"][0]

svg, PIN, BOXES, SEGS = [], {}, [], []
INK, MUTED = "#1B2430", "#6B7785"


# ----------------------------------------------------------------- primitives
def text(x, y, t, size=11, anchor="start", fill=INK, bold=False):
    svg.append(f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}" '
               f'fill="{fill}" font-weight="{600 if bold else 400}">{t}</text>')


def section(x, y, label):
    text(x, y, label, 12.5, "start", "#95A2B0", True)
    svg.append(f'<line x1="{x}" y1="{y+7}" x2="{x+210}" y2="{y+7}" '
               f'stroke="#CBD4DD" stroke-width="1.5"/>')


def box(ref, title, sub, x, y, w, h, fill, left=(), right=(), tcol="#fff", note=None):
    BOXES.append((ref, x, y, w, h))
    svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="{fill}" '
               f'stroke="{INK}" stroke-width="2"/>')
    text(x + w / 2, y + h - 17, title, 12.5, "middle", tcol, True)
    if sub:
        text(x + w / 2, y + h - 4, sub, 9, "middle", tcol)
    if note:
        text(x + w / 2, y - 8, note, 9, "middle", MUTED)
    for label, py in left:
        svg.append(f'<rect x="{x-5}" y="{py-4.5}" width="14" height="9" rx="2" '
                   f'fill="#F0C04A" stroke="{INK}" stroke-width="0.8"/>')
        text(x + 14, py + 3.5, label, 9.5, "start", tcol)
        PIN[(ref, label)] = (x - 5, py)
    for label, py in right:
        svg.append(f'<rect x="{x+w-9}" y="{py-4.5}" width="14" height="9" rx="2" '
                   f'fill="#F0C04A" stroke="{INK}" stroke-width="0.8"/>')
        text(x + w - 14, py + 3.5, label, 9.5, "end", tcol)
        PIN[(ref, label)] = (x + w + 5, py)


def wire(net, pts, w=2.6, dash=False):
    for a, b in zip(pts, pts[1:]):
        SEGS.append((net, a, b))
    d = " ".join(f"{px},{py}" for px, py in pts)
    da = ' stroke-dasharray="9 6"' if dash else ''
    svg.append(f'<polyline points="{d}" fill="none" stroke="{NET[net]}" stroke-width="{w}" '
               f'stroke-linejoin="round" stroke-linecap="round"{da}/>')


def dot(x, y, net):
    svg.append(f'<circle cx="{x}" cy="{y}" r="4.2" fill="{NET[net]}"/>')


def to_rail(net, pt, rail, colx):
    """Route a pin horizontally to its own lane, then down to a power rail."""
    ry = RAIL[rail][0]
    wire(net, [pt, (colx, pt[1]), (colx, ry)])
    dot(colx, ry, net)


def res_h(x, y, val, ref=""):
    svg.append(f'<rect x="{x}" y="{y-8}" width="66" height="16" rx="2.5" fill="#E8D5A8" '
               f'stroke="#6B5836" stroke-width="1.4"/>')
    for i, c in enumerate(["#6B3A1F", "#111", "#C4881F"]):
        svg.append(f'<rect x="{x+13+i*13}" y="{y-8}" width="4.5" height="16" fill="{c}"/>')
    text(x + 33, y - 12, val, 9, "middle", MUTED)
    if ref:
        text(x + 33, y + 19, ref, 8.5, "middle", "#95A2B0")
    return (x, y), (x + 66, y)


def res_v(x, y, val, ref=""):
    svg.append(f'<rect x="{x-8}" y="{y}" width="16" height="58" rx="2.5" fill="#E8D5A8" '
               f'stroke="#6B5836" stroke-width="1.4"/>')
    for i, c in enumerate(["#6B3A1F", "#111", "#C4881F"]):
        svg.append(f'<rect x="{x-8}" y="{y+11+i*12}" width="16" height="4.5" fill="{c}"/>')
    text(x + 13, y + 24, val, 9, "start", MUTED)
    if ref:
        text(x + 13, y + 36, ref, 8.5, "start", "#95A2B0")
    return (x, y), (x, y + 58)


def diode_v(x, y, ref, part, length=56):
    """Vertical diode, cathode bar at the TOP (conducts upward)."""
    svg.append(f'<line x1="{x}" y1="{y}" x2="{x}" y2="{y+length}" stroke="{INK}" stroke-width="2"/>')
    mid = y + length / 2
    svg.append(f'<polygon points="{x-11},{mid+8} {x+11},{mid+8} {x},{mid-8}" fill="{INK}"/>')
    svg.append(f'<line x1="{x-13}" y1="{mid-8}" x2="{x+13}" y2="{mid-8}" stroke="{INK}" stroke-width="3"/>')
    text(x + 18, mid - 2, ref, 9.5, "start", INK, True)
    text(x + 18, mid + 10, part, 8.5, "start", MUTED)
    return (x, y), (x, y + length)


def diode_h(x, y, ref, part):
    """Horizontal diode, conducts left to right, cathode bar on the right."""
    svg.append(f'<line x1="{x}" y1="{y}" x2="{x+56}" y2="{y}" stroke="{INK}" stroke-width="2"/>')
    svg.append(f'<polygon points="{x+18},{y-11} {x+18},{y+11} {x+36},{y}" fill="{INK}"/>')
    svg.append(f'<line x1="{x+36}" y1="{y-13}" x2="{x+36}" y2="{y+13}" stroke="{INK}" stroke-width="3"/>')
    text(x + 28, y - 19, ref, 9.5, "middle", INK, True)
    text(x + 28, y + 27, part, 8.5, "middle", MUTED)
    return (x, y), (x + 56, y)


def cap_v(x, top, bottom, val, ref, polarized=False):
    """Capacitor drawn between two y values, symbol centred."""
    mid = (top + bottom) / 2
    svg.append(f'<line x1="{x}" y1="{top}" x2="{x}" y2="{mid-7}" stroke="{INK}" stroke-width="2"/>')
    svg.append(f'<line x1="{x-13}" y1="{mid-7}" x2="{x+13}" y2="{mid-7}" stroke="{INK}" stroke-width="3"/>')
    if polarized:
        svg.append(f'<path d="M {x-13} {mid+8} q 13 -9 26 0" fill="none" stroke="{INK}" stroke-width="3"/>')
        text(x - 17, mid - 10, "+", 11, "end", INK, True)
    else:
        svg.append(f'<line x1="{x-13}" y1="{mid+7}" x2="{x+13}" y2="{mid+7}" stroke="{INK}" stroke-width="3"/>')
    svg.append(f'<line x1="{x}" y1="{mid+8}" x2="{x}" y2="{bottom}" stroke="{INK}" stroke-width="2"/>')
    text(x + 17, mid - 2, val, 9, "start", MUTED)
    text(x + 17, mid + 10, ref, 8.5, "start", "#95A2B0")


def fuse_h(x, y, label):
    svg.append(f'<rect x="{x}" y="{y-10}" width="58" height="20" rx="10" fill="#FFF" '
               f'stroke="{INK}" stroke-width="2"/>')
    svg.append(f'<line x1="{x+6}" y1="{y}" x2="{x+52}" y2="{y}" stroke="{INK}" stroke-width="1.6"/>')
    text(x + 29, y - 16, label, 9, "middle", MUTED)
    return x + 58


def switch_h(x, y, label):
    svg.append(f'<circle cx="{x}" cy="{y}" r="4" fill="{INK}"/>')
    svg.append(f'<circle cx="{x+50}" cy="{y}" r="4" fill="{INK}"/>')
    svg.append(f'<line x1="{x}" y1="{y}" x2="{x+42}" y2="{y-16}" stroke="{INK}" stroke-width="2.4"/>')
    text(x + 25, y - 25, label, 9, "middle", MUTED)
    return x + 50


# ----------------------------------------------------------------- title
svg.append(f'<rect width="{W}" height="{H}" fill="#F4F7FA"/>')
text(44, 46, "N.E.D.S — Smart Lathi: complete system schematic (rev C)", 25, "start", INK, True)
text(44, 72, "Team Venjex · SIH 2026 · PS 26026  —  eight 16-bit gas channels on two ADS1115 converters, "
             "environmental compensation, thermal-desorption heater, pump, GPS, battery monitoring & protected power chain.", 12.5, "start", MUTED)
text(44, 91, "Filled dot = junction. Plain crossing = not connected. Every 5 V sensor "
             "output passes its own 10k/10k divider before it reaches a 3.3 V input.",
     11.5, "start", MUTED)

# shared routing lanes
L_MQ5, L_SGND, L_DIV = 240, 274, 395      # left of the sensors and dividers
L_BME33 = 560                              # BME 3.3 V feed
L_ESP5, L_ESPG = 945, 960                  # right of the ESP32 (clear of ESP x=915)
L_RGND, L_R33, L_R5 = 1150, 1166, 1182     # right service lane

# ----------------------------------------------------------------- sensors
section(60, 132, "GAS SENSOR ARRAY")

SENSORS_REV_C = [
    ("U1", "MQ-2", "LPG, smoke, broad VOC", 150, "ADS1", "A0", "R1", "R2"),
    ("U2", "MQ-3", "alcohol, solvent vapour", 310, "ADS1", "A1", "R3", "R4"),
    ("U3", "MQ-5", "LPG, methane, coal gas", 470, "ADS1", "A2", "R5", "R6"),
    ("U4", "MQ-137", "ammonia (NH₃)", 630, "ADS1", "A3", "R7", "R8"),
    ("U5", "MQ-138", "ketones, aldehydes", 790, "ADS2", "A0", "R9", "R10"),
]
for ref, name, det, y, _ads, _pin, _ra, _rb in SENSORS_REV_C:
    box(ref, name, det, 60, y, 170, 128, "#1F5FBF",
        right=[("VCC", y + 26), ("GND", y + 52), ("D0", y + 78), ("AOUT", y + 104)])

box("U6", "MiCS-6814", "NO₂ / CO / NH₃ — three elements", 60, 990, 170, 154,
    "#B71C1C", right=[("VCC", 1016), ("GND", 1042), ("OX NO₂", 1068),
                      ("RED CO", 1094), ("NH₃", 1120)])

box("U7", "BME688", "temperature / humidity / pressure", 60, 1190, 170, 116, "#6A3FB5",
    right=[("VCC", 1216), ("GND", 1240), ("SCL", 1264), ("SDA", 1288)])

section(410, 132, "16-BIT ANALOG FRONT END")

# ADS1115 Converters
box("U8", "ADS1115 #1", "ADDR → GND · 0x48", 410, 176, 170, 560, "#0F6F7A",
    left=[("A0", 254), ("A1", 414), ("A2", 574), ("A3", 734)],
    right=[("VCC", 230), ("GND", 270), ("SCL", 330), ("SDA", 370)],
    tcol="#fff")

box("U9", "ADS1115 #2", "ADDR → VDD · 0x49", 410, 750, 170, 420, "#0F6F7A",
    left=[("A0", 894), ("A1", 1068), ("A2", 1094), ("A3", 1120)],
    right=[("VCC", 790), ("GND", 830), ("SCL", 890), ("SDA", 930)],
    tcol="#fff")

text(410, 150, "Every 5 V sensor output passes a 10k/10k divider before reaching a 16-bit ADS1115 ADC.",
     9, "start", MUTED)

# Level shifter chains for U1..U6 to ADS1115 #1 and #2
ALL_LEVEL_SHIFTS = [
    ("U1", "AOUT", 150 + 104, "R1", "R2", "U8", "A0"),
    ("U2", "AOUT", 310 + 104, "R3", "R4", "U8", "A1"),
    ("U3", "AOUT", 470 + 104, "R5", "R6", "U8", "A2"),
    ("U4", "AOUT", 630 + 104, "R7", "R8", "U8", "A3"),
    ("U5", "AOUT", 790 + 104, "R9", "R10", "U9", "A0"),
    ("U6", "OX NO₂", 1068, "R11", "R12", "U9", "A1"),
    ("U6", "RED CO", 1094, "R21", "R22", "U9", "A2"),
    ("U6", "NH₃", 1120, "R23", "R24", "U9", "A3"),
]

for s_ref, pin_lbl, ay, ra, rb, ads_ref, ads_pin in ALL_LEVEL_SHIFTS:
    res_h(250, ay, "10k", ra)
    wire("ana", [PIN[(s_ref, pin_lbl)], (250, ay)])
    node = (335, ay)
    wire("ana", [(316, ay), node])
    dot(*node, "ana")
    
    res_v(335, ay + 15, "10k", rb)
    wire("ana", [node, (335, ay + 15)])
    wire("gnd", [(335, ay + 73), (L_DIV, ay + 73), (L_DIV, GND_Y)])
    
    wire("ana", [node, PIN[(ads_ref, ads_pin)]])

dot(L_DIV, GND_Y, "gnd")

# D0 unused indicators for sensors U1..U5
for ref, _n, _d, y, _a, _p, _ra, _rb in SENSORS_REV_C:
    ux, uy = PIN[(ref, "D0")]
    wire("ana", [(ux, uy), (225, uy)], w=1.9)
    svg.append(f'<circle cx="231" cy="{uy}" r="5.5" fill="none" stroke="{NET["ana"]}" stroke-width="2.2"/>')
    text(238, uy + 3.5, "D0 unused", 8.5, "start", "#C62F28")

# I2C lines from ADS1115 #1 and #2 to BME688 & ESP32
wire("scl", [PIN[("U8", "SCL")], (605, 330), (605, 1264)])
wire("i2c", [PIN[("U8", "SDA")], (620, 370), (620, 1288)])
wire("scl", [PIN[("U9", "SCL")], (605, 890)])
dot(605, 890, "scl")
wire("i2c", [PIN[("U9", "SDA")], (620, 930)])
dot(620, 930, "i2c")

to_rail("v33", PIN[("U8", "VCC")], "3V3", 590)
wire("gnd", [PIN[("U8", "GND")], (580, 270), (580, GND_Y)])
dot(580, GND_Y, "gnd")

to_rail("v33", PIN[("U9", "VCC")], "3V3", 590)
wire("gnd", [PIN[("U9", "GND")], (580, 830), (580, GND_Y)])
dot(580, GND_Y, "gnd")

# ----------------------------------------------------------------- ESP32
ESP_X, ESP_Y, ESP_W, ESP_H = 670, 150, 245, 1190
ESP_LEFT = [
    ("GPIO35 (bat)", 1200),
    ("GPIO22 SCL", 1264),
    ("GPIO21 SDA", 1288),
    ("GND", 1316),
]
ESP_RIGHT = [
    ("GPIO25", 210), ("GPIO26", 362), ("GPIO27", 512),
    ("GPIO14", 666), ("GPIO19", 692), ("GPIO5", 718),
    ("GPIO16", 836), ("GPIO17", 862),
    ("GPIO13", 1006), ("GPIO18", 1060), ("GPIO23", 1146),
    ("GPIO33 TX", 1200), ("GPIO32 RX", 1226),
    ("VIN 5V", 1250), ("GND2", 1276),
]
box("ESP", "ESP32 DevKit V1", "2 × ADS1115 converters handling eight 16-bit channels",
    ESP_X, ESP_Y, ESP_W, ESP_H, "#1F3A5F", left=ESP_LEFT, right=ESP_RIGHT,
    note="GPIO34, GPIO36 and GPIO39 are now free spare analog inputs")

# Battery Monitor Divider on GPIO35
res_v(630, 1110, "33k", "R25")
res_v(620, 1220, "10k", "R26")
node_bat = (630, 1200)
wire("bat", [(2600, RAIL["BAT+ 7.4 V"][0]), (2600, 1440), (630, 1440), (630, 1110)])
dot(2600, RAIL["BAT+ 7.4 V"][0], "bat")
wire("bat", [(630, 1168), node_bat])
dot(*node_bat, "bat")
wire("bat", [node_bat, PIN[("ESP", "GPIO35 (bat)")]])
wire("gnd", [(620, 1220), (610, 1220), (610, 1278), (580, 1278), (580, GND_Y)])
dot(580, GND_Y, "gnd")
text(530, 1180, "BAT+ → 33k → GPIO35 → 10k → GND", 8.5, "end", MUTED)
text(530, 1192, "8.4 V max → 1.95 V to ESP32 ADC", 8.5, "end", MUTED)

# I2C: BME688 straight across to the ESP32
wire("scl", [PIN[("U7", "SCL")], (605, 1264), PIN[("ESP", "GPIO22 SCL")]])
dot(605, 1264, "scl")
wire("i2c", [PIN[("U7", "SDA")], (620, 1288), PIN[("ESP", "GPIO21 SDA")]])
dot(620, 1288, "i2c")
text(300, 1258, "I²C bus: 0x48, 0x49 and 0x76", 9, "start", MUTED)
text(300, 1310, "Supplies eight 16-bit gas channels plus temperature and humidity", 9, "start", MUTED)
text(300, 1321, "figures that correct every MQ reading for drift.", 9, "start", MUTED)

# sensor power
for ref, _n, _d, y, _ad, _p, _ra, _rb in SENSORS_REV_C:
    wire("mq5", [PIN[(ref, "VCC")], (L_MQ5, y + 26), (L_MQ5, RAIL["MQ_5V (switched)"][0])])
    wire("gnd", [PIN[(ref, "GND")], (L_SGND, y + 52), (L_SGND, GND_Y)])
wire("mq5", [PIN[("U6", "VCC")], (L_MQ5, 1016), (L_MQ5, RAIL["MQ_5V (switched)"][0])])
wire("gnd", [PIN[("U6", "GND")], (L_SGND, 1042), (L_SGND, GND_Y)])
dot(L_MQ5, RAIL["MQ_5V (switched)"][0], "mq5")
dot(L_SGND, GND_Y, "gnd")

# U7 (BME688) 3V3 power line
to_rail("v33", PIN[("U7", "VCC")], "3V3", L_BME33)
wire("gnd", [PIN[("U7", "GND")], (L_SGND, 1240), (L_SGND, GND_Y)])

# ESP32 power
to_rail("v5", PIN[("ESP", "VIN 5V")], "+5 V regulated", L_ESP5)
to_rail("gnd", PIN[("ESP", "GND2")], "GND (common)", L_ESPG)
wire("gnd", [PIN[("ESP", "GND")], (600, 1316), (600, GND_Y)])
dot(600, GND_Y, "gnd")

# ----------------------------------------------------------------- drivers
section(935, 132, "SWITCHING — HEATER, PUMP, SENSOR RAIL")

MOS_X, MOS_W = 1200, 150


def gate_network(gate_pin, series_val, series_ref, pd_ref, node_x, pd_x):
    """220R in series from the GPIO, 10k pull-down holding the gate at 0 V on boot."""
    gx, gy = PIN[("ESP", gate_pin)]
    res_h(970, gy, series_val, series_ref)
    wire("ctl", [(gx, gy), (970, gy)])
    node = (node_x, gy)
    wire("ctl", [(1036, gy), node])
    dot(*node, "ctl")
    res_v(pd_x, gy + 20, "10k", pd_ref)
    wire("ctl", [node, (pd_x, gy + 20)])
    wire("gnd", [(pd_x, gy + 78), (pd_x, gy + 100), (L_RGND, gy + 100), (L_RGND, GND_Y)])
    return node


# Q1 nichrome heater
n1 = gate_network("GPIO25", "220R", "R13", "R16", 1110, 1110)
box("Q1", "Q1  IRLB8721", "N-channel, low side", MOS_X, 176, MOS_W, 80, "#C9A46A",
    left=[("G", 204), ("S", 232)], right=[("D", 210)], tcol="#2B2212")
wire("ctl", [n1, PIN[("Q1", "G")]])
wire("gnd", [PIN[("Q1", "S")], (L_RGND, 232), (L_RGND, GND_Y)])

# Q2 diaphragm pump
n2 = gate_network("GPIO26", "220R", "R14", "R17", 1110, 1110)
box("Q2", "Q2  IRLB8721", "N-channel, low side", MOS_X, 328, MOS_W, 80, "#C9A46A",
    left=[("G", 356), ("S", 384)], right=[("D", 362)], tcol="#2B2212")
wire("ctl", [n2, PIN[("Q2", "G")]])
wire("gnd", [PIN[("Q2", "S")], (L_RGND, 384), (L_RGND, GND_Y)])

# Q4 NPN drives the gate of the high-side P-MOSFET Q3
n3 = gate_network("GPIO27", "1k", "R15", "R18", 1096, 1096)
box("Q4", "Q4  2N2222", "NPN level shifter", MOS_X, 480, 110, 70, "#C9A46A",
    left=[("B", 512)], right=[("C", 498), ("E", 528)], tcol="#2B2212")
wire("ctl", [n3, PIN[("Q4", "B")]])
wire("gnd", [PIN[("Q4", "E")], (1340, 528), (1340, 600), (L_RGND, 600), (L_RGND, GND_Y)])
dot(L_RGND, GND_Y, "gnd")

box("Q3", "Q3  IRF9540N", "P-channel, high side", 1430, 470, 165, 90, "#C9A46A",
    left=[("G", 505)], right=[("S  5 V in", 492), ("D  MQ_5V", 536)], tcol="#2B2212")
wire("ctl", [PIN[("Q4", "C")], (1390, 498), (1390, 505), PIN[("Q3", "G")]])
res_v(1390, 408, "10k", "R19")
wire("ctl", [(1390, 466), (1390, 505)])
dot(1390, 505, "ctl")
wire("v5", [(1390, 408), (1390, 300), (1640, 300), (1640, RAIL["+5 V regulated"][0])])
dot(1640, RAIL["+5 V regulated"][0], "v5")
text(1400, 294, "R19 holds the P-MOSFET OFF until Q4 pulls its gate down,", 9, "start", MUTED)
text(1400, 283, "so the sensor rail is dead while the ESP32 boots.", 9, "start", MUTED)
to_rail("v5", PIN[("Q3", "S  5 V in")], "+5 V regulated", 1664)
to_rail("mq5", PIN[("Q3", "D  MQ_5V")], "MQ_5V (switched)", 1688)

# ----------------------------------------------------------------- loads
section(1700, 132, "LOADS")

box("HT1", "Nichrome mesh", "thermal desorption, ≈180 °C", 1760, 176, 190, 80,
    "#ECEFF3", left=[("to Q1 D", 210)], right=[("to BAT+", 234)], tcol=INK)
wire("bat", [PIN[("Q1", "D")], PIN[("HT1", "to Q1 D")]])
to_rail("bat", PIN[("HT1", "to BAT+")], "BAT+ 7.4 V", 2010)

box("M1", "370 diaphragm pump", "draws vapour across the array", 1760, 328, 190, 80,
    "#ECEFF3", left=[("− to Q2 D", 362)], right=[("+ to BAT+", 386)], tcol=INK)
wire("bat", [PIN[("Q2", "D")], PIN[("M1", "− to Q2 D")]])
to_rail("bat", PIN[("M1", "+ to BAT+")], "BAT+ 7.4 V", 2050)

# D1 flyback across the pump: cathode to +, anode to -
diode_h(1800, 296, "D1", "1N5819")
wire("bat", [(1800, 296), (1730, 296), (1730, 362)])
dot(1730, 362, "bat")
wire("bat", [(1856, 296), (2050, 296)])
dot(2050, 296, "bat")
text(1700, 440, "D1 shorts the collapsing-field spike the moment the pump switches off.",
     9.5, "start", "#C62F28")
text(1700, 452, "Leave it out and that spike resets or destroys the ESP32.",
     9.5, "start", "#C62F28")

# ----------------------------------------------------------------- MAX6675
section(935, 606, "HEATER TEMPERATURE READBACK")
box("TC1", "MAX6675", "K-type amplifier, SPI", 980, 648, 175, 116, "#0F6F7A",
    left=[("SCK", 680), ("SO", 706), ("CS", 732)],
    right=[("VCC", 680), ("GND", 706), ("T+", 732), ("T−", 754)])
for pin, lbl, lane in [("GPIO14", "SCK", 930), ("GPIO19", "SO", 945), ("GPIO5", "CS", 960)]:
    ex, ey = PIN[("ESP", pin)]
    tx, ty = PIN[("TC1", lbl)]
    wire("spi", [(ex, ey), (lane, ey), (lane, ty), (tx, ty)])
to_rail("v33", PIN[("TC1", "VCC")], "3V3", L_R33)
wire("gnd", [PIN[("TC1", "GND")], (L_RGND, 706), (L_RGND, GND_Y)])

box("TC2", "K-type probe", "bonded to the mesh", 1200, 706, 175, 72, "#ECEFF3",
    left=[("+", 732), ("−", 754)], tcol=INK)
wire("ctl", [PIN[("TC1", "T+")], PIN[("TC2", "+")]], w=2.2)
wire("ctl", [PIN[("TC1", "T−")], PIN[("TC2", "−")]], w=2.2)
text(1200, 800, "Proves the mesh really reaches 180 °C. Without it", 9, "start", MUTED)
text(1200, 812, "the duty-cycle figure in the deck is just an assertion.", 9, "start", MUTED)

# ----------------------------------------------------------------- GPS
section(935, 852, "LOCATION  +  TIME")
box("GPS", "NEO-6M GPS", "UART1 at 9600 baud", 980, 894, 175, 102, "#1F5FBF",
    left=[("TX", 920), ("RX", 946)], right=[("VCC", 920), ("GND", 946)])
ex, ey = PIN[("ESP", "GPIO16")]
wire("uart", [(ex, ey), (930, ey), (930, 920), PIN[("GPS", "TX")]])
ex, ey = PIN[("ESP", "GPIO17")]
wire("uart", [(ex, ey), (945, ey), (945, 946), PIN[("GPS", "RX")]])
to_rail("v5", PIN[("GPS", "VCC")], "+5 V regulated", L_R5)
wire("gnd", [PIN[("GPS", "GND")], (L_RGND, 946), (L_RGND, GND_Y)])
text(980, 1026, "Stamps every alert with position and UTC time — the", 9, "start", MUTED)
text(980, 1038, "traceable evidence trail an NDPS case needs.", 9, "start", MUTED)

# ----------------------------------------------------------------- officer UI
section(935, 1070, "OFFICER INTERFACE")
box("BZ1", "Buzzer", "piezo, 85 dB", 980, 1106, 160, 64, "#ECEFF3",
    left=[("+", 1138)], right=[("−", 1138)], tcol=INK)
ex, ey = PIN[("ESP", "GPIO13")]
wire("ui", [(ex, ey), (930, ey), (930, 1138), PIN[("BZ1", "+")]])
wire("gnd", [PIN[("BZ1", "−")], (L_RGND, 1138), (L_RGND, GND_Y)])

res_h(1006, 1216, "220R", "R20")
ex, ey = PIN[("ESP", "GPIO18")]
wire("ui", [(ex, ey), (955, ey), (955, 1216), (1006, 1216)])
svg.append('<circle cx="1102" cy="1216" r="13" fill="#FF5252" stroke="#1B2430" stroke-width="2"/>')
wire("ui", [(1072, 1216), (1089, 1216)])
text(1122, 1220, "LED1  status", 9.5, "start", INK)
wire("gnd", [(1115, 1216), (L_RGND, 1216), (L_RGND, GND_Y)])

box("SW2", "SCAN button", "to GND, internal pull-up", 980, 1256, 160, 64, "#ECEFF3",
    left=[("1", 1288)], right=[("2", 1288)], tcol=INK)
ex, ey = PIN[("ESP", "GPIO23")]
wire("ui", [(ex, ey), (960, ey), (960, 1288), PIN[("SW2", "1")]])
wire("gnd", [PIN[("SW2", "2")], (L_RGND, 1288), (L_RGND, GND_Y)])

# ----------------------------------------------------------------- power chain
section(2180, 132, "PROTECTED POWER CHAIN")

box("BT1", "2S LiPo  7.4 V", "2200–5000 mAh", 2180, 176, 180, 80, "#244A8F",
    right=[("+", 204), ("−", 232)])
fx = fuse_h(2392, 204, "F1  10 A")
wire("bat", [PIN[("BT1", "+")], (2392, 204)])
sx = switch_h(2470, 204, "SW1  master")
wire("bat", [(fx, 204), (2470, 204)])
diode_v(2560, 232, "D2", "SS54 reverse-polarity", length=56)
wire("bat", [(sx, 204), (2560, 204), (2560, 232)])
wire("bat", [(2560, 288), (2560, 330)])

# D3 TVS clamps switching spikes across the battery rail
diode_v(2400, 352, "D3", "SMAJ15A TVS", length=56)
wire("bat", [(2560, 330), (2400, 330), (2400, 352)])
dot(2460, 330, "bat")
wire("gnd", [(2400, 408), (2400, 440), (2074, 440), (2074, GND_Y)])
wire("bat", [(2560, 330), (2560, RAIL["BAT+ 7.4 V"][0])])
dot(2560, RAIL["BAT+ 7.4 V"][0], "bat")
wire("gnd", [PIN[("BT1", "−")], (2376, 232), (2376, 290), (2074, 290), (2074, GND_Y)])
dot(2074, GND_Y, "gnd")

box("PS1", "LM2596 buck", "7.4 V → 5.0 V · sensors, ESP32, GPS", 2140, 520, 200, 96,
    "#1A4FA0", left=[("OUT+", 550), ("OUT−", 580)],
    right=[("IN+", 550), ("IN−", 580)])
wire("bat", [PIN[("PS1", "IN+")], (2460, 550), (2460, 330)])
wire("gnd", [PIN[("PS1", "IN−")], (2490, 580), (2490, GND_Y)])
dot(2490, GND_Y, "gnd")
to_rail("v5", PIN[("PS1", "OUT+")], "+5 V regulated", 2100)
to_rail("gnd", PIN[("PS1", "OUT−")], "GND (common)", 2074)

box("PS2", "5 A synchronous buck", "7.4 V → 5.1 V · Raspberry Pi only",
    2140, 680, 200, 96, "#0F6F7A",
    left=[("OUT+", 710), ("OUT−", 740)], right=[("IN+", 710), ("IN−", 740)])
wire("bat", [PIN[("PS2", "IN+")], (2524, 710), (2524, 330)])
dot(2524, 330, "bat")
wire("gnd", [PIN[("PS2", "IN−")], (2554, 740), (2554, GND_Y)])
dot(2554, GND_Y, "gnd")
text(2140, 800, "A Pi 4 draws 1.2–2.0 A by itself. Share the LM2596 with it and it", 9,
     "start", "#C62F28")
text(2140, 812, "browns out mid-scan, which shows up as a corrupted SD card.", 9,
     "start", "#C62F28")

# ----------------------------------------------------------------- Pi & Wired UART
section(2180, 850, "AI BRAIN  +  DASHBOARD")
box("PI", "Raspberry Pi 4", "RF / SVM inference · Flask dashboard · SQLite evidence log",
    2180, 892, 260, 150, "#1C7A3C",
    left=[("pin 2  5V", 940), ("pin 6  GND", 976), ("pin 10  RX", 1000), ("pin 8  TX", 1024)])
wire("v5", [PIN[("PS2", "OUT+")], (2130, 710), (2130, 940), PIN[("PI", "pin 2  5V")]])
wire("gnd", [PIN[("PS2", "OUT−")], (2086, 740), (2086, 976), PIN[("PI", "pin 6  GND")]])

# Wired UART Fallback connections (ESP32 GPIO33 TX -> Pi pin 10 RX, ESP32 GPIO32 RX <- Pi pin 8 TX)
wire("uart", [PIN[("ESP", "GPIO33 TX")], (965, 1200), (965, 1400), (2030, 1400), (2030, 1000), PIN[("PI", "pin 10  RX")]])
wire("uart", [PIN[("ESP", "GPIO32 RX")], (975, 1226), (975, 1416), (2020, 1416), (2020, 1024), PIN[("PI", "pin 8  TX")]])
text(980, 1412, "WIRED UART FALLBACK — GPIO33/32 active when Wi-Fi signal is lost. Cross pair: ESP32 TX -> Pi RX, ESP32 RX <- Pi TX. Both see 3.3 V logic levels.",
     9, "start", NET["uart"], True)

# Wi-Fi link
wire("wifi", [(920, 140), (920, 112), (2630, 112), (2630, 950), (2452, 950)], dash=True)
svg.append('<polygon points="2440,950 2452,944 2452,956" fill="#1F6FEB"/>')
text(1500, 105, "Wi-Fi · MQTT to neds/neds-01/sensors, one reading per second — the primary link",
     11.5, "middle", NET["wifi"], True)
text(2180, 1060, "Wi-Fi is the primary link. If signal drops, GPIO33/32 picks up via wired UART.", 9,
     "start", MUTED)

# ----------------------------------------------------------------- rails
for name, (y, net) in RAIL.items():
    svg.append(f'<line x1="{RAIL_X0}" y1="{y}" x2="{RAIL_X1}" y2="{y}" stroke="{NET[net]}" '
               f'stroke-width="7" stroke-linecap="round"/>')
    text(RAIL_X0 + 6, y - 11, name, 12, "start", NET[net], True)

# bulk and decoupling capacitors
cap_v(1760, RAIL["+5 V regulated"][0], GND_Y, "470µF", "C1", polarized=True)
dot(1760, RAIL["+5 V regulated"][0], "v5")
dot(1760, GND_Y, "gnd")
cap_v(1900, RAIL["BAT+ 7.4 V"][0], GND_Y, "1000µF", "C2", polarized=True)
dot(1900, RAIL["BAT+ 7.4 V"][0], "bat")
dot(1900, GND_Y, "gnd")
text(1920, 1404, "C2 sits beside the heater and absorbs the 180 °C burst current.",
     8.5, "start", MUTED)
for cx, ref in [(1180, "C3"), (1260, "C4"), (1340, "C5")]:
    cap_v(cx, RAIL["3V3"][0], GND_Y, "100nF", ref)
    dot(cx, RAIL["3V3"][0], "v33")
    dot(cx, GND_Y, "gnd")
text(1180, 1556, "C3–C5: 100 nF decoupling — one at each ADS1115, BME688 and the MAX6675.",
     8.5, "start", MUTED)

# ----------------------------------------------------------------- legend
LY = 1672
items = [("Battery 7.4 V", "bat"), ("+5 V", "v5"), ("MQ_5V switched", "mq5"),
         ("3.3 V", "v33"), ("GND", "gnd"), ("Analog", "ana"), ("I²C SDA", "i2c"),
         ("I²C SCL", "scl"), ("Gate drive", "ctl"), ("SPI", "spi"),
         ("UART", "uart"), ("Buzzer / LED", "ui"), ("Wi-Fi", "wifi")]
for i, (lbl, net) in enumerate(items):
    x = 70 + (i % 7) * 168
    y = LY + (i // 7) * 24
    svg.append(f'<line x1="{x}" y1="{y}" x2="{x+26}" y2="{y}" stroke="{NET[net]}" '
               f'stroke-width="5" stroke-linecap="round"/>')
    text(x + 33, y + 4, lbl, 10.5, "start", MUTED)

svg.append(f'<rect x="1300" y="{LY-26}" width="1300" height="78" rx="10" fill="#FFF8E6" '
           f'stroke="#E0B94A"/>')
text(1316, LY - 6, "What changed in rev C, and why nothing burns", 11.5, "start", "#7A5A00", True)
for i, n in enumerate([
    "Two ADS1115 converters replace the ESP32's own ADC: eight 16-bit channels instead of six 12-bit ones, so the MiCS-6814's CO and NH₃ elements are finally read.",
    "GPIO35 measures Vbat (33k/10k). Wired UART fallback to Pi (pin 8/10); GPIO33/32 picks up if Wi-Fi drops. GPIO34, 36 and 39 stay spare.",
    "Discharge protection: R1–R24 halve every sensor output. R13–R15 limit gate current. R16–R19 hold all three MOSFETs OFF at boot. D1 flyback. D2 reverse-polarity. D3 TVS. F1 fuse. SW1 master cut-off.",
]):
    text(1316, LY + 12 + i * 13, n, 9, "start", "#7A5A00")


# ----------------------------------------------------------------- self-check
def hits(a, b, bx, by, bw, bh):
    return (min(a[0], b[0]) < bx + bw - 6 and max(a[0], b[0]) > bx + 6
            and min(a[1], b[1]) < by + bh - 6 and max(a[1], b[1]) > by + 6)


problems = []
for net, a, b in SEGS:
    for ref, bx, by, bw, bh in BOXES:
        if hits(a, b, bx, by, bw, bh):
            problems.append(f"{net} {a}->{b} crosses {ref}")
for i in range(len(SEGS)):
    for j in range(i + 1, len(SEGS)):
        n1, a1, b1 = SEGS[i]
        n2, a2, b2 = SEGS[j]
        if n1 == n2:
            continue
        if a1[0] == b1[0] and a2[0] == b2[0] and abs(a1[0] - a2[0]) < 4:
            l1, h1 = sorted([a1[1], b1[1]])
            l2, h2 = sorted([a2[1], b2[1]])
            if min(h1, h2) - max(l1, l2) > 14:
                problems.append(f"overlap {n1}/{n2} vertical near x={a1[0]}")
        if a1[1] == b1[1] and a2[1] == b2[1] and abs(a1[1] - a2[1]) < 4:
            l1, h1 = sorted([a1[0], b1[0]])
            l2, h2 = sorted([a2[0], b2[0]])
            if min(h1, h2) - max(l1, l2) > 14:
                problems.append(f"overlap {n1}/{n2} horizontal near y={a1[1]}")

print(f"boxes {len(BOXES)}   segments {len(SEGS)}   named pins {len(PIN)}   "
      f"problems {len(problems)}")
for p in sorted(set(problems)):
    print("  !", p)

svg_str = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="Segoe UI, Noto Sans, system-ui, sans-serif">' + "".join(svg) + "</svg>"

open(os.path.join(OUT, "neds-v3-schematic.svg"), "w", encoding="utf-8").write(svg_str)
open(os.path.join(OUT, "neds-full-schematic.svg"), "w", encoding="utf-8").write(svg_str)
print("Successfully generated neds-v3-schematic.svg and neds-full-schematic.svg!")
