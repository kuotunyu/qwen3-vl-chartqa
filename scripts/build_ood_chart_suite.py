"""Generate a deterministic synthetic out-of-domain chart evaluation suite."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "assets" / "ood_charts"
WIDTH, HEIGHT = 1000, 700
INK = "#15202b"
MUTED = "#667085"
GRID = "#d7dce2"
PAPER = "#fffdf7"
COLORS = ["#ff5a36", "#58718f", "#c6f04d", "#f4c64d", "#8c6bc7"]


def font(size: int, bold: bool = False):
    candidates = [
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
        Path(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
            if bold
            else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        ),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default(size=size)


F12, F16, F20, F26, F34 = (font(x) for x in (12, 16, 20, 26, 34))
B16, B20, B28 = (font(x, True) for x in (16, 20, 28))


def canvas(title: str, subtitle: str = "Synthetic OOD validation"):
    image = Image.new("RGB", (WIDTH, HEIGHT), PAPER)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, WIDTH, 92), fill=INK)
    draw.text((46, 24), title, font=B28, fill="white")
    draw.text((48, 61), subtitle, font=F16, fill="#cad2dc")
    return image, draw


def axes(draw: ImageDraw.ImageDraw, y_max: int, step: int):
    left, top, right, bottom = 110, 135, 930, 610
    draw.line((left, top, left, bottom), fill=INK, width=3)
    draw.line((left, bottom, right, bottom), fill=INK, width=3)
    for value in range(0, y_max + 1, step):
        y = bottom - (bottom - top) * value / y_max
        draw.line((left, y, right, y), fill=GRID, width=1)
        draw.text((54, y - 10), str(value), font=F16, fill=MUTED)
    return left, top, right, bottom


def save(image: Image.Image, case_id: str) -> str:
    filename = f"{case_id}.png"
    image.save(OUTPUT / filename, optimize=True)
    return filename


def vertical_bar():
    image, draw = canvas("Regional revenue · FY2025", "USD millions")
    values = {"North": 42, "East": 57, "South": 49, "West": 68}
    left, top, right, bottom = axes(draw, 80, 20)
    width, gap = 120, 70
    x = left + 75
    for index, (label, value) in enumerate(values.items()):
        y = bottom - (bottom - top) * value / 80
        draw.rectangle(
            (x, y, x + width, bottom), fill=COLORS[index], outline=INK, width=2
        )
        draw.text((x + 42, y - 30), str(value), font=B20, fill=INK)
        draw.text((x + 24, bottom + 16), label, font=F20, fill=INK)
        x += width + gap
    return image


def grouped_bar():
    image, draw = canvas(
        "Quarterly product sales", "Units sold · Product A vs Product B"
    )
    labels = ["Q1", "Q2", "Q3", "Q4"]
    product_a = [36, 51, 63, 78]
    product_b = [44, 58, 72, 69]
    left, top, right, bottom = axes(draw, 90, 15)
    for idx, label in enumerate(labels):
        group_x = left + 65 + idx * 190
        for offset, value, color in [
            (0, product_a[idx], COLORS[0]),
            (58, product_b[idx], COLORS[1]),
        ]:
            y = bottom - (bottom - top) * value / 90
            draw.rectangle(
                (group_x + offset, y, group_x + offset + 48, bottom),
                fill=color,
                outline=INK,
                width=2,
            )
            draw.text((group_x + offset + 8, y - 25), str(value), font=F16, fill=INK)
        draw.text((group_x + 38, bottom + 16), label, font=B20, fill=INK)
    draw.rectangle((720, 105, 748, 125), fill=COLORS[0])
    draw.text((758, 103), "Product A", font=F16, fill=INK)
    draw.rectangle((840, 105, 868, 125), fill=COLORS[1])
    draw.text((878, 103), "Product B", font=F16, fill=INK)
    return image


def line_chart():
    image, draw = canvas("Monthly active users", "Thousands of users")
    years = ["2020", "2021", "2022", "2023", "2024", "2025"]
    values = [12, 18, 27, 31, 45, 52]
    left, top, right, bottom = axes(draw, 60, 10)
    points = []
    for idx, value in enumerate(values):
        x = left + 45 + idx * 145
        y = bottom - (bottom - top) * value / 60
        points.append((x, y))
        draw.text((x - 22, bottom + 16), years[idx], font=F16, fill=INK)
    draw.line(points, fill=COLORS[0], width=6, joint="curve")
    for (x, y), value in zip(points, values, strict=True):
        draw.ellipse((x - 8, y - 8, x + 8, y + 8), fill=PAPER, outline=INK, width=3)
        draw.text((x - 12, y - 34), str(value), font=B16, fill=INK)
    return image


def stacked_bar():
    image, draw = canvas("Support tickets by month", "Resolved and open tickets")
    labels = ["March", "April", "May", "June"]
    resolved = [55, 62, 70, 76]
    opened = [25, 28, 30, 24]
    left, top, right, bottom = axes(draw, 110, 20)
    for idx, label in enumerate(labels):
        x = left + 80 + idx * 190
        open_y = bottom - (bottom - top) * opened[idx] / 110
        total_y = bottom - (bottom - top) * (opened[idx] + resolved[idx]) / 110
        draw.rectangle(
            (x, open_y, x + 90, bottom), fill=COLORS[1], outline=INK, width=2
        )
        draw.rectangle(
            (x, total_y, x + 90, open_y), fill=COLORS[2], outline=INK, width=2
        )
        draw.text(
            (x + 24, total_y - 28), str(opened[idx] + resolved[idx]), font=B20, fill=INK
        )
        draw.text((x + 10, bottom + 16), label, font=F16, fill=INK)
    draw.rectangle((700, 105, 728, 125), fill=COLORS[2])
    draw.text((738, 103), "Resolved", font=F16, fill=INK)
    draw.rectangle((815, 105, 843, 125), fill=COLORS[1])
    draw.text((853, 103), "Open", font=F16, fill=INK)
    return image


def scatter_chart():
    image, draw = canvas("Campaign efficiency", "Spend index vs conversion rate (%)")
    left, top, right, bottom = axes(draw, 8, 1)
    points = {"A": (2, 3), "B": (4, 5), "C": (6, 4), "D": (7, 7), "E": (9, 6)}
    for label, (spend, conversion) in points.items():
        x = left + (right - left) * spend / 10
        y = bottom - (bottom - top) * conversion / 8
        draw.ellipse(
            (x - 13, y - 13, x + 13, y + 13), fill=COLORS[0], outline=INK, width=3
        )
        draw.text((x + 16, y - 15), f"Campaign {label}", font=B16, fill=INK)
    draw.text((430, 650), "Spend index", font=F20, fill=INK)
    return image


def donut_chart():
    image, draw = canvas("Traffic by device", "Share of sessions")
    shares = [("Mobile", 48), ("Desktop", 37), ("Tablet", 15)]
    box = (160, 145, 660, 645)
    start = -90
    for idx, (_label, value) in enumerate(shares):
        end = start + value * 3.6
        draw.pieslice(
            box, start=start, end=end, fill=COLORS[idx], outline=PAPER, width=4
        )
        start = end
    draw.ellipse((295, 280, 525, 510), fill=PAPER, outline=INK, width=2)
    draw.text((360, 355), "100%", font=B28, fill=INK)
    for idx, (label, value) in enumerate(shares):
        y = 250 + idx * 92
        draw.rectangle((725, y, 765, y + 28), fill=COLORS[idx], outline=INK, width=2)
        draw.text((782, y - 2), f"{label}  {value}%", font=B20, fill=INK)
    return image


def heatmap():
    image, draw = canvas("Team quality score", "Weekly scorecard · darker means higher")
    teams = ["Team A", "Team B", "Team C"]
    weeks = ["Week 1", "Week 2", "Week 3", "Week 4"]
    values = [[76, 82, 73, 86], [69, 75, 79, 83], [81, 87, 91, 88]]
    x0, y0, cell_w, cell_h = 230, 180, 165, 120
    for col, week in enumerate(weeks):
        draw.text((x0 + col * cell_w + 42, 135), week, font=B16, fill=INK)
    for row, team in enumerate(teams):
        draw.text((85, y0 + row * cell_h + 42), team, font=B20, fill=INK)
        for col, value in enumerate(values[row]):
            intensity = int(238 - (value - 65) * 4.2)
            color = (255, max(90, intensity), max(70, intensity - 35))
            box = (
                x0 + col * cell_w,
                y0 + row * cell_h,
                x0 + (col + 1) * cell_w,
                y0 + (row + 1) * cell_h,
            )
            draw.rectangle(box, fill=color, outline=INK, width=2)
            draw.text((box[0] + 64, box[1] + 40), str(value), font=B28, fill=INK)
    return image


def horizontal_bar():
    image, draw = canvas("Process cycle time", "Average minutes · lower is better")
    values = [("Alpha", 45), ("Beta", 34), ("Gamma", 29), ("Delta", 21)]
    x0, y0, scale = 210, 165, 14
    for idx, (label, value) in enumerate(values):
        y = y0 + idx * 105
        draw.text((85, y + 15), label, font=B20, fill=INK)
        draw.rectangle(
            (x0, y, x0 + value * scale, y + 58), fill=COLORS[idx], outline=INK, width=2
        )
        draw.text((x0 + value * scale + 16, y + 13), str(value), font=B20, fill=INK)
    draw.line((x0, 140, x0, 600), fill=INK, width=3)
    return image


def annotated_line():
    image, draw = canvas(
        "Sensor calibration drift", "Fine annotation / OCR stress case"
    )
    labels = ["2021", "2022", "2023", "2024"]
    values = [102, 107, 113, 119]
    left, top, right, bottom = axes(draw, 120, 20)
    points = []
    for idx, value in enumerate(values):
        x = left + 80 + idx * 230
        y = bottom - (bottom - top) * value / 120
        points.append((x, y))
        draw.text((x - 24, bottom + 16), labels[idx], font=F16, fill=INK)
    draw.line(points, fill=COLORS[1], width=5)
    for (x, y), value in zip(points, values, strict=True):
        draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill=COLORS[2], outline=INK, width=2)
        draw.rounded_rectangle(
            (x - 24, y - 45, x + 34, y - 17), radius=6, fill="#ffffff", outline=MUTED
        )
        draw.text((x - 12, y - 43), str(value), font=F12, fill=INK)
    return image


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    cases = [
        (
            "ood_01_vertical_bar",
            "bar",
            vertical_bar(),
            "Which region has the highest revenue?",
            "West",
            "comparison",
        ),
        (
            "ood_02_grouped_bar",
            "grouped_bar",
            grouped_bar(),
            "How many units of Product B were sold in Q3?",
            "72",
            "lookup",
        ),
        (
            "ood_03_line",
            "line",
            line_chart(),
            "How many thousand monthly active users were there in 2024?",
            "45",
            "lookup",
        ),
        (
            "ood_04_stacked",
            "stacked_bar",
            stacked_bar(),
            "What was the total number of support tickets in May?",
            "100",
            "arithmetic",
        ),
        (
            "ood_05_scatter",
            "scatter",
            scatter_chart(),
            "Which campaign has the highest conversion rate?",
            "Campaign D",
            "comparison",
        ),
        (
            "ood_06_donut",
            "donut",
            donut_chart(),
            "What percentage of sessions came from Mobile?",
            "48%",
            "lookup",
        ),
        (
            "ood_07_heatmap",
            "heatmap",
            heatmap(),
            "Which team had the highest score in Week 3?",
            "Team C",
            "comparison",
        ),
        (
            "ood_08_horizontal",
            "horizontal_bar",
            horizontal_bar(),
            "What is the difference in cycle time between Alpha and Delta?",
            "24",
            "arithmetic",
        ),
        (
            "ood_09_ocr",
            "annotated_line",
            annotated_line(),
            "What was the calibration reading in 2024?",
            "119",
            "ocr-heavy",
        ),
    ]
    manifest = []
    for case_id, chart_type, image, question, expected, skill in cases:
        manifest.append(
            {
                "id": case_id,
                "chart_type": chart_type,
                "image": save(image, case_id),
                "question": question,
                "expected": expected,
                "skill": skill,
                "source": "synthetic-ood-v1",
            }
        )
    sheet = Image.new("RGB", (960, 720), "#e7e0d0")
    for index, item in enumerate(manifest):
        preview = Image.open(OUTPUT / item["image"]).convert("RGB")
        preview.thumbnail((300, 210))
        x = 15 + (index % 3) * 315
        y = 15 + (index // 3) * 235
        sheet.paste(preview, (x, y))
        ImageDraw.Draw(sheet).text((x, y + 212), item["id"], font=F12, fill=INK)
    sheet.save(OUTPUT / "contact_sheet.png", optimize=True)
    (OUTPUT / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"OOD suite generated: {len(manifest)} cases -> {OUTPUT}")


if __name__ == "__main__":
    main()
