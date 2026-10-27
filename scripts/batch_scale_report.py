"""Reproduce the batch-scaling data table and README figures.

With --score/--value-loss/--entropy, import the three W&B chart exports.
Without them, use the normalized data checked into docs/data.
Requires Pillow for PNG output; CSV parsing and all calculations use stdlib.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "docs" / "data" / "batch_experiment_metrics.csv"
FIGURE_DIR = ROOT / "docs" / "figures"
BATCHES = (256, 1024, 10240)
COLORS = {256: "#37649D", 1024: "#D98532", 10240: "#128077"}
METRICS = {
    "score": "mean_episode_score",
    "value_loss": "value_loss",
    "entropy": "entropy",
}
MILLION = 1_000_000
HALF_WINDOW = 10 * MILLION


def import_export(path: Path, metric: str) -> dict[int, dict[int, float]]:
    result = {batch: {} for batch in BATCHES}
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            samples = int(row["total_samples"])
            for batch in BATCHES:
                key = f"E001_batch{batch} - train_samples/{metric}"
                value = row[key]
                if value:
                    number = float(value)
                    if not math.isfinite(number) or samples in result[batch]:
                        raise ValueError(f"Invalid or duplicate {metric}: batch={batch}, samples={samples}")
                    result[batch][samples] = number
    return result


def write_normalized(score: Path, value_loss: Path, entropy: Path) -> None:
    sources = {
        "mean_episode_score": import_export(score, METRICS["score"]),
        "value_loss": import_export(value_loss, METRICS["value_loss"]),
        "entropy": import_export(entropy, METRICS["entropy"]),
    }
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    with DATA_PATH.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["batch_size", "total_samples", *sources])
        for batch in BATCHES:
            coordinates = set(sources["mean_episode_score"][batch])
            if any(set(data[batch]) != coordinates for data in sources.values()):
                raise ValueError(f"Metric sample coordinates differ for batch={batch}")
            for samples in sorted(coordinates):
                writer.writerow([
                    batch,
                    samples,
                    *(sources[name][batch][samples] for name in sources),
                ])


def read_normalized() -> dict[int, dict[str, list[tuple[int, float]]]]:
    series = {batch: {name: [] for name in METRICS.values()} for batch in BATCHES}
    with DATA_PATH.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            batch = int(row["batch_size"])
            samples = int(row["total_samples"])
            for name in METRICS.values():
                value = float(row[name])
                if not math.isfinite(value):
                    raise ValueError(f"Non-finite {name}: batch={batch}, samples={samples}")
                series[batch][name].append((samples, value))
    for batch in BATCHES:
        for name, points in series[batch].items():
            if not points or any(left[0] >= right[0] for left, right in zip(points, points[1:])):
                raise ValueError(f"Empty or unsorted {name} for batch={batch}")
    return series


class SampleCurve:
    """Piecewise-linear curve and its exact trapezoidal area over samples."""

    def __init__(self, points: list[tuple[int, float]]) -> None:
        self.x = [point[0] for point in points]
        self.y = [point[1] for point in points]
        self.area = [0.0]
        for i in range(1, len(points)):
            width = self.x[i] - self.x[i - 1]
            self.area.append(self.area[-1] + width * (self.y[i - 1] + self.y[i]) / 2)

    def area_at(self, x: float) -> float:
        if not self.x[0] <= x <= self.x[-1]:
            raise ValueError(f"Samples {x} outside observed range")
        index = bisect.bisect_right(self.x, x) - 1
        if index == len(self.x) - 1:
            return self.area[-1]
        offset = x - self.x[index]
        slope = (self.y[index + 1] - self.y[index]) / (self.x[index + 1] - self.x[index])
        return self.area[index] + self.y[index] * offset + slope * offset * offset / 2

    def mean(self, start: float, end: float) -> float:
        return (self.area_at(end) - self.area_at(start)) / (end - start)

    def smoothed(self, center: float) -> float:
        start = max(self.x[0], center - HALF_WINDOW)
        end = min(self.x[-1], center + HALF_WINDOW)
        return self.mean(start, end)

    def plot_points(self, first: int = 0) -> list[tuple[float, float]]:
        start = max(first, self.x[0])
        grid = list(range(math.ceil(start / MILLION) * MILLION, self.x[-1], MILLION))
        return [
            (start, self.smoothed(start)),
            *((x, self.smoothed(x)) for x in grid),
            (self.x[-1], self.smoothed(self.x[-1])),
        ]

    def sustained_peak(self) -> tuple[int, float]:
        start = math.ceil((self.x[0] + HALF_WINDOW) / MILLION) * MILLION
        stop = self.x[-1] - HALF_WINDOW
        return max(
            ((x, self.smoothed(x)) for x in range(start, math.floor(stop / MILLION) * MILLION + 1, MILLION)),
            key=lambda item: item[1],
        )


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    names = ["segoeuib.ttf" if bold else "segoeui.ttf", "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"]
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_chart(
    curves: dict[int, SampleCurve],
    title: str,
    y_label: str,
    y_max: float,
    y_ticks: list[float],
    output: Path,
    x_min: int = 0,
    y_format: str = "{:.0f}",
    subtitle: str = "Centered 20M-transition moving average",
) -> None:
    width, height = 1600, 840
    left, top, right, bottom = 155, 165, 85, 140
    plot_width, plot_height = width - left - right, height - top - bottom
    x_max = 660 * MILLION
    image = Image.new("RGB", (width, height), "#FFFFFF")
    draw = ImageDraw.Draw(image)
    title_font, text_font, small_font = font(38, True), font(26), font(22)
    draw.text((left, 35), title, font=title_font, fill="#24313C")
    draw.text((left, 92), subtitle, font=text_font, fill="#687784")

    def xy(x: float, y: float) -> tuple[float, float]:
        return (
            left + (x - x_min) / (x_max - x_min) * plot_width,
            top + (y_max - y) / y_max * plot_height,
        )

    for tick in y_ticks:
        y = xy(x_min, tick)[1]
        draw.line((left, y, width - right, y), fill="#E6EBEF", width=2)
        label = y_format.format(tick)
        box = draw.textbbox((0, 0), label, font=small_font)
        draw.text((left - 24 - (box[2] - box[0]), y - 13), label, font=small_font, fill="#52616E")
    x_ticks = [x_min // MILLION, 100, 200, 300, 400, 500, 600] if x_min else [0, 100, 200, 300, 400, 500, 600]
    for tick in x_ticks:
        x = xy(tick * MILLION, 0)[0]
        draw.line((x, top, x, height - bottom), fill="#F0F3F5", width=2)
        label = str(tick)
        box = draw.textbbox((0, 0), label, font=small_font)
        draw.text((x - (box[2] - box[0]) / 2, height - bottom + 17), label, font=small_font, fill="#52616E")
    draw.line((left, top, left, height - bottom), fill="#8C9AA4", width=2)
    draw.line((left, height - bottom, width - right, height - bottom), fill="#8C9AA4", width=2)

    for batch in BATCHES:
        points = [(x, y) for x, y in curves[batch].plot_points(x_min) if x >= x_min]
        pixels = [xy(x, min(y_max, max(0, y))) for x, y in points]
        draw.line(pixels, fill=COLORS[batch], width=5, joint="curve")
        if pixels:
            end_x, end_y = pixels[-1]
            draw.ellipse((end_x - 5, end_y - 5, end_x + 5, end_y + 5), fill=COLORS[batch])

    draw.text((left, height - 74), "Environment transitions (millions)", font=text_font, fill="#33424E")
    draw.text((left, top - 38), y_label, font=small_font, fill="#33424E")
    for index, batch in enumerate(BATCHES):
        x = 910 + 208 * index
        draw.line((x, 105, x + 38, 105), fill=COLORS[batch], width=6)
        draw.text((x + 50, 90), str(batch), font=small_font, fill="#33424E")
    draw.text(
        (left, height - 34),
        "Single training seed (42). Batch 256 export ends at 285.6M transitions.",
        font=small_font,
        fill="#687784",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, optimize=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score", type=Path)
    parser.add_argument("--value-loss", type=Path)
    parser.add_argument("--entropy", type=Path)
    args = parser.parse_args()
    exports = (args.score, args.value_loss, args.entropy)
    if any(exports):
        if not all(exports):
            parser.error("Supply all three W&B exports together")
        write_normalized(*exports)

    data = read_normalized()
    curves = {
        name: {batch: SampleCurve(data[batch][name]) for batch in BATCHES}
        for name in METRICS.values()
    }
    print("batch,points,last_samples,score_235-285M,score_600-650M,value_loss_235-285M,value_loss_600-650M,entropy_235-285M,entropy_600-650M,sustained_score_peak")
    for batch in BATCHES:
        score = curves["mean_episode_score"][batch]
        end = score.x[-1]
        def window(name: str, start: int, stop: int) -> str:
            curve = curves[name][batch]
            if stop * MILLION > curve.x[-1]:
                return ""
            return f"{curve.mean(start * MILLION, stop * MILLION):.3f}"
        print(",".join([
            str(batch), str(len(score.x)), str(end),
            window("mean_episode_score", 235, 285),
            window("mean_episode_score", 600, 650),
            window("value_loss", 235, 285),
            window("value_loss", 600, 650),
            window("entropy", 235, 285),
            window("entropy", 600, 650),
            f"{score.sustained_peak()[1]:.3f}",
        ]))

    draw_chart(
        curves["mean_episode_score"],
        "Training score by batch scale",
        "Mean score of completed episodes",
        110,
        [0, 20, 40, 60, 80, 100],
        FIGURE_DIR / "batch-score.png",
    )
    draw_chart(
        curves["value_loss"],
        "PPO value loss by batch scale",
        "Mean squared value error",
        0.24,
        [0, 0.04, 0.08, 0.12, 0.16, 0.20, 0.24],
        FIGURE_DIR / "batch-value-loss.png",
        y_format="{:.2f}",
    )
    draw_chart(
        curves["entropy"],
        "Policy entropy after initial exploration",
        "Entropy (nats)",
        0.16,
        [0, 0.04, 0.08, 0.12, 0.16],
        FIGURE_DIR / "batch-entropy.png",
        x_min=30 * MILLION,
        y_format="{:.2f}",
        subtitle="20M-transition average; shown from 30M samples",
    )


if __name__ == "__main__":
    main()
