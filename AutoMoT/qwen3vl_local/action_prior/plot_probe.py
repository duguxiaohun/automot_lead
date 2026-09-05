"""把 probe case 中的预测/GT 轨迹画成可独立分享的 PNG。"""

import argparse
import json
from pathlib import Path


def render(directory):
    """显示 ego (x forward, y left)，坐标轴单位米。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for path in sorted(Path(directory).glob("rank*_case*.json")):
        row = json.loads(path.read_text())
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        for ax, pred, gt, title in zip(
            axes,
            ("pred_route", "pred_waypoints"),
            ("gt_route", "gt_waypoints"),
            ("Route", "Waypoints (2 s)"),
        ):
            for key, color, label in (
                (gt, "tab:green", "GT"),
                (pred, "tab:red", "Prediction"),
            ):
                points = row[key][0]
                ax.plot(
                    [-p[1] for p in points],
                    [p[0] for p in points],
                    "-o",
                    color=color,
                    label=label,
                )
            ax.scatter([0], [0], marker="^", c="black", label="Ego")
            ax.set(xlabel="Right (-ego y), m", ylabel="Forward (ego x), m", title=title)
            ax.set_aspect("equal", adjustable="datalim")
            ax.grid(True, alpha=0.3)
            ax.legend()
        sample = row["sample"]
        fig.suptitle(
            f"{sample['scenario']}/{sample['run_id']} @ {sample['anchor']}\ninvalid: {row['invalid']}",
            fontsize=8,
        )
        fig.tight_layout()
        fig.savefig(path.with_suffix(".png"), dpi=130)
        plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("cases_dir")
    render(p.parse_args().cases_dir)
