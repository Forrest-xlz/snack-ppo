"""Recompute the README evidence and figures from unsampled W&B exports.

Run after export_wandb_runs.py. Evaluation is never smoothed. Training-only
curves use a sample-distance EMA; raw values remain in the exported data.
"""

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd


EVAL = "eval_samples/mean_score"
TRAIN = "train_samples/mean_episode_score"
COLORS = ["#2463A6", "#C59222", "#CE6230", "#778337", "#BD6285"]
GROUP_COLORS = {"A": COLORS[0], "B": COLORS[1], "C": COLORS[2], "D": COLORS[3], "E": COLORS[4], "F": "#555D68"}
EPOCH_COLORS = {1: "#555D68", 2: COLORS[0], 4: COLORS[1], 8: COLORS[2]}
STYLES = ["-", "--", "-.", ":", "-"]
DIAGNOSTICS = [
    "entropy", "value_loss", "policy_approx_kl", "clip_fraction",
    "raw_adv_mean", "raw_adv_std", "critic_explained_variance", "return_mean", "return_std",
    "reverse_action_rate", "policy_loss", "total_loss", "optimizer_steps_total",
    "wall_time_seconds", "samples_per_second", "window_samples",
    "teacher_kl_coef", "teacher_kl_loss", "teacher_kl_penalty",
]


def series(run, metric):
    if metric not in run["frame"]:
        return pd.DataFrame(columns=["total_samples", metric])
    return run["frame"][["total_samples", metric]].dropna().sort_values("total_samples").drop_duplicates("total_samples", keep="last")


def ema(x, y, half_life=25e6):
    if len(y) == 0:
        return np.array([])
    out = [float(y[0])]
    for index in range(1, len(y)):
        alpha = -np.expm1(-np.log(2) * (x[index] - x[index - 1]) / half_life)
        out.append(out[-1] + alpha * (float(y[index]) - out[-1]))
    return np.array(out)


def at_sample(run, target):
    frame = series(run, EVAL)
    if frame.empty:
        return None
    row = frame.iloc[np.abs(frame.total_samples.to_numpy() - target).argmin()]
    return float(row[EVAL]) if abs(row.total_samples - target) <= run["R"] + 1 else None


def score_line(ax, run, label, color, style="-", offset=0):
    frame = series(run, EVAL)
    ax.plot((frame.total_samples - offset) / 1e6, frame[EVAL], label=label,
            color=color, linestyle=style, marker="o", markersize=2.5, linewidth=1.7)


def diagnostic_line(ax, run, metric, label, color, style="-", half_life=25e6):
    frame = series(run, metric)
    if frame.empty:
        return
    x, y = frame.total_samples.to_numpy(), frame[metric].to_numpy()
    ax.plot(x / 1e6, y, color=color, alpha=0.12, linewidth=0.6)
    ax.plot(x / 1e6, ema(x, y, half_life), color=color, linestyle=style, label=label, linewidth=1.7)


def axes_style(ax, title, ylabel="Eval mean score", score=True, xmax=None):
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold", pad=10)
    ax.set_xlabel("Environment samples (M)")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color="#DFE4EA", linewidth=0.65)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["bottom", "left"]].set_color("#9AA5B3")
    ax.xaxis.set_major_locator(MaxNLocator(6))
    if score:
        ax.set_ylim(0, 145)
    if xmax is not None:
        ax.set_xlim(0, xmax + 20 if xmax == 1200 else xmax)
        if xmax == 1200:
            ax.set_xticks(np.arange(0, 1201, 200))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("analysis/wandb/raw"))
    parser.add_argument("--output", type=Path, default=Path("docs/analysis"))
    args = parser.parse_args()
    data_dir, figures = args.output / "data", args.output / "figures"
    data_dir.mkdir(parents=True, exist_ok=True)
    (figures / "runs").mkdir(parents=True, exist_ok=True)
    inventory = json.loads((args.input / "inventory.json").read_text(encoding="utf-8"))
    assert not inventory.get("failed_runs"), "Finish the export before analyzing"
    runs, checks, rows_out, configs, eval_frames, diag_frames = [], [], [], {}, [], []
    for record in inventory["runs"]:
        folder = args.input / record["id"]
        metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
        raw = folder / "history.jsonl.gz"
        assert hashlib.sha256(raw.read_bytes()).hexdigest() == metadata["history_sha256"]
        with gzip.open(raw, "rt", encoding="utf-8") as stream:
            rows = [json.loads(line) for line in stream]
        assert len(rows) == metadata["history_rows"]
        frame = pd.DataFrame(rows).sort_values("_step")
        assert not frame._step.duplicated().any(), metadata["id"]
        assert frame.total_samples.dropna().is_monotonic_increasing, metadata["id"]
        config = metadata["config"]
        env, ppo = config["environment"], config["ppo"]
        R = env["num_envs"] * env["rollout_steps"]
        pretrained = config["training"].get("pretrained_env_samples", 0)
        run = {**metadata, "frame": frame, "R": R, "pretrained": pretrained}
        run["study"] = metadata["name"].split()[0]
        if run["study"] in ("E001", "E004"):
            run["variant"] = metadata["name"].split("exp-")[-1]
        else:
            run["variant"] = metadata["name"].split()[-1]
        runs.append(run)
        ev = series(run, EVAL)
        assert not ev.empty, f"No eval data for {run['id']}"
        assert ev[EVAL].between(0, 141).all(), run["id"]
        best_row = ev.loc[ev[EVAL].idxmax()]
        last = ev.iloc[-1]
        final_samples = float(frame.total_samples.max())
        assert final_samples == float(metadata["summary"]["total_samples"]), run["id"]
        assert (frame.get("finetune_samples", frame.total_samples - pretrained).dropna()
                == (frame.total_samples - pretrained).loc[frame.get("finetune_samples", frame.total_samples - pretrained).dropna().index]).all()
        steps = frame["train_samples/optimizer_steps_total"].dropna()
        assert abs(float(steps.iloc[-1]) - (final_samples - pretrained) / ppo["batch_size"] * ppo["epochs"]) < 1e-6, run["id"]
        stop = series(run, "eval_samples/early_stop_triggered").iloc[-1, 1]
        if run["state"] != "finished":
            end_reason = run["state"] + " (partial)"
        elif stop == 1:
            end_reason = "eval early stop"
        elif final_samples - pretrained >= config["training"]["max_env_samples"]:
            end_reason = "sample cap"
        else:
            end_reason = "ended before configured cap"
        result = {
            "run_id": run["id"], "study": run["study"], "variant": run["variant"],
            "name": run["name"], "url": run["url"], "state": run["state"], "end_reason": end_reason,
            "model_version": config["model"].get("version", "mlp_v1"),
            "device": config["device"], "seed": config["seed"], "batch_size": ppo["batch_size"],
            "rollout_samples": R, "num_envs": env["num_envs"], "rollout_steps": env["rollout_steps"],
            "epochs": ppo["epochs"], "learning_rate": ppo["learning_rate"],
            "pretrained_samples": pretrained, "last_logged_samples": final_samples,
            "last_eval_samples": float(last.total_samples), "last_eval": float(last[EVAL]),
            "best_eval": float(best_row[EVAL]), "best_eval_samples": float(best_row.total_samples),
            "last4_eval_mean": float(ev.tail(4)[EVAL].mean()), "last4_eval_count": min(4, len(ev)),
            "peak_to_last_drop": float(best_row[EVAL] - last[EVAL]),
            "eval_count": len(ev), "history_rows": len(frame),
            "logged_wall_time_seconds": float(series(run, "train_samples/wall_time_seconds").iloc[-1,1]),
            "optimizer_steps_total": int(steps.iloc[-1]),
            "eval_at_300M": at_sample(run, 300e6), "eval_at_600M": at_sample(run, 600e6),
            "eval_at_1200M": at_sample(run, 1200e6),
            "eval_300M_after_switch": at_sample(run, pretrained + 300e6) if pretrained else None,
            "initial_eval": float(ev.iloc[0][EVAL]) if pretrained else None,
        }
        for threshold in [90, 100, 110, 120, 130]:
            reached = ev[ev[EVAL] >= threshold]
            result[f"first_eval_ge_{threshold}_samples"] = float(reached.iloc[0].total_samples) if len(reached) else None
        result["last_entropy"] = float(series(run, "train_samples/entropy").iloc[-1, 1])
        if (config.get("teacher_kl") or {}).get("enabled"):
            teacher = frame[["total_samples", EVAL, "eval_samples/teacher_mean_score", "eval_samples/teacher_updated"]].dropna()
            assert np.allclose(teacher["eval_samples/teacher_mean_score"], teacher[EVAL].cummax())
            assert np.array_equal(teacher["eval_samples/teacher_updated"].to_numpy(),
                                  (teacher[EVAL] > teacher[EVAL].cummax().shift(1).fillna(-np.inf)).astype(float))
            result["teacher_updates"] = int(teacher["eval_samples/teacher_updated"].sum())
        run["result"] = result
        rows_out.append(result)
        configs[run["id"]] = {key: run[key] for key in ["id", "name", "group", "state", "url", "created_at", "updated_at", "config"]}
        ef = frame[[col for col in frame if col.startswith("eval_samples/") or col in ["total_samples", "finetune_samples", "_step", "_timestamp"]]].dropna(subset=[EVAL]).copy()
        ef.insert(0, "run_id", run["id"])
        eval_frames.append(ef)
        df = frame[[col for col in ["total_samples", "finetune_samples", "_step", "_timestamp", TRAIN] + ["train_samples/" + k for k in DIAGNOSTICS] if col in frame]].copy()
        df.insert(0, "run_id", run["id"])
        diag_frames.append(df)
        checks.append({"run_id": run["id"], "history_sha256": metadata["history_sha256"],
                       "rows": len(frame), "max_step": int(frame._step.max()), "max_samples": final_samples,
                       "evals": len(ev), "checks_passed": ["archive hash", "row count", "unique steps", "monotone samples", "score bounds", "summary coverage", "sample clock", "optimizer-step accounting"]})

    result_frame = pd.DataFrame(rows_out).sort_values(["study", "variant"])
    result_frame.to_csv(data_dir / "run_results.csv", index=False)
    pd.concat(eval_frames, ignore_index=True).to_csv(data_dir / "eval_history.csv", index=False)
    pd.concat(diag_frames, ignore_index=True).to_csv(data_dir / "training_diagnostics.csv.gz", index=False, compression="gzip")
    (data_dir / "run_configs.json").write_text(json.dumps(configs, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (data_dir / "inventory.json").write_text(json.dumps(inventory, indent=2) + "\n", encoding="utf-8")
    (data_dir / "validation.json").write_text(json.dumps({"verified_at_utc": datetime.now(timezone.utc).isoformat(), "runs": checks}, indent=2) + "\n", encoding="utf-8")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.labelsize": 10,
                         "figure.facecolor": "white", "axes.facecolor": "white",
                         "savefig.facecolor": "white", "legend.frameon": False,
                         "svg.fonttype": "none"})

    def save(fig, filename, title, note="Raw held-out evaluation means; seed 42. Curves end at the last recorded evaluation."):
        fig.suptitle(title, x=0.065, ha="left", fontsize=14, fontweight="bold")
        fig.text(0.065, 0.012, note, ha="left", va="bottom", fontsize=8.2, color="#4C5665")
        fig.tight_layout(rect=[0.01, 0.055, 1, 0.95], h_pad=2.2, w_pad=2.1)
        fig.savefig(figures / filename, dpi=160)
        plt.close(fig)

    lookup = {(r["study"], r["variant"]): r for r in runs}
    baselines = {key: lookup[("E001", key)] for key in "ABCDEF"}
    fig, ax = plt.subplots(figsize=(10.8, 5.3))
    for key in "ABC":
        r = baselines[key]
        score_line(ax, r, f"{key}: b={r['config']['ppo']['batch_size']:,}, R={r['R']:,}", GROUP_COLORS[key], STYLES["ABC".index(key)])
    axes_style(ax, "Joint scaling at fixed R / b = 8", xmax=1200)
    ax.legend(loc="lower right")
    save(fig, "e001_joint_scale.png", "E001 - Mini-batch and rollout scaling")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for ax, pair, title in zip(axes, ["BD", "CE"], ["Fixed R=8,192: b=1,024 vs 256", "Fixed R=81,920: b=10,240 vs 1,024"]):
        for j, key in enumerate(pair):
            score_line(ax, baselines[key], key, GROUP_COLORS[key], STYLES[j])
        axes_style(ax, title, xmax=1200)
        ax.legend()
    save(fig, "e001_fixed_rollout.png", "E001 - Mini-batch comparisons at fixed rollout")

    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.7), sharey=True)
    for ax, pair, title in zip(axes, ["AD", "BE", "FC"], ["b=256", "b=1,024", "b=10,240"]):
        for j, key in enumerate(pair):
            score_line(ax, baselines[key], f"{key}: R={baselines[key]['R']:,}", GROUP_COLORS[key], STYLES[j])
        axes_style(ax, title, xmax=1200)
        ax.legend(fontsize=9, loc="lower right")
    save(fig, "e001_fixed_batch.png", "E001 - Rollout comparisons at fixed mini-batch")

    diagnostics = [("entropy", "Policy entropy"), ("value_loss", "Value loss"), ("policy_approx_kl", "PPO approximate KL"), ("clip_fraction", "Clip fraction"), ("raw_adv_std", "Raw advantage std"), ("critic_explained_variance", "Critic explained variance")]
    fig, axes = plt.subplots(2, 3, figsize=(14.2, 8.1))
    for ax, (metric, title) in zip(axes.flat, diagnostics):
        for j, key in enumerate("ABC"):
            diagnostic_line(ax, baselines[key], "train_samples/" + metric, key, GROUP_COLORS[key], STYLES[j])
        axes_style(ax, title, title, score=False, xmax=1200)
        if metric in ["entropy", "value_loss", "policy_approx_kl"]:
            ax.set_yscale("log")
        ax.legend(fontsize=8)
    save(fig, "e001_diagnostics.png", "E001 - Training diagnostics", "Raw + 25M-sample-half-life EMA. Entropy, value loss and approx KL use log axes. No seed uncertainty bands.")

    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.8), sharey=True)
    for ax, key in zip(axes, "ABC"):
        for j, epoch in enumerate([1, 2, 4, 8]):
            r = baselines[key] if epoch == 1 else lookup[("E002", f"{key}-ep{epoch}")]
            score_line(ax, r, f"epochs={epoch}", EPOCH_COLORS[epoch], STYLES[j])
        axes_style(ax, f"{key}: b={baselines[key]['config']['ppo']['batch_size']:,}", xmax=1200)
        ax.legend(fontsize=9)
    save(fig, "e002_epochs.png", "E002 - Data reuse across PPO epochs")

    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.8), sharey=True)
    for ax, key in zip(axes, "ABC"):
        for j, epoch in enumerate([1, 2, 4, 8]):
            r = baselines[key] if epoch == 1 else lookup[("E002", f"{key}-ep{epoch}")]
            score_line(ax, r, f"epochs={epoch}", EPOCH_COLORS[epoch], STYLES[j])
        axes_style(ax, f"{key}: b={baselines[key]['config']['ppo']['batch_size']:,}", xmax=200)
        ax.set_xticks([0, 50, 100, 150, 200])
        ax.legend(fontsize=9)
    save(fig, "e002_early_learning.png", "E002 - Initial learning speed", "Raw evaluation scores during the first 200M environment samples. The complete trajectories are shown separately.")

    collapse = lookup[("E002", "A-ep2")]
    fig, axes = plt.subplots(2, 3, figsize=(14.2, 8.1))
    collapse_metrics = [(EVAL, "Eval mean score"), (TRAIN, "Training rolling mean score"), ("train_samples/entropy", "Policy entropy"), ("train_samples/policy_approx_kl", "PPO approximate KL"), ("train_samples/raw_adv_std", "Raw advantage std"), ("train_samples/critic_explained_variance", "Critic explained variance")]
    for ax, (metric, title) in zip(axes.flat, collapse_metrics):
        for j, r in enumerate([baselines["A"], collapse]):
            if metric == EVAL:
                score_line(ax, r, ["A epoch 1", "A epoch 2"][j], ["#68717D", COLORS[2]][j], STYLES[j])
            else:
                diagnostic_line(ax, r, metric, ["A epoch 1", "A epoch 2"][j], ["#68717D", COLORS[2]][j], STYLES[j], half_life=5e6)
            axes_style(ax, title, title, score=metric in [EVAL, TRAIN])
        ax.set_xlim(600, 850)
        if metric not in [EVAL, TRAIN]:
            visible = np.concatenate([series(r, metric).loc[lambda f:f.total_samples.between(600e6,850e6),metric].to_numpy() for r in [baselines["A"],collapse]])
            low, high = float(np.min(visible)), float(np.max(visible))
            margin = max((high-low)*0.08,1e-6)
            ax.set_ylim(low-margin,high+margin)
        ax.legend(fontsize=8)
    save(fig, "e002_collapse.png", "E002 - Late regression in A at epoch 2", "Eval is unsmoothed. Training diagnostics: faint raw values and 5M-sample-half-life EMA. Association is not causal identification.")

    directions = [("A", "B"), ("A", "C"), ("B", "C"), ("C", "B")]
    for additional in [False, True]:
        fig, axes = plt.subplots(2, 2, figsize=(13.4, 9.0), sharey=True)
        for ax, (source, dest) in zip(axes.flat, directions):
            branches = sorted([r for r in runs if r["study"] == "E003" and r["variant"].startswith(source + "-") and r["variant"].endswith("to-" + dest)], key=lambda r:r["pretrained"])
            for j, r in enumerate(branches):
                score_line(ax, r, f"Switch at {r['pretrained']/1e6:.0f}M", COLORS[j], STYLES[j], r["pretrained"] if additional else 0)
            score_line(ax, baselines[dest], f"{dest} from scratch", "#424A56", "--")
            if not additional:
                score_line(ax, baselines[source], f"{source} from scratch", "#9BA4B1", ":")
            axes_style(ax, f"{source} -> {dest}", xmax=1200 if additional else 1800)
            if additional:
                ax.set_xlabel("Additional samples after switch (M)")
            ax.legend(fontsize=8, loc="lower right")
        save(fig, "e003_additional.png" if additional else "e003_total.png", "E003 - Switching the complete batch/rollout scale",
             "Post-switch clock excludes prefix cost; scratch reference is a learning-speed reference, not equal total compute." if additional else "Total samples include pretraining. Adam resets at each switch; same-scale reset controls were not run.")

    e004 = {r["variant"]:r for r in runs if r["study"] == "E004"}
    architecture_labels = {"A":"A CNN / 3 planes", "B-v7":"B small", "C":"C base (partial)", "D":"D small + teacher KL (partial)"}
    fig, ax = plt.subplots(figsize=(11.6, 5.8))
    score_line(ax, baselines["C"], "E001 C MLP / 3 planes", "#626B77", "--")
    for j, key in enumerate(["A", "B-v7", "C", "D"]):
        score_line(ax, e004[key], architecture_labels[key], COLORS[j], STYLES[j])
    axes_style(ax, "CNN, entity Transformer small/base and teacher KL", xmax=1200)
    ax.legend(fontsize=8.5, loc="lower right", ncol=2)
    save(fig, "e004_architectures.png", "E004 - Feature and model architecture experiments")

    small, teacher = e004["B-v7"], e004["D"]
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.0))
    for j, r in enumerate([small, teacher]):
        score_line(axes[0], r, ["Small, no teacher", "Small + teacher KL"][j], [COLORS[2], COLORS[4]][j], STYLES[j])
        diagnostic_line(axes[1], r, "train_samples/entropy", ["Small, no teacher", "Small + teacher KL"][j], [COLORS[2], COLORS[4]][j], STYLES[j])
    teacher_scores = series(teacher, "eval_samples/teacher_mean_score")
    axes[0].step(teacher_scores.total_samples / 1e6, teacher_scores.iloc[:, 1], where="post", color="#555D68", linestyle=":", label="Recorded teacher score", linewidth=1.4)
    axes_style(axes[0], "Student evaluation and teacher record", xmax=950)
    axes_style(axes[1], "Training policy entropy", "Entropy (nats)", score=False, xmax=950)
    axes[1].set_yscale("log")
    for ax in axes:
        ax.legend(fontsize=8)
    save(fig, "e004_teacher_comparison.png", "E004 - Small model with and without teacher KL", "First eval already differs before KL starts. D is incomplete and still regresses. Entropy axis is logarithmic.")

    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.8))
    diagnostic_line(axes[0], teacher, "train_samples/teacher_kl_loss", "Teacher || student", COLORS[4])
    axes_style(axes[0], "Teacher KL", "Unweighted forward KL", score=False, xmax=950)
    diagnostic_line(axes[1], teacher, "train_samples/teacher_kl_penalty", "KL penalty", COLORS[4])
    ep = series(teacher, "train_samples/entropy")
    axes[1].plot(ep.total_samples / 1e6, ema(ep.total_samples.to_numpy(), ep.iloc[:,1].to_numpy()*teacher["config"]["ppo"]["entropy_coef"]), color=COLORS[1], linestyle="--", label="Entropy bonus magnitude")
    axes_style(axes[1], "Loss-term magnitudes", "Coefficient times metric", score=False, xmax=950)
    axes[1].ticklabel_format(axis="y", style="sci", scilimits=(-3,3))
    for j, (r,label) in enumerate([(small,"No teacher"),(teacher,"Teacher KL")]):
        diagnostic_line(axes[2],r,"train_samples/critic_explained_variance",label,[COLORS[2],COLORS[4]][j],STYLES[j])
    axes_style(axes[2], "Critic explained variance", "Explained variance", score=False, xmax=950)
    axes[2].set_ylim(0.8,1.0)
    for ax in axes: ax.legend(fontsize=8)
    save(fig,"e004_teacher_diagnostics.png","E004 - Teacher regularization diagnostics", "25M-sample-half-life EMA; critic y-axis focuses on 0.8-1.0. Loss magnitudes do not measure gradient influence or establish causality.")

    for run in runs:
        fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.4), sharey=True)
        score_line(axes[0],run,"Held-out deterministic eval",COLORS[0])
        axes_style(axes[0],"Held-out evaluation")
        diagnostic_line(axes[1],run,TRAIN,"Training rolling score",COLORS[2],half_life=10e6)
        axes_style(axes[1],"Stochastic training games","Training rolling mean score")
        for ax in axes:
            ax.set_xlim(max(0,run["pretrained"]/1e6-15),max(35,run["result"]["last_logged_samples"]/1e6+15))
        save(fig,f"runs/{run['id']}.png",f"{run['study']} {run['variant']} | {run['id']}",
             f"{run['state']}; eval is raw. Training: raw upload values + 10M-sample-half-life EMA. Total samples include any prefix.")

    print(result_frame[["study","variant","model_version","last_logged_samples","last_eval","best_eval","last4_eval_mean","eval_at_300M","end_reason"]].to_string(index=False))
    print(f"Validated {len(runs)} runs, {sum(len(r['frame']) for r in runs):,} history rows; wrote {len(list(figures.rglob('*.png')))} figures")


if __name__ == "__main__":
    main()
