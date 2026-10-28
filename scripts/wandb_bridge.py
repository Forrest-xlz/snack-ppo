import json
import sys

import wandb


def main():
    run = None
    for line in sys.stdin:
        message = json.loads(line)
        kind = message.pop("kind")
        if kind == "init":
            run = wandb.init(
                project=message["project"],
                entity=message.get("entity"),
                group=message.get("group"),
                name=message.get("run_name"),
                mode=message["mode"],
                config=message["config"],
            )
            run.define_metric("total_samples")
            run.define_metric("train_samples/*", step_metric="total_samples")
            run.define_metric("eval_samples/*", step_metric="total_samples")
            run.define_metric("early_stop_samples/*", step_metric="total_samples")
        elif kind == "log":
            if run is None:
                raise RuntimeError("received log before init")
            run.log(message["metrics"])
        elif kind == "finish":
            if run is not None:
                run.finish()
            return
        else:
            raise ValueError(f"unknown message kind: {kind}")


if __name__ == "__main__":
    main()
