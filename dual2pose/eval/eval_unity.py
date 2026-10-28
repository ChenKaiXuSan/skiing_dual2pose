from dual2pose.training_protocol import evaluation_checkpoint_from_env
#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/code/dual2pose/main.py
Project: /workspace/code/dual2pose
Created Date: Tuesday April 22nd 2025
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Thursday May 1st 2025 8:34:05 pm
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2025 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
"""

import logging
import os
import sys
from pathlib import Path
import hydra

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import (
    RichProgressBar,
)
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dual2pose.dataloader.data_loader import UnityDataModule
from dual2pose.trainer.train_crossview_fusion import CrossViewFusionTrainer
from dual2pose.eval.pose_metrics import EVALUATION_PROTOCOL

logger = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path="../../configs",  # * the config_path is relative to location of the python script
    config_name="dual2pose.yaml",
)
def init_params(config):
    config.train.gpu = 0

    seed_everything(42, workers=True)

    classification_module = CrossViewFusionTrainer(config)
    unity_data_module = UnityDataModule(config)

    # for the tensorboard
    tb_logger = TensorBoardLogger(
        save_dir=os.path.join(config.log_path, "tb_logs"), name="test"
    )

    # 初始化 CSVLogger
    cvs_logger = CSVLogger(
        save_dir=os.path.join(config.log_path, "csv_logs"),
        name="test",
    )

    # some callbacks
    # progress_bar = TQDMProgressBar(refresh_rate=10)
    progress_bar = RichProgressBar(refresh_rate=10, leave=True)

    trainer = Trainer(
        devices=[
            int(config.train.gpu),
        ],
        accelerator="gpu",
        max_epochs=config.train.max_epochs,
        logger=[tb_logger, cvs_logger],
        callbacks=[
            progress_bar,
        ],
    )

    # save the metrics to file
    ckpt_path = str(evaluation_checkpoint_from_env())
    metrics = trainer.test(
        classification_module,
        unity_data_module,
        ckpt_path=ckpt_path,
        weights_only=False,
    )

    logger.info(f"Test metrics: {metrics}")

    # persist metrics to output directory
    run_tag = Path(ckpt_path).stem if ckpt_path else "run"
    out_root = Path(os.environ.get("EVAL_OUTPUT_PATH", config.log_path))
    out_dir = out_root / "unity_eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    # write JSON
    out_json = out_dir / f"metrics_{run_tag}.json"
    try:
        import json

        with open(out_json, "w", encoding="utf-8") as fp:
            json.dump({"evaluation_protocol": dict(EVALUATION_PROTOCOL), "metrics": metrics}, fp, ensure_ascii=False, indent=2)
    except Exception:
        logger.exception("Failed to write JSON metrics")

    # write CSV (flatten first record if present)
    out_csv = out_dir / f"metrics_{run_tag}.csv"
    try:
        if metrics and isinstance(metrics, list) and len(metrics) > 0 and isinstance(metrics[0], dict):
            import csv

            with open(out_csv, "w", encoding="utf-8", newline="") as fp:
                writer = csv.DictWriter(fp, fieldnames=list(metrics[0].keys()))
                writer.writeheader()
                for row in metrics:
                    writer.writerow({k: (v if not isinstance(v, (list, dict)) else str(v)) for k, v in row.items()})
    except Exception:
        logger.exception("Failed to write CSV metrics")


if __name__ == "__main__":
    os.environ["HYDRA_FULL_ERROR"] = "1"
    init_params()
