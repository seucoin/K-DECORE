import os
import sys
import random
import subprocess

from llamafactory.extras.misc import get_device_count
from llamafactory.extras import logging
from llamafactory import launcher

from typing import TYPE_CHECKING, Any, Dict, List, Optional
from llamafactory.hparams.parser import get_train_args
from llamafactory.train.callbacks import LogCallback
from workflow import run_sft


if TYPE_CHECKING:
    from transformers import TrainerCallback

logger = logging.get_logger(__name__)

def run_exp(args: Optional[Dict[str, Any]] = None, callbacks: List["TrainerCallback"] = []) -> None:
    callbacks.append(LogCallback())
    model_args, data_args, training_args, finetuning_args, generating_args = get_train_args(args)

    if finetuning_args.stage == "sft":
        run_sft(model_args, data_args, training_args, finetuning_args, generating_args, callbacks)
    else:
        raise ValueError(f"Unknown task: {finetuning_args.stage}.")


def main():
    force_torchrun = os.getenv("FORCE_TORCHRUN", "0").lower() in ["true", "1"]
    if force_torchrun or get_device_count() > 1:
        master_addr = os.getenv("MASTER_ADDR", "127.0.0.1")
        master_port = os.getenv("MASTER_PORT", str(random.randint(20001, 29999)))
        logger.info_rank0(f"Initializing distributed tasks at: {master_addr}:{master_port}")
        process = subprocess.run(
            (
                "torchrun --nnodes {nnodes} --node_rank {node_rank} --nproc_per_node {nproc_per_node} "
                "--master_addr {master_addr} --master_port {master_port} {file_name} {args}"
            )
            .format(
                nnodes=os.getenv("NNODES", "1"),
                node_rank=os.getenv("NODE_RANK", "0"),
                nproc_per_node=os.getenv("NPROC_PER_NODE", str(get_device_count())),
                master_addr=master_addr,
                master_port=master_port,
                file_name=launcher.__file__,
                args=" ".join(sys.argv[1:]),
            )
            .split()
        )
        sys.exit(process.returncode)
    else:
        run_exp()


if __name__ == "__main__":
    main()