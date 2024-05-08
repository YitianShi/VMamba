import sys, os, glob
import copy
sys.path.append("../../")
sys.path.append("../../google-bnn-hmc/")
sys.path.append("..")
sys.path.append(".")

import torch
import torch.nn as nn
import torch.nn.functional as F

import time
import math
import itertools
import yaml
import logging
import config

from src.algos.laplace_approx import LaplaceApprox
from src.eval.calibration import ClassificationCalibrationResults
from src.algos.ensemble import DeepEnsemble
from src.algos.util import wilson_scheduler
from experiments.base import cifar
from models import get_model
from experiments.base.multiclass_classification import _analyze_output


import wandb

def eval_model_on_dataset(model, device, config, loader):
    outputs = []
    targets = []
    with torch.no_grad():
        for input, target in loader:
            input = input.to(device)
            with torch.autocast(device_type="cuda", enabled=config["use_amp"]):
                output = model.predict(lambda m: m(input), samples=config["eval_samples"])
            output = torch.logsumexp(output.float(), dim=0).cpu() - math.log(output.shape[0])
            outputs.append(output)
            targets.append(target)

    return torch.cat(outputs), torch.cat(targets)

def eval_model(model, config, device, split="test", subsample=None):
    model.eval()

    if split == "test":
        loader = cifar.cifar10_testloader(config["data_path"], config["eval_batch_size"], True)
    else:
        loader = cifar.cifar10_corrupted_testloader(config["data_path"], split, config["batch_size"], True) # Shuffle as datapoints are sorted by corruption type

    if subsample is not None:
        loader = itertools.islice(loader, subsample)
    
    outputs, targets = eval_model_on_dataset(model, device, config, loader)
    errors, confidences, log_likelihoods, _, _ = _analyze_output(outputs, targets)

    calibration = ClassificationCalibrationResults(config["ece_bins"], errors, confidences)
    return {
        "accuracy": errors.sum() / len(errors),
        "log_likelihood": log_likelihoods.mean(),
        "ece": calibration.ece,
        "sece": calibration.signed_ece,
        "bin_accuracies": calibration.bin_accuracys,
        "bin_confidences": calibration.bin_confidences,
        "bin_counts": calibration.bin_counts
    }

def run(device, config, out_path, log, rep):
    if config.get("share_file_system", False):
        torch.multiprocessing.set_sharing_strategy('file_system')
    wandb_mode = "disabled" if config.get("disable_wandb") else "online"

    wandb_tags = [config["model"]]

    if config["model"] == "mcd":
        wandb_tags.append(f"p-{config['p']}")

    wandb.init(
        name=f"{config['model']}_{config['members']}-({rep})", 
        project="cifar_10", 
        group=config["model"],
        config=config,
        tags=wandb_tags,
        mode=wandb_mode)

    model = get_model(config["model"], config, device)
    print(model)

    train_model(model, device, config, log, out_path)
    torch.save(model.state_dict(), out_path + f"{config['model']}_final.tar")

    if "laplace" in config["model"]:
        fit_laplace(model, config, log)

    #model.load_state_dict(torch.load(out_path + f"{config['model']}_final.tar"))

    eval_time = time.time()

    test_results = eval_model(model, config, device, split="test", subsample=None)
    log.info(f"Test: {test_results}")

    #c1_results = eval_model(model, config, device, split=0, subsample=10)
    #log.info(f"Corrupted (1): {c1_results}")

    #c3_results = eval_model(model, config, device, split=2, subsample=10)
    #log.info(f"Corrupted (3): {c3_results}")

    #c5_results = eval_model(model, config, device, split=4, subsample=10)
    #log.info(f"Corrupted (5): {c5_results}")

    wandb.log({
        "test_results": test_results,
        #"c1_results": c1_results,
        #"c3_results": c3_results,
        #"c5_results": c5_results
    })
    log.info(f"Eval time: {time.time() - eval_time}s")

def load_model(model_idx, model, scaler, optimizer, out_path, config, log):
    ckpt = None
    start_epoch = 0

    # Load checkpoint and scaler if available
    if config.get("use_checkpoint", None):
        try:
            ckpt_paths = glob.glob(out_path + f"{config['model']}_chkpt_{model_idx}_*.pth")
            ckpt_paths.sort(key=os.path.getmtime)
            ckpt = torch.load(ckpt_paths[-1]) 
            model.load_state_dict(ckpt['model_state_dict'])
            start_epoch = ckpt["epoch"] + 1
            scaler.load_state_dict(ckpt["scaler_state_dict"])
            log.info(f"Loaded checkpoint for model {model_idx} at epoch {start_epoch}")
        except:
            log.info(f"Failed to load checkpoint for model {model_idx}")

    optimizer.init_grad_scaler(scaler)

    # Load optimizer state if available
    # Base optimizer state is loaded separately if available
    if ckpt is not None: 
        try:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if ckpt.get("base_optimizer") is not None:
                optimizer.get_base_optimizer().load_state_dict(ckpt["base_optimizer"])
            log.info(f"Loaded base optimizer state for model {model_idx}")
        except:
            log.info(f"Failed to load optimizer state for model {model_idx}")

    # Load scheduler state if available
    if config["lr_schedule"]:
        scheduler = wilson_scheduler(optimizer.get_base_optimizer(), config["epochs"], config["lr"], None)
        if ckpt is not None:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            log.info(f"Loaded scheduler state for model {model_idx}")
    else:
        scheduler = None
    
    return start_epoch, model, optimizer, scaler, scheduler

def save_model(model, optimizer, scheduler, scaler, out_path, config, model_idx, epoch):
    state_dict = {
                    'epoch': epoch,
                    'model_idx': model_idx,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else 'None'
                }
    if hasattr(optimizer, "base_optimizer"):
        state_dict['base_optimizer'] = optimizer.get_base_optimizer().state_dict()
    torch.save(state_dict, out_path + f"{config['model']}_ckpt_{model_idx}_{epoch}.pth")

def train_model(ensemble: DeepEnsemble, device, config, log, out_path):

    ensemble.to(device)

    before_all = time.time()

    for model_idx, (model, optimizer) in enumerate(ensemble.models_and_optimizers):
        log.info(f"==================================================")
        log.info(f"Training model {model_idx}")
        log.info(f"==================================================")

        scaler = torch.cuda.amp.GradScaler(enabled=config["use_amp"])

        start_epoch, model, optimizer, scaler, scheduler = load_model(model_idx, model, scaler, optimizer, out_path, config, log)

        trainloader = cifar.cifar10_trainloader(config["data_path"], config["batch_size"])

        log.info(f"Training on {len(trainloader)} minibatches")
        before = time.time()
        for epoch in range(start_epoch, config["epochs"]):
            ensemble.train()
            epoch_loss = torch.tensor(0.0, device=device)
            for input, target in trainloader:
                input, target = input.to(device), target.to(device)

                def forward():
                    with torch.autocast(device_type="cuda", enabled=config["use_amp"]):
                        return F.nll_loss(model(input), target)

                def backward(loss):
                    scaler.scale(loss).backward()

                loss = optimizer.step(forward, backward, grad_scaler=scaler)
                #print(loss.detach())
                scaler.update()
                epoch_loss += loss.detach()
            optimizer.complete_epoch()
            if scheduler is not None:
                scheduler.step()
            epoch_loss /= len(trainloader)

            if epoch % 10 == 0 and epoch > start_epoch:
                save_model(model, optimizer, scheduler, scaler, out_path, config, model_idx, epoch)
            log.info(f"Epoch {epoch}: train loss {(epoch_loss):.5f}")
            wandb.log({"train_loss": epoch_loss}, step=(epoch + model_idx * config["epochs"]))
        log.info(f"Final loss: {epoch_loss:.5f}")
        log.info(f"Training time: {time.time() - before}s")
    
    log.info(f"==================================================")
    log.info(f"Finished training")
    log.info(f"Total training time {time.time() - before_all}s")
    log.info(f"==================================================")

def fit_laplace(ensemble, config, log):
    for i in range(len(ensemble.models)):
        trainloader = cifar.cifar10_trainloader(config["data_path"], config["batch_size"])
        class DatasetMock:
            def __len__(self):
                return len(trainloader.dataset)

        class LoaderMock:
            def __init__(self):
                self.dataset = DatasetMock()

            def __iter__(self):
                return iter(map(lambda x: (x[0], x[1]), iter(trainloader)))

        log.info("Fitting laplace...")
        laplace = LaplaceApprox(ensemble.models[i][0], regression=False, out_activation=torch.log, hessian="full")
        laplace.fit(LoaderMock())

        log.info("Optimizing prior prec...")
        laplace.optimize_prior_prec()
        log.info("Done")

        ensemble.models[i] = laplace


def wilson_scheduler(optimizer, pretrain_epochs, lr_init, swag_lr=None):
    def wilson_schedule(epoch):
        t = (epoch) / pretrain_epochs
        lr_ratio = swag_lr / lr_init if swag_lr is not None else 0.01
        if t <= 0.5:
            factor = 1.0
        elif t <= 0.9:
            factor = 1.0 - (1.0 - lr_ratio) * (t - 0.5) / 0.4
        else:
            factor = lr_ratio
        return factor
    return torch.optim.lr_scheduler.LambdaLR(optimizer, wilson_schedule)

def deep_merge_dicts(source, destination):
    """
    Recursively updates a dictionary with values from another dictionary.
    Values in the source dictionary take precedence over those in the destination.
    """
    for key, value in source.items():
        if isinstance(value, dict):
            # Get node or create one
            node = destination.setdefault(key, {})
            deep_merge_dicts(value, node)
        else:
            destination[key] = value

    return destination

def main(config_path, name):
    with open(config_path, 'r') as file:
        configs = list(yaml.safe_load_all(file))

    default_config = configs[0]
    for config in configs:
        if config.get('name') == name:
            break

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()

    if torch.cuda.is_available():
        logger.info("Using the GPU")
        device = torch.device("cuda")
    else:
        logger.info("Using the CPU")
        device = torch.device("cpu")
    
    config = deep_merge_dicts(default_config, config)
    config["params"]["data_path"] = os.path.join(os.path.expanduser("~"),"Research/dataset")
    # set up the path to save the model, the location is in the same folder as this script
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ckpt/")
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    run(device, config["params"], out_path, logger, rep=0)

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, "cifar.yaml")
    
    main(config_path, name="BBB")