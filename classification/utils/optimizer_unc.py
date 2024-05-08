from functools import partial
import torch
from torch import optim as optim


from models.bnn.src.algos.swag import SwagOptimizer
from models.bnn.src.algos.bbb import GaussianPrior, BBBOptimizer
from models.bnn.src.algos.bbb_layers import BBBLinear
from models.bnn.src.algos.rank1 import Rank1Linear
from models.bnn.src.algos.ensemble import DeepEnsemble
from models.bnn.src.algos.pp import MAPOptimizer
from models.bnn.src.algos.ivorn import iVONOptimizer
from models.bnn.src.algos.svgd import SVGDOptimizer
from models.bnn.src.algos.dropout import patch_dropout, FixableDropout
from models.bnn.src.algos.kernel.sngp import SNGPWrapper, SNGPOptimizer
from models.bnn.src.algos.kernel.base import spectrally_normalize_module
from models.bnn.src.algos.util import reset_model_params

def build_optimizer_unc(config, model, logger, **kwargs):
    """
    Build optimizer, set weight decay of normalization to 0 by default.
    """
    logger.info(f"==============> building optimizer {config.TRAIN.OPTIMIZER.NAME}....................")
    skip = {}
    skip_keywords = {}
    if hasattr(model, 'no_weight_decay'):
        skip = model.no_weight_decay()
    if hasattr(model, 'no_weight_decay_keywords'):
        skip_keywords = model.no_weight_decay_keywords()
    parameters, no_decay_names = set_weight_decay(model, skip, skip_keywords)
    logger.info(f"No weight decay list: {no_decay_names}")

    opt_lower = config.TRAIN.OPTIMIZER.NAME.lower()
    optimizer = None
    optimizer = optim.SGD(parameters, momentum=config.TRAIN.OPTIMIZER.MOMENTUM, nesterov=True,
                              lr=config.TRAIN.BASE_LR, weight_decay=config.TRAIN.WEIGHT_DECAY)
    #if opt_lower == 'sgd':
    #    optimizer = optim.SGD(parameters, momentum=config.TRAIN.OPTIMIZER.MOMENTUM, nesterov=True,
    #                         lr=config.TRAIN.BASE_LR, weight_decay=config.TRAIN.WEIGHT_DECAY)
    #elif opt_lower == 'adamw':
    #   optimizer = optim.AdamW(parameters, eps=config.TRAIN.OPTIMIZER.EPS, betas=config.TRAIN.OPTIMIZER.BETAS,
    #                          lr=config.TRAIN.BASE_LR, weight_decay=config.TRAIN.WEIGHT_DECAY)
    #else:
        #raise NotImplementedError
    
    """ 
    if uncertainty == "bbb":
        prior = GaussianPrior(0, config_unc["prior_std"])
        optimizer = BBBOptimizer(parameters, optimizer, prior=prior, **config_unc[uncertainty])
    elif uncertainty == "rank1":
        prior = GaussianPrior(0, config_unc["prior_std"])
        optimizer = BBBOptimizer(parameters, optimizer, prior=prior, **config_unc[uncertainty])
    elif uncertainty == "swag":
        optimizer = SwagOptimizer(parameters, optimizer, **config_unc[uncertainty])
    elif uncertainty == "ivon":
        optimizer = iVONOptimizer(parameters, **config_unc[uncertainty])
    elif uncertainty == "mcd" or uncertainty == "map":
        optimizer = MAPOptimizer(parameters, optimizer)
    else:    
        raise NotImplementedError
    """
    return optimizer


def set_weight_decay(model, skip_list=(), skip_keywords=()):
    has_decay = []
    no_decay = []
    no_decay_names = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or (name in skip_list) or \
                check_keywords_in_name(name, skip_keywords):
            no_decay.append(param)
            no_decay_names.append(name)
            # print(f"{name} has no weight decay")
        else:
            has_decay.append(param)
    return [{'params': has_decay},
            {'params': no_decay, 'weight_decay': 0.}], no_decay_names 


def check_keywords_in_name(name, keywords=()):
    isin = False
    for keyword in keywords:
        if keyword in name:
            isin = True
    return isin


def BBB_loss(outputs, targets, criterion, config_unc, kl_params, prior):
        
    mc_samples = config_unc["mc_samples"]
    dataset_size = config_unc["dataset_size"]
    kl_rescaling = config_unc["kl_rescaling"]

    total_data_loss = torch.tensor(0.0, device=targets.device)
    for output in outputs:
       total_data_loss += criterion(output, targets)

    # Collect KL loss & reg only once
    total_kl_loss = torch.tensor(0.0, device=targets.device)
    for param in kl_params:
        total_kl_loss += param.get_parameter_kl(prior)

    pi = kl_rescaling / dataset_size
    # don't divide the kl loss by the mc sample count as we have only been collection it once
    loss = pi * total_kl_loss + total_data_loss / mc_samples

    return loss
    