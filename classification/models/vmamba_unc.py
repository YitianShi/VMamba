import os
import time
import math
import copy
from functools import partial
from typing import Optional, Callable, Any
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from timm.models.layers import DropPath, trunc_normal_
from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count, parameter_count
from torchvision.models import VisionTransformer

DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

from .bnn.src.algos.bbb_layers import make_module_bbb, BBBLinear
from .bnn.src.algos.bbb import GaussianPrior
from .bnn.src.algos.rank1 import make_module_rank1, Rank1Linear
from .bnn.src.algos.dropout import patch_dropout, FixableDropout

from .vmamba import *

def build_vssmunc_model(config, is_pretrain=False, uncertainty="bbb"):
    model_type = config.MODEL.TYPE
    if model_type in ["vssm"]:
        model = VSSMUnc(
            patch_size=config.MODEL.VSSM.PATCH_SIZE, 
            in_chans=config.MODEL.VSSM.IN_CHANS, 
            num_classes=config.MODEL.NUM_CLASSES, 
            depths=config.MODEL.VSSM.DEPTHS, 
            dims=config.MODEL.VSSM.EMBED_DIM, 
            # ===================
            ssm_d_state=config.MODEL.VSSM.SSM_D_STATE,
            ssm_ratio=config.MODEL.VSSM.SSM_RATIO,
            ssm_rank_ratio=config.MODEL.VSSM.SSM_RANK_RATIO,
            ssm_dt_rank=("auto" if config.MODEL.VSSM.SSM_DT_RANK == "auto" else int(config.MODEL.VSSM.SSM_DT_RANK)),
            ssm_act_layer=config.MODEL.VSSM.SSM_ACT_LAYER,
            ssm_conv=config.MODEL.VSSM.SSM_CONV,
            ssm_conv_bias=config.MODEL.VSSM.SSM_CONV_BIAS,
            ssm_drop_rate=config.MODEL.VSSM.SSM_DROP_RATE,
            ssm_init=config.MODEL.VSSM.SSM_INIT,
            forward_type=config.MODEL.VSSM.SSM_FORWARDTYPE,
            # ===================
            mlp_ratio=config.MODEL.VSSM.MLP_RATIO,
            mlp_act_layer=config.MODEL.VSSM.MLP_ACT_LAYER,
            mlp_drop_rate=config.MODEL.VSSM.MLP_DROP_RATE,
            # ===================
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            patch_norm=config.MODEL.VSSM.PATCH_NORM,
            norm_layer=config.MODEL.VSSM.NORM_LAYER,
            downsample_version=config.MODEL.VSSM.DOWNSAMPLE,
            patchembed_version=config.MODEL.VSSM.PATCHEMBED,
            gmlp=config.MODEL.VSSM.GMLP,
            use_checkpoint=config.TRAIN.USE_CHECKPOINT,
            # ===================
            posembed=config.MODEL.VSSM.POSEMBED,
            imgsize=config.DATA.IMG_SIZE,
            # ===================
            uncertainty=uncertainty,
        )
        return model

    return None

class SS2DUnc(nn.Module, mamba_init, SS2Dv0, SS2Dv2, SS2Dv3):
    def __init__(
        self,
        # basic dims ===========
        d_model=96,
        d_state=16,
        ssm_ratio=2.0,
        dt_rank="auto",
        act_layer=nn.SiLU,
        # dwconv ===============
        d_conv=3, # < 2 means no conv 
        conv_bias=True,
        # ======================
        dropout=0.0,
        bias=False,
        # dt init ==============
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        initialize="v0",
        # ======================
        forward_type="v2",
        channel_first=False,
        # ======================
        uncertainty="bbb", # bbb, rank1, dropout_patch
        proj_only=False,
        weight_prior=None,
        bias_prior=None,
        # ======================
        sigma=0.1,
        components=5,
        **kwargs,    
    ):
        super().__init__()
        self.d_inner = int(ssm_ratio * d_model)
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        kwargs.update(
            d_model=d_model, d_state=d_state, ssm_ratio=ssm_ratio, dt_rank=dt_rank,
            act_layer=act_layer, d_conv=d_conv, conv_bias=conv_bias, dropout=dropout, bias=bias,
            dt_min=dt_min, dt_max=dt_max, dt_init=dt_init, dt_scale=dt_scale, dt_init_floor=dt_init_floor,
            initialize=initialize, forward_type=forward_type, channel_first=channel_first,
        )
        if forward_type in ["v0", "v0seq"]:
            self.__initv0__(seq=("seq" in forward_type), **kwargs)
        elif forward_type.startswith("xv"):
            self.__initxv__(**kwargs)
        else:
            self.__initv2__(**kwargs)

        # bbb, rank1, dropout_patch ==============================
        if weight_prior is None:
            self.prior = GaussianPrior(0, sigma)
            if bias_prior is None and uncertainty == "bbb":
                self.bias_prior = GaussianPrior(0, sigma)

        if uncertainty == "bbb":
            self.kl = 0
            del self.x_proj_weight
            if not proj_only: 
                make_module_bbb(self, prior=self.prior)
            self.x_proj = [
            BBBLinear(self.d_inner, (self.dt_rank + d_state * 2), bias=bias, weight_prior=self.prior, bias_prior=self.bias_prior, components=components)
            for _ in range(4)
            ]
            self.x_proj_weight = nn.ParameterList([self.x_proj[i].weight for i in range(4)])
            self.x_proj_bias = nn.ParameterList([self.x_proj[i].bias for i in range(4)]) if self.x_proj[0].bias is not None else None
            
        elif uncertainty == "rank1":
            self.comp=0
            self.components = components
            if not proj_only:
                make_module_rank1(self, prior=self.prior, components=components)
            self.x_proj = [
            Rank1Linear(self.d_inner, (self.dt_rank + d_state * 2), bias=bias, prior=self.prior, components=components)
            for _ in range(4)
            ]
            self.x_proj_s = nn.ParameterList([nn.ParameterList([self.x_proj[i].s[t] for i in range(4)]) for t in range(components)]) 
            self.x_proj_r = nn.ParameterList([nn.ParameterList([self.x_proj[i].r[t] for i in range(4)]) for t in range(components)])
            self.x_proj_bias = nn.Parameter(torch.stack([t.bias for t in self.x_proj], dim=0)) if self.x_proj[0].bias is not None else None

        del self.x_proj
        self.uncertainty = uncertainty
        
        

    def forward_corev2(
        self,
        x: torch.Tensor=None, 
        # ==============================
        to_dtype=True, # True: final out to dtype
        force_fp32=False, # True: input fp32
        # ==============================
        ssoflex=True, # True: out fp32 in SSOflex; else, SSOflex is the same as SSCore
        # ==============================
        SelectiveScan=SelectiveScanOflex,
        CrossScan=CrossScan,
        CrossMerge=CrossMerge,
        no_einsum=False, # replace einsum with linear or conv1d to raise throughput
        # ==============================
        cascade2d=True,
        **kwargs,
    ):
        dt_projs_weight = self.dt_projs_weight
        dt_projs_bias = self.dt_projs_bias
        A_logs = self.A_logs
        Ds = self.Ds
        delta_softplus = True
        out_norm = getattr(self, "out_norm", None)
        channel_first = self.channel_first
        to_fp32 = lambda *args: (_a.to(torch.float32) for _a in args)

        B, D, H, W = x.shape
        D, N = A_logs.shape
        K, D, R = dt_projs_weight.shape
        L = H * W

        def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True):
            return SelectiveScan.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, -1, -1, ssoflex)
        
        def dbl_bbb(xs, B, D, L, K, cascade2d=False, width=True):
            x_proj_weight_mean = torch.stack([x_proj_weight.mean for x_proj_weight in self.x_proj_weight])
            x_proj_weight_std = torch.stack([x_proj_weight.std for x_proj_weight in self.x_proj_weight])
            x_proj_bias_mean = torch.stack([x_proj_bias.mean for x_proj_bias in self.x_proj_bias]) if self.x_proj_bias is not None else None
            x_proj_bias_std = torch.stack([x_proj_bias.std for x_proj_bias in self.x_proj_bias]) if self.x_proj_bias is not None else None

            if cascade2d:
                if width:
                    x_proj_weight_mean = x_proj_weight_mean.view(4, -1, D)[:2].contiguous()
                    x_proj_weight_std = x_proj_weight_std.view(4, -1, D)[:2].contiguous()
                    x_proj_bias_mean = x_proj_bias_mean.view(4, -1)[:2].contiguous() if x_proj_bias_mean is not None else None
                    x_proj_bias_std = x_proj_bias_std.view(4, -1)[:2].contiguous() if x_proj_bias_std is not None else None
                else:
                    x_proj_weight_mean = x_proj_weight_mean.view(4, -1, D)[2:].contiguous()
                    x_proj_weight_std = x_proj_weight_std.view(4, -1, D)[2:].contiguous()
                    x_proj_bias_mean = x_proj_bias_mean.view(4, -1)[2:].contiguous() if x_proj_bias_mean is not None else None
                    x_proj_bias_std = x_proj_bias_std.view(4, -1)[2:].contiguous() if x_proj_bias_std is not None else None

            if no_einsum:
                x_dbl_mean = F.conv1d(xs.view(B, -1, L), x_proj_weight_mean.view(-1, D, 1), 
                                        bias=(x_proj_bias_mean.view(-1) if x_proj_bias_mean is not None else None), groups=K)
                x_dbl_var = F.conv1d((xs.view(B, -1, L)**2).clamp(min=1e-4), (x_proj_weight_std**2).view(-1, D, 1).clamp(min=1e-4), 
                                        bias=(x_proj_bias_std.view(-1)**2 if x_proj_bias_std is not None else None), groups=K)
                x_dbl_std = torch.sqrt(x_dbl_var)

                x_dbl = self.sample_parameters(x_dbl_mean, x_dbl_std)
            
            else:
                x_dbl_mean = torch.einsum("b k d l, k c d -> b k c l", xs, x_proj_weight_mean)
                x_dbl_std = torch.einsum("b k d l, k c d -> b k c l", xs**2, x_proj_weight_std**2).clamp(min=1e-4).sqrt()
                x_dbl = self.sample_parameters(x_dbl_mean, x_dbl_std)
                if x_proj_bias_mean is not None:
                    return x_dbl + self.sample_parameters(x_proj_bias_mean, x_proj_bias_std)
            
            # KL divergence
            if self.training:
                self.kl = self.prior.kl_divergence(x_proj_weight_mean, x_proj_weight_std)
                if self.x_proj_bias is not None:
                    self.kl += self.prior.kl_divergence(x_proj_bias_mean, x_proj_weight_std)

            return x_dbl

        def dbl_rank1(xs, B, D, L, K, cascade2d=False, width=True):
            x_proj_weight = self.x_proj_weight
            x_proj_bias = self.x_proj_bias if self.x_proj_bias is not None else None
            x_proj_r = self.x_proj_r[self.comp]
            x_proj_s = self.x_proj_s[self.comp]
            x_proj_r = torch.stack([r.sample() for r in x_proj_r], dim=0)
            x_proj_s = torch.stack([s.sample() for s in x_proj_s], dim=0)

            if cascade2d:
                if width:
                    x_proj_weight = x_proj_weight.view(4, -1, D)[:2].contiguous()
                    x_proj_r = x_proj_r.view(4, -1)[:2].contiguous()
                    x_proj_s = x_proj_s.view(4, -1)[:2].contiguous()
                else:
                    x_proj_weight = x_proj_weight.view(4, -1, D)[2:].contiguous()
                    x_proj_r = x_proj_r.view(4, -1)[2:].contiguous()
                    x_proj_s = x_proj_s.view(4, -1)[2:].contiguous()

            if no_einsum:
                x_dbl = F.conv1d(xs.view(B, -1, L)*x_proj_s.view(1, -1, 1), x_proj_weight.view(-1, D, 1), 
                                 bias=(x_proj_bias.view(-1) if x_proj_bias is not None else None), groups=K)
                x_dbl *= x_proj_r.view(1, -1, 1)    
            else:
                x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs*x_proj_s.unsqueeze(0).unsqueeze(-1), x_proj_weight)
                x_dbl *= x_proj_r.unsqueeze(0).unsqueeze(-1)
                if x_proj_bias is not None:
                    return x_dbl + x_proj_bias.view(1, K, -1, 1)
            
            if not cascade2d or not width:
                self.comp = (self.comp + 1) % self.components
            return x_dbl
            
        if self.uncertainty == "bbb":
            dbl = dbl_bbb
        elif self.uncertainty == "rank1":
            dbl = dbl_rank1
        
        if cascade2d:
            def scan_rowcol(
                x: torch.Tensor, 
                dt_weight: torch.Tensor, 
                dt_bias: torch.Tensor, # (2*c)
                _As: torch.Tensor, # As = -torch.exp(A_logs.to(torch.float))[:2,] # (2*c, d_state)
                _Ds: torch.Tensor,
                width = True,
            ):
                # x: (B, D, H, W)
                # proj_weight: (2 * D, (R+N+N))
                XB, XD, XH, XW = x.shape
                if width:
                    _B, _D, _L = XB * XH, XD, XW
                    xs = x.permute(0, 2, 1, 3).contiguous()
                else:
                    _B, _D, _L = XB * XW, XD, XH
                    xs = x.permute(0, 3, 1, 2).contiguous()
                xs = torch.stack([xs, xs.flip(dims=[-1])], dim=2) # (B, H, 2, D, W)
                x_dbl = dbl(xs, _B, _D, _L, 2, cascade2d=True, width=width)
                if no_einsum:
                    dts, Bs, Cs = torch.split(x_dbl.view(_B, 2, -1, _L), [R, N, N], dim=2)
                    dts = F.conv1d(dts.contiguous().view(_B, -1, _L), dt_weight.view(2 * _D, -1, 1), groups=2)
                else:
                    dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
                    dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_weight)

                xs = xs.view(_B, -1, _L)
                dts = dts.contiguous().view(_B, -1, _L)
                As = _As.view(-1, N).to(torch.float)
                Bs = Bs.contiguous().view(_B, 2, N, _L)
                Cs = Cs.contiguous().view(_B, 2, N, _L)
                Ds = _Ds.view(-1)
                delta_bias = dt_bias.view(-1).to(torch.float)

                if force_fp32:
                    xs = xs.to(torch.float)
                dts = dts.to(xs.dtype)
                Bs = Bs.to(xs.dtype)
                Cs = Cs.to(xs.dtype)

                ys: torch.Tensor = selective_scan(
                    xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus
                ).view(_B, 2, -1, _L)
                return ys
            
            As = -torch.exp(A_logs.to(torch.float)).view(4, -1, N)
            y_row = scan_rowcol(
                x,
                dt_weight = dt_projs_weight.view(4, D, -1)[:2].contiguous(),
                dt_bias = (dt_projs_bias.view(4, -1)[:2].contiguous() if dt_projs_bias is not None else None),
                _As = As[:2].contiguous().view(-1, N),
                _Ds = Ds.view(4, -1)[:2].contiguous().view(-1),
                width=True,
            ).view(B, H, 2, -1, W).sum(dim=2).permute(0, 2, 1, 3)
            y_col = scan_rowcol(
                y_row,
                dt_weight = dt_projs_weight.view(4, D, -1)[2:].contiguous().to(y_row.dtype),
                dt_bias = (dt_projs_bias.view(4, -1)[2:].contiguous().to(y_row.dtype) if dt_projs_bias is not None else None),
                _As = As[2:].contiguous().view(-1, N),
                _Ds = Ds.view(4, -1)[2:].contiguous().view(-1),
                width=False,
            ).view(B, W, 2, -1, H).sum(dim=2).permute(0, 2, 3, 1)
            y = y_col
        else:
            xs = CrossScan.apply(x)
            
            if no_einsum:
                dts, Bs, Cs = torch.split(dbl(xs, B, D, L, K).view(B, K, -1, L), [R, N, N], dim=2)
                dts = F.conv1d(dts.contiguous().view(B, -1, L), dt_projs_weight.view(K * D, -1, 1), groups=K)
            else:
                dts, Bs, Cs = torch.split(dbl(xs, B, D, L, K), [R, N, N], dim=2)
                dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_projs_weight)

            xs = xs.view(B, -1, L)
            dts = dts.contiguous().view(B, -1, L)
            As = -torch.exp(A_logs.to(torch.float)) # (k * c, d_state)
            Bs = Bs.contiguous().view(B, K, N, L)
            Cs = Cs.contiguous().view(B, K, N, L)
            Ds = Ds.to(torch.float) # (K * c)
            delta_bias = dt_projs_bias.view(-1).to(torch.float)

            if force_fp32:
                xs, dts, Bs, Cs = to_fp32(xs, dts, Bs, Cs)

            ys: torch.Tensor = selective_scan(
                xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus
            ).view(B, K, -1, H, W)
            
            y: torch.Tensor = CrossMerge.apply(ys)

            if getattr(self, "__DEBUG__", False):
                setattr(self, "__data__", dict(
                    A_logs=A_logs, Bs=Bs, Cs=Cs, Ds=Ds,
                    us=xs, dts=dts, delta_bias=delta_bias,
                    ys=ys, y=y,
                ))

        y = y.view(B, -1, H, W)
        if not channel_first:
            y = y.view(B, -1, H * W).transpose(dim0=1, dim1=2).contiguous().view(B, H, W, -1) # (B, L, C)
        y = out_norm(y)

        return (y.to(x.dtype) if to_dtype else y)

    @staticmethod
    def sample_parameters(mu, sigma):
        epsilon = torch.empty(sigma.shape, device=mu.device).normal_(0, 1)
        return mu + sigma * epsilon
    
# =====================================================
class VSSUncBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: nn.Module = nn.LayerNorm,
        channel_first=False,
        # =============================
        ssm_d_state: int = 16,
        ssm_ratio=2.0,
        ssm_dt_rank: Any = "auto",
        ssm_act_layer=nn.SiLU,
        ssm_conv: int = 3,
        ssm_conv_bias=True,
        ssm_drop_rate: float = 0,
        ssm_init="v0",
        forward_type="v2",
        # =============================
        mlp_ratio=4.0,
        mlp_act_layer=nn.GELU,
        mlp_drop_rate: float = 0.0,
        gmlp=False,
        # =============================
        use_checkpoint: bool = False,
        post_norm: bool = False,
        # =============================
        uncertainty="bbb",
        components=4,
        proj_only=True,
        # =============================
        **kwargs,
    ):
        super().__init__()
        self.ssm_branch = ssm_ratio > 0
        self.mlp_branch = mlp_ratio > 0
        self.use_checkpoint = use_checkpoint
        self.post_norm = post_norm

        if self.ssm_branch:
            self.norm = norm_layer(hidden_dim)
            self.op = SS2DUnc(
                d_model=hidden_dim, 
                d_state=ssm_d_state, 
                ssm_ratio=ssm_ratio,
                dt_rank=ssm_dt_rank,
                act_layer=ssm_act_layer,
                # ==========================
                d_conv=ssm_conv,
                conv_bias=ssm_conv_bias,
                # ==========================
                dropout=ssm_drop_rate,
                # bias=False,
                # ==========================
                # dt_min=0.001,
                # dt_max=0.1,
                # dt_init="random",
                # dt_scale="random",
                # dt_init_floor=1e-4,
                initialize=ssm_init,
                # ==========================
                forward_type=forward_type,
                channel_first=channel_first,
                # ==========================
                uncertainty=uncertainty,
                components=components,
                proj_only = proj_only,
                # ==========================
            )
        
        self.drop_path = DropPath(drop_path)
        
        if self.mlp_branch:
            _MLP = Mlp if not gmlp else gMlp
            self.norm2 = norm_layer(hidden_dim)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            self.mlp = _MLP(in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=mlp_act_layer, drop=mlp_drop_rate, channels_first=channel_first)

    def _forward(self, input: torch.Tensor):
        x = input
        if self.ssm_branch:
            if self.post_norm:
                x = x + self.drop_path(self.norm(self.op(x)))
            else:
                x = x + self.drop_path(self.op(self.norm(x)))
        if self.mlp_branch:
            if self.post_norm:
                x = x + self.drop_path(self.norm2(self.mlp(x))) # FFN
            else:
                x = x + self.drop_path(self.mlp(self.norm2(x))) # FFN
        return x

    def forward(self, input: torch.Tensor):
        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, input)
        else:
            return self._forward(input)

# =====================================================
class VSSMUnc(nn.Module):
    def __init__(
        self, 
        patch_size=4, 
        in_chans=3, 
        num_classes=1000, 
        depths=[2, 2, 9, 2], 
        dims=[96, 192, 384, 768], 
        # =========================
        ssm_d_state=16,
        ssm_ratio=2.0,
        ssm_dt_rank="auto",
        ssm_act_layer="silu",        
        ssm_conv=3,
        ssm_conv_bias=True,
        ssm_drop_rate=0.0, 
        ssm_init="v0",
        forward_type="v2",
        # =========================
        mlp_ratio=4.0,
        mlp_act_layer="gelu",
        mlp_drop_rate=0.0,
        gmlp=False,
        # =========================
        drop_path_rate=0.1, 
        patch_norm=True, 
        norm_layer="LN", # "BN", "LN2D"
        downsample_version: str = "v2", # "v1", "v2", "v3"
        patchembed_version: str = "v1", # "v1", "v2"
        use_checkpoint=False,  
        # =========================
        posembed=False,
        imgsize=224,
        # =========================
        uncertainty="bbb",
        **kwargs,
    ):
        super().__init__()
        self.channel_first = (norm_layer.lower() in ["bn", "ln2d"])
        self.num_classes = num_classes
        self.num_layers = len(depths)
        if isinstance(dims, int):
            dims = [int(dims * 2 ** i_layer) for i_layer in range(self.num_layers)]
        self.num_features = dims[-1]
        self.dims = dims
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        
        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            ln2d=LayerNorm2d,
            bn=nn.BatchNorm2d,
        )

        _ACTLAYERS = dict(
            silu=nn.SiLU, 
            gelu=nn.GELU, 
            relu=nn.ReLU, 
            sigmoid=nn.Sigmoid,
        )

        norm_layer: nn.Module = _NORMLAYERS.get(norm_layer.lower(), None)
        ssm_act_layer: nn.Module = _ACTLAYERS.get(ssm_act_layer.lower(), None)
        mlp_act_layer: nn.Module = _ACTLAYERS.get(mlp_act_layer.lower(), None)

        self.pos_embed = self._pos_embed(dims[0], patch_size, imgsize) if posembed else None

        _make_patch_embed = dict(
            v1=self._make_patch_embed, 
            v2=self._make_patch_embed_v2,
        ).get(patchembed_version, None)
        self.patch_embed = _make_patch_embed(in_chans, dims[0], patch_size, patch_norm, norm_layer, channel_first=self.channel_first)

        _make_downsample = dict(
            v1=PatchMerging2D, 
            v2=self._make_downsample, 
            v3=self._make_downsample_v3, 
            none=(lambda *_, **_k: None),
        ).get(downsample_version, None)

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            downsample = _make_downsample(
                self.dims[i_layer], 
                self.dims[i_layer + 1], 
                norm_layer=norm_layer,
                channel_first=self.channel_first,
            ) if (i_layer < self.num_layers - 1) else nn.Identity()

            self.layers.append(self._make_layer(
                dim = self.dims[i_layer],
                drop_path = dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                use_checkpoint=use_checkpoint,
                norm_layer=norm_layer,
                downsample=downsample,
                channel_first=self.channel_first,
                # =================
                ssm_d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_dt_rank=ssm_dt_rank,
                ssm_act_layer=ssm_act_layer,
                ssm_conv=ssm_conv,
                ssm_conv_bias=ssm_conv_bias,
                ssm_drop_rate=ssm_drop_rate,
                ssm_init=ssm_init,
                forward_type=forward_type,
                # =================
                mlp_ratio=mlp_ratio,
                mlp_act_layer=mlp_act_layer,
                mlp_drop_rate=mlp_drop_rate,
                gmlp=gmlp,
                # =================
                uncertainty=uncertainty,
                proj_only=True,
            ))

        self.classifier = nn.Sequential(OrderedDict(
            norm=norm_layer(self.num_features), # B,H,W,C
            permute=(Permute(0, 3, 1, 2) if not self.channel_first else nn.Identity()),
            avgpool=nn.AdaptiveAvgPool2d(1),
            flatten=nn.Flatten(1),
            head=nn.Linear(self.num_features, num_classes),
        ))

        self.apply(self._init_weights)

    @staticmethod
    def _pos_embed(embed_dims, patch_size, img_size):
        patch_height, patch_width = (img_size // patch_size, img_size // patch_size)
        pos_embed = nn.Parameter(torch.zeros(1, embed_dims, patch_height, patch_width))
        trunc_normal_(pos_embed, std=0.02)
        return pos_embed

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # used in building optimizer
    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed"}

    # used in building optimizer
    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {}

    @staticmethod
    def _make_patch_embed(in_chans=3, embed_dim=96, patch_size=4, patch_norm=True, norm_layer=nn.LayerNorm, channel_first=False):
        # if channel first, then Norm and Output are both channel_first
        return nn.Sequential(
            nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=True),
            (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            (norm_layer(embed_dim) if patch_norm else nn.Identity()),
        )

    @staticmethod
    def _make_patch_embed_v2(in_chans=3, embed_dim=96, patch_size=4, patch_norm=True, norm_layer=nn.LayerNorm, channel_first=False):
        # if channel first, then Norm and Output are both channel_first
        stride = patch_size // 2
        kernel_size = stride + 1
        padding = 1
        return nn.Sequential(
            nn.Conv2d(in_chans, embed_dim // 2, kernel_size=kernel_size, stride=stride, padding=padding),
            (nn.Identity() if (channel_first or (not patch_norm)) else Permute(0, 2, 3, 1)),
            (norm_layer(embed_dim // 2) if patch_norm else nn.Identity()),
            (nn.Identity() if (channel_first or (not patch_norm)) else Permute(0, 3, 1, 2)),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding),
            (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            (norm_layer(embed_dim) if patch_norm else nn.Identity()),
        )
    
    @staticmethod
    def _make_downsample(dim=96, out_dim=192, norm_layer=nn.LayerNorm, channel_first=False):
        # if channel first, then Norm and Output are both channel_first
        return nn.Sequential(
            (nn.Identity() if channel_first else Permute(0, 3, 1, 2)),
            nn.Conv2d(dim, out_dim, kernel_size=2, stride=2),
            (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            norm_layer(out_dim),
        )

    @staticmethod
    def _make_downsample_v3(dim=96, out_dim=192, norm_layer=nn.LayerNorm, channel_first=False):
        # if channel first, then Norm and Output are both channel_first
        return nn.Sequential(
            (nn.Identity() if channel_first else Permute(0, 3, 1, 2)),
            nn.Conv2d(dim, out_dim, kernel_size=3, stride=2, padding=1),
            (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            norm_layer(out_dim),
        )

    @staticmethod
    def _make_layer(
        dim=96, 
        drop_path=[0.1, 0.1], 
        use_checkpoint=False, 
        norm_layer=nn.LayerNorm,
        downsample=nn.Identity(),
        channel_first=False,
        # ===========================
        ssm_d_state=16,
        ssm_ratio=2.0,
        ssm_dt_rank="auto",       
        ssm_act_layer=nn.SiLU,
        ssm_conv=3,
        ssm_conv_bias=True,
        ssm_drop_rate=0.0, 
        ssm_init="v0",
        forward_type="v2",
        # ===========================
        mlp_ratio=4.0,
        mlp_act_layer=nn.GELU,
        mlp_drop_rate=0.0,
        gmlp=False,
        # ===========================
        uncertainty = "bbb",
        proj_only = True,
        **kwargs,
    ):
        # if channel first, then Norm and Output are both channel_first
        depth = len(drop_path)
        blocks = []
        for d in range(depth):
            blocks.append(VSSUncBlock(
                hidden_dim=dim, 
                drop_path=drop_path[d],
                norm_layer=norm_layer,
                channel_first=channel_first,
                ssm_d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_dt_rank=ssm_dt_rank,
                ssm_act_layer=ssm_act_layer,
                ssm_conv=ssm_conv,
                ssm_conv_bias=ssm_conv_bias,
                ssm_drop_rate=ssm_drop_rate,
                ssm_init=ssm_init,
                forward_type=forward_type,
                mlp_ratio=mlp_ratio,
                mlp_act_layer=mlp_act_layer,
                mlp_drop_rate=mlp_drop_rate,
                gmlp=gmlp,
                use_checkpoint=use_checkpoint,
                uncertainty=uncertainty,
                proj_only=proj_only,
            ))
        
        return nn.Sequential(OrderedDict(
            blocks=nn.Sequential(*blocks,),
            downsample=downsample,
        ))

    def forward(self, x: torch.Tensor):
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            pos_embed = self.pos_embed.permute(0, 2, 3, 1) if not self.channel_first else self.pos_embed
            x = x + pos_embed
        for layer in self.layers:
            x = layer(x)
        x = self.classifier(x)
        return x

    def flops(self, shape=(3, 224, 224), verbose=True):
        # shape = self.__input_shape__[1:]
        supported_ops={
            "aten::silu": None, # as relu is in _IGNORED_OPS
            "aten::neg": None, # as relu is in _IGNORED_OPS
            "aten::exp": None, # as relu is in _IGNORED_OPS
            "aten::flip": None, # as permute is in _IGNORED_OPS
            # "prim::PythonOp.CrossScan": None,
            # "prim::PythonOp.CrossMerge": None,
            "prim::PythonOp.SelectiveScanMamba": partial(selective_scan_flop_jit, flops_fn=flops_selective_scan_fn, verbose=verbose),
            "prim::PythonOp.SelectiveScanOflex": partial(selective_scan_flop_jit, flops_fn=flops_selective_scan_fn, verbose=verbose),
            "prim::PythonOp.SelectiveScanCore": partial(selective_scan_flop_jit, flops_fn=flops_selective_scan_fn, verbose=verbose),
            "prim::PythonOp.SelectiveScanNRow": partial(selective_scan_flop_jit, flops_fn=flops_selective_scan_fn, verbose=verbose),
        }

        model = copy.deepcopy(self)
        model.cuda().eval()

        input = torch.randn((1, *shape), device=next(model.parameters()).device)
        params = parameter_count(model)[""]
        Gflops, unsupported = flop_count(model=model, inputs=(input,), supported_ops=supported_ops)

        del model, input
        return sum(Gflops.values()) * 1e9
        return f"params {params} GFLOPs {sum(Gflops.values())}"

    # used to load ckpt from previous training code
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):

        def check_name(src, state_dict: dict = state_dict, strict=False):
            if strict:
                if prefix + src in list(state_dict.keys()):
                    return True
            else:
                key = prefix + src
                for k in list(state_dict.keys()):
                    if k.startswith(key):
                        return True
            return False

        def change_name(src, dst, state_dict: dict = state_dict, strict=False):
            if strict:
                if prefix + src in list(state_dict.keys()):
                    state_dict[prefix + dst] = state_dict[prefix + src]
                    state_dict.pop(prefix + src)
            else:
                key = prefix + src
                for k in list(state_dict.keys()):
                    if k.startswith(key):
                        new_k = prefix + dst + k[len(key):]
                        state_dict[new_k] = state_dict[k]
                        state_dict.pop(k)

        if check_name("pos_embed", strict=True):
            srcEmb: torch.Tensor = state_dict[prefix + "pos_embed"]
            state_dict[prefix + "pos_embed"] = F.interpolate(srcEmb.float(), size=self.pos_embed.shape[2:4], align_corners=False, mode="bicubic").to(srcEmb.device)

        change_name("patch_embed.proj", "patch_embed.0")
        change_name("patch_embed.norm", "patch_embed.2")
        for i in range(100):
            for j in range(100):
                change_name(f"layers.{i}.blocks.{j}.ln_1", f"layers.{i}.blocks.{j}.norm")
                change_name(f"layers.{i}.blocks.{j}.self_attention", f"layers.{i}.blocks.{j}.op")
        change_name("norm", "classifier.norm")
        change_name("head", "classifier.head")

        return super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
