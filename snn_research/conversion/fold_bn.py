# ファイルパス: snn_research/conversion/fold_bn.py
# (新規作成)
# Title: BatchNorm Folding ユーティリティ
# Description:
# ANNからSNNへの変換精度を向上させるため、Convolution層とそれに続くBatchNorm層を
# 単一のConvolution層に統合（folding）する機能を提供する。

import torch
import torch.nn as nn
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def fold_conv_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convolution層とBatchNorm層を統合し、新しい重みとバイアスを計算する。

    Args:
        conv (nn.Conv2d): 畳み込み層。
        bn (nn.BatchNorm2d): バッチ正規化層。

    Returns:
        tuple[torch.Tensor, torch.Tensor]: 統合後の新しい重みとバイアス。
    """
    w = conv.weight.clone().detach()
    if conv.bias is None:
        bias = torch.zeros(w.size(0), device=w.device)
    else:
        bias = conv.bias.clone().detach()

    # BN params
    eps = bn.eps
    gamma = bn.weight.clone().detach()
    beta = bn.bias.clone().detach()
    running_mean = bn.running_mean.clone().detach()
    running_var = bn.running_var.clone().detach()

    denom = torch.sqrt(running_var + eps)
    w_fold = w * (gamma / denom).reshape(-1, 1, 1, 1)
    b_fold = (bias - running_mean) * (gamma / denom) + beta
    
    return w_fold, b_fold

def fold_all_batchnorms(model: nn.Module) -> nn.Module:
    """
    モデル内のすべてのConv-BNペアを探索し、インプレースで統合する。
    注意: この関数はモジュール自体を置き換えます。

    Args:
        model (nn.Module): 変更対象のモデル。

    Returns:
        nn.Module: BatchNormが統合されたモデル。
    """
    model.eval()
    
    # モジュールのリストを名前付きで取得
    module_list = list(model.named_children())
    
    for i in range(len(module_list) - 1):
        name, module = module_list[i]
        next_name, next_module = module_list[i+1]
        
        if isinstance(module, nn.Conv2d) and isinstance(next_module, nn.BatchNorm2d):
            logging.info(f"Folding BatchNorm layer '{next_name}' into Conv layer '{name}'")
            
            # 新しい重みとバイアスを計算
            new_weight, new_bias = fold_conv_bn(module, next_module)
            
            # 新しいConv層を作成
            new_conv = nn.Conv2d(
                in_channels=module.in_channels,
                out_channels=module.out_channels,
                kernel_size=module.kernel_size,
                stride=module.stride,
                padding=module.padding,
                dilation=module.dilation,
                groups=module.groups,
                bias=True
            )
            new_conv.weight.data.copy_(new_weight)
            new_conv.bias.data.copy_(new_bias)
            
            # 元のConv層を新しいものに置き換え
            setattr(model, name, new_conv)
            # BN層をIdentity（何もしない層）に置き換え
            setattr(model, next_name, nn.Identity())

    # 再帰的に子モジュールにも適用
    for name, m in model.named_children():
        if len(list(m.children())) > 0:
            fold_all_batchnorms(m)
            
    return model