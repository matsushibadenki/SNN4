# matsushibadenki/snn4/snn_research/core/neurons.py
"""
AdaptiveLIFNeuron and IzhikevichNeuron implementations based on expert feedback and documentation.
- BPTT-enabled state updates
- Correct surrogate gradient usage
- Docstrings and type hints
- Vectorized updates (batch x units)
- device/dtype-aware
"""
from typing import Optional, Tuple
import torch
from torch import Tensor, nn
import torch.nn.functional as F
import math
from spikingjelly.activation_based import surrogate, base # type: ignore

class AdaptiveLIFNeuron(base.MemoryModule):
    """
    Adaptive Leaky Integrate-and-Fire (LIF) neuron with threshold adaptation.
    Designed for vectorized operations and to be BPTT-friendly.
    """
    def __init__(
        self,
        features: int,
        tau_mem: float = 20.0,
        base_threshold: float = 1.0,
        adaptation_strength: float = 0.1,
        target_spike_rate: float = 0.02,
        noise_intensity: float = 0.0,
    ):
        super().__init__()
        self.features = features
        self.mem_decay = math.exp(-1.0 / tau_mem)
        self.base_threshold = nn.Parameter(torch.full((features,), base_threshold))
        self.adaptation_strength = adaptation_strength
        self.target_spike_rate = target_spike_rate
        self.noise_intensity = noise_intensity
        self.surrogate_function = surrogate.ATan(alpha=2.0)

        self.register_buffer("mem", None)
        self.register_buffer("adaptive_threshold", None)
        self.register_buffer("spikes", torch.zeros(features))
        self.register_buffer("total_spikes", torch.tensor(0.0))
        self.stateful = False

    def set_stateful(self, stateful: bool):
        """時系列データの処理モードを設定"""
        self.stateful = stateful
        if not stateful:
            self.reset()

    def reset(self):
        """Resets the neuron's state variables."""
        super().reset()
        self.mem = None
        self.adaptive_threshold = None
        self.spikes.zero_()
        self.total_spikes.zero_()

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Processes one timestep of input current."""
        if not self.stateful:
            self.mem = None
            self.adaptive_threshold = None

        if self.mem is None or self.mem.shape != x.shape:
            self.mem = torch.zeros_like(x)
        if self.adaptive_threshold is None or self.adaptive_threshold.shape != x.shape:
            self.adaptive_threshold = torch.zeros_like(x)

        self.mem = self.mem * self.mem_decay + x
        
        if self.training and self.noise_intensity > 0:
            self.mem += torch.randn_like(self.mem) * self.noise_intensity

        current_threshold = self.base_threshold + self.adaptive_threshold
        spike = self.surrogate_function(self.mem - current_threshold)
        
        self.spikes = spike.mean(dim=0) if spike.ndim > 1 else spike
        
        with torch.no_grad():
            self.total_spikes += spike.detach().sum()

        reset_mask = spike.detach() 
        self.mem = self.mem * (1.0 - reset_mask)

        if self.training:
            self.adaptive_threshold = (
                self.adaptive_threshold * self.mem_decay + 
                self.adaptation_strength * spike.detach()
            )
        else:
            with torch.no_grad():
                self.adaptive_threshold = (
                    self.adaptive_threshold * self.mem_decay + 
                    self.adaptation_strength * spike
                )
        
        return spike, self.mem

    def get_spike_rate_loss(self) -> torch.Tensor:
        """スパイク率の目標値からの乖離を損失として返す"""
        current_rate = self.spikes.mean()
        # target_spike_rateがスカラー値であることを想定
        target = torch.tensor(self.target_spike_rate, device=current_rate.device)
        return F.mse_loss(current_rate, target)

class IzhikevichNeuron(base.MemoryModule):
    """
    Izhikevich neuron model, capable of producing a wide variety of firing patterns.
    """
    def __init__(
        self,
        features: int,
        a: float = 0.02,
        b: float = 0.2,
        c: float = -65.0,
        d: float = 8.0,
        dt: float = 0.5,
    ):
        super().__init__()
        self.features = features
        # a: 回復変数uの時定数。小さいほど回復が遅い (e.g., 0.02 for regular spiking)
        self.a = a
        # b: 膜電位vに対する回復変数uの感受性。大きいほどvとuの結合が強い (e.g., 0.2 for regular spiking)
        self.b = b
        # c: スパイク後の膜電位vのリセット値 (e.g., -65 mV)
        self.c = c
        # d: スパイク後の回復変数uの増加量 (e.g., 2 for regular spiking, 8 for chattering)
        self.d = d
        # dt: シミュレーションの時間刻み幅
        self.dt = dt
        self.v_peak = 30.0
        self.surrogate_function = surrogate.ATan(alpha=2.0)

        self.register_buffer("v", None)
        self.register_buffer("u", None)
        self.register_buffer("spikes", torch.zeros(features))
        self.register_buffer("total_spikes", torch.tensor(0.0))

    def reset(self):
        super().reset()
        self.v = None
        self.u = None
        self.spikes.zero_()
        self.total_spikes.zero_()

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Processes one timestep of input current with corrected Izhikevich dynamics.
        """
        if self.v is None or self.v.shape != x.shape:
            self.v = torch.full_like(x, self.c)
        if self.u is None or self.u.shape != x.shape:
            self.u = torch.full_like(x, self.b * self.c)

        dv = 0.04 * self.v**2 + 5 * self.v + 140 - self.u + x
        du = self.a * (self.b * self.v - self.u)
        
        self.v = self.v + dv * self.dt
        self.u = self.u + du * self.dt
        
        spike = self.surrogate_function(self.v - self.v_peak)
        self.spikes = spike.mean(dim=0) if spike.ndim > 1 else spike
        
        with torch.no_grad():
            self.total_spikes += spike.detach().sum()
        
        reset_mask = (self.v >= self.v_peak).detach()
        self.v = torch.where(reset_mask, torch.full_like(self.v, self.c), self.v)
        self.u = torch.where(reset_mask, self.u + self.d, self.u)
        
        # クランプ範囲は、発散を防ぎ、数値的安定性を保つためのものです。
        self.v = torch.clamp(self.v, min=-100.0, max=50.0)

        return spike, self.v