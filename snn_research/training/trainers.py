# ファイルパス: snn_research/training/trainers.py
# (省略...)
# 修正点 (v8): ParticleFilterTrainerのコンストラクタを修正し、
#           KeyError: 'training' を解消する。

# (...ファイルの先頭は省略...)

class BPTTTrainer:
    # (...このクラスは変更なし...)
    pass

class ParticleFilterTrainer:
    """
    逐次モンテカルロ法（パーティクルフィルタ）を用いて、微分不可能なSNNを学習するトレーナー。
    CPU上での実行を想定し、GPU依存から脱却するアプローチ。
    """
    def __init__(self, base_model: BioSNN, config: Dict[str, Any], device: str):
        self.base_model = base_model.to(device)
        self.device = device
        self.config = config
        # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓修正開始◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
        # configの階層構造を変更
        self.num_particles = config['num_particles']
        self.noise_std = config['noise_std']
        # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑修正終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
        
        self.particles = [copy.deepcopy(self.base_model) for _ in range(self.num_particles)]
        self.particle_weights = torch.ones(self.num_particles, device=self.device) / self.num_particles
        print(f"🌪️ ParticleFilterTrainer initialized with {self.num_particles} particles.")

    def train_step(self, data: torch.Tensor, targets: torch.Tensor) -> float:
        """1ステップの学習（予測、尤度計算、再サンプリング）を実行する。"""
        data = data.to(self.device)
        targets = targets.to(self.device)

        for particle in self.particles:
            with torch.no_grad():
                for param in particle.parameters():
                    param.add_(torch.randn_like(param) * self.noise_std)
        
        log_likelihoods = []
        for particle in self.particles:
            particle.eval()
            with torch.no_grad():
                squeezed_data = data.squeeze(0) if data.dim() > 1 else data
                input_spikes = (torch.rand_like(squeezed_data) > 0.5).float().to(self.device)
                outputs, _ = particle(input_spikes)
                
                squeezed_targets = targets.squeeze(0) if targets.dim() > 1 else targets
                loss = F.mse_loss(outputs, squeezed_targets)
                log_likelihoods.append(-loss)
        
        log_likelihoods_tensor = torch.tensor(log_likelihoods, device=self.device)
        self.particle_weights *= torch.exp(log_likelihoods_tensor - log_likelihoods_tensor.max())
        
        if self.particle_weights.sum() > 0:
            self.particle_weights /= self.particle_weights.sum()
        else:
            self.particle_weights.fill_(1.0 / self.num_particles)

        if 1. / (self.particle_weights**2).sum() < self.num_particles / 2:
            indices = torch.multinomial(self.particle_weights, self.num_particles, replacement=True)
            new_particles = [copy.deepcopy(self.particles[i]) for i in indices]
            self.particles = new_particles
            self.particle_weights.fill_(1.0 / self.num_particles)
        
        best_particle_loss = -log_likelihoods_tensor.max().item()
        return best_particle_loss
