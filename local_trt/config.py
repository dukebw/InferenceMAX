"""Configuration for TRT-LLM benchmarking."""

from dataclasses import dataclass, field
from pathlib import Path
import os


@dataclass
class Config:
    """TRT-LLM benchmark configuration."""

    # Model settings
    model: str = "nvidia/DeepSeek-R1-0528-FP4-V2"
    tp: int = 8
    ep_size: int = 1
    dp_attention: bool = False
    max_model_len: int = 160000
    mtp_layers: int = 1  # 0=disabled, 1-5=enabled

    # Benchmark settings
    num_prompts: int = 300
    input_len: int = 75000
    output_len: int = 300
    concurrency: int = 8

    # NSYS profiling
    nsys: bool = False
    nsys_duration: int = 0  # 0=until benchmark ends
    nsys_delay: int = 0
    nsys_sqlite: bool = False

    # Runtime flags
    skip_server: bool = False

    # Hardcoded settings (not exposed as CLI args)
    port: int = field(default=8888, repr=False)
    container_name: str = field(default="trtllm-local", repr=False)
    image: str = field(default="nvcr.io/nvidia/tensorrt-llm/release:1.2.0rc5", repr=False)
    hf_cache: Path = field(default_factory=lambda: Path.home() / ".cache/huggingface", repr=False)
    modular_repo: Path = field(default_factory=lambda: Path.home() / "work/modular", repr=False)

    @property
    def output_dir(self) -> Path:
        return Path(__file__).parent / "output"

    @property
    def result_filename(self) -> str:
        return f"dsr1_fp4_trt_tp{self.tp}_ep{self.ep_size}_dpa_{str(self.dp_attention).lower()}_conc{self.concurrency}"

    @property
    def nsys_output(self) -> Path:
        return self.output_dir / f"{self.result_filename}.nsys-rep"

    def validate(self) -> None:
        """Validate configuration."""
        if self.tp not in (1, 2, 4, 8):
            raise ValueError(f"TP must be 1, 2, 4, or 8, got {self.tp}")
        if self.mtp_layers < 0 or self.mtp_layers > 5:
            raise ValueError(f"MTP layers must be 0-5, got {self.mtp_layers}")
        if self.concurrency < 1:
            raise ValueError(f"Concurrency must be >= 1, got {self.concurrency}")
        if not os.environ.get("HF_TOKEN"):
            raise ValueError("HF_TOKEN environment variable is required")
        if not self.modular_repo.exists():
            raise ValueError(f"Modular repo not found at {self.modular_repo}")
