"""Configuración central: rutas, nombres de archivos e hiperparámetros del Módulo 1.

Se carga desde variables de entorno (.env) con valores por defecto razonables.
Todas las rutas se resuelven relativas a la raíz del proyecto.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Raíz del repo = dos niveles arriba de este archivo (src/triager_omega/config.py).
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Configuración global del proyecto."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Rutas base ---
    data_dir: Path = Field(default=PROJECT_ROOT / "data")
    artifacts_dir: Path = Field(default=PROJECT_ROOT / "artifacts")

    # --- Nombres reales de los parquets en data/ (difieren del PLAN.md) ---
    bugs_file: str = "bug_metadata.parquet"
    contributors_file: str = "contributors.parquet"
    comments_file: str = "bug_comments.parquet"

    # --- Módulo 2: Destilación (backend-agnóstico) ---
    # backend: "ollama"   (local, óptimo, plan para el batch completo) ·
    #          "lmstudio" (local, OpenAI-compat) · "google" (API, ojo rate limits).
    distill_backend: str = "ollama"
    ollama_base_url: str = "http://localhost:11434/v1"
    ollama_model: str = "gemma4:e4b-it-qat"  # modelo en Ollama (ver scripts/test_ollama.py)
    ollama_think: bool = False               # desactiva el "thinking" → JSON directo, sin preámbulo
    lm_studio_base_url: str = "http://127.0.0.1:1234/v1"
    lm_studio_model: str = "google/gemma-4-e4b"
    google_model: str = "gemma-4-26b-a4b-it"  # modelo Gemma vía google.genai (ver test_api.py)
    grok_api_key: str = ""
    grok_base_url: str = "https://api.x.ai/v1"
    grok_model: str = "grok-3-mini"
    distill_max_comment_chars: int = 1500      # truncado del primer comentario en la entrada
    distill_max_tokens: int = 2048             # holgado; con think=False Gemma no genera preámbulo
    torch_device: str = "mps"

    # --- Piloto del CBR (subconjunto escala TriagerX para validar el diseño) ---
    pilot_n_devs: int = 20     # top-N devs más activos (TriagerX usó 17-41)
    pilot_cap: int = 300       # tope de bugs/dev en train (aplana la cabeza)
    pilot_eval_cap: int = 100  # tope de bugs/dev en val y test (acota la destilación)

    # --- Módulo 1: directorio activo y balanceo ---
    active_threshold: int = 50  # 'Assigned To and Fixed' mínimo (decisión: ≥50, 538 devs, 91% cobertura)
    last_activity_horizon_months: int | None = None  # filtro opcional por recencia; None = desactivado

    # --- Splits temporales ---
    train_frac: float = 0.70
    val_frac: float = 0.15
    # test_frac = 1 - train - val = 0.15

    # --- Limpieza de texto ---
    max_stacktrace_lines: int = 30

    # --- Reproducibilidad ---
    seed: int = 42

    # --- Marcadores de usuarios automáticos a excluir como label ---
    automated_user_markers: tuple[str, ...] = ("nobody",)

    # --- Minería de repositorio (MSR) para enriquecer el IBR ---
    repo_url: str = "https://github.com/mozilla/gecko-dev.git"
    repo_branch: str = "master"
    repo_mine_since: str = "2021-01-01"  # alineado con el rango de Creation Time de los bugs

    # --- Módulo 4: IBR (SBERT + decaimiento + Interaction Points, estilo TriagerX) ---
    # Valores alineados al triagerx_config.yaml real (máxima fidelidad a TriagerX).
    sbert_model: str = "sentence-transformers/all-mpnet-base-v2"
    ibr_tau: float = 0.6            # umbral de similitud coseno (TriagerX: similarity_threshold=0.6)
    ibr_top_k_retrieve: int = 20    # cap de issues similares por query (paper: 20)
    ibr_lambda: float = 0.01        # decaimiento temporal (1/día) (TriagerX: time_decay_factor=0.01)
    ibr_w_f: float = 0.7            # peso del IBR en FS = NPS + W_f·NIS (TriagerX: similarity_prediction_weight=0.7)
    # Interaction Points: 3 tipos como TriagerX (contribution/assignment/discussion).
    # commit Y review se fusionan en `contribution` (TriagerX no separa revisores:
    # pull_request+commits comparten contribution_score). Valores = triagerx_config.yaml.
    ip_contribution: float = 1.5   # commit + review (gecko-dev) → un solo peso, como TriagerX
    ip_assignment: float = 0.5
    ip_discussion: float = 0.1

    # ----- Rutas derivadas a artefactos del Módulo 1 -----
    @property
    def active_candidates_path(self) -> Path:
        return self.artifacts_dir / "active_candidates.json"

    @property
    def label_encoder_path(self) -> Path:
        return self.artifacts_dir / "label_encoder.json"

    @property
    def splits_path(self) -> Path:
        return self.artifacts_dir / "splits.parquet"

    @property
    def sample_weights_path(self) -> Path:
        return self.artifacts_dir / "sample_weights_train.npy"

    # ----- Rutas del piloto del CBR (Módulo 2-3) -----
    @property
    def pilot_dir(self) -> Path:
        return self.artifacts_dir / "pilot"

    @property
    def pilot_splits_path(self) -> Path:
        return self.pilot_dir / "splits.parquet"

    @property
    def pilot_label_encoder_path(self) -> Path:
        return self.pilot_dir / "label_encoder.json"

    @property
    def distillations_path(self) -> Path:
        return self.pilot_dir / "distillations.parquet"

    # ----- Rutas a los parquets de entrada -----
    @property
    def bugs_path(self) -> Path:
        return self.data_dir / self.bugs_file

    @property
    def contributors_path(self) -> Path:
        return self.data_dir / self.contributors_file

    @property
    def comments_path(self) -> Path:
        return self.data_dir / self.comments_file


settings = Settings()
