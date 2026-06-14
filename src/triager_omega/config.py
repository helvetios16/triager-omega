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


def _detect_torch_device() -> str:
    """Autodetecta el acelerador: CUDA (RTX) > MPS (Apple) > CPU.

    Evita el valor fijo "mps" que rompía SBERT/torch en equipos sin MPS
    (p.ej. el Windows con RTX 5060). Override por env: TORCH_DEVICE=...
    """
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


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
    # non-reasoning: la destilación es transformación determinista (temp=0) → no
    # necesita razonamiento; el variante non-reasoning es ~2-4× más barato y rápido.
    # ("grok-3-mini" era un alias legacy que xAI redirige silenciosamente a grok-4.3.)
    grok_model: str = "grok-4.20-0309-non-reasoning"
    distill_max_comment_chars: int = 1500      # truncado del primer comentario en la entrada
    distill_max_tokens: int = 2048             # holgado; con think=False Gemma no genera preámbulo
    torch_device: str = Field(default_factory=_detect_torch_device)

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

    # --- Validación en OpenJ9 (dataset de TriagerX, etiqueta = fixer/owner) ---
    # Permite probar el IBR "completo" (con `contribution` activo) en el régimen donde
    # TriagerX brilla, y comparar Hit@K/MRR contra el paper. Ver docs/gecko-dev-mining-status.md.
    triagerx_repo: Path = Path("/Users/sebastian/Documents/Python/triagerX")
    openj9_gh_repo: str = "eclipse-openj9/openj9"  # owner/repo para la GitHub API
    github_api_base: str = "https://api.github.com"
    github_token: str = ""  # token de lectura pública; se lee de GITHUB_TOKEN en .env

    # --- Módulo 4: IBR (SBERT + decaimiento + Interaction Points, estilo TriagerX) ---
    # Valores alineados al triagerx_config.yaml real (máxima fidelidad a TriagerX).
    sbert_model: str = "sentence-transformers/all-mpnet-base-v2"
    ibr_tau: float = 0.6            # umbral de similitud coseno (TriagerX: similarity_threshold=0.6)
    ibr_top_k_retrieve: int = 20    # cap de issues similares por query (paper: 20)
    ibr_lambda: float = 0.01        # decaimiento temporal (1/día) (TriagerX: time_decay_factor=0.01)
    # peso del IBR en FS = NPS + W_f·NIS. TriagerX usa 0.7 (similarity_prediction_weight,
    # calibrado en OpenJ9), pero en nuestro piloto Mozilla el CBR ≫ IBR, así que un W_f
    # alto degrada; sintonizado en validación da ~0.1 (mejor Hit@1/MRR). Re-tunear con
    # `aggregator eval --grid` si cambian datos/modelo.
    ibr_w_f: float = 0.1
    # Interaction Points: 3 tipos como TriagerX (contribution/assignment/discussion).
    # commit Y review se fusionan en `contribution` (TriagerX no separa revisores:
    # pull_request+commits comparten contribution_score).
    # VALORES RE-TUNEADOS PARA MOZILLA (ablación §11.3 + grid §10.2). TriagerX usa
    # 1.5/0.5/0.1 (OpenJ9, etiqueta=contribuidor de código), pero aquí la etiqueta es
    # `Assigned To`: `contribution` (commit/review) apunta a revisores ≠ assignee y
    # NO aporta (incluso daña), mientras assignment y discussion son las útiles.
    # Como el NIS se normaliza min-max, solo importa el RATIO → 0/0.5/0.5 ≡ 0/1/1.
    # Re-tuneado: +1.4pp Hit@1 sobre el default TriagerX. Re-confirmar a escala 450 devs
    # (con el CBR más débil, `contribution` podría volver a aportar).
    ip_contribution: float = 0.0   # commit + review (gecko-dev): apagado en Mozilla (TriagerX: 1.5)
    ip_assignment: float = 0.5     # (TriagerX: 0.5)
    ip_discussion: float = 0.5     # (TriagerX: 0.1)

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

    # ----- Rutas del IBR (Módulo 4) -----
    @property
    def repo_interactions_path(self) -> Path:
        """Interacciones commit/review minadas de gecko-dev (data/repo_miner.py)."""
        return self.artifacts_dir / "repo_interactions.parquet"

    @property
    def discussion_interactions_path(self) -> Path:
        """Interacciones discussion limpias (scripts/build_discussion_interactions.py)."""
        return self.artifacts_dir / "discussion_interactions.parquet"

    @property
    def ibr_embeddings_path(self) -> Path:
        """Matriz de embeddings SBERT de los bugs de train del piloto (índice IBR)."""
        return self.pilot_dir / "ibr_embeddings.npy"

    @property
    def ibr_bug_ids_path(self) -> Path:
        """Bug Ids alineados fila a fila con ibr_embeddings.npy."""
        return self.pilot_dir / "ibr_bug_ids.npy"

    # ----- Rutas OpenJ9 (validación del IBR) -----
    @property
    def openj9_dir(self) -> Path:
        return self.artifacts_dir / "openj9"

    @property
    def openj9_train_csv(self) -> Path:
        return self.triagerx_repo / "assets" / "openj9_train_17.csv"

    @property
    def openj9_test_csv(self) -> Path:
        return self.triagerx_repo / "assets" / "openj9_test_17.csv"

    @property
    def openj9_interactions_path(self) -> Path:
        """Tabla larga (issue_number, dev, kind, timestamp) minada de la GitHub API."""
        return self.openj9_dir / "openj9_interactions.parquet"

    @property
    def openj9_issue_meta_path(self) -> Path:
        """(issue_number, created_at) — t_now de las queries y fecha del assignment."""
        return self.openj9_dir / "openj9_issue_meta.parquet"

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
