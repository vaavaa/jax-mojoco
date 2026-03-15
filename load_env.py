"""Load .env and set WANDB_API_KEY for wandb. Import this before using wandb."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")
# WANDB_API_KEY is now in os.environ when defined in .env
