# Design and Implementation of a Hybrid Agentic Architecture for Financial Analysis with Locally Deployed Large Language Models and Model Context Protocol (MCP) Governed Trade Execution

This thesis presents the design and implementation of an autonomous cloud based conversational system for real time financial analysis of a specific stock and human supervised stock trade execution based on the results of that analysis. The system combines cloud hosted and locally run artificial intelligence components to analyse market data, interpret news sentiment, forecast short-term trends, and propose as well as execute trading actions, all orchestrated through the Model Context Protocol (MCP). The architecture emphasizes modularity,availability and safe human-in-the-loop supervision so that final trade decisions remain under explicit user control.
The system leverages Autogen for multi agent coordination, Azure OpenAI’s GPT-4o for natural-language reasoning , FinGPT (a LLaMA-2 model fine-tuned with LoRA adapters) for short-horizon financial trend forecasting, the Bing Grounding Tool for retrieving timely market moving news, and yfinance for essential technical indicators. These components operate within an orchestrated agent pipeline in which specialized agents independently handle news extraction, financial metric computation, trend analysis, decision synthesis and communicate insights to the end user via Slack. The Slack component provides the users with a conversational interface to the trading platform, i.e. Alpaca, which subsequently retrieves or executes real time market orders according to the user’s messages. 
This integration is a classical demonstration of how agentic AI architectures can bridge cloud services, local computational resources,communication channels such as Slack, and trading platforms such as Alpaca into a cohesive and extensible platform capable of informed decision making and the automated execution and retrieval of trades, positions, and other brokerage data.


A multi-agent stock analysis system that combines:

- **Azure OpenAI (GPT-4o)** for reasoning & orchestration  
- **Bing Grounding Tool** for real news search  
- **yfinance** for market data  
- **Local FinGPT adapter (LLaMA-2 + LoRA)** for short-term trend forecasting  
<img width="1987" height="2705" alt="Untitled diagram-2025-11-30-161432" src="https://github.com/user-attachments/assets/e5245736-6af7-45bb-8200-08eded7b21ff" />

The system runs a round-robin agent team to gather news, fetch financials, forecast trends, and make structured Buy/Sell/Hold decisions.

---

## ⚡ Recommended Setup (GPU + Local FinGPT Adapter)

This is the workflow validated on an **RTX 4060 Laptop GPU (8GB)** with **Python 3.11**.

### 0) Prerequisites

- Windows 10/11  
- Python **3.11.x**  
- Git  
- NVIDIA driver (RTX 40-series or similar)  
- Hugging Face account (for LLaMA + FinGPT LoRA)  
- Azure subscription (for GPT-4o + Bing Grounding)

### 1) Clone & Environment

```powershell
git clone https://github.com/pranavb1211/AutoAgentThesis.git
cd AutoAgentThesis

py -3.11 -m venv kello
.\kello\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
```

Add to `.gitignore`:

```
kello/
__pycache__/
*.pyc
```

### 2) Install PyTorch (CUDA build)

```powershell
python -m pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1
```

Verify:

```powershell
python - << 'PY'
import torch
print("torch:", torch.__version__)
print("CUDA:", torch.cuda.is_available(), "device:", torch.cuda.get_device_name(0))
PY
```

### 3) Install Repo Dependencies

```powershell
pip install -r requirements.txt
```

> If `requirements.txt` pins `torch==…+cpu`, comment it out.

### 4) Download Models

```powershell
pip install huggingface_hub
huggingface-cli login
```

Accept license for **meta-llama/Llama-2-7b-chat-hf**.

Download:

```powershell
# Base LLaMA2 (~13 GB)
huggingface-cli download meta-llama/Llama-2-7b-chat-hf --local-dir C:\hf\models\llama2-7b-chat

# FinGPT LoRA (hundreds MB)
huggingface-cli download FinGPT/fingpt-forecaster_dow30_llama2-7b_lora --local-dir C:\hf\models\fingpt_adapter
```

### 5) Configure `.env`

In repo root, create `.env`:

```env
# Azure OpenAI
AZURE_OPENAI_API_KEY=your-key
AZURE_ENDPOINT=https://<your-endpoint>.openai.azure.com
MODEL_DEPLOYMENT_NAME=gpt-4o
MODEL_API_VERSION=2024-12-01-preview
OPENAI_API_VERSION=2024-12-01-preview

# Azure project for Bing
AZURE_PROJECT_ENDPOINT=https://<project>.services.ai.azure.com/api/projects/<project>
BING_CONNECTION_NAME=BingGroundingTool

# Local FinGPT model paths
FINGPT_BASE_DIR=C:\hf\models\llama2-7b-chat
FINGPT_LORA_DIR=C:\hf\models\fingpt_adapter
```

### 6) Test Local FinGPT

```powershell
python - << 'PY'
from adapters.fingpt_local import your_fingpt_analyze_function
print(your_fingpt_analyze_function("One-sentence market outlook for the Dow Jones today.", max_new_tokens=64))
PY
```

### 7) Run Agents

```powershell
python app.py
```

Agents will:
1. Fetch news (NewsAnalyzer via Bing).  
2. Pull financials (yfinance).  
3. Pass context to FinGPT local forecaster.  
4. DecisionAgent issues Buy/Sell/Hold with confidence.  

Logs:
- `predictions.jsonl` → structured outputs  
- `fingpt_outputs.jsonl` → raw FinGPT runs

---

## 🧾 Legacy Setup (CPU + Full FinGPT Clone)

This was the **initial setup path** used before the lightweight local adapter. Useful if you want the full FinGPT repo experience.

### 1) Clone FinGPT repo

```powershell
cd C:\Users\<you>\Desktop
git clone https://github.com/AI4Finance-Foundation/FinGPT.git
cd FinGPT
pip install -e .
```

### 2) Pin to specific commit (stable)

```powershell
git checkout 4e53f8d7f3d3342d7f9cfa9fb6681609e9703dea
```

### 3) Integrate with AutoAgentThesis

In your thesis repo (`adapters/fingpt_local.py`), import functions from the installed FinGPT library instead of using only local LoRA weights.

---

## 📂 Project Structure

```
AutoAgentThesis/
  adapters/
    fingpt_local.py   # minimal LLaMA2 + FinGPT LoRA loader
  app.py              # orchestrator: Azure + Bing + yfinance + FinGPT agents
  newsagent.py        # sample FinGPT call
  requirements.txt
  .env
  kello/              # venv (ignored)
```

---

## 🛠 Troubleshooting

- **Torch not compiled with CUDA** → reinstall CUDA wheels (see above).  
- **OOM on 8 GB GPU** → reduce `max_new_tokens` to 32–64.  
- **env vars missing** → check `.env` placement & `load_dotenv()`.  
- **10k untracked files** → venv inside repo; add `kello/` to `.gitignore`.  
- **OneDrive permission errors** → keep repo outside OneDrive.  

---

## 🎯 One-liner Recap

```powershell
git clone https://github.com/pranavb1211/AutoAgentThesis.git
cd AutoAgentThesis
py -3.11 -m venv kello; .\kello\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1
pip install -r requirements.txt
huggingface-cli login
huggingface-cli download meta-llama/Llama-2-7b-chat-hf --local-dir C:\hf\models\llama2-7b-chat
huggingface-cli download FinGPT/fingpt-forecaster_dow30_llama2-7b_lora --local-dir C:\hf\models\fingpt_adapter
# create .env
python app.py
```
