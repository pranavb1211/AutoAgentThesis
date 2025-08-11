# AutoAgent + FinGPT + LLaMA Setup Guide

This guide documents the exact working setup steps for integrating **AutoGen**, **FinGPT**, and **LLaMA** on **Python 3.11** (Windows).

---

## 🚀 1. Create & Activate Virtual Environment

```powershell
# Create venv (Python 3.11)
py -3.11 -m venv .venv

# Activate
.\.venv\Scripts\Activate.ps1

# Upgrade pip tools
python -m pip install --upgrade pip setuptools wheel
```

---

## ⚙️ 2. Install AutoGen + Azure Stack

```powershell
pip install autogen-agentchat autogen-core "autogen-ext[azure,openai]" asyncio python-dotenv openai tiktoken "aiohttp>=3.8.0" yfinance
```

---

## 🖥️ 3. Install PyTorch (CPU build)

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

> 💡 **GPU Option:** If you have CUDA, replace the above with:
> ```powershell
> pip install torch --index-url https://download.pytorch.org/whl/cu121
> ```

---

## 📦 4. Install Hugging Face + FinGPT Dependencies

```powershell
pip install transformers datasets accelerate peft sentencepiece safetensors scikit-learn tqdm huggingface-hub
```

---

## 📥 5. Clone & Install FinGPT (Pinned Commit)

```powershell
git clone https://github.com/AI4Finance-Foundation/FinGPT.git
cd FinGPT
git checkout 4e53f8d7f3d3342d7f9cfa9fb6681609e9703dea
pip install -e .
cd ..
```

---

## 🔑 6. Hugging Face Login & LLaMA Access

1. **Generate Token:** Go to [Hugging Face Tokens](https://huggingface.co/settings/tokens) → New token (Read access).  
2. **Accept License:** On the LLaMA model page (e.g., `meta-llama/Llama-2-7b-chat-hf`), click **"Access repository"** and accept the terms.  
3. **Login:**  
```powershell
huggingface-cli login
```
Paste your token when prompted.

Or set in `.env`:
```
HUGGINGFACE_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxx
```

---

## ⚙️ 7. Environment Variables Setup

Create a `.env` file in the project root:

```env
# === Azure OpenAI ===
AZURE_OPENAI_ENDPOINT=
AZURE_OPENAI_API_KEY=
AZURE_OPENAI_API_VERSION=2024-xx-xx

# === Bing Search (optional) ===
BING_SEARCH_API_KEY=

# === Hugging Face (required for LLaMA) ===
HUGGINGFACE_TOKEN=

# === FinGPT (optional overrides) ===
FIN_GPT_MODEL_PATH=models/finbert
```

> **Important:** Add `.env` to `.gitignore` to avoid committing secrets.

---

## ▶️ 8. Run the Project

```powershell
python your_autogen_script.py --ticker AAPL
```

---

## ✅ Summary

Following these steps ensures:
- Python 3.11 compatibility
- Stable AutoGen + Azure integration
- Reproducible FinGPT installation (pinned commit)
- Proper LLaMA model access via Hugging Face
