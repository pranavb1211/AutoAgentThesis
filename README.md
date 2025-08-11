# AutoAgent + FinGPT Setup

## 🚀 Setup Instructions

### 1️⃣ Clone this repository
```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
```

---

### 2️⃣ Create and activate a virtual environment
```bash
# Create venv
python -m venv tisenv

# Activate venv
# Windows:
tisenv\Scripts\activate
# macOS/Linux:
source tisenv/bin/activate
```

---

### 3️⃣ Install project dependencies

**Install AutoGen + Azure stack**
```bash
pip install autogen-agentchat autogen-core "autogen-ext[azure,openai]" asyncio python-dotenv openai tiktoken "aiohttp>=3.8.0" yfinance
```

**Install Hugging Face + FinGPT stack**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu  # Change for GPU if available
pip install transformers datasets accelerate peft sentencepiece safetensors scikit-learn tqdm huggingface-hub
```

---

### 4️⃣ Clone and install FinGPT
```bash
git clone https://github.com/AI4Finance-Foundation/FinGPT.git
cd FinGPT
git checkout 4e53f8d7f3d3342d7f9cfa9fb6681609e9703dea
pip install -e .
cd ..
```

---

### 5️⃣ Configure environment variables
Create a `.env` file in the project root:
```env
AZURE_OPENAI_ENDPOINT=...
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_API_VERSION=2024-xx-xx
BING_SEARCH_API_KEY=...
HUGGINGFACE_TOKEN=...             # Optional, only if model requires login
FIN_GPT_MODEL_PATH=models/finbert # Optional override
```

> **Important:** Never commit `.env`. Add it to `.gitignore`:
```bash
echo .env >> .gitignore
```

---

### 6️⃣ Run the project
```bash
python your_autogen_script.py --ticker AAPL
```

---

### 7️⃣ (Optional) Using GPU for PyTorch
If you have a CUDA-capable GPU, install the correct PyTorch build:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```
