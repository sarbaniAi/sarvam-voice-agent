# Yatra — Sovereign Voice Travel Assistant

**Sarvam AI + Databricks | 22 Indian Languages | 100% India-Hosted**

Yatra is a voice-enabled travel assistant that runs entirely on Indian infrastructure, demonstrating how Databricks can power sovereign AI applications where no data leaves the country. Users can ask travel queries in Hindi, English, Tamil, Telugu, or any of 22 Indian languages and get spoken responses back.

## Architecture

```
User Voice → [Sarvam Saaras v3 STT] → Text → [Sarvam-M LLM] → Response → [Sarvam Bulbul v2 TTS] → Audio
                   (India)                      (India)                         (India)
```

| Component | Model | Provider | Purpose |
|-----------|-------|----------|---------|
| Speech-to-Text | Sarvam Saaras v3 | Sarvam AI (India) | Converts voice to text, auto-detects language |
| LLM | Sarvam-M (24B) / Sarvam-30B | Sarvam AI (India) | Generates travel responses in user's language |
| Text-to-Speech | Sarvam Bulbul v2 | Sarvam AI (India) | Converts response to natural speech |
| App Hosting | Databricks Apps | Azure Central India | Hosts the Gradio web application |
| Model Serving | Databricks Model Serving | Azure Central India | Self-hosts Sarvam-30B (optional) |

## Supported Languages

Hindi, English, Tamil, Telugu, Kannada, Malayalam, Bengali, Gujarati, Marathi, Punjabi, Odia — and more via Sarvam's 22 Indian language support.

## Project Structure

```
sarvam-voice-agent/
├── app.py                      # Main Gradio app (Text + Voice tabs)
├── app.yaml                    # Databricks App configuration
├── requirements.txt            # Python dependencies
├── 01_deploy_sarvam_model.py   # Notebook: Deploy Sarvam-30B on Databricks Model Serving
└── README.md
```

## Quick Start

### Prerequisites

- A [Sarvam AI](https://www.sarvam.ai/) API key
- A [Databricks](https://www.databricks.com/) workspace 
- Databricks CLI installed (`pip install databricks-cli`)

### 1. Clone and configure

```bash
git clone https://github.com/sarbanimaiti/sarvam-voice-agent.git
cd sarvam-voice-agent
```

Edit `app.yaml` and set your Sarvam API key:

```yaml
env:
  - name: SARVAM_API_KEY
    value: "your-sarvam-api-key"
```

### 2. Deploy as Databricks App

```bash
# Authenticate with your Databricks workspace
databricks auth login <your-workspace-url>

# Upload files to workspace
databricks workspace import-dir . /Workspace/Users/<your-email>/sarvam-voice-agent --overwrite

# Create the app
databricks apps create yatra-voice-agent \
  --description "Sovereign Voice Travel Assistant" \
  --source-code-path /Workspace/Users/<your-email>/sarvam-voice-agent

# Deploy
databricks apps deploy yatra-voice-agent \
  --source-code-path /Workspace/Users/<your-email>/sarvam-voice-agent
```

### 3. Run locally (optional)

```bash
pip install -r requirements.txt
export SARVAM_API_KEY="your-sarvam-api-key"
python app.py
```

Open http://localhost:8000



## How It Works

**Text Flow:**
1. User types a query in any language
2. Language is auto-detected from script (Devanagari → Hindi, Tamil script → Tamil, etc.)
3. Sarvam-M LLM generates a response in the same language
4. Sarvam Bulbul TTS converts the response to audio with auto-play

**Voice Flow:**
1. User records audio via browser microphone (JavaScript MediaRecorder API)
2. Audio is sent as base64 to Sarvam Saaras STT
3. STT returns transcript + detected language
4. LLM generates response → TTS speaks it back

**LLM Fallback Chain:**
1. First tries Databricks Model Serving (`sarvam-30b-serving` endpoint)
2. Falls back to Sarvam API (`sarvam-m` model) if endpoint unavailable

## Sarvam Models Used

| Model | Type | Parameters | License | Notes |
|-------|------|-----------|---------|-------|
| [Sarvam-30B](https://huggingface.co/sarvamai/sarvam-30b) | LLM (MoE) | 30B total / 2.4B active | Apache 2.0 | Can be self-hosted on Databricks |
| Sarvam-M | LLM | 24B (Mistral-based) | API | Production-grade multilingual |
| Saaras v3 | STT | — | API | 22 languages, auto language detection |
| Bulbul v2 | TTS | — | API | Natural Indian voices |

## Why Data Sovereignty Matters

For Indian BFSI customers (banks, insurance, securities), regulatory bodies like RBI, SEBI, and IRDAI require that customer data stays within India. This architecture ensures:

- All AI processing happens in India (Azure Central India)
- No data is sent to US-hosted LLMs (OpenAI, Anthropic, etc.)
- Models can be fully self-hosted on Databricks for complete control
- Databricks Unity Catalog provides model governance and lineage

## License

MIT

## Credits

- [Sarvam AI](https://www.sarvam.ai/) — Indian language AI models
- [Databricks](https://www.databricks.com/) — AI platform, Apps, Model Serving
- Built by Sarbani Maiti, SSA — AI Product Group, Databricks APJ
