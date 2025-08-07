# CloudWalk Chatbot 💬

## Task
Build a chatbot that explains what CloudWalk is, its products, mission, and brand values.

- **Goal:** Build a chatbot that explains what CloudWalk is, its products (like InfinitePay), mission, and brand values.
- **Input:** User questions via chat interface.
- **Output:** Natural language answers (optionally Markdown or links).

**Requirements:**
- Retrieval-augmented generation (RAG) from public sources
- 3 sample conversations in README

## Solution

**How to use**:
- Clone this repository into your local development environment.
- Install ``uv`` with pip: ``pip install uv``
- Create your virtual environment with the ``uv venv .venv``.
- Initialize your virtual environment:
  - Linux/macOS: ``source .venv/bin/activate``
  - Windows (cmd.exe): ``.venv\Scripts\activate``
  - Windows (PowerShell): ``.venv\Scripts\Activate.ps1``
- Install the required libraries with ``uv pip install -r requirements.txt``.
- Create a huggingface at ``https://huggingface.co/settings/tokens``
- Run the desired app with ``python3 app.py``; 
- Paste the huggingface token;
- Acess the local URL:  http://127.0.0.1:7861;
- Or, acess the online chatbot at huggingface spaces: https://huggingface.co/spaces/k3ybladewielder/cloudwalk_chatbot

## Demo
<img src="demo.gif"> 
