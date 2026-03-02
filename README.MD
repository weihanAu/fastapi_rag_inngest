# 1. get a virtual python env
uv venv

# 2. run a inngest server using node
npx inngest-cli@latest dev -u http://127.0.0.1:8000/api/inngest --no-discovery

# 3. run fastapi server
uv run uvicorn main:app --reload