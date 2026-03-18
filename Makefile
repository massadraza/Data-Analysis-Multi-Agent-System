.PHONY: train serve frontend test docker-up docker-down lint

train:
	python scripts/train.py

serve:
	uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

frontend:
	streamlit run frontend/app.py --server.port 8501

test:
	pytest tests/ -v

lint:
	ruff check .

docker-up:
	docker compose up --build -d

docker-down:
	docker compose down
