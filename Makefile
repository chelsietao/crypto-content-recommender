.PHONY: install data test pipeline app clean

install:
	python -m pip install -r requirements.txt

data:
	python -m src.data_pipeline

test:
	python -m pytest -q

pipeline:
	python -m src.run_pipeline

app:
	streamlit run app.py

clean:
	find src tests -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache

.PHONY: neural neural-check
neural:
	python -m src.run_neural

neural-check:
	python -m src.run_neural --verify-only

.PHONY: service service-benchmark
service:
	python -m src.serve_neural

service-benchmark:
	python -m src.benchmark_service
