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
