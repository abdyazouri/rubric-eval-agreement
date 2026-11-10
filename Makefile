.PHONY: setup test run clean

setup:            ## install dependencies into the current environment
	pip install -r requirements.txt

test:             ## run the test suite (incl. the cross-checked kappa implementations)
	pytest -q

run:              ## generate synthetic data, score it, and write reports/agreement_report.md
	python -m rubric_eval.cli --items 300 --seed 7 --n-boot 2000

clean:
	rm -rf data reports __pycache__ rubric_eval/__pycache__ tests/__pycache__ .pytest_cache
