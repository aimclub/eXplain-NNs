format:
	add-trailing-comma ./**/*.py
	brunette ./**/*.py
	isort .

check:
	brunette ./**/*.py --check
	isort . --check
	flake8 .
	