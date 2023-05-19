format:
	add-trailing-comma ./eXNN/**/*.py ./examples/**/*.py ./tests/*.py --py36-plus
	brunette ./eXNN/**/*.py ./examples/**/*.py ./tests/*.py
	isort .

check:
	brunette ./eXNN/**/*.py ./examples/**/*.py ./tests/*.py --check --config=setup.cfg
	flake8 .
	
