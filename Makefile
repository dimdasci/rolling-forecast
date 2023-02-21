install: dirs jupyter

## make project directories
dirs:
	mkdir -p notebooks
	mkdir -p notebooks/data
	mkdir -p models
	mkdir -p logs

## install local package and dependencies
requirements:
	pip install -U pip setuptools wheel
	pip install --no-cache-dir -r requirements.txt

## install jupyter extensions
jupyter: requirements
	jupyter contrib nbextension install --user
	jupyter nbextension enable code_prettify/code_prettify 
	jupyter nbextension enable toc2/main
	jupyter nbextension enable collapsible_headings/main

## run jupyter server
run:
	jupyter notebook --ip 0.0.0.0 --no-browser notebooks

## Format using Black
format: 
	black src

## Lint using flake8
lint:
	flake8 src
