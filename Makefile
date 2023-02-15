install: dirs jupyter

dirs:
	mkdir -p notebooks
	mkdir -p notebooks/data

requirements:
	pip install --no-cache-dir -r requirements.txt

jupyter: requirements
	jupyter contrib nbextension install --user
	jupyter nbextension enable code_prettify/code_prettify 
	jupyter nbextension enable toc2/main
	jupyter nbextension enable collapsible_headings/main

run:
	jupyter notebook --ip 0.0.0.0 --no-browser notebooks
