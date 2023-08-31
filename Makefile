
all:
	python3 -m pip install ../genotate/

clean:
	rm -fr build/
	rm -fr dist/
	rm -fr genotate.egg-info/
	pip uninstall -y genotate

build:
	python3 -m build
	python3 -m twine upload dist/*
