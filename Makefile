
all:
	python3 -m pip install ../genotate/ --user

clean:
	rm -fr build/
	rm -fr dist/
	rm -fr genotate.egg-info/
	pip uninstall -y genotate
