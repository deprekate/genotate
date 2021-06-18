
all:
	python3 setup.py install --user

clean:
	rm -fr build/
	rm -fr dist/
	rm -fr genotate.egg-info/
	pip uninstall -y genotate
