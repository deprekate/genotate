#!/usr/bin/env python3
# encoding: utf-8

import os
#from distutils.core import setup, Extension
from setuptools import setup, Extension, find_packages


def extension():
	#os.environ["CC"] = "gcc"
	#compile_args = ["-g -Wall -O2"]
	link_args	= ["-lm"]

	ext = Extension('genotate.windows',
				language='gcc',
				#extra_compile_args=compile_args,
				extra_link_args=link_args,
				include_dirs=[
							 '.',
							 '...',
							 os.path.join(os.getcwd(), 'include'),
				],
				library_dirs = [os.getcwd(),],
				sources = ['src/windows.c'])
	return ext

def readme():
	with open("README.md", "r") as fh:
		long_desc = fh.read()
	return long_desc

def get_version():
    with open("VERSION", 'r') as f:
        v = f.readline().strip()
        return v

def main():
	setup (
		name = 'genotate',
		version = get_version(),
		author = "Katelyn McNair",
		author_email = "deprekate@gmail.com",
		description = 'A a tool to annotate microbial genomes',
		long_description = readme(),
		long_description_content_type="text/markdown",
		url =  "https://github.com/deprekate/genotate",
		scripts=['genotate.py'],
		classifiers=[
			"Programming Language :: Python :: 3",
			"License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
			"Operating System :: OS Independent",
		],
		python_requires='>3.5.2',
		packages=find_packages(),
		#install_requires=['fastpath>=1.4'],
		ext_modules = [extension()]
	)


if __name__ == "__main__":
	main()
