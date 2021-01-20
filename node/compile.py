from distutils.core import setup, Extension
from Cython.Build import cythonize

if __name__ == '__main__':
	ext = Extension(
		'node',
		sources=['src/node.cpp', 'node.pyx'],
		language='c++',
		extra_compile_args=['-std=c++11']
	)
	setup(ext_modules=cythonize(ext, annotate=True, compiler_directives={'language_level': 3}))
