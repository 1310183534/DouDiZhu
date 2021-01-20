from distutils.core import setup, Extension
from Cython.Build import cythonize

if __name__ == '__main__':
	ext = Extension(
		'game',
		sources=['src/Action.cpp', 'src/Env.cpp', 'src/Game.cpp', 'src/utils.cpp', 'game.pyx'],
		language='c++',
		extra_compile_args=['-std=c++11']
	)
	setup(ext_modules=cythonize(ext, annotate=True, compiler_directives={'language_level': 3}))
