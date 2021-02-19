from setuptools import setup
import seq2att

setup(name='seq2att',
	version=seq2att.__version__,
	description='seq2att is a command line interface to train a Read2Pheno model on customized 16S rRNA dataset.',
	url='https://github.com/z2e2/seq2att',
	author='Zhengqiao Zhao',
        author_email='zz374@drexel.edu',
	classifiers=['Development Status :: 1 - Beta',
	'Environment :: Console',
	'Intended Audience :: Science/Research',
	'License :: OSI Approved :: BSD License',
	'Natural Language :: English',
	'Operating System :: POSIX :: Linux',
	'Topic :: Scientific/Engineering :: Bio-Informatics'],
        license='BSD',
	packages=['seq2att'],
        python_requires='==3.5.4',
	install_requires=['matplotlib>=2.1.1','pandas>=0.20.3',
                      'numpy>=1.13.3','biopython>=1.71',
                      'tensorflow==1.9.0', 'keras==2.2.2',
                      'scikit-learn>=0.19.1', 'scipy>=1.0.0'
                     ],
	entry_points = {
        'console_scripts': ['seq2att=seq2att.seq2att:main'],
    },
	zip_safe=False)
