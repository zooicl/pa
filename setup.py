from setuptools import setup, find_packages

VERSION = 0.1
setup(
    name='pa_algorithm',
    version=VERSION,
    packages=find_packages(exclude=['tests']),
    # packages=['classification', 'helper'],
    # package_dir={'classification': 'src/classification', 'helper': 'src/helper'},
    install_requires=[
        'scikit-learn',
    ],
    license='MIT License',
    author='zooicl',
    author_email='aiden.hyochan.song@gmail.com',
    url='https://github.com/zooicl/pa',
    description='Warping scikit-learn algorithms',
    # long_description='Excelpy can add new sheets, copy sheets, delete sheets, and edit string and number type datas.',
    keywords=['algorithm', 'sklearn', 'machine learning'],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
        'Topic :: Database',
        'Topic :: Office/Business',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
)
