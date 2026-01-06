"""Module setuptools script."""
from setuptools import setup

# Read long description from README
try:
    with open('README.md', encoding='utf-8') as f:
        long_description = f.read()
except FileNotFoundError:
    long_description = ""

setup(
    name='autoLeague',
    version='1.1.0',
    description='League of Legends replay data extraction library with LCU API support',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='hanueluni1106',
    author_email='sykim1106@naver.com',
    license='GPL-3.0',
    keywords=[
        'League of Legends',
        'Machine Learning',
        'Reinforcement Learning',
        'Supervised Learning',
        'TLoL',
        'Dataset Generation',
        'Data Scraping',
        'Replay Extraction',
        'LCU API'
    ],
    url='https://github.com/league-of-legends-replay-extractor/pyLoL',
    packages=[
        'autoLeague',
        'autoLeague.bin',
        'autoLeague.replays',
        'autoLeague.dataset',
        'autoLeague.preprocess',
        'autoLeague.utils'
    ],
    install_requires=[
        'absl-py',
        'requests',
        'psutil',
        'tqdm',
        'python-dotenv',
        'pandas',
        'mss',
        'opencv-python',
        'pyautogui'
    ],
    python_requires='>=3.7',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'Operating System :: Microsoft :: Windows',
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ]
)
