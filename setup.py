import setuptools


setuptools.setup(
    name="bert-text-summarizer",
    version="0.0.2",
    author="David Brown",
    author_email="davewbrwn@gmail.com",
    description="A BERT-based text summarization tool",
    url="https://github.com/david-wb/bert-text-summarizer",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    entry_points={
        'console_scripts': ['bert-text-summarizer=bert_text_summarizer.cli.main:main'],
    },
    install_requires=[
        'tf-nightly',
        'tf-models-official',
        'tensorflow-hub>=0.7.0',
        'sentencepiece==0.1.85',
        'nltk>=3.5',
    ],
)
