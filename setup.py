from distutils.core import setup

setup(name='Gecko',
    version='1.0.0',
    description='Gecko - Graph Embeddings for Classification and Knowledge Observation',
    packages=['Gecko'],
    install_requires=['scikit-learn >= 0.18.1',
        'gem>=1.0.0',
        'pandas >= 0.19.2',
        'scipy >= 0.19.0',
        'numpy>=1.14.2',
        'matplotlib>=2.2.2',
        'hdbscan>=0.8.18'],
    include_package_data=True,
)