from distutils.core import setup

setup(name='Sloth',
    version='2.0.2',
    description='Time series tools for classification, forecasting and clustering',
    packages=['Sloth'],
    install_requires=['scikit-learn >= 0.18.1',
        'fastdtw>=0.3.2',
        'pandas >= 0.19.2',
        'scipy >= 0.19.0',
        'numpy>=1.14.2',
        'matplotlib>=2.2.2',
        'statsmodels>=0.9.0',
        'pyramid-arima>=0.6.5',
        'cython>=0.28.5',
        'tslearn>=0.1.21',
        'hdbscan>=0.8.18'],
    include_package_data=True,
)