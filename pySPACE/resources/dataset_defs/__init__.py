""" Store and load accumulations of datasets together with meta data

Datasets are normally several data sets arranged in several folders with
some additional meta data, giving information, e.g. about origin and type.
The *type* specifies witch dataset class should handle the data. When writing new
datasets it is important to follow this naming rules:

    - *type* and *module name* of the new dataset in lower-case with
      underscores to separate words, e.g. time_series and time_series.py
    - *class name* of the new dataset in camel-case and with ending
      'Dataset', e.g. TimeSeriesDataset

When stored as results of operations or operation chains,
datasets already have the needed format, but some
datasets are also able to transform data to the needed format.

Nevertheless, you will need a `metadata.yaml` file.
For more details have a look at: :ref:`storage`.

Datasets are mostly stored as results of an
:mod:`operation <pySPACE.missions.operations>`.
"""
