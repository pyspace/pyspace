""" Store only meta data but no real data (except from store state of nodes) """

import logging
import os
import yaml
from pySPACE.resources.dataset_defs.base import BaseDataset

class DummyDataset(BaseDataset):
    """ Class to store only meta data of collection

    This class overrides the 'store' method
    in a way that only the collection meta data files are stored.

    This type is intended to be passed to pySPACE as a result
    by the NilSinkNode.

    **Parameters**

        :dataset_md:
            The meta data of the current dataset.

            (*optional, default: None*)

    :Author: David Feess (david.feess@dfki.de)
    :Created: 2010/03/30
    """
    def __init__(self, dataset_md =  None):
        super(DummyDataset, self).__init__(dataset_md = dataset_md)

    def store(self, result_dir, s_format = "None"):
        if not s_format == "None":
            self._log("The format %s is not supported!"%s_format, level=logging.CRITICAL)
            return
        # Update the meta data
        try:
            import platform
            CURRENTOS = platform.system()
            if CURRENTOS == "Windows":
                import getpass
                author = getpass.getuser()
            else:
                import pwd
                author = pwd.getpwuid(os.getuid())[4]
        except:
            author = "unknown"
            self._log("Author could not be resolved.",level=logging.WARNING)
        self.update_meta_data({"type": "only output of individual nodes stored",
                                      "storage_format": s_format,
                                      "author" : author,
                                      "data_pattern": "no data stored"})

        # Store meta data
        BaseDataset.store_meta_data(result_dir,self.meta_data)