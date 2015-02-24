Code structure of the Tractor
=============================

Miscellaneous notes on control flow and call stack.

::

    Tractor.optimize()
        Tractor.getDerivs()
            Image.getParamDerivatives()
            Tractor.getModelImage()
            Source.getParamDerivatives()
        Tractor.getUpdateDirection()
            Tractor.getChiImage()
            scipy...lsqr()
        Tractor.tryUpdates()
            Tractor.getLogProb()

