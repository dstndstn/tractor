Code structure of the Tractor
=============================

Miscellaneous notes on control flow and call stacks.

::

    Tractor.optimize()
        Tractor.getDerivs()
            Image.getParamDerivatives()
            Tractor.getModelImage()
            Tractor._getSourceDerivatives()
                Source.getParamDerivatives()
        Tractor.getUpdateDirection()
            Tractor.getChiImage()
            scipy...lsqr()
        Tractor.tryUpdates()
            Tractor.getLogProb()


::

   Tractor.getLogProb()
        Tractor.getLogPrior()
        Tractor.getLogLikelihood()
            Tractor.getChiImages()
                Tractor.getModelImages()
                    Tractor.getModelImage()
                        Image.getSky()
                        Tractor.getModelPatch()
                            Tractor.getModelPatchNoCache()
                                Source.getModelPatch()



