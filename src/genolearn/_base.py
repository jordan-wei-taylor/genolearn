class Dict(dict):
    """
    Dictionary wrapper with additional `rank` method
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def rank(self, ascending = False):
        """
        Returns
        -------
        dict
            Dictionary with same keys as parent dictionary. Values have been computed using
            numpy.argsort(value, axis = -1)
        """
        ret = {}
        for key in self:
            rank     = self[key].argsort(axis = -1)
            ret[key] = rank if ascending else rank[::-1]
        return ret
