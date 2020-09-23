class PostConstraint():
    def __init__(self, key, lower=None, upper=None):
        self.key = key
        self.lower = lower
        self.upper = upper

    def __getitem__(self, index):
        return [self.key, self.lower, self.upper][index]

    def __len__(self):
        return 3
