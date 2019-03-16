class TokenFilter:
    def __init__(self, stopword_file):
        self.stopwords = set()
        self.load_stopwords(stopword_file)

    def load_stopwords(self, file):
        """loads all stopwords into a set"""
        with open(file) as f:
            for row in f:
                self.stopwords.add(row.replace("\n", "").lower())

    def filter(self, token: str):
        """filters all stopwords and characters that are not relevant"""
        if token not in self.stopwords:
            return token
        else:
            return None
