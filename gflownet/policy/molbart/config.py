import os


DEFAULT_SMILES_VOCAB_PATH = os.path.join(os.path.dirname(__file__), "bart_vocab.txt")
DEFAULT_SELFIES_VOCAB_PATH = os.path.join(os.path.dirname(__file__), "selfies.txt")
DEFAULT_SMILES_CHEM_TOKEN_START = 6
DEFAULT_SELFIES_CHEM_TOKEN_START = 6
REGEX_SMILES = "\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9]"
REGEX_SELFIES = "\[[^\]]+]"