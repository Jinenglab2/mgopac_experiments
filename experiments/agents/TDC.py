from agents.MGOPAC import MGOPAC
from utils.dict import merge

class TDC(MGOPAC):
    def __init__(self, features, params):
        # TDC is just an instance of MGOPAC where beta = 0
        super().__init__(features, merge(params, { 'beta': 0 }))
