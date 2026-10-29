from utils.io.targetparsing import load_cdte_fits, parse_target_intervals

class CdTeDataParsing:
    ''' This class takes draft data and saves versions parsed in time for each target '''

    CDTE_DICT = {
        'CdTe1': {'number': 1},
        'CdTe3': {'number': 3},
        'CdTe4': {'number': 4},
        'CdTe5': {'number': 5},
    }

    def __init__(self, cdte_filelist):
        for cdte, cdtekey in zip(cdte_filelist, list(self.CDTE_DICT.keys())):
            self.save_cdte_data(cdte, cdtekey)
            self.make_targetfiltered_cdte(cdtekey)

    def save_cdte_data(self, cdte, cdtekey):
        '''Saves CdTe data as an astropy Table and header info into CDTE_DICT'''
        data, header = load_cdte_fits(cdte)
        self.CDTE_DICT[cdtekey]['Data'] = data
        self.CDTE_DICT[cdtekey]['Header'] = header

    def make_targetfiltered_cdte(self, cdtekey):
        '''For each target, data for that time range is saved in CDTE_DICT'''
        data = self.CDTE_DICT[cdtekey]['Data']
        header = self.CDTE_DICT[cdtekey]['Header']
        
        parsed_targets = parse_target_intervals(data, header)
        self.CDTE_DICT[cdtekey].update(parsed_targets)