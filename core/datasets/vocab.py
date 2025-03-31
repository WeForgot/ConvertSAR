class Vocabulary(object):
    def __init__(self, layer_names=None, verbose=False):
        self.layer_to_idx = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2}
        self.idx_to_layer = {0: '<PAD>', 1: '<SOS>', 2: '<EOS>'}
        self.verbose = verbose
        if layer_names is not None:
            for layer in layer_names:
                cur_idx = len(self.layer_to_idx)
                self.layer_to_idx[layer] = cur_idx
                self.idx_to_layer[cur_idx] = layer
    
    def __len__(self):
        return len(self.layer_to_idx)
    
    def __repr__(self):
        to_return = ''
        for x in self.idx_to_layer:
            to_return += '{}: {}\n'.format(x, self.idx_to_layer[x])
        return to_return
    
    def __getitem__(self, idx):
        if isinstance(idx, str):
            if idx.isnumeric():
                idx = str(int(idx) + 1)
            if idx not in self.layer_to_idx:
                if self.verbose:
                    print('Adding {}'.format(idx))
                self.layer_to_idx[idx] = len(self.layer_to_idx)
                self.idx_to_layer[self.layer_to_idx[idx]] = idx
            return self.layer_to_idx[idx]
        elif isinstance(idx, int):
            return self.idx_to_layer[idx]
        else:
            raise ValueError('Vocabulary indices can only be strings or integers')