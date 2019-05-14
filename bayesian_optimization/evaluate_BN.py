import numpy as np
import os

class Eval_BN(object):
    def __init__(self, save_dir, R_script='compute_score.R'):
        self.save_dir = save_dir
        self.R_script = R_script

    def eval(self, input_string):
        input_matrix = np.array([int(x) for x in input_string.split()]).reshape(8, 8)
        tmp_file = os.path.join(self.save_dir, 'temp_BN_matrix')
        np.savetxt(tmp_file, input_matrix)
        return self.compute_score(tmp_file)

    def compute_score(self, input_file, output_file=None):
        if output_file is None:
            output_file = input_file + "_score"
        cmd = 'Rscript compute_score.R %s %s' % (input_file, output_file)
        os.system(cmd)
        score = np.loadtxt(output_file, ndmin=1)
        return float(score[0])
