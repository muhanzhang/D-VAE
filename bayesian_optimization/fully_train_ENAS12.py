import os
import pdb
import numpy as np



gpu_id = 0

# DVAE_fast
arcs_scores = '''
2 2 0 4 0 0 1 0 0 0 0 1 0 1 0 4 1 1 0 1 1 0 0 1 0 0 1 0 2 0 0 0 1 1 0 1 1 0 1 0 1 1 0 0 0 3 0 1 0 1 0 0 1 0 1 0 0 1 1 0 0 1 1 0 0 0 0 1 0 1 1 1 0 0 0 1 1 0 0.755
1 2 0 3 0 0 1 0 0 0 3 1 1 1 0 0 0 1 0 1 1 1 0 0 0 1 1 0 0 0 0 0 1 1 0 1 1 0 0 1 0 0 0 1 0 1 1 0 0 1 0 0 1 0 1 5 0 0 1 0 1 0 1 0 0 0 0 1 0 0 0 1 0 0 0 1 0 0 0.7536
2 0 0 2 0 0 4 0 0 0 5 1 0 0 1 4 1 1 0 0 1 0 0 1 0 1 1 0 3 0 0 0 0 1 0 1 2 0 0 1 0 1 0 0 0 3 0 1 1 1 1 0 1 0 0 3 1 0 1 1 1 1 1 0 0 0 3 0 0 0 0 0 0 0 1 0 0 0 0.7534
1 2 0 4 0 0 4 0 0 0 4 0 1 0 1 2 0 1 1 0 1 2 1 0 0 1 1 1 3 0 0 0 0 1 1 0 0 0 0 0 0 0 0 1 1 5 0 1 0 1 1 0 1 0 0 3 0 0 1 0 1 1 0 0 0 0 2 1 1 0 0 1 1 1 0 0 0 0 0.753
3 5 0 3 0 0 3 0 0 0 2 0 0 0 1 5 1 1 1 0 1 4 0 0 1 0 1 0 5 0 0 0 1 1 0 1 0 0 0 0 0 0 0 1 0 1 1 0 1 1 0 0 1 1 1 1 1 0 1 1 1 0 1 0 0 0 2 0 1 1 1 1 0 1 0 0 1 0 0.7524
0 2 0 3 0 0 3 0 0 0 3 0 1 1 0 4 1 1 0 1 1 0 0 1 0 0 1 0 3 0 0 0 0 1 0 1 5 0 1 0 0 1 0 1 0 2 1 1 1 0 0 0 1 0 1 1 1 1 1 0 1 1 0 0 0 1 1 0 0 1 0 0 0 0 0 1 0 1 0.7522
1 1 1 5 0 0 0 0 0 1 2 0 0 0 1 0 0 1 1 0 0 0 0 1 0 1 1 0 4 0 0 0 0 1 1 1 2 0 0 0 0 1 0 1 0 1 0 1 1 0 0 0 1 1 0 5 1 0 1 1 1 1 0 0 0 0 2 0 0 1 1 1 0 0 1 1 0 0 0.7522
0 4 0 2 0 0 3 0 0 0 0 1 1 0 0 1 0 1 1 1 1 0 0 0 0 0 1 0 3 0 0 0 0 1 0 1 1 0 1 1 0 1 0 0 0 5 0 1 1 1 0 0 0 0 1 5 0 0 1 0 1 1 1 0 0 0 5 0 0 0 1 1 0 0 0 0 0 0 0.7518
0 2 0 4 0 0 1 0 0 0 0 0 1 1 0 4 1 1 0 1 1 2 0 1 0 1 1 0 3 0 0 0 0 1 0 1 5 0 1 1 0 1 0 0 0 4 0 1 1 1 1 0 1 0 1 5 0 0 1 0 1 0 1 0 0 0 4 0 1 1 0 0 0 1 0 1 1 0 0.7516
3 0 0 0 0 0 5 0 0 0 5 1 0 0 0 4 1 1 1 1 1 3 0 0 0 1 1 0 3 0 0 0 0 1 0 1 5 0 0 0 0 1 0 1 0 5 0 0 1 0 1 0 0 0 1 1 0 0 1 1 1 1 0 1 0 0 5 1 0 1 1 0 0 0 0 1 0 0 0.7512
2 2 0 0 0 0 5 0 0 1 0 0 1 0 0 2 0 1 0 1 1 0 1 0 1 0 1 0 3 0 0 0 0 1 0 1 2 0 0 0 1 0 0 1 0 3 0 0 1 1 1 0 1 0 1 1 1 0 1 0 1 0 1 0 1 0 2 1 1 1 0 1 0 0 0 1 1 1 0.751
2 0 0 0 0 0 3 0 0 0 0 0 0 0 1 5 0 1 1 0 1 0 0 1 0 0 1 0 4 0 0 0 0 1 0 1 2 0 1 0 0 1 0 1 0 3 1 0 1 1 1 0 1 0 0 1 1 0 1 1 1 0 1 0 0 1 1 0 0 0 1 1 1 0 0 1 0 1 0.7506
4 5 0 4 0 0 2 0 0 0 5 1 0 0 1 5 0 1 0 0 1 1 0 0 0 0 1 0 3 0 1 0 0 1 1 1 1 0 0 0 1 1 0 0 0 2 0 0 0 0 1 0 0 0 1 2 0 0 0 1 1 1 1 1 0 0 0 1 1 0 0 1 0 0 1 1 0 0 0.7502
0 0 0 4 0 0 0 0 0 0 4 0 0 1 0 3 0 1 0 1 1 3 0 0 0 1 1 0 3 0 0 0 0 1 0 1 5 0 0 1 0 1 0 0 0 3 0 0 1 1 0 0 1 0 1 5 1 0 1 0 1 1 0 0 0 0 3 0 1 1 1 1 0 0 1 0 0 0 0.7502
3 3 0 2 0 0 3 0 0 0 2 1 0 0 1 0 0 1 0 0 1 2 1 1 0 0 1 0 3 0 0 0 0 1 1 1 2 0 1 1 0 1 0 0 0 5 0 1 0 1 1 0 0 1 0 5 0 0 0 1 1 0 1 0 0 0 2 1 0 0 0 1 1 0 1 0 0 1 0.7502
'''

enas_pos = '../software/enas/'

arcs = [x.split('.')[0][:-2] for x in arcs_scores.strip().split('\n')]
print(arcs)

scores = []

for arc in arcs:
    print('Fully training ENAS architecture ' + arc)
    save_appendix = ''.join(arc.split()) + '_256filters'
    if not os.path.exists(enas_pos + 'outputs_' + save_appendix):
        pwd = os.getcwd()
        os.chdir(enas_pos)
        os.system('CUDA_VISABLE_DEVICES={} ./scripts/custom_cifar10_macro_final_12.sh'.format(gpu_id) + ' "' + arc + '" ' + save_appendix + " 256")
        os.chdir(pwd)
    with open(enas_pos + 'outputs_' + save_appendix + '/stdout', 'r') as f:
        last_line = f.readlines()[-1]
        scores.append(last_line)


new_arcs_scores = [x+' '+y for x, y in zip(arcs_scores.strip().split('\n'), scores)]
new_arcs_scores = ''.join(new_arcs_scores)
print()
print('Fully trained architecture, WS acc, and test acc:')
print(new_arcs_scores)
print('Average score is {}'.format(np.mean([float(x.split()[1]) for x in scores])))

pdb.set_trace()
