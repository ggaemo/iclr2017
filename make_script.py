import collections
import itertools

num_part_list = [2, 5, 10]
num_output_list = [20, 50, 100]

num_part_list = [2, 3]
num_output_list = [10, 20, 50]

num_part_list_comb = itertools.combinations_with_replacement(num_part_list, 2)
num_output_list_comb = itertools.combinations_with_replacement(num_output_list, 2)


f = open('job.sh', 'w')
f.write('#!/usr/bin/env bash\n')
for config in itertools.product(num_part_list_comb, num_output_list_comb):
    num_part = ' '.join([str(x) for x in config[0]])
    num_output = ' '.join([str(x) for x in config[1]])
    data_dir = 'cifar-10'
    test_data_size = 10000
    test_batch_size = 10000
    for reg_w in [1, 10, 100]:
        for reg_type in ['dotOrthogonalProduct', 'dotProductHuber']:
            f.write('python train.py -np {0} -no {1} -reg {2} -reg_w {'
                    '3} -bn '
                    '-data_dir '
                    '{4} -test_data_size {5} -test_batch_size {6}\n'.format(num_part,
                                                                            num_output,
                                                                            reg_type,
                                                                            reg_w,
                                                                            data_dir,
                                                                            test_data_size,
                                                                            test_batch_size))
f.close()

