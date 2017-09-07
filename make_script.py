import itertools

num_part_list = [2, 3]
num_output_list = [10, 50]




def make_script(num_part_list, num_output_list, num_comb, reg_w_list,
                delta_list):
    num_part_list_comb = itertools.combinations_with_replacement(num_part_list, num_comb)
    num_output_list_comb = itertools.combinations_with_replacement(num_output_list,
                                                                   num_comb)
    f = open('job.sh', 'w')
    f.write('#!/usr/bin/env bash\n')
    for config in itertools.product(num_part_list_comb, num_output_list_comb):
        num_part = ' '.join([str(x) for x in config[0]])
        num_output = ' '.join([str(x) for x in config[1]])
        data_dir = 'higgs'
        test_data_size = 10000
        test_batch_size = 10000

        for reg_w in reg_w_list:
            for reg_type in ['dotProductHuber']:
                if reg_type == 'dotProductHuber':
                    for delta in delta_list:

                        f.write('python train.py -np {0} -no {1} -reg {2} -reg_w {'
                                '3} -bn '
                                '-data_dir '
                                '{4} -test_data_size {5} -test_batch_size {6} '
                                '-huber_delta {7} -num_target_class {8}\n'.format(
                            num_part,
                                                                                        num_output,
                                                                                        reg_type,
                                                                                        reg_w,
                                                                                        data_dir,
                                                                                        test_data_size,
                                                                                        test_batch_size,
                        delta, 2))
                else:
                    f.write('python train.py -np {0} -no {1} -reg {2} -reg_w {'
                            '3} -bn '
                            '-data_dir '
                            '{4} -test_data_size {5} -test_batch_size {6} -huber_delta {'
                            '7} '
                            '-num_target_class {8}'
                            '\n'.format(
                        num_part,
                        num_output,
                        reg_type,
                        reg_w,
                        data_dir,
                        test_data_size,
                        test_batch_size,
                        1.0, 2))
    f.close()



make_script(num_part_list, num_output_list, 3, [1.0, 5.0, 10.0], [1.0, 5.0, 10.0])