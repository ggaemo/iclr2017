import itertools

num_part_list = [2, 3]
num_output_list = [10, 50]


def make_script(data, num_batch, num_train_size, num_part_list, num_output_list, num_comb,
                reg_w_list,
                delta_list, num_target_class, test_data_size,
                test_batch_size, model_option):
    num_part_list_comb = itertools.combinations_with_replacement(num_part_list, num_comb)
    num_output_list_comb = itertools.combinations_with_replacement(num_output_list,
                                                                   num_comb)

    num_cycle = int(num_train_size / num_batch) + 1
    f = open('job.sh', 'w')
    f.write('#!/usr/bin/env bash\n')
    for config in itertools.product(num_part_list_comb, num_output_list_comb):
        num_part = ' '.join([str(x) for x in config[0]])
        num_output = ' '.join([str(x) for x in config[1]])
        data_dir = data

        for reg_w in reg_w_list:
            for reg_type in ['DotHuber']:
                if reg_type == 'DotHuber':
                    for delta in delta_list:
                        config = 'python train.py -b {} -np {} -no {} -reg {} -reg_w {} ' \
                                 '-bn -data_dir {} -test_data_size {} -test_batch_size {} ' \
                                 '-huber_delta {} -num_target_class {} ' \
                                 '-num_cycle {} -option {}\n'.format(
                                    num_batch,
                                    num_part,
                                    num_output,
                                    reg_type,
                                    reg_w,
                                    data_dir,
                                    test_data_size,
                                    test_batch_size,
                                    delta,
                                    num_target_class,
                                    num_cycle,
                                    model_option)
                        f.write(config)
                else:
                    config = 'python train.py -b {} -np {} -no {} -reg {} -reg_w {} ' \
                             '-bn -data_dir {} -test_data_size {} -test_batch_size {} ' \
                             '-num_target_class {} -num_cycle {} -option {}\n'.format(
                        num_batch,
                        num_part,
                        num_output,
                        reg_type,
                        reg_w,
                        data_dir,
                        test_data_size,
                        test_batch_size,
                        num_target_class,
                        num_cycle,
                        model_option)
                    f.write(config)

    f.close()


cifar_10_args = ('cifar-10', 256, 50000, num_part_list, num_output_list, 3, [1.0, 5.0,
                                                                             10.0],
                 [1.0, 5.0,10.0], 10, 10000, 10000, 'attention'

                 )
make_script(*cifar_10_args)