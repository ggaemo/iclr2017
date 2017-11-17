
def make_script(data, num_batch, train_data_size, encoder_list, part_num_list,
                part_dim_list,
                decoder_list,
                test_data_size, test_batch_size, discriminator_list, option, num_split
                ):
    total_configs = len(encoder_list) * len(part_dim_list) * len(discriminator_list) * \
                    len(part_num_list) * 4 * 2 # 3은
    # activation 2는
    # is_bn

    num_config_per_partition = int(total_configs / num_split)

    num_config_list = [num_config_per_partition] * num_split

    if total_configs % num_split != 0:
        num_config_list[-1] += total_configs % num_split

    num_cycle = int(train_data_size / num_batch) + 1

    # f = open('job_{}_{}_part_{}.sh'.format(data, option, partition_num), 'w')

    f_list = [open('job_{}_{}_part_{}.sh'.format(data, option, split_num), 'w') for
              split_num in range(num_split)]

    # f.write('#!/usr/bin/env bash\n')

    file_num = 0
    count = 0
    for encoder, decoder in zip(encoder_list, decoder_list):
        encoder = ' '.join([str(x) for x in encoder])
        decoder = ' '.join([str(x) for x in decoder])
        for part_dim in part_dim_list:
            for part_num in part_num_list:
                for discrim in discriminator_list:
                    discrim = ' '.join([str(x) for x in discrim])
                    for disentangled_activation_fn in ['relu', 'tanh', 'sigmoid',
                                                       'linear']:
                        for is_bn in [' -bn', '']:
                            if part_dim == 2 and part_num == 2:
                                z_plot = '-z_plot'
                            else:
                                z_plot = ''
                            if count > num_config_list[file_num]:
                                count = 0
                                file_num += 1
                            f = f_list[file_num]
                            config = 'python train_disentagled_vae.py -enc {} -part_num {} ' \
                                     '-part_dim {} -dec {} -data_dir {} -test_data_size {} ' \
                                     '-test_batch_size {} -discrim {} -num_cycle {}  ' \
                                     '-disen_act_fn {} {} {} -option {} \n'.format(encoder,
                                     part_num, part_dim, decoder,
                                                data,
                                               test_data_size, test_batch_size, discrim,
                                               num_cycle, disentangled_activation_fn, z_plot,
                                    is_bn ,option)
                            f.write(config)
                            count += 1

    [f.close() for f in f_list]

# data = 'mnist'
# encoder_list = [[500, 250], [500, 500, 500], [500, 250, 250], [500, 250, 125]]
# decoder_list = [[250, 500], [500, 500, 500], [250, 250, 500], [125, 250, 500]]
# part_num_list = [2, 10, 2, 5, 2, 5]
# part_dim_list = [2, 2, 10, 10, 50, 50]
# train_data_size = 20
# test_data_size = 10000
# test_batch_size = 10000
# discriminator_list = [[5, 5], [40, 40], [25, 25], [40, 40], [25, 25], [25, 10]]



data = 'mnist'
encoder_list = [[500, 500, 500], [500, 250, 250], [500, 250, 125]]
decoder_list = [[500, 500, 500], [250, 250, 500], [125, 250, 500]]
part_num_list = [2, 5, 20]
part_dim_list = [2, 10, 50]
train_data_size = 50000
test_data_size = 10000
test_batch_size = 10000
discriminator_list = [[5, 5], [20, 20], [40, 40]]
num_split = 4

mnist_args = {'data' : data, 'num_batch' : 256,'encoder_list' : encoder_list,
              'decoder_list' : decoder_list,
              'part_num_list' : part_num_list, 'part_dim_list' : part_dim_list,
              'train_data_size' : train_data_size, 'test_data_size' : test_data_size,
              'test_batch_size' : test_batch_size, 'discriminator_list'
              : discriminator_list, 'option' : 'no_reg',
              'num_split' : num_split}
make_script(**mnist_args)