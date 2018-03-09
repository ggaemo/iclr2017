import re


def make_script(data, input_dim, model_type, encoder_list, decoder_list, fc_dim,
                z_dim_list, beta_s_list, beta_c_list,
                discriminator_list,
                interval_list,
                num_split, d_lambda_list):

    alternate_list =['', '-alternate']
    style_list = ['','-style_included']
    alternate_list = ['']
    style_list = ['-style_included']
    deterministic_c_list = ['-deterministic_c']
    total_configs = len(discriminator_list) * len(z_dim_list) * len(beta_s_list) * len(
        beta_c_list) * len(interval_list) * len(alternate_list) * len(style_list) * len(
        d_lambda_list) * len(deterministic_c_list)
    print(total_configs)

    num_config_per_partition = int(total_configs / num_split)

    num_config_list = [num_config_per_partition] * num_split

    if total_configs % num_split != 0:
        num_config_list[-1] += total_configs % num_split

    f_list = [open('job_{}_part_{}.sh'.format(data, split_num), 'w') for
              split_num in range(num_split)]

    # f.write('#!/usr/bin/env bash\n')

    file_num = 0
    count = 0

    encoder = ' '.join([str(x) for x in encoder_list])
    decoder = ' '.join([str(x) for x in decoder_list])
    for interval in interval_list:
        for z_dim in z_dim_list:
            for beta_s in beta_s_list:
                for beta_c in beta_c_list:
                    for discrim in discriminator_list:
                        discrim = ' '.join([str(x) for x in discrim])
                        for alternate in alternate_list:
                            for style in style_list:
                                for d_lambda in d_lambda_list:
                                    for deterministic_c in deterministic_c_list:
                                        if count >= num_config_list[file_num]:
                                            count = 0
                                            file_num += 1
                                        f = f_list[file_num]
                                        config = 'python train_beta_vae_discrim.py -data_dir {} ' \
                                                 '-input_dim {} ' \
                                                 '-m {} -output_prob bernoulli ' \
                                                 '-enc {} -dec {} ' \
                                                 '-fc_dim {} -z_dim {} ' \
                                                 '-final_act_fn linear -beta_s {} -beta_c {} ' \
                                                 '-learning_rate 1e-4 -optimizer Adam ' \
                                                 '-discrim {} ' \
                                                 '-context_class_num 3 ' \
                                                 '{} -interval {} {} -d_lambda {} ' \
                                                 '{} -option ' \
                                                 'gan_discrim_multiple_loss_8_16_4' \
                                                 .format(data,
                                                                                  input_dim,
                                                                                  model_type,
                                                                          encoder,
                                                                          decoder,
                                                                          fc_dim,
                                                                                  z_dim,
                                                                                  beta_s,
                                                                                  beta_c,
                                                                                  discrim,
                                                                                alternate,
                                                                                  interval,
                                                                               style, d_lambda,
                                                             deterministic_c
                                                                                  )


                                        config = re.sub('\s+', ' ', config)
                                        config = config + '\n'
                                        f.write(config)
                                        count += 1

    [f.close() for f in f_list]


data = 'dsprite'
input_dim = 64
model_type = 'FC'
encoder_list = [1200]
decoder_list = [1200, 1200, 4096]
fc_dim = 1200
z_dim_list = [10, 20]
beta_s_list = [1, 2, 4]
beta_c_list = [1]
discriminator_list = [[1200, 1200, 1200, 500]]
interval = [0]
num_split = 1
d_lambda_list = [1]

make_script(data, input_dim, model_type, encoder_list, decoder_list, fc_dim,
                z_dim_list, beta_s_list, beta_c_list,
                discriminator_list,
                interval,
                num_split, d_lambda_list)