
# coding: utf-8

# In[1]:

import re
import argparse
import os
import time
import numpy as np
import utils
import tensorflow as tf
import re
from proposed_model import Beta_VAE_Discrim, Disentagled_VAE_FC_Discrim
# from tensorflow.python.client import timeline
import matplotlib.pyplot as plt
import matplotlib


def get_image(data_num):
    data = np.load("dsprite/data/data_batch_{}.npz".format(data_num))

    img = data['img'].astype(np.float32)
    latent = data['latent']
    img = np.expand_dims(img, 3)
    return img, latent

img_0, label_0 = get_image(0)
img_1, label_1 = get_image(1)
img_2, label_2 = get_image(2)

def save_plot_per_model(folder):

    input_dim = 64
    data_dir = 'dsprite'


    # In[9]:

    def make_bool(expr):
        return expr == 'True'


    # In[10]:

    model_name = re.match('model-([A-Z]+)_', folder).group(1)
    batch_size = 99
    encoder_layer = [1200]
    fc_dim = int(re.search('fc_dim-((\d(-)?)+)', folder).group(1))
    z_dim = int(re.search('z_dim-(\d+)', folder).group(1))
    beta_s = float(re.search('beta-(\d\.\d)_\d\.\d', folder).group(1))
    beta_c = float(re.search('beta-\d\.\d_(\d\.\d)', folder).group(1))
    decoder_layer = [1200, 1200, 4096]
    final_act_fn = 'linear'
    output_prob = 'bernoulli'
    learning_rate = float(re.search('l_rate-(\d.\d+)', folder).group(1))
    optimizer = re.search('optim-([A-Za-z]+)', folder).group(1)
    discriminator_layer = [int(x) for x in re.search('discrim-(\d+)', folder).group(1).split('-')]
    alter = make_bool(re.search('alter-([A-Za-z]+)', folder).group(1))
    style_included =  make_bool(re.search('style-([A-Za-z]+)', folder).group(1))
    num_partition = 2
    interval = float(re.search('interval-(\d)', folder).group(1))
    discrim_lambda = float(re.search('discrim_lambda-(\d+)', folder).group(1))
    context_class_num = 3
    option = re.search('option_(.*)', folder)
    if option:
        option = option.group(1)


    # In[11]:

    if model_name == 'FC':
        model = Disentagled_VAE_FC_Discrim

    save_dir = data_dir + '/summary/' + folder +'/'


    if final_act_fn == 'relu':
        final_act_fn = tf.nn.relu
    elif final_act_fn == 'tanh':
        final_act_fn = tf.nn.tanh
    elif final_act_fn == 'sigmoid':
        final_act_fn = tf.nn.sigmoid
    elif final_act_fn == 'linear':
        final_act_fn = None

    if model_name == 'CNN':
        tmp = [encoder_layer[i:i+3] for i in range(0, len(encoder_layer), 3)]
        encoder_layer = tmp

        tmp = [decoder_layer[i:i+3] for i in range(0, len(decoder_layer), 3)]
        decoder_layer = tmp


    # In[13]:

    tf.reset_default_graph()


    # In[14]:

    input_img = tf.placeholder(tf.float32, [None, 64, 64, 1])
    input_label = tf.placeholder(tf.int32, [None, 5])
    train_inputs = {'img': input_img, 'latent' : input_label}


    # In[15]:

    with tf.variable_scope('Model'):
        trn_model = model(
            encoder_layer, decoder_layer, input_dim, fc_dim, z_dim,
                     False, batch_size, train_inputs, final_act_fn, beta_s, beta_c,
                     output_prob,
                     learning_rate, optimizer, num_partition, discriminator_layer,
                     context_class_num, style_included, discrim_lambda)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    config.allow_soft_placement=True


    # In[16]:

    sess = tf.Session(config=config)
    tf.set_random_seed(1)

    saver = tf.train.Saver()


    def find_best_recon_ckpt(save_dir):
        all_ckpt = os.listdir(save_dir)
        recon_ckpt = [x for x in all_ckpt if re.match('model.ckpt-\d+.meta', x)]
        best_recon = sorted([int(re.match('model.ckpt-(\d+).meta', x).group(1)) for x in recon_ckpt])
        return 'model.ckpt-'+str(best_recon[-1])

    best_recon_ckpt = find_best_recon_ckpt(save_dir)
    path = save_dir + best_recon_ckpt


    # In[35]:

    saver.restore(sess, path)


    def get_latent(img):
        if img.ndim == 3:
            img = np.expand_dims(img, 0)
        elif img.ndim == 2:
            img = np.expand_dims(img, 0)
            img = np.expand_dims(img, 3)

        recon, m_c, m_s, sig_c, sig_s, v_c, v_s = sess.run([trn_model.recon,
                                                      trn_model.z_m_c,trn_model.z_m_s,
                                                      trn_model.z_log_sigma_sq_c,trn_model.z_log_sigma_sq_s,
                                                      trn_model.z_v_c,trn_model.z_v_s
                                                           ],
                                                     feed_dict={input_img : img})
        return recon, m_c, m_s, np.exp(sig_c), np.exp(sig_s), v_c, v_s


    # In[80]:

    def get_span_matrix(z_latent, span_min_val, span_max_val, span_len):
        z_latent_span = np.tile(z_latent, span_len * z_dim).reshape(-1, z_dim)
        span_value_range = np.linspace(span_min_val, span_max_val, span_len)
        for i in range(z_dim):
            for j, val in zip(range(span_len), span_value_range):
                z_latent_span[i * span_len + j , i] = val
        return z_latent_span, span_value_range


    # In[81]:

    def merge_latent(span_matrix, fixed_latent, fixed_type):
        tiled = np.tile(fixed_latent, span_matrix.shape[0]).reshape(-1, z_dim)
        if fixed_type == 'c':
            merged = np.hstack((tiled, span_matrix))
        elif fixed_type == 's':
            merged = np.hstack((span_matrix, tiled))
        return merged


    # In[82]:

    def get_recon(z_latent):
        return np.squeeze(sess.run(trn_model.recon, feed_dict={trn_model.latent : z_latent}))


    # In[83]:

    def plot_figures(figures, span_value_range, title, text, cmap=plt.gray()):
        plt.suptitle(title +'_' + text, fontsize=20)
        z_values = span_value_range
        plt.subplots_adjust(wspace=0.05, hspace=0.05)
        span_len = len(span_value_range)
        z_dim = int(figures.shape[0] / span_len)

        for i in range(span_len):
            for j in range(z_dim):
                fig = figures[j * span_len + i]
                ax = plt.subplot2grid((span_len, z_dim), (i, j))
                plt.imshow(fig, cmap = cmap)
                if j == 0:
                    ax.set_ylabel('value : {}'.format(z_values[i]), fontsize=20)
                if i == 0:
                    ax.set_title('factor : {}'.format(j), fontsize=20)
                ax.set_yticklabels([])
                ax.set_xticklabels([])

        plt.tight_layout()
        plt.savefig(save_dir+title+'.png')


    # In[102]:

    def plot_multiple(*imgs, names, type):
        plot_len = len(imgs)
        for idx, img in enumerate(imgs):
            ax = plt.subplot2grid((1, plot_len), (0, idx))
            ax.set_title(names[idx], fontsize=30)
            ax.imshow(np.squeeze(img))
    #     plt.suptitle(version, fontsize=40)
        plt.subplots_adjust(top=1.4)
        plt.savefig(save_dir+type+'_org_recon.png')



    # In[103]:

    def get_all_plots(img, swap_img, img_num, swap_img_num, span_min_val, span_max_val, span_len):
        print('s_swap')
        img_sample = img[img_num]
        swap_sample = swap_img[swap_img_num]

    #     z_v_recon, z_latent, z_m, z_sigma_sq = get_latent(img_sample)
        recon_swap, z_m_c_swap, z_m_s_swap, z_sig_c, z_sig_s, z_v_c_swap, z_v_s_swap = get_latent(swap_sample)
        recon, z_m_c, z_m_s, z_sig_c, z_sig_s, z_v_c, z_v_s = get_latent(img_sample)
    #     recon, z_m_c, z_m_s_swap, z_sig_c, z_sig_s, z_v_c_swap, z_v_s = get_latent(swap_sample)
        print('z_m :', z_m_s)
        print('z_latent :', z_v_s)
        print('z_sigma_sq :', z_sig_s)
        print('z_sigma argsort :', np.argsort(z_sig_s))

        text = str(z_sig_s)+'_'+str(np.argsort(z_sig_s))
        plt.figure()
        plot_multiple(img_sample, recon, swap_sample, recon_swap,
                      names=['original','recon', 'swap_org', 'swap_recon'], type='s')

        def _plot(z_m_c, z_m_s, z_v_c, z_v_s, version):
            z_m_s_latent = np.hstack((z_m_c, z_m_s))
            z_v_s_latent = np.hstack((z_v_c, z_v_s))

            z_m_s_recon = get_recon(z_m_s_latent)
            z_v_s_recon = get_recon(z_v_s_latent)

            z_m_s_span, span_value_range = get_span_matrix(z_m_s, span_min_val, span_max_val, span_len)
            z_v_s_span, span_value_range = get_span_matrix(z_v_s, span_min_val, span_max_val, span_len)

            merged_m = merge_latent(span_matrix=z_m_s_span, fixed_latent=z_m_c, fixed_type = 'c')
            merged_v = merge_latent(span_matrix=z_v_s_span, fixed_latent=z_m_c, fixed_type = 'c')
        #     z_m_span, span_value_range = get_span_matrix(z_, span_min_val, span_max_val, span_len)
        #     z_span_recon = get_recon(z_latent_span)
            z_m_s_span_recon = get_recon(merged_m)
            z_v_s_span_recon = get_recon(merged_v)

            plt.figure()
            plot_figures(z_v_s_span_recon, span_value_range, 'latent_{}'.format(version), text)
            plt.figure()
            plot_figures(z_m_s_span_recon, span_value_range, 'z_m_{}'.format(version), text)

        _plot(z_m_c, z_m_s, z_v_c, z_v_s, version='s_plain')
        _plot(z_m_c_swap, z_m_s_swap, z_v_c_swap, z_v_s_swap, version='s_swapped')


    # In[104]:

    def get_all_plots_c(img, swap_img, img_num, swap_img_num, span_min_val, span_max_val, span_len):
        print('c_swap')
        img_sample = img[img_num]
        swap_sample = swap_img[swap_img_num]

        recon_swap, z_m_c_swap, z_m_s_swap, z_sig_c, z_sig_s, z_v_c_swap, z_v_s_swap = get_latent(swap_sample)
        recon, z_m_c, z_m_s, z_sig_c, z_sig_s, z_v_c, z_v_s = get_latent(img_sample)

        print('z_m :', z_m_c)
        print('z_latent :', z_v_c)
        print('z_sigma_sq :', z_sig_c)
        print('z_sigma argsort :', np.argsort(z_sig_c))
        text = str(z_sig_s)+'_'+str(np.argsort(z_sig_s))

        plt.figure()
        plot_multiple(img_sample, recon, swap_sample, recon_swap,
                      names=['original','recon', 'swap_org', 'swap_recon'], type='c')

        def _plot(z_m_c, z_m_s, z_v_c, z_v_s, version):
            z_m_s_latent = np.hstack((z_m_c, z_m_s))
            z_v_s_latent = np.hstack((z_v_c, z_v_s))

            z_m_s_recon = get_recon(z_m_s_latent)
            z_v_s_recon = get_recon(z_v_s_latent)

            z_m_c_span, span_value_range = get_span_matrix(z_m_c, span_min_val, span_max_val, span_len)
            z_v_c_span, span_value_range = get_span_matrix(z_v_c, span_min_val, span_max_val, span_len)

            merged_m = merge_latent(span_matrix=z_m_c_span, fixed_latent=z_m_s, fixed_type = 's')
            merged_v = merge_latent(span_matrix=z_v_c_span, fixed_latent=z_m_s, fixed_type = 's')
        #     z_m_span, span_value_range = get_span_matrix(z_, span_min_val, span_max_val, span_len)
        #     z_span_recon = get_recon(z_latent_span)
            z_m_c_span_recon = get_recon(merged_m)
            z_v_c_span_recon = get_recon(merged_v)

            plt.figure()
            plot_figures(z_v_c_span_recon, span_value_range, 'latent_{}'.format(version), text)
            plt.figure()
            plot_figures(z_m_c_span_recon, span_value_range, 'z_m_{}'.format(version), text)

        _plot(z_m_c, z_m_s, z_v_c, z_v_s, version='c_plain')
        _plot(z_m_c_swap, z_m_s_swap, z_v_c_swap, z_v_s_swap, version='c_swapped')


    # In[88]:

    matplotlib.rcParams['figure.figsize'] = (20.0, 20.0)


    # In[30]:




    # In[105]:

    get_all_plots(img_0, img_1, 0, 100, -3, 3, 5)


    # In[106]:

    get_all_plots_c(img_0, img_1, 0, 100, -3, 3, 5)


# In[ ]:



