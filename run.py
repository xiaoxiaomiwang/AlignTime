import torch
import numpy as np
import random
from exp.exp_main import Exp_Main
import argparse
import time
import datetime

fix_seed = 1024
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Multivariate Time Series Forecasting')

    # basic config
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model', type=str, required=True, default='AlignTime',
                        help='model name, options: [Transformer, Linear, NLinear, DLinear, SCINet, AlignTime]')

    # data loader
    parser.add_argument('--data', type=str, required=True, default='ETTh1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./dataset/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--seg', type=int, default=20, help='prediction plot segments')
    parser.add_argument('--mat', type=int, default=0, help='option: [0-random, 1-identity]')
    parser.add_argument('--highest_freq', type=int, default=2,
                        help='the number of downsampling in factorized temporal interaction')

    # model
    parser.add_argument('--embed_type', type=int, default=0,
                        help='0: default 1: value embedding + temporal embedding + positional embedding 2: value embedding + positional embedding')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=1, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

    # optimization
    parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='optimizer learning rate')
    parser.add_argument('--loss', type=str, default='mse', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
    parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage')

    # Get current timestamp
    current_time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    args = parser.parse_args()

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.dvices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    f = open("result_long_term_forecast.txt", 'a')
    f.write('Args in experiment:')
    f.write('\n')
    f.close()  # 在这里关闭文件
    print(args)
    f = open("result_long_term_forecast.txt", 'a')
    f.write(str(args))
    f.write('\n')
    f.close()  # 在这里关闭文件

    Exp = Exp_Main

    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            setting = '{}_{}_ft{}_sl{}_ll{}_pl{}_lr{}_dm{}_df{}_nh{}_el{}_dl{}_fc{}_eb{}_{}_time{}'.format(
                args.model,
                args.data_path[:-4],
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.learning_rate,
                args.d_model,
                args.d_ff,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.factor,
                args.embed, ii,
                current_time
            )

            exp = Exp(args)  # set experiments
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            f = open("result_long_term_forecast.txt", 'a')
            f.write('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            f.close()  # 在这里关闭文件
            exp.train(setting)

            time_now = time.time()
            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            f = open("result_long_term_forecast.txt", 'a')
            f.write('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            f.close()  # 在这里关闭文件


            exp.test(setting)
            inference_time = time.time() - time_now
            print('Inference time: ', inference_time)

            with open("result_long_term_forecast.txt", 'a') as f:
                f.write(f'Inference time: {inference_time}\n')
                f.write('\n')
                f.write('\n')

            if args.do_predict:
                print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                f = open("result_long_term_forecast.txt", 'a')
                f.write('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                f.close()  # 在这里关闭文件
                exp.predict(setting, True)

            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = '{}_{}_ft{}_sl{}_ll{}_pl{}_lr{}_dm{}_df{}_nh{}_el{}_dl{}_fc{}_eb{}_{}_time{}'.format(args.model_id,
                                                                                                       args.model,
                                                                                                       args.data_path[
                                                                                                       :-4],
                                                                                                       args.features,
                                                                                                       args.seq_len,
                                                                                                       args.label_len,
                                                                                                       args.pred_len,
                                                                                                       args.learning_rate,
                                                                                                       args.d_model,
                                                                                                       args.d_ff,
                                                                                                       args.n_heads,
                                                                                                       args.e_layers,
                                                                                                       args.d_layers,
                                                                                                       args.factor,
                                                                                                       args.embed, ii,
                                                                                                       current_time
                                                                                                       )

        exp = Exp(args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        f = open("result_long_term_forecast.txt", 'a')
        f.write('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        f.close()  # 在这里关闭文件
        exp.test(setting, test=1)
        torch.cuda.empty_cache()
