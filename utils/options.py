import argparse
import time
import os

DEFAULTS = {'nyu_data_path':{'your-pc-name-here': 'your-data-path-here'},
            'max_depth':{'nyu':10, 'kitti':80},
            'min_depth':1e-3}

class Options:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Uncertainty-depth options")

        self.pc_name = os.popen('hostname').read().strip()

        self.parser.add_argument("--dataset",
                                 type=str,
                                 help="Dataset to use",
                                 choices=['nyu', 'kitti'],
                                 default='nyu')

        self.parser.add_argument("--encoder",
                                 type=str,
                                 help="Encoder type",
                                 choices=['resnet', 'densenet', 'swin', 'vit', 'efficientnet'],
                                 default='resnet')

        self.parser.add_argument("--decoder",
                                 type=str,
                                 help="Decoder type",
                                 choices=['simple', 'bts'],
                                 default='simple')

        self.parser.add_argument("--reg_mode",
                                 type=str,
                                 help="Regression method, `direct` for regression, `lin_cls` for classification",
                                 choices=['direct', 'lin_cls'],
                                 default='lin_cls')
                                
        self.parser.add_argument("--reg_supervision",
                                 type=str,
                                 help="regression supervision method",
                                 choices=['regression_l1', 'none'],
                                 default='regression_l1')
        
        self.parser.add_argument("--prob_supervision",
                                 type=str,
                                 help="classification supervision method",
                                 choices=['soft_label', 'none'],
                                 default='soft_label')
                                
        self.parser.add_argument("--uncert_supervision",
                                 type=str,
                                 help="uncertainty supervision method",
                                 choices=['error_uncertainty_ranking',
                                          'error_uncertainty_ranking_noclamp', 
                                          'error_uncertainty_l1', 'none'],
                                 default='error_uncertainty_ranking')

        self.parser.add_argument("--aleatoric_uncertainty",
                                 type=str,
                                 help="aleatoric uncertainty method",
                                 choices=['mc_dropout', 'multi_head', 'noise', 'none'],
                                 default='none')

        self.parser.add_argument("--nyu_data_path",
                                 type=str,
                                 help="Path to the training data",
                                 default=DEFAULTS['nyu_data_path'][self.pc_name])

        self.parser.add_argument("--model_name",
                                 type=str,
                                 help="Description of the experiment",
                                 default=str(time.strftime("%Y_%m_%d-%H:%M:%S"))
                                )
        
        self.parser.add_argument("--log_path",
                                 type=str,
                                 help="log path",
                                 default="./logs/")

        self.parser.add_argument("--save_path",
                                 type=str,
                                 help="path of the models to save",
                                 default="./ckpt/")

        self.parser.add_argument("--min_depth",
                                 type=float,
                                 help="minimum depth for disp-depth convert",
                                 default=1e-4)

        ####### training configs #######
        self.parser.add_argument('--epochs', 
                                 type=int, 
                                 default=10, 
                                 help='epoch number')

        self.parser.add_argument("--workers",
                                 type=int,
                                 help="Number of workers for dataloader",
                                 default=4)

        self.parser.add_argument("--batch_size",
                                 type=int,
                                 help="Batch size for dataloader",
                                 default=4)

        self.parser.add_argument("--lr_gen",
                                 type=float,
                                 help="learning rate for opt",
                                 default=1e-4)
        
        self.parser.add_argument('--decay_rate', 
                                 type=float, 
                                 default=0.8, 
                                 help='decay rate of learning rate')
        
        self.parser.add_argument('--decay_epoch', 
                                 type=int, 
                                 default=2, 
                                 help='every n epochs decay learning rate')

        self.parser.add_argument('--feat_channel', 
                                 type=int, 
                                 default=64, 
                                 help='reduced channel of saliency feat')

        self.parser.add_argument("--print_freq",
                                 type=int,
                                 help="logging frequency",
                                 default=200)

        ####### distribution #######
        self.parser.add_argument("--no-ddp",
                                 help="if set, not use ddp",
                                 action="store_true")

        self.parser.add_argument("--local_rank",
                                 type=int,
                                 help="local rank for DDP",
                                 default=0)

        ####### phone notify #######
        self.parser.add_argument("--phone-notify",
                                 help="if set, push messages to your phone through ifttt if bug occurs",
                                 action="store_true")
        

        # setup guide: https://pythondict.com/life-intelligent/python-send-notification-ifttt/
        self.parser.add_argument("--ifttt_key",
                                 type=str,
                                 help="your key of the ifttt webhooks",
                                 default="YOUR_KEY_HERE")

    def parse(self):
        self.options = self.parser.parse_args()
        return self.options
