import configargparse


def config_parser():
    parser = configargparse.ArgumentParser()
    # general
    parser.add_argument('--config', is_config_file=True, help='config file path')
    parser.add_argument("--log_dir", type=str, default='./logs/',
                        help='where to store ckpts and logs')
    parser.add_argument("--sample_dir", type=str, default='./samples/',
                        help='where to store ckpts and logs')
    parser.add_argument("--expname", type=str,default='exp_name', help='experiment name')
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--num_vis_step", type=int, default=100)
    parser.add_argument("--num_vis", type=int, default=64)
    parser.add_argument("--num_step", type=int, default=1000, help='number of diffusion steps')
    parser.add_argument('--save_imgs', default=True, action="store_true",
                        help='save GT image for each step')

    parser.add_argument("--batchSize", type=int, default=128)
    parser.add_argument('--optimizer', type=str, default='adam',
                        help='optimizer type',
                        choices=['sgd', 'adam', 'radam', 'ranger'])
    parser.add_argument('--lr', type=float, default=8e-5,
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='learning rate momentum')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay')
    parser.add_argument('--lr_scheduler', type=str, default='cosine',
                        help='scheduler type',
                        choices=['steplr', 'cosine', 'poly'])
    #### params for warmup, only applied when optimizer == 'sgd' or 'adam'
    parser.add_argument('--warmup_multiplier', type=float, default=1.0,
                        help='lr is multiplied by this factor after --warmup_epochs')
    parser.add_argument('--warmup_epochs', type=int, default=0,
                        help='Gradually warm-up(increasing) learning rate in optimizer')
    ###########################
    #### params for steplr ####
    parser.add_argument('--decay_step', nargs='+', type=int, default=[20],
                        help='scheduler decay step')
    parser.add_argument('--decay_gamma', type=float, default=0.1,
                        help='learning rate decay amount')
    ###########################
    #### params for poly ####
    parser.add_argument('--poly_exp', type=float, default=0.9,
                        help='exponent for polynomial learning rate decay')

    parser.add_argument('--grad_clip', type=float, default=1.0,
                        help='exponent for polynomial learning rate decay')

    return parser.parse_args()


