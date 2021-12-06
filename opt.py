import configargparse


def config_parser():
    parser = configargparse.ArgumentParser()
    # general
    parser.add_argument('--config', is_config_file=True, help='config file path')
    parser.add_argument("--log_dir", type=str, default='./logs/',
                        help='where to store ckpts and logs')
    parser.add_argument("--expname", type=str,default='./exp_name/', help='experiment name')
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--N_vis", type=int, default=2)


    return parser.parse_args()


