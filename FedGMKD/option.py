import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    #Data specifc paremeters
    parser.add_argument('--dataset', default='OfficeHome',
                        help='CIFAR10, CIFAR100, SVHN') 
    #Training specifc parameters
    parser.add_argument('--log_frq', type=int, default=5,
                        help='frequency of logging')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='minibatch size')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='number of epochs')  
    parser.add_argument('--clip_grad', type=float, default=None,
                        help='gadient clipping')        
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate')
    parser.add_argument('--lr_sh_rate', type=int, default=10,
                        help='number of steps to drop the lr')
    parser.add_argument('--use_lrschd', action="store_true", default=True,
                        help='Use lr rate scheduler')
    parser.add_argument('--num_clients',  type=int, default=4,
                        help='number of local models')
    
    parser.add_argument('--num_classes', type=int,default=65,
                        help='number of classes')
    
    parser.add_argument('--sampling_rate', type=float,default=1,
                        help='frac of local models to update')
    parser.add_argument('--local_ep',type=int, default=3,
                        help='iterations of local updating')
    parser.add_argument('--beta', type=float,default=0.5,
                        help='beta for non-iid distribution')
    parser.add_argument('--seed', type=int,default=0,
                        help='random seed for generating datasets')
    parser.add_argument('--code_len', type=int,default=32,
                        help='length of code')
    parser.add_argument('--alg', default='FedGMKD',
                        help='FedAvg, FedProx, FedMD, Fedproto, FedGMKD')
    
    parser.add_argument('--lam', type=float, default=0.06,
                        help='hyper-parameter for loss2')
    
    parser.add_argument('--gamma', type=float, default=0.06,
                        help='hyper-parameter for loss3')
    
    parser.add_argument('--std', type=float, default=0,
                        help='std of gaussian noise ')
    
    parser.add_argument('--part', type=float,default=0.1,
                        help='percentage of each local data')
    
    
    parser.add_argument('--temp', type=float,default=0.6,
                        help='temperture for soft prediction')
    
    parser.add_argument('--model', default= 'resnet18',
                        help='resnet18')
    
    parser.add_argument('--upload_model', action="store_true", default= True,
                        help='allow clients to upload models to the server')
    
    parser.add_argument('--eval_only', action="store_true", default=True,help='evaluate the model')
   
    parser.add_argument('--moon_mu', type=float, default=0.3, help='MOON ')


    args = parser.parse_args()
    return args