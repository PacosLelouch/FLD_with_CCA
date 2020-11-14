import os
from arg import argument_parser
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from models import Network
from utils import cal_loss, Evaluator
import utils

parser = argument_parser()
args = parser.parse_args()

def main():
    # random seed
    seed = 1234
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # load dataset
    if args.dataset[0] == 'deepfashion':
        ds = pd.read_csv('./Anno/df_info.csv')
        from dataset import DeepFashionDataset as DataManager
    elif args.dataset[0] == 'fld':
        ds = pd.read_csv('./Anno/fld_info.csv')
        from dataset import FLDDataset as DataManager
    else :
        raise ValueError

    print('dataset : %s' % (args.dataset[0]))
    if not args.evaluate:
        train_dm = DataManager(ds[ds['evaluation_status'] == 'train'], root=args.root)
        train_dl = DataLoader(train_dm, batch_size=args.batchsize, shuffle=True)

        if os.path.exists('models') is False:
            os.makedirs('models')

    test_dm = DataManager(ds[ds['evaluation_status'] == 'test'], root=args.root)
    #test_dl = DataLoader(test_dm, batch_size=args.batchsize, shuffle=False)
    test_dl = DataLoader(test_dm, batch_size=1, shuffle=False)

    # Load model
    print("Load the model...")
    net = torch.nn.DataParallel(Network(dataset=args.dataset, flag=args.cca)).cuda()
    if not args.weight_file == None:
        weights = torch.load(args.weight_file)
        if args.update_weight:
            weights = utils.load_weight(net, weights)
        net.load_state_dict(weights)

    # evaluate only
    if args.evaluate:
        print("Evaluation only")
        test(net, test_dl, 0)
        return
    
    # base epoch
    if args.base_epoch:
        base_epoch = args.base_epoch
    else:
        base_epoch = 0

    # learning parameters
    optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.decay_epoch, args.gamma)

    print('Start training')
    for epoch in range(args.epoch):
        lr_scheduler.step()
        train(net, optimizer, train_dl, epoch, base_epoch)
        test(net, test_dl, epoch)


def train(net, optimizer, trainloader, epoch, base_epoch=0):
    train_step = len(trainloader)
    net.train()
    for i, sample in enumerate(trainloader):
        for key in sample:
            if key[0:2] != 'D_':
                sample[key] = sample[key].cuda()
        output = net(sample)
        loss = cal_loss(sample, output, loss_type=args.loss_type)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 10 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, args.epoch, i + 1, train_step, loss.item()))
    save_file = 'model_%03d.pkl'
    print('Saving Model : ' + save_file % (base_epoch + epoch + 1))
    torch.save(net.state_dict(), './models/'+ save_file % (base_epoch + epoch + 1))


def test(net, test_loader, epoch):
    net.eval()
    test_step = len(test_loader)
    print('\nEvaluating...')
    with torch.no_grad():
        evaluator = Evaluator()
        for i, sample in enumerate(test_loader):
            another_ret = test_loader.dataset.another_ret
            for key in sample:
                if key[0:2] != 'D_':
                    sample[key] = sample[key].cuda()
            output = net(sample)
            evaluator.add(output, sample, another_ret)
            
            if (i + 1) % 100 == 0:
                print('Val Step [{}/{}]'.format(i + 1, test_step)) # j + 1 ?

        results = evaluator.evaluate()
        print_string = 'Epoch {}/{}'.format(epoch + 1, args.epoch) + '\n' \
                    + '|  L.Collar  |  R.Collar  |  L.Sleeve  |  R.Sleeve  |   L.Waist  |   R.Waist  |    L.Hem   |   R.Hem    |     ALL    |' + '\n' \
                    + '|   {:.5f}  |   {:.5f}  |   {:.5f}  |   {:.5f}  |   {:.5f}  |   {:.5f}  |   {:.5f}  |   {:.5f}  |   {:.5f}  |' \
              .format(results['lm_dist'][0], results['lm_dist'][1], results['lm_dist'][2], results['lm_dist'][3], \
                      results['lm_dist'][4], results['lm_dist'][5], results['lm_dist'][6], results['lm_dist'][7], \
                      results['lm_dist_all']) + '\n'
        
        print_string += 'PDL:\n'
        PDLs = results['lm_PDLs']
        for i in range(0, len(PDLs)):
            print_string += '--Dist = %d:\n'%i
            print_string += '----|  L.Collar  |  R.Collar  |  L.Sleeve  |  R.Sleeve  |   L.Waist  |   R.Waist  |    L.Hem   |   R.Hem    |     ALL    |' + '\n' \
                + '----|   {:.5f}  |   {:.5f}  |   {:.5f}  |   {:.5f}  |   {:.5f}  |   {:.5f}  |   {:.5f}  |   {:.5f}  |   {:.5f}  |' \
              .format(PDLs[i][0], PDLs[i][1], PDLs[i][2], PDLs[i][3], \
                      PDLs[i][4], PDLs[i][5], PDLs[i][6], PDLs[i][7], \
                      results['lm_PDLs_all'][i]) + '\n\n'
        
        print(print_string)
        #print('Epoch {}/{}'.format(epoch + 1, args.epoch))
        #print('|  L.Collar  |  R.Collar  |  L.Sleeve  |  R.Sleeve  |   L.Waist  |   R.Waist  |    L.Hem   |   R.Hem    |     ALL    |')
        #print('|   {:.5f}  |   {:.5f}  |   {:.5f}  |   {:.5f}  |   {:.5f}  |   {:.5f}  |   {:.5f}  |   {:.5f}  |   {:.5f}  |'
        #      .format(results['lm_dist'][0], results['lm_dist'][1], results['lm_dist'][2], results['lm_dist'][3],
        #              results['lm_dist'][4], results['lm_dist'][5], results['lm_dist'][6], results['lm_dist'][7],
        #              results['lm_dist_all']))
        file = open('results_%s_lr_%.4f_base_%d_de_%d_g_%.2f.txt'% \
                    (args.dataset, \
                     args.learning_rate, \
                     args.base_epoch, \
                     args.decay_epoch, \
                     args.gamma), 'a')
        file.write(print_string + '\n')
        #file.write('Epoch {}\n'.format(args.base_epoch + epoch + 1))
        #file.write('|  L.Collar  |  R.Collar  |  L.Sleeve  |  R.Sleeve  |   L.Waist  |   R.Waist  |    L.Hem   |   R.Hem    |     ALL    |\n')
        #file.write('|   {:.5f}  |   {:.5f}  |   {:.5f}  |   {:.5f}  |   {:.5f}  |   {:.5f}  |   {:.5f}  |   {:.5f}  |   {:.5f}  |\n\n'
        #      .format(results['lm_dist'][0], results['lm_dist'][1], results['lm_dist'][2], results['lm_dist'][3],
        #              results['lm_dist'][4], results['lm_dist'][5], results['lm_dist'][6], results['lm_dist'][7],
        #              results['lm_dist_all']))
        file.close()


if __name__ == '__main__':
    main()
