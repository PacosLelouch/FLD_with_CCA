import os
from arg import argument_parser
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from models import Network
from utils import cal_loss, Evaluator
import utils
import matplotlib.pyplot as plt

parser = argument_parser()
args = parser.parse_args()

if args.dataset[0] == 'fld':
    weight_cca = './models_fld_avg/model_324.pkl'
    weight_fpn = './models_fld_avg/model_010.pkl'
    target_dir = './predict_fld_avg_ori/'
elif args.dataset[0] == 'deepfashion':
    weight_cca = './models_df_avg/model_109.pkl'
    weight_fpn = './models_df_avg/model_009.pkl'
    target_dir = './predict_df_avg_ori/'

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
    test_dl = DataLoader(test_dm, batch_size=args.batchsize, shuffle=False)

    # Load model
    print("Load the model...")
    net_cca = torch.nn.DataParallel(Network(dataset=args.dataset, flag=1)).cuda()
    
    net_fpn = torch.nn.DataParallel(Network(dataset=args.dataset, flag=0)).cuda()
    
    weights = torch.load(weight_cca)
    net_cca.load_state_dict(weights)
    
    weights = torch.load(weight_fpn)
    net_fpn.load_state_dict(weights)

    #print('net:\n' + str(net.module))#TEST

    print("Prediction only")
    predict(net_cca, net_fpn, test_dl, 0)

def predict(net_cca, net_fpn, test_loader, epoch):
    net_cca.eval()
    net_fpn.eval()
    test_step = len(test_loader)
    #postfix = '_cca' if args.cca else '_fpn'
    fmt = '.png'
    paint = ['ro', 'go']
    ms = 16.0
    columns = ['no.'] + ['x%d'%i for i in range(1, 9)] + ['y%d'%i for i in range(1, 9)]
    predict_lm_cca = pd.DataFrame(columns=columns)
    predict_lm_fpn = pd.DataFrame(columns=columns)
    
    print('\nPredicting...')
    with torch.no_grad():
        evaluator = Evaluator()
        for count, sample in enumerate(test_loader):
            image_name_ori = sample['D_image_name']
            images = sample['D_image_bb_ori'] #sample['image_ori']
            lm_gt = sample['D_landmark_pos_ori'] #sample['landmark_pos']
            lm_vis = landmark_vis_count = sample['landmark_vis'].cpu().numpy()
            #print(lm_gt) #TEST
            #print(lm_vis) #TEST
            
            lm_pos_x_gth = np.array([[lm_gt[j, i, 0] for i in range(8)] for j in range(lm_gt.shape[0])])
            lm_pos_y_gth = np.array([[lm_gt[j, i, 1] for i in range(8)] for j in range(lm_gt.shape[0])])
            #print(lm_pos_x_gth) #TEST

            for key in sample:
                if key[0:2] != 'D_':
                    sample[key] = sample[key].cuda()
            output_cca = net_cca(sample)
            output_fpn = net_fpn(sample)
            
            #back to original image
            x1, x2, y1, y2 = np.array(sample['D_image_x1']), np.array(sample['D_image_x2']), \
                            np.array(sample['D_image_y1']), np.array(sample['D_image_y2'])
            height, width = \
                np.array([image.shape[0] for image in images]).astype(np.float32), \
                np.array([image.shape[1] for image in images]).astype(np.float32)
            #print("x1 =", x1)#TEST
            #print("width =", width)#TEST
            x1x2 = [x1, x2]
            y1y2 = [y1, y2]
            
            predict_index_cca, lm_pos_x_cca, lm_pos_y_cca = \
                evaluator.predict(output_cca, sample, count + 1, \
                    x1x2=x1x2, y1y2=y1y2)
            predict_index_fpn, lm_pos_x_fpn, lm_pos_y_fpn = \
                evaluator.predict(output_fpn, sample, count + 1, \
                    x1x2=x1x2, y1y2=y1y2)
            
            for row in range(len(predict_index_cca)):
                row_data_cca = list(lm_pos_x_cca[row]) + \
                           list(lm_pos_y_cca[row])

                row_data_fpn = list(lm_pos_x_fpn[row]) + \
                           list(lm_pos_y_fpn[row])

                image_name_ori_r = image_name_ori[row].replace('/', '_')
                image = images[row].numpy()
                #image = np.transpose(image, (1, 2, 0))
                
                img_name = image_name_ori_r + '_cca' + fmt
                    #'%08d'%predict_index_cca[row] + '_cca' + fmt
                img_num = predict_index_cca[row]
                predict_lm_cca.loc[img_name] = [img_num] + row_data_cca
                plt.figure(1)
                plt.axis('off')
                plt.imshow(image)
                for i in range(lm_pos_x_cca[row].shape[0]):
                    x = lm_pos_x_cca[row, i]
                    y = lm_pos_y_cca[row, i]
                    plt.plot(x, y, paint[lm_vis[row, i]], markersize=ms)
                plt.savefig(target_dir + img_name, bbox_inches = 'tight')
                plt.close()

                img_name = image_name_ori_r + '_fpn' + fmt
                        #'%08d'%predict_index_fpn[row] + '_fpn' + fmt
                img_num = predict_index_fpn[row]
                predict_lm_fpn.loc[img_name] = [img_num] + row_data_fpn
                plt.figure(1)
                plt.axis('off')
                plt.imshow(image)
                for i in range(lm_pos_x_fpn[row].shape[0]):
                    x = lm_pos_x_fpn[row, i]
                    y = lm_pos_y_fpn[row, i]
                    plt.plot(x, y, paint[lm_vis[row, i]], markersize=ms)
                plt.savefig(target_dir + img_name, bbox_inches = 'tight')
                plt.close()

                img_name = image_name_ori_r + '_gth' + fmt
                        #'%08d'%predict_index_fpn[row] + '_gth' + fmt
                img_num = predict_index_fpn[row]
                plt.figure(1)
                plt.axis('off')
                plt.imshow(image)
                for i in range(lm_pos_x_fpn[row].shape[0]):
                    x = lm_pos_x_gth[row, i]
                    y = lm_pos_y_gth[row, i]
                    plt.plot(x, y, paint[lm_vis[row, i]], markersize=ms)
                plt.savefig(target_dir + img_name, bbox_inches = 'tight')
                plt.close()

                img_name = image_name_ori_r + '_ori' + fmt
                        #'%08d'%predict_index_cca[row] + '_ori' + fmt
                img_num = predict_index_cca[row]
                plt.figure(1)
                plt.axis('off')
                plt.imshow(image)
                plt.savefig(target_dir + img_name, bbox_inches = 'tight')
                plt.close()
                
            #print(predict_name, lm_pos_x, lm_pos_y) #TEST
            #break #TEST
            
            if (count + 1) % 100 == 0:
                print('Pred Step [{}/{}]'.format(count + 1, test_step))
        predict_lm_cca.to_csv(target_dir + 'predict_lm_cca.csv',
                          index=True,
                          index_label='predict_name')
        predict_lm_fpn.to_csv(target_dir + 'predict_lm_fpn.csv',
                          index=True,
                          index_label='predict_name')
        
if __name__ == '__main__':
    main()
