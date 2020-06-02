import torch
import torch.nn as nn
import numpy as np

def load_weight(model, weights):
    model_dict =  model.state_dict()
    state_dict = {k:v for k,v in weights.items() if k in model_dict.keys()}
    model_dict.update(state_dict)
    return model_dict

def loss_mse(pred, gt, mask):
    return torch.pow(mask * (pred - gt), 2).mean()

def loss_cross_entropy(pred, gt, mask):
    positive_weight = 1#torch.where(mask == 0, torch.tensor(1.0).cuda(), torch.tensor(0.0).cuda())
    loss = -(positive_weight * gt * torch.log(torch.sigmoid(pred) + 1e-10)
             + (1 - gt) * torch.log(1 + 1e-10 - torch.sigmoid(pred))).mean()
    return loss

def cal_loss(sample, output, loss_type='mse'):
    batch_size, _, pred_w, pred_h = sample['image'].size()

    lm_size = int(output['lm_pos_map'].size(2))
    visibility = sample['landmark_vis']
    vis_mask = torch.cat([visibility.reshape(batch_size* 8, -1)] * lm_size * lm_size, dim=1).float()
    lm_map_gt = sample['landmark_map%d' % lm_size].reshape(batch_size * 8, -1)
    lm_pos_map = output['lm_pos_map']
    lm_map_pred =lm_pos_map.reshape(batch_size * 8, -1)

    loss_func = {
            'cross_entropy':loss_cross_entropy,
            'mse':loss_mse,
        }
    def error(*args, **kwargs):
        raise NotImplementedError('%s not implemented.'%loss_type)
    func = loss_func.get(loss_type, error)
    loss = func(lm_map_pred, lm_map_gt, vis_mask)
        
    #loss = torch.pow(vis_mask * (lm_map_pred - lm_map_gt), 2).mean()

    return loss

class Evaluator(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.lm_vis_count_all = np.array([0.] * 8)
        self.lm_dist_all = np.array([0.] * 8)

    def add(self, output, sample):
        landmark_vis_count = sample['landmark_vis'].cpu().numpy().sum(axis=0)
        landmark_vis_float = torch.unsqueeze(sample['landmark_vis'].float(), dim=2)
        landmark_vis_float = torch.cat([landmark_vis_float, landmark_vis_float], dim=2).cpu().detach().numpy()

        lm_pos_map = output['lm_pos_map']
        batch_size, _, pred_h, pred_w = lm_pos_map.size()
        lm_pos_reshaped = lm_pos_map.reshape(batch_size, 8, -1)

        lm_pos_y, lm_pos_x = np.unravel_index(torch.argmax(lm_pos_reshaped, dim=2).cpu().numpy(), (pred_h, pred_w))
        lm_pos_output = np.stack([lm_pos_x / (pred_w - 1), lm_pos_y / (pred_h - 1)], axis=2)

        landmark_dist = np.sum(np.sqrt(np.sum(np.square(
            landmark_vis_float * lm_pos_output - landmark_vis_float * sample['landmark_pos_normalized'].cpu().numpy(),
        ), axis=2)), axis=0)

        self.lm_vis_count_all += landmark_vis_count
        self.lm_dist_all += landmark_dist

    def evaluate(self):
        lm_dist = self.lm_dist_all / self.lm_vis_count_all
        lm_dist_all = (self.lm_dist_all / self.lm_vis_count_all).mean()

        return {'lm_dist' : lm_dist,
                'lm_dist_all' : lm_dist_all}

    def predict(self, output, sample, index, x1x2=[], y1y2=[]):
        landmark_vis_count = sample['landmark_vis'].cpu().numpy().sum(axis=0)
        landmark_vis_float = torch.unsqueeze(sample['landmark_vis'].float(), dim=2)
        landmark_vis_float = torch.cat([landmark_vis_float, landmark_vis_float], dim=2).cpu().detach().numpy()

        lm_pos_map = output['lm_pos_map']
        batch_size, _, pred_h, pred_w = lm_pos_map.size()
        lm_pos_reshaped = lm_pos_map.reshape(batch_size, 8, -1)

        lm_pos_y, lm_pos_x = np.unravel_index(torch.argmax(lm_pos_reshaped, dim=2).cpu().numpy(), (pred_h, pred_w))
        if len(x1x2) == 2 and len(y1y2) == 2:
            lm_pos_x_n, lm_pos_y_n = lm_pos_x / pred_w, lm_pos_y / pred_h
            lm_pos_x = (1 - lm_pos_x_n) * x1x2[0] + lm_pos_x_n * x1x2[1]
            lm_pos_y = (1 - lm_pos_y_n) * y1y2[0] + lm_pos_y_n * y1y2[1]
        predict_index = np.array([i for i in range(index, index + batch_size)])
        return predict_index, lm_pos_x, lm_pos_y
