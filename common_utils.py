'''
Author: rxin rxin@connect.ust.hk
Date: 2023-02-16 15:19:45
LastEditors: rxin rxin@connect.ust.hk
LastEditTime: 2023-02-16 15:19:45
FilePath: /DIPP/common_utils.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import torch


def save_checkpoint(epoch, save_name, cfg, model):
    """ Save model to file. """
    model_dict = {'epoch': epoch,
                  'state_dict': model.state_dict(),
                  'model_cfg': cfg['training']['model_params'],
                  'loss_params': cfg['training']['loss_params']}

    print("Saving model to {}".format(save_name))
    torch.save(model_dict, save_name)

def load_checkpoint(model_file):
    """ Load a model from a file. """
    print("Loading model from {}".format(model_file))
    model_dict = torch.load(model_file)
    ## save loaded model params
    print('>>>> Model config is:')
    for k,v in model_dict['model_cfg'].items():
        print('--',k,v)
    print('>>>> Loss config is:')
    for k,v in model_dict['loss_params'].items():
        print('--',k,v)
    model = model_selection[f"{model_dict['model_cfg']['name']}"](**model_dict['model_cfg'])
    model.load_state_dict(state_dict=model_dict['state_dict'])
    loss_func = loss_selection[model_dict['loss_params']['loss_name']] \
        (**model_dict['loss_params'])
    epoch = model_dict['epoch']+1
    print('load succeed')
    return model, loss_func, epoch