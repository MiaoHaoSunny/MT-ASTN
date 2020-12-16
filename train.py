import random
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import time
# import matplotlib.pyplot as plt

from STMain import *
from PeopleBasic.PeopleSparse_loader import *
from PeopleBasic.PeopleFlowDataLoader import *
from Model.Backbone.Functions import MSE, DiffLoss
from Model.Backbone.preiter import *

import logging


def get_log(file_name):
    logger = logging.getLogger('train')  # 设定logger的名字
    logger.setLevel(logging.INFO)  # 设定logger得等级

    ch = logging.StreamHandler()  # 输出流的hander，用与设定logger的各种信息
    ch.setLevel(logging.INFO)  # 设定输出hander的level

    fh = logging.FileHandler(file_name, mode='a')  # 文件流的hander，输出得文件名称，以及mode设置为覆盖模式
    fh.setLevel(logging.INFO)  # 设定文件hander得lever

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)  # 两个hander设置个是，输出得信息包括，时间，信息得等级，以及message
    fh.setFormatter(formatter)
    logger.addHandler(fh)  # 将两个hander添加到我们声明的logger中去
    logger.addHandler(ch)
    return logger


Flow_train_loader = Flow_train_dataloader
len_flow_train_loader = len(Flow_train_loader)

Flow_validation_loader = Flow_validate_dataloader
len_flow_validation_loader = len(Flow_validation_loader)

OD_train_loader = od_train_dataloader
len_od_train_loader = len(OD_train_loader)

OD_validation_loader = od_val_dataloader
len_od_validation_loader = len(OD_validation_loader)

node_features = node_train_dataloader
adjs = adjs_train_dataloader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger = get_log('/ceph_10826/halemiao/log_taxi.txt')
# logger1 = get_log('epoch_log1.txt')

cuda = True
cudnn.benchmark = True

lr = 0.0001
n_epoch = 150
step_decay_weight = 0.95
lr_decay_step = len_flow_train_loader * 20
weight_decay = 5e-4
# alpha_weight = 0.01
# beta_weight = 0.075
# gamma_weight = 0.25

manual_seed = random.randint(1, 10000)
random.seed(manual_seed)
torch.manual_seed(manual_seed)

my_net = STMultiTask()
# my_net.load_state_dict(torch.load('STMulti_new20200420_120.pkl'))
print('net load over')
# flow_loss_list = []
# od_loss_list = []
# file = open("log.txt", 'w+')


# def exp_lr_scheduler(optimizer, step, init_lr=lr, lr_decay_step=lr_decay_step, step_decay_weight=step_decay_weight):
#     current_lr = init_lr * (step_decay_weight ** (step / lr_decay_step))
#
#     if step % lr_decay_step == 0:
#         print('Learning rate is set to %f' % current_lr)
#
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = current_lr
#
#     return optimizer


optimizer = optim.Adam(my_net.parameters(), lr=lr, weight_decay=weight_decay)
shedule = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)

# optimizer_od = optim.Adam(my_net.parameters(), lr=lr, weight_decay=weight_decay)

loss_task = nn.MSELoss()
loss_discriminator = nn.CrossEntropyLoss()
loss_diff = DiffLoss()

my_net = my_net.to(device)
loss_task = loss_task.to(device)
loss_discriminator = loss_discriminator.to(device)
loss_diff = loss_diff.to(device)

current_step = 0
best_MSE = 10000000.0

for epoch in range(n_epoch):
    # t1
    t1 = time.time()
    my_net.train()
    # shedule.step()
    # flow_iter = data_prefetcher(Flow_train_loader)
    # flow_validate_iter = data_prefetcher(Flow_validation_loader)

    flow_iter = iter(Flow_train_loader)
    flow_validate_iter = iter(Flow_validation_loader)

    # od_iter = data_prefetcher(OD_train_loader)
    # od_validate_iter = data_prefetcher(OD_validation_loader)

    od_iter = iter(OD_train_loader)
    od_validate_iter = iter(OD_validation_loader)

    # node_features_iter = data_prefetcher(node_features)
    # node_features_val_iter = data_prefetcher(node_val_dataloader)

    node_features_iter = iter(node_features)
    node_features_val_iter = iter(node_val_dataloader)

    # adjs_iter = data_prefetcher(adjs)
    # adjs_val_iter = data_prefetcher(adjs_val_dataloader)

    adjs_iter = iter(adjs)
    adjs_val_iter = iter(adjs_val_dataloader)

    t = time.time()
    print('Data load cost: {:.4f}min'.format((t-t1)/60))

    i = 0

    flow_train_average_MSE = 0
    od_train_average_MSE = 0

    validate_average_MSE = 0

    while i < len_flow_train_loader:
        t2 = time.time()
        # with torch.autograd.set_detect_anomaly(True):
        # flow_img, flow_label = flow_iter.next()
        # od_img, od_label = od_iter.next()

        flow_img, flow_label = next(flow_iter)
        od_img, od_label = next(od_iter)

        node_feature, _ = next(node_features_iter)
        adj, _ = next(adjs_iter)

        node_feature = node_feature.permute((0, 1, 3, 2))
        adj = adj.permute((0, 1, 3, 2))

        flow_img, flow_label, od_img, od_label, node_feature, adj = flow_img.float().to(device), flow_label.float(). \
            to(device), od_img.float().to(device), od_label.float().to(device), node_feature. \
                                                                        float().to(device), adj.float().to(device)

        # t3 = time.time()
        # print('Data iter cost: {:.4f} min'.format((t3-t2)/60))

        p = float(i + epoch * len_flow_train_loader) / n_epoch / len_flow_train_loader
        alpha = 2./(1.+np.exp(-10*p)) - 1


        flow_img, od_img = flow_img.permute((0, 2, 1, 3, 4)), od_img.permute((0, 4, 1, 2, 3))
        # print(flow_img.shape, od_img.shape)
        flow_result, od_result = my_net(flow_img, od_img, node_feature, adj, alpha)
        # print('Net calculate over')
        # t4 = time.time()
        # print('Net over!, cost: {:.4f}min'.format((t4-t3)/60))
        flow_domain_true_label = torch.zeros(batch_size).long().to(device)
        od_domain_true_label = torch.ones(batch_size).long().to(device)

        # flow train
        flow_private_feature, flow_private_domain_label, flow_shared_feature, flow_domain_label, flow_out = flow_result

        flow_task_loss = loss_task(flow_out, flow_label)
        flow_shared_loss = loss_discriminator(flow_domain_label, flow_domain_true_label)
        flow_private_loss = loss_discriminator(flow_private_domain_label, flow_domain_true_label)
        # torch.cuda.empty_cache()
        # print(flow_private_feature.shape, flow_shared_feature.shape)
        # flow_diff_loss = loss_diff(flow_private_feature, flow_shared_feature)

        # flow_loss = flow_task_loss + flow_shared_loss + flow_private_loss + flow_diff_loss
        flow_loss = flow_task_loss + 1*(flow_shared_loss + flow_private_loss)
        flow_train_average_MSE += flow_task_loss.item()
        # print('Flow loss cal over')

        optimizer.zero_grad()
        flow_loss.backward()
        optimizer.step()

        # print('flow loss over')

        # od train
        od_private_feature, od_private_domain_label, od_shared_feature, od_domain_label, od_out = od_result

        # print(od_out.shape, od_label.shape)

        od_task_loss = loss_task(od_out, od_label.permute(0, 1, 4, 2, 3))
        od_shared_loss = loss_discriminator(od_domain_label, od_domain_true_label)
        od_private_loss = loss_discriminator(od_private_domain_label, od_domain_true_label)
        # od_diff_loss = loss_diff(od_private_feature, od_shared_feature)

        # od_loss = od_task_loss + od_shared_loss + od_private_loss + od_diff_loss
        od_loss = 1*od_task_loss + od_shared_loss + od_private_loss
        od_train_average_MSE += od_task_loss.item()
        # print('OD loss cal over')
        optimizer.zero_grad()
        od_loss.backward()
        optimizer.step()
        # print('od loss over')

        # all_loss = flow_loss + od_loss
        # optimizer.zero_grad()
        # all_loss = 0.01*flow_loss + od_loss
        # # all_loss = flow_loss
        # all_loss.backward()
        # optimizer.step()
        # t2 = time.time()
        # print('Epoch: [{}/{}], Flow Loss: {:.4f}, OD Loss: {:.4f}, Cost: {:.4f}'.format(epoch, n_epoch,
        #                                                                                 flow_train_average_MSE,
        #                                                                                 od_train_average_MSE,
        #                                                                                 (t2-t1)/60))
        if (i+1) % 40 == 0:
            t5 = time.time()
            # print('Epoch: [{}/{}], step: [{}/{}], flow loss: {:.4f}, flow task loss: {:.4f}, od loss: {:.4f}, '
            #       'od task loss: {:.4f} '
            #       'cost: {:.4f}min'.format(epoch+1, n_epoch, i+1, len_flow_train_loader, flow_loss.item(),
            #                                flow_task_loss.item(), od_loss.item(), od_task_loss.item(), (t2-t1)/60))
            logger.info('Epoch: [{}/{}], step: [{}/{}], flow loss: {:.4f}, flow task loss: {:.4f}, od loss: {:.4f}, '
                        'od task loss: {:.4f} '
                        'cost: {:.4f}min'.format(epoch+1, n_epoch, i+1, len_flow_train_loader, flow_loss.item(),
                                                 flow_task_loss.item(), od_loss.item(), od_task_loss.item(), (t5-t)/60))
        i += 1
        # print('Over')
    flow_train_average_MSE /= len_flow_train_loader
    # flow_loss_list.append(flow_train_average_MSE)
    od_train_average_MSE /= len_od_train_loader
    # od_loss_list.append(od_train_average_MSE)
    t6 = time.time()
    # torch.save(my_net.state_dict(), 'STMulti_new20200415.pkl')
    # if (epoch+1) > 99 and (epoch+1) % 10 == 0:
    #     torch.save(my_net.state_dict(), 'STMulti_new20200427_{}.pkl'.format(epoch+1))
    logger.info('Epoch: [{}/{}], Flow Loss: {:.4f}, OD Loss: {:.4f}, Cost: {:.4f}min'.format(epoch+1, n_epoch,
                                                                                             flow_train_average_MSE,
                                                                                             od_train_average_MSE,
                                                                                             (t6-t1)/60))
    if (epoch + 1) % 5 == 0:
        my_net.eval()
        with torch.no_grad():
            j = 0
            while j < len_flow_validation_loader:
                flow_val_img, flow_val_label = flow_validate_iter.next()
                flow_val_img, flow_val_label = flow_val_img.permute(0, 2, 1, 3, 4).float().to(device), flow_val_label.permute(0, 2,
                                                                                                                      1, 3,
                                                                                                                      4).float().to(
                    device)
                od_val_img, od_val_label = od_validate_iter.next()
                od_val_img, od_val_label = od_val_img.permute((0, 4, 1, 2, 3)).float().to(device), od_val_label.permute(
                    (0, 4, 1, 2, 3)).float().to(device)
    
                node_val_feature, _ = node_features_val_iter.next()
                node_val_feature = node_val_feature.permute((0, 1, 3, 2)).float().to(device)
                adjs_val, _ = adjs_val_iter.next()
                adjs_val = adjs_val.permute((0, 1, 3, 2)).float().to(device)
    
                p = float(i + epoch * len_flow_validation_loader) / n_epoch / len_flow_validation_loader
                alpha = 2./(1.+np.exp(-10*p)) - 1
    
                flow_val_result, od_val_result = my_net(flow_val_img, od_val_img, node_val_feature, adjs_val, alpha)
                _, _, _, _, flow_val_out = flow_val_result
                _, _, _, _, od_val_out = od_val_result
                flow_val_loss = loss_task(flow_val_out, flow_val_label)
                od_val_loss = loss_task(od_val_out, od_val_label)
                mse = flow_val_loss.item() + od_val_loss.item()
                validate_average_MSE += mse
                j += 1
            average_mse = validate_average_MSE / len_flow_validation_loader
            if average_mse < best_MSE:
                best_MSE = average_mse
                torch.save(my_net.state_dict(), '/ceph_10826/halemiao/STMulti_20201012.pkl')
    
    logger.info('Epoch: [{}/{}], Flow Loss: {:.4f}, OD Loss: {:.4f}, best MSE: {:.4f}'.format(epoch+1, n_epoch,
                                                                                              flow_train_average_MSE,
                                                                                              od_train_average_MSE,
                                                                                              best_MSE))
# f.close()
#flow_loss = np.array(flow_loss_list)
#np.save('/ceph_10826/halemiao/flow_loss20201010.npy', flow_loss)
#od_loss = np.array(od_loss_list)
#np.save('/ceph_10826/halemiao/od_loss20201010.npy', od_loss)
