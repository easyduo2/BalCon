import numpy as np
import torch
import os
from utils import load_data, set_params, clustering_metrics
from module.BalCon import *
from module.preprocess import *
import warnings
import datetime
import random
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
warnings.filterwarnings('ignore')
args = set_params()

if torch.cuda.is_available():
    device = torch.device("cuda:" + str(args.gpu))
    torch.cuda.set_device(args.gpu)
else:
    device = torch.device("cpu")

## random seed ##
def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def format_time(time):
    elapsed_rounded = int(round((time)))
    return str(datetime.timedelta(seconds=elapsed_rounded))


def train(args):
    
    feat, adjs, label = load_data(args.dataset)
    nb_classes = label.shape[-1]
    print("cluster number: ", nb_classes)
    num_target_node = len(feat)

    print_flag = True
    plot_flag = False

    feats_dim = feat.shape[1]
    sub_num = int(len(adjs))
    print("Dataset: ", args.dataset)
    print("The number of meta-paths: ", sub_num)
    print("Number of target nodes:", num_target_node)
    print("The dim of target' nodes' feature: ", feats_dim)
    print("Label: ", label.sum(dim=0))
    print(args)

    if torch.cuda.is_available():
        print('Using CUDA')
        adjs = [adj.cuda() for adj in adjs]
        feat = feat.cuda()

    adjs_o = graph_process(adjs, feat, args)

    f_list = APPNP([feat for _ in range(sub_num)], adjs_o, args.nlayer, args.filter_alpha)
    dominant_index = pre_compute_dominant_view(f_list, feat)

    model = BalCon(feats_dim, sub_num, args.hidden_dim, args.embed_dim, nb_classes, args.tau, args.dropout, len(feat), dominant_index, args.nlayer, device, args.alpha, args.beta)
    
    optimizer_discriminator = torch.optim.Adam(filter(lambda p: p.requires_grad, model.discriminator.parameters()), lr=args.lr, weight_decay=args.l2_coef)
    optimizer_encoder = torch.optim.Adam(filter(lambda p: p.requires_grad, model.online_encoder.parameters()), lr=args.lr, weight_decay=args.l2_coef)
    optimizer_decoder = torch.optim.Adam(filter(lambda p: p.requires_grad, model.decoder.parameters()), lr=args.lr, weight_decay=args.l2_coef)

    def set_train():
        model.online_encoder.train()
        model.decoder.train()
        model.discriminator.train()
    
    def set_eval():
        model.online_encoder.eval()
        model.decoder.eval()
        model.discriminator.eval()

    if torch.cuda.is_available():
        model.cuda()

    period = 50
    fh = open("result_" + args.dataset + "_NMI&ARI.txt", "a")
    print(args, file=fh)
    fh.write('\r\n')
    fh.flush()
    fh.close()

    if args.load_parameters == False:

        for epoch in range(args.nb_epochs):
            flag = False
            set_train()

            """discriminator update"""
            # print(feat.shape)
            loss_d = model.forward_discriminator(feat, f_list)

            optimizer_discriminator.zero_grad()
            loss_d.backward()
            optimizer_discriminator.step()

            """autoencoder update"""
            optimizer_encoder.zero_grad()
            optimizer_decoder.zero_grad()
            if (epoch+1) % period == 0:
                flag = True
            warm_up = True if epoch < 100 else False
            loss, result = model(feat, f_list, flag, warm_up)

            loss.backward()
            optimizer_encoder.step()
            optimizer_decoder.step()

            print("Epoch:", epoch)
            if epoch == 0:
                loss_history = {
                    'ae_loss': [],
                    'err_loss': [], 
                    'clu_loss': [],
                    'contrastive_loss': [],
                    'd_loss': []
                }
            
            loss_history['ae_loss'].append(result['ae_loss'])
            loss_history['err_loss'].append(result['err_loss'])
            loss_history['clu_loss'].append(result['clu_loss']) 
            loss_history['contrastive_loss'].append(result['contrastive_loss'])
            loss_history['d_loss'].append(loss_d.item())

            if print_flag:
                print('ae_loss: ', result['ae_loss'], 'err_loss: ', result['err_loss'], 
                      'clu_loss: ', result['clu_loss'], 'contrastive_loss: ', result['contrastive_loss'],
                      'd_loss: ', loss_d.item())
            if epoch == args.nb_epochs - 1 and plot_flag:
                for loss_type in loss_history.keys():
                    plt.figure(figsize=(10,6))
                    plt.plot(loss_history[loss_type], label=loss_type)
                    plt.xlabel('Epoch')
                    plt.ylabel('Loss')
                    plt.legend()
                    plt.title(f'{loss_type} Training Curve')
                    plt.savefig(f'{loss_type}_{args.dataset}.png')
                    plt.close()
            if (epoch + 1) % period == 0:
                model.eval()
                embeds = model.get_embeds(f_list).cpu().numpy()

                estimator = KMeans(n_clusters=nb_classes, random_state=args.seed)
                ACC_list = []
                F1_list = []
                NMI_list = []
                ARI_list = []
                for _ in range(10):
                    estimator.fit(embeds)
                    y_pred = estimator.predict(embeds)
                    cm = clustering_metrics(torch.argmax(label, dim=-1).numpy(), y_pred, args.dataset)
                    ac, nm, f1, ari = cm.evaluationClusterModelFromLabel()

                    ACC_list.append(ac)
                    F1_list.append(f1)
                    NMI_list.append(nm)
                    ARI_list.append(ari)
                acc = sum(ACC_list) / len(ACC_list)
                f1 = sum(F1_list) / len(F1_list)
                ari = sum(ARI_list) / len(ARI_list)
                nmi = sum(NMI_list) / len(NMI_list)

                if print_flag:
                    print(
                        '\t[Clustering] ACC: {:.2f}   F1: {:.2f}  NMI: {:.2f}   ARI: {:.2f} \n'.format(np.round(acc * 100, 2),
                                                                                                    np.round(f1 * 100, 2),
                                                                                                np.round(nmi * 100, 2),
                                                                                                np.round(ari * 100, 2)))
                fh = open("result_" + args.dataset + "_NMI&ARI.txt", "a")
                fh.write(
                    'ACC=%f, f1_macro=%f,  NMI=%f, ADJ_RAND_SCORE=%f' % (acc, f1, nmi, ari))
                fh.write('\r\n')
                fh.flush()
                fh.close()
        if not os.path.exists('./checkpoint/' + args.dataset):
            os.makedirs('./checkpoint/' + args.dataset)

        torch.save(model.state_dict(), './checkpoint/' + args.dataset + '/best_' + str(args.seed) + '.pth')

    else:
        model.load_state_dict(torch.load('./best/' + args.dataset + '/best_' + str(0) + '.pth'))

    model.cuda()
    print("---------------------------------------------------")
    model.eval()
    embeds = model.get_embeds(f_list).cpu().numpy()

    estimator = KMeans(n_clusters=nb_classes)
    ACC_list = []
    F1_list = []
    NMI_list = []
    ARI_list = []
    for _ in range(10):
        estimator.fit(embeds)
        y_pred = estimator.predict(embeds)
        cm = clustering_metrics(torch.argmax(label, dim=-1).numpy(), y_pred, args.dataset)
        ac, nm, f1, ari = cm.evaluationClusterModelFromLabel()

        ACC_list.append(ac)
        F1_list.append(f1)
        NMI_list.append(nm)
        ARI_list.append(ari)
    acc = sum(ACC_list) / len(ACC_list)
    f1 = sum(F1_list) / len(F1_list)
    ari = sum(ARI_list) / len(ARI_list)
    nmi = sum(NMI_list) / len(NMI_list)

    print('\t[Clustering] ACC: {:.2f}   F1: {:.2f}  NMI: {:.2f}   ARI: {:.2f} \n'.format(np.round(acc*100,2), np.round(f1*100,2),
                                                                                      np.round(nmi*100,2), np.round(ari*100,2)))
    fh = open("result_" + args.dataset + "_NMI&ARI.txt", "a")
    fh.write(
        'ACC=%f, f1_macro=%f,  NMI=%f, ADJ_RAND_SCORE=%f' % (acc, f1, nmi, ari))
    fh.write('\r\n')
    fh.write('---------------------------------------------------------------------------------------------------')
    fh.write('\r\n')
    fh.flush()
    fh.close()
    return np.round(acc*100,2), np.round(f1*100,2), np.round(nmi*100,2), np.round(ari*100,2)


if __name__ == '__main__':

    set_seed(args.seed)
    train(args)

