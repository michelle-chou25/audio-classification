import sys
import os
import datetime

sys.path.append(os.path.dirname(os.path.dirname(sys.path[0])))
from utilities import *
import time
import torch
from torch import nn
import numpy as np
import pickle
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from generate_log import logger


class FocalLoss(nn.Module):
    def __init__(self, class_num, alpha=1., gamma=3, size_average=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha

        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets, device=0):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs, dim=1)

        class_mask = targets

        # if inputs.is_cuda and not self.alpha.is_cuda:
        #     self.alpha = self.alpha.cuda(device)
        alpha = self.alpha

        probs = (P * class_mask).sum(1).view(-1, 1)

        log_p = probs.log()
        batch_loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p
        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss


def train(audio_model, train_loader, test_loader, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info('running on ' + str(device))
    if args.tensorboard:
        logdir = args.tensorboard
        writer = SummaryWriter(logdir=logdir)
    torch.set_grad_enabled(True)

    # Initialize all of the statistics we want to keep track of
    batch_time = AverageMeter()
    per_sample_time = AverageMeter()
    data_time = AverageMeter()
    per_sample_data_time = AverageMeter()
    loss_meter = AverageMeter()
    per_sample_dnn_time = AverageMeter()
    progress = []
    # best_ensemble_mAP is checkpoint ensemble from the first epoch to the best epoch
    best_epoch, best_ensemble_epoch, best_mAP, best_acc, best_ensemble_mAP = 0, 0, -np.inf, -np.inf, -np.inf
    global_step, epoch = 0, 0
    start_time = time.time()
    exp_dir = args.exp_dir

    def _save_progress():
        progress.append([epoch, global_step, best_epoch, best_mAP, time.time() - start_time])
        with open("%s/progress.pkl" % exp_dir, "wb") as f:
            pickle.dump(progress, f)

    if not isinstance(audio_model, nn.DataParallel):
        audio_model = nn.DataParallel(audio_model)

    audio_model = audio_model.to(device)
    # Set up the optimizer
    trainables = [p for p in audio_model.parameters() if p.requires_grad]
    logger.info('Total parameter number is : {:.3f} million'.format(sum(p.numel() for p in audio_model.parameters()) / 1e6))
    logger.info('Total trainable parameter number is : {:.3f} million'.format(sum(p.numel() for p in trainables) / 1e6))
    optimizer = torch.optim.Adam(trainables, args.lr, weight_decay=args.weight_decay, betas=(0.95, 0.999))

    # dataset specific settings
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=args.lr_patience, verbose=True)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, list(range(args.lrscheduler_start, 1000, 5)),
                                                     gamma=args.lrscheduler_decay, last_epoch=epoch - 1)
    main_metrics = args.metrics
    if args.loss == 'BCE':
        loss_fn = nn.BCELoss()
        # loss_fn = FocalLoss(2, size_average=True)
    elif args.loss == 'CE':
        loss_fn = nn.CrossEntropyLoss()

    warmup = args.warmup
    args.loss_fn = loss_fn
    logger.info('now training with {:s}, main metrics: {:s}, loss function: {:s}, learning rate scheduler: {:s}'.format(
        str(args.dataset), str(main_metrics), str(loss_fn), str(scheduler)))
    logger.info('The learning rate scheduler starts at {:d} epoch with decay rate of {:.3f} '.format(args.lrscheduler_start,
                                                                                               args.lrscheduler_decay))

    epoch += 1

    logger.info("current #steps=%s, #epochs=%s" % (global_step, epoch))
    logger.info("start training...")
    result = np.zeros([args.n_epochs, 10])
    audio_model.train()
    while epoch < args.n_epochs + 1:
        begin_time = time.time()
        end_time = time.time()
        audio_model.train()
        logger.info('---------------')
        logger.info(datetime.datetime.now())
        logger.info("current #epochs=%s, #steps=%s" % (epoch, global_step))

        for i, (audio_input, labels) in enumerate(train_loader):

            B = audio_input.size(0)
            audio_input = audio_input.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            data_time.update(time.time() - end_time)
            per_sample_data_time.update((time.time() - end_time) / audio_input.shape[0])
            dnn_start_time = time.time()

            # first several steps for warm-up
            if global_step <= 1000 and global_step % 50 == 0 and warmup == True:
                warm_lr = (global_step / 1000) * args.lr
                for param_group in optimizer.param_groups:
                    param_group['lr'] = warm_lr
                logger.info('warm-up learning rate is {:f}'.format(optimizer.param_groups[0]['lr']))

            audio_output = audio_model(audio_input)
            if isinstance(loss_fn, torch.nn.CrossEntropyLoss):
                loss = loss_fn(audio_output, torch.argmax(labels.long(), axis=1))
            else:
                epsilon = 1e-7
                audio_output = torch.clamp(audio_output, epsilon, 1. - epsilon)
                # labels[labels == 1.] = 0.94
                # labels[labels == 0.] = 0.06
                loss = loss_fn(audio_output, labels)

            # optimization if amp is not used
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # record loss
            loss_meter.update(loss.item(), B)
            batch_time.update(time.time() - end_time)
            per_sample_time.update((time.time() - end_time) / audio_input.shape[0])
            per_sample_dnn_time.update((time.time() - dnn_start_time) / audio_input.shape[0])

            print_step = global_step % args.n_print_steps == 0
            early_print_step = epoch == 0 and global_step % (args.n_print_steps / 10) == 0
            print_step = print_step or early_print_step

            # Generate step-loss curve
            if args.tensorboard:
                writer.add_scalar(
                    tag="loss/step",
                    scalar_value=loss.item(),
                    global_step=global_step)
            end_time = time.time()
            global_step += 1

            if print_step and global_step == len(train_loader):
                logger.info('Epoch: [{0}][{1}/{2}]\t'
                      'Per Sample Total Time {per_sample_time.avg:.5f}\t'
                      'Per Sample Data Time {per_sample_data_time.avg:.5f}\t'
                      'Per Sample DNN Time {per_sample_dnn_time.avg:.5f}\t'
                      'Train Loss {loss_meter.avg:.4f}\t'.format(
                    epoch, i, len(train_loader), per_sample_time=per_sample_time,
                    per_sample_data_time=per_sample_data_time,
                    per_sample_dnn_time=per_sample_dnn_time, loss_meter=loss_meter))
                if np.isnan(loss_meter.avg):
                    logger.info("training diverged...")
                    return

        logger.info('start validation')
        logger.info('validate:')
        stats, valid_loss, class_acc = validate(audio_model, test_loader, args, epoch)

        # ensemble results
        logger.info('validate ensemble:')
        ensemble_stats = validate_ensemble(args, epoch)
        ensemble_mAP = np.mean([stat['AP'] for stat in ensemble_stats])
        ensemble_mAUC = np.mean([stat['auc'] for stat in ensemble_stats])
        ensemble_acc = ensemble_stats[0]['acc']

        mAP = np.mean([stat['AP'] for stat in stats])
        mAUC = np.mean([stat['auc'] for stat in stats])
        acc = stats[0]['acc']

        # Generate epoch-accuracy curve
        if args.tensorboard:
            writer.add_scalar(
                tag="accuracy/epoch",
                scalar_value=acc,
                global_step=epoch)
        # Generate the loss of validation dataset - epoch  curve
        if args.tensorboard:
            writer.add_scalar(
                tag="valid_loss/epoch",
                scalar_value=valid_loss,
                global_step=epoch)

        middle_ps = [stat['precisions'][int(len(stat['precisions']) / 2)] for stat in stats]
        middle_rs = [stat['recalls'][int(len(stat['recalls']) / 2)] for stat in stats]
        average_precision = np.mean(middle_ps)
        average_recall = np.mean(middle_rs)

        if main_metrics == 'mAP':
            logger.info("mAP: {:.6f}".format(mAP))
        else:
            logger.info("acc: {:.4f} (ng:{:.3f}, ok:{:.3f})".format(acc, class_acc["ng"], class_acc["ok"]))
        logger.info("AUC: {:.6f}".format(mAUC))
        logger.info("Avg Precision: {:.6f}".format(average_precision))
        logger.info("Avg Recall: {:.6f}".format(average_recall))
        logger.info("d_prime: {:.6f}".format(d_prime(mAUC)))
        logger.info("train_loss: {:.6f}".format(loss_meter.avg))
        logger.info("valid_loss: {:.6f}".format(valid_loss))

        if main_metrics == 'mAP':
            result[epoch - 1, :] = [mAP, mAUC, average_precision, average_recall, d_prime(mAUC), loss_meter.avg,
                                    valid_loss, ensemble_mAP, ensemble_mAUC, optimizer.param_groups[0]['lr']]
        else:
            result[epoch - 1, :] = [acc, mAUC, average_precision, average_recall, d_prime(mAUC), loss_meter.avg,
                                    valid_loss, ensemble_acc, ensemble_mAUC, optimizer.param_groups[0]['lr']]
        np.savetxt(exp_dir + '/result.csv', result, delimiter=',')
        logger.info('validation finished')

        if mAP > best_mAP:
            best_mAP = mAP
            if main_metrics == 'mAP':
                best_epoch = epoch

        if acc > best_acc:
            best_acc = acc
            if main_metrics == 'acc':
                best_epoch = epoch
                best_loss = valid_loss

        if ensemble_mAP > best_ensemble_mAP:
            best_ensemble_epoch = epoch
            best_ensemble_mAP = ensemble_mAP
        logger.info(f'@@@@@@@@@@@@@@@@@@@@best epoch:{best_epoch} acc:{best_acc} loss:{best_loss}')
        if best_epoch == epoch:
            torch.save(audio_model.state_dict(), "%s/models/best_audio_model.pth" % (exp_dir))
            torch.save(optimizer.state_dict(), "%s/models/best_optim_state.pth" % (exp_dir))

        torch.save(audio_model.state_dict(), "%s/models/audio_model.%d.pth" % (exp_dir, epoch))
        if len(train_loader.dataset) > 2e5:
            torch.save(optimizer.state_dict(), "%s/models/optim_state.%d.pth" % (exp_dir, epoch))

        scheduler.step()
        # Generate epoch-learning rate curve
        if args.tensorboard:
            writer.add_scalar(
                tag="learning_rate/epoch",
                scalar_value=optimizer.param_groups[0]['lr'],
                global_step=epoch)
        logger.info('Epoch-{0} lr: {1}'.format(epoch, optimizer.param_groups[0]['lr']))

        with open(exp_dir + '/stats_' + str(epoch) + '.pickle', 'wb') as handle:
            pickle.dump(stats, handle, protocol=pickle.HIGHEST_PROTOCOL)
        _save_progress()

        finish_time = time.time()
        logger.info('epoch {:d} training time: {:.3f}'.format(epoch, finish_time - begin_time))

        epoch += 1

        batch_time.reset()
        per_sample_time.reset()
        data_time.reset()
        per_sample_data_time.reset()
        loss_meter.reset()
        per_sample_dnn_time.reset()

    # if test weight averaging
    if args.wa == True:
        stats = validate_wa(audio_model, test_loader, args, args.wa_start, args.wa_end)
        mAP = np.mean([stat['AP'] for stat in stats])
        mAUC = np.mean([stat['auc'] for stat in stats])
        middle_ps = [stat['precisions'][int(len(stat['precisions']) / 2)] for stat in stats]
        middle_rs = [stat['recalls'][int(len(stat['recalls']) / 2)] for stat in stats]
        average_precision = np.mean(middle_ps)
        average_recall = np.mean(middle_rs)
        wa_result = [mAP, mAUC]
        logger.info('---------------Training Finished---------------')
        # logger.info('On Validation Set')
        # logger.info('weighted averaged model results')
        # logger.info("mAP: {:.6f}".format(mAP))
        # logger.info("AUC: {:.6f}".format(mAUC))
        # logger.info("d_prime: {:.6f}".format(d_prime(mAUC)))
        np.savetxt(exp_dir + '/wa_result.csv', wa_result)


def validate(audio_model, val_loader, args, epoch, eval_target=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_time = AverageMeter()
    if not isinstance(audio_model, nn.DataParallel):
        audio_model = nn.DataParallel(audio_model)
    audio_model = audio_model.to(device)
    # switch to evaluate mode   
    audio_model.eval()
    end = time.time()
    A_predictions = []
    A_targets = []
    A_loss = []
    with torch.no_grad():
        for i, (audio_input, labels) in enumerate(val_loader):
            audio_input = audio_input.to(device)
            # compute output    
            audio_output = audio_model(audio_input)
            predictions = audio_output.to('cpu').detach()
            A_predictions.append(predictions)
            A_targets.append(labels)
            # compute the loss  
            labels = labels.to(device)
            epsilon = 1e-7
            audio_output = torch.clamp(audio_output, epsilon, 1. - epsilon)
            if isinstance(args.loss_fn, torch.nn.CrossEntropyLoss):
                loss = args.loss_fn(audio_output, torch.argmax(labels.long(), axis=1))
            else:
                loss = args.loss_fn(audio_output, labels)
            A_loss.append(loss.to('cpu').detach())
            batch_time.update(time.time() - end)
            end = time.time()
        audio_output = torch.cat(A_predictions)
        target = torch.cat(A_targets)
        loss = np.mean(A_loss)
        stats, class_acc = calculate_stats(audio_output, target)
        # save the prediction here  
        exp_dir = args.exp_dir
        if os.path.exists(exp_dir + '/predictions') == False:
            os.mkdir(exp_dir + '/predictions')
            np.savetxt(exp_dir + '/predictions/target.csv', target, delimiter=',')
        np.savetxt(exp_dir + '/predictions/predictions_' + str(epoch) + '.csv', audio_output, delimiter=',')
        # save the target for the separate eval set if there's one. 
        if eval_target == True and os.path.exists(exp_dir + '/predictions/eval_target.csv') == False:
            np.savetxt(exp_dir + '/predictions/eval_target.csv', target, delimiter=',')
    return stats, loss, class_acc


def validate_ensemble(args, epoch):
    exp_dir = args.exp_dir
    target = np.loadtxt(exp_dir + '/predictions/target.csv', delimiter=',')
    if epoch == 1:
        ensemble_predictions = np.loadtxt(exp_dir + '/predictions/predictions_1.csv', delimiter=',')
    else:
        ensemble_predictions = np.loadtxt(exp_dir + '/predictions/ensemble_predictions.csv', delimiter=',') * (
                epoch - 1)
        predictions = np.loadtxt(exp_dir + '/predictions/predictions_' + str(epoch) + '.csv', delimiter=',')
        ensemble_predictions = ensemble_predictions + predictions
        # remove the prediction file to save storage space
        os.remove(exp_dir + '/predictions/predictions_' + str(epoch - 1) + '.csv')

    ensemble_predictions = ensemble_predictions / epoch
    np.savetxt(exp_dir + '/predictions/ensemble_predictions.csv', ensemble_predictions, delimiter=',')

    stats, _ = calculate_stats(ensemble_predictions, target)
    return stats


def validate_wa(audio_model, val_loader, args, start_epoch, end_epoch):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp_dir = args.exp_dir

    sdA = torch.load(exp_dir + '/models/audio_model.' + str(start_epoch) + '.pth', map_location=device)

    model_cnt = 1
    for epoch in range(start_epoch, end_epoch + 1):
        sdB = torch.load(exp_dir + '/models/audio_model.' + str(epoch) + '.pth', map_location=device)
        for key in sdA:
            sdA[key] = sdA[key] + sdB[key]
        model_cnt += 1

        # if choose not to save models of epoch, remove to save space
        if args.save_model == False:
            os.remove(exp_dir + '/models/audio_model.' + str(epoch) + '.pth')

    # averaging
    for key in sdA:
        sdA[key] = sdA[key] / float(model_cnt)

    audio_model.load_state_dict(sdA)

    torch.save(audio_model.state_dict(), exp_dir + '/models/audio_model_wa.pth')

    stats, loss, _ = validate(audio_model, val_loader, args, 'wa')
    return stats
