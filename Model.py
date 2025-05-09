from architectures.ArchitecturesUtils import *
from architectures.Evaluations import *
from architectures.UNet import *
from interpret_segmentation import hdm
import datetime
import numpy as np
import os
import matplotlib.pyplot as plt
import json


class TrainerUnet():
    def __init__(self, input_shape, opt):
        self.c, self.h, self.w = input_shape
        self.tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.epochs = opt.n_epochs
        self.batch_size = opt.batch_size
        today = datetime.date.today()
        self.today = today.strftime("%d_%m_%y")

        self.bce = nn.BCEWithLogitsLoss()

        self.net = Unet(self.c, self.c).to(self.device)

        self.optimizer = torch.optim.RMSprop(self.net.parameters(), lr=opt.lr, weight_decay=1e-7, momentum=0.9)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5)


        self.logs = {'epochs': [], 'loss': [], 'mIoU': [], 'mDice': []}
        self.model_dir = os.path.join("ExportedModels")
        return

    def train(self, data_loader_t, data_loader_v, sample_interval=10, data_name='lung'):
        best_iou = 0
        best_dice = 0
        for epoch in range(self.epochs):
            loss_list = []
            self.net.train()
            for batch_i, (imgs, segs) in enumerate(data_loader_t):
                start_time = datetime.datetime.now()

                imgs = imgs.to(self.device)
                segs = segs.to(self.device)

                self.optimizer.zero_grad()

                fake_segs = self.net(imgs)

                segmentation_loss = self.bce(fake_segs, segs)
                loss = segmentation_loss

                loss.backward()
                nn.utils.clip_grad_value_(self.net.parameters(), 0.1)
                self.optimizer.step()

                loss_list.append(loss.item())

                elapsed_time = datetime.datetime.now() - start_time

                # print the progress
                print(
                    "[ Epoch %d/%d ] [ Batch %d/%d ] [ loss: %06f ] [ time: %s ]" \
                    % (epoch, self.epochs,
                       batch_i, len(data_loader_t),
                       loss.item(),
                       elapsed_time))


            if epoch % sample_interval == 0:
                self.sampleImages(epoch, batch_i, data_loader_v, data_name)
            m_iou, m_dice = self.evaluate(data_loader_v)
            self.scheduler.step(m_dice)
            if m_iou > best_iou and m_dice > best_dice:
                self.saveModels(data_name)
                best_dice = m_dice
                best_iou = m_iou

            # save epochs statistics
            self.logs['epochs'].append(epoch)
            self.logs['loss'].append(loss_list)
            self.logs['mIoU'].append(m_iou)
            self.logs['mDice'].append(m_dice)

        json.dump(self.logs, open("{:}_{:}_train_stat.json".format(data_name, self.epochs), "w"))
        return

    def sampleImages(self, epoch, batch_i, data_loader, data_name):

        self.net.eval()

        r, c = 1, 3
        for img_idx, (imgs, segs) in enumerate(data_loader):
            imgs = imgs.to(self.device)
            segs = segs.to(self.device)

            os.makedirs('images/%s' % data_name, exist_ok=True)

            # Translate images to the other domain
            fake_segs = self.net(imgs)
            fake_segs = torch.sigmoid(fake_segs)

            gen_imgs = [imgs, segs, fake_segs]
            titles = ['Img', 'Seg', 'Gen Seg']
            fig, axs = plt.subplots(r, c)
            cnt = 0
            for i in range(r):
                for j in range(c):
                    if j == 2:
                        im = gen_imgs[cnt]
                        im = im.squeeze().cpu().detach().numpy()
                        im = im > 0.5
                        im = im.astype(np.uint8)
                    else:
                        im = 0.5 * gen_imgs[cnt].squeeze().cpu().detach().numpy() + 0.5
                    axs[j].imshow(im)
                    axs[j].set_title(titles[j])
                    axs[j].axis('off')
                    cnt += 1
            fig.savefig("images/%s/%d_%d_%d.png" % (data_name, img_idx, epoch, batch_i))
            plt.close()

    def evaluate(self, loader):
        self.net.eval()
        ious = np.zeros(len(loader))
        dices = np.zeros(len(loader))
        print(15 * "~-")
        for i, (im, seg) in enumerate(loader):
            im = im.to(self.device)
            seg = seg

            pred = self.net(im)
            pred = torch.sigmoid(pred)
            pred = pred.cpu()
            pred = pred.detach().numpy()[0, 0, :, :]
            mask = seg.numpy()[0, 0, :, :]

            # Binarize masks
            gt = mask > 0.5
            pr = pred > 0.5

            ious[i] = IoU(gt, pr)
            dices[i] = Dice(gt, pr)
            print("results: IoU: {:f}, DICE: {:f}".format(ious[i], dices[i]))

        print('Mean IoU:', ious.mean())
        print('Mean Dice:', dices.mean())
        print(15 * "~-")
        return ious.mean(), dices.mean()

    def saveModels(self, data_name):
        print("Saving Model")
        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)

        torch.save(self.net.state_dict(), os.path.join(self.model_dir, "{:}_unet_E_{:}.pt".format(data_name,
                                                                                                      self.epochs)))
        return


class TesterUnet():
    def __init__(self, input_shape, opt):
        self.c, self.h, self.w = input_shape
        self.tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.epochs = opt.n_epochs
        self.batch_size = opt.batch_size

        self.net = Unet(self.c, self.c).to(self.device)

    def test(self, data_loader, data_name):
        g_path = os.path.join("ExportedModels",
                              "{:}_unet_E_{:}.pt".format(data_name, self.epochs))
        if self.device == 'cuda':
            self.net.load_state_dict(torch.load(g_path))
        else:
            self.net.load_state_dict(torch.load(g_path, map_location = 'cpu'))

        self.net.eval()
        ious = np.zeros(len(data_loader))
        dices = np.zeros(len(data_loader))
        hds = np.zeros(len(data_loader))
        HD = hdm.HausdorffDistanceMasks(256, 256)
        print(15 * "~-")
        for i, (im, seg) in enumerate(data_loader):
            im = im.to(self.device)
            seg = seg

            pred = self.net(im)
            pred = torch.sigmoid(pred)
            pred = pred.cpu()
            pred = pred.detach().numpy()[0, 0, :, :]
            mask = seg.detach().numpy()[0, 0, :, :]
            # Binarize masks
            gt = mask > 0.5
            pr = pred > 0.5

            ious[i] = IoU(gt, pr)
            dices[i] = Dice(gt, pr)
            hds[i] = HD.calculate_distance(pr, gt)
            print("results: IoU: {:f}, DICE: {:f}, HD: {:f}".format(ious[i], dices[i], hds[i]))

        print('Mean IoU:', ious.mean())
        print('Mean Dice:', dices.mean())
        print('Mean HD:', hds.mean())
        print(15 * "~-")
        return







