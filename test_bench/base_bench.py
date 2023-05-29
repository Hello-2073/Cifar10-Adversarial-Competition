import abc

class BaseBench(abc.ABC):
    @abc.abstractclassmethod
    def __init__(self, n_cls):
        pass

    @abc.abstractclassmethod
    def test(self, imgs):
        pass

    @abc.abstractclassmethod
    def model(self):
        pass

    def train(self, data, iteration=200, lr=1e-3, betas=(5e-3, 5e-3), save_root='./checkpoints/'):
        optim = torch.optim.Adam(self.resnet.fc.parameters(), lr=lr, betas=betas)
        self.resnet.eval()
        for epoch in range(iteration):
            self.resnet.fc.train()
            total_loss = 0
            for step, (imgs, labels) in enumerate(data):
                if self.cuda:
                    imgs = imgs.cuda()
                    labels = labels.cuda()
                preds = self.resnet(imgs)
                loss = F.cross_entropy(preds, labels)
                optim.zero_grad()
                loss.backward()
                optim.step()
                total_loss += loss.item()
            print(total_loss)
            
            if epoch % 5 == 0:
                self.save(os.path.join(save_root, self.name, 'epoch{}.pth'.format(epoch)))
