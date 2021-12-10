import  os
import  torch
import  tqdm
import  numpy as np
from torch.utils.data import  Dataset, DataLoader
from sklearn.metrics import classification_report, accuracy_score
from torch.optim.optimizer import Optimizer

from src.base.logger import TxtLogger
from src.base.meter import AverageMeter

class Text_Classification_Learner():
    def __init__(self,
                 cfg,
                 model : torch.nn.Module,
                 loss_fn,
                 optimizer: Optimizer,
                 scheduler,
                 logger : TxtLogger,
                 save_dir : str,
                 log_steps = 100,
                 device = 'cuda',
                 gradient_accum_steps = 1,
                 max_grad_norm = 1.0,
                 early_stop_n = 4,
                 ):
        self.cfg = cfg
        self.model  = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_dir = save_dir
        self.log_steps = log_steps
        self.logger = logger
        self.device = device
        self.gradient_accum_steps = gradient_accum_steps
        self.max_grad_norm = max_grad_norm
        self.early_stop_n = early_stop_n
        self.global_step = 0

    def step(self, step_n,  label, text):
        loggits = self.model(text)
        label = label.squeeze()
        loss = self.loss_fn(loggits, label)
        loss = loss.mean()
        if self.gradient_accum_steps > 1:
            loss = loss / self.gradient_accum_steps
        loss.backward()

        loggits = loggits.detach().cpu().numpy()
        pred = np.argmax(loggits, axis=1)
        label = label.detach().cpu().numpy()
        acc = self._accuracy(pred, label)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        if (step_n + 1) % self.gradient_accum_steps == 0:
            self.optimizer.step()
            self.scheduler.step()  # Update learning rate schedule
            self.model.zero_grad()
            self.global_step += 1
        return  loss,acc

    def _accuracy(self,preds, labels):
        return accuracy_score(y_pred= preds, y_true= labels)#(preds == labels).mean()

    def _acc_and_f1(self, preds, labels):
        acc = self._accuracy(preds, labels)
        print(classification_report(y_pred= preds, y_true= labels))
        return {
            "acc": acc,
        }


    def val(self, val_dataloader : DataLoader):
        eval_loss = 0.0
        all_preds = None
        all_labels = None
        self.model.eval()
        for label, text in tqdm.tqdm(val_dataloader):
            with torch.no_grad():
                label, text = label.to(self.device), text.to(self.device)
                pred_loggits = self.model( text)
                label = label.squeeze()
                loss = self.loss_fn(pred_loggits,label )
                eval_loss += loss.mean().item()
            if all_preds is None:
                all_preds = pred_loggits.detach().cpu().numpy()
                all_labels = label.detach().cpu().numpy()
            else:
                all_preds = np.append(all_preds, pred_loggits.detach().cpu().numpy(), axis=0)
                all_labels = np.append(all_labels, label.detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / len(val_dataloader)
        all_preds = np.argmax(all_preds, axis=1)
        self.logger.write("steps: {} ,mean eval loss : {:.4f} ". \
                                      format(self.global_step, eval_loss))
        print("all preds shape: ", all_preds.shape)
        result =   self._acc_and_f1(all_preds, all_labels)
        return result

    def train(self, train_dataloader : DataLoader,
              val_dataloader : DataLoader,
              epoches = 100):
        best_score = 0
        early_n = 0
        for epo in range(epoches):
            step_n = 0
            train_avg_loss = AverageMeter()
            train_avg_acc = AverageMeter()
            data_iter = tqdm.tqdm(train_dataloader)
            for label, text in data_iter:
                label, text = label.to(self.device), text.to(self.device)
                self.model.train()
                train_loss, acc = self.step(step_n, label, text)
                train_avg_loss.update(train_loss.item(),1)
                train_avg_acc.update(acc,1)
                status = '[{0}] lr= {1:.6f} loss= {2:.3f} avg_loss= {3:.4f} avg_acc={4:.3f} '.format(
                    epo + 1, self.scheduler.get_lr()[0],
                    train_loss.item(), train_avg_loss.avg, train_avg_acc.avg )
                #if step_n%self.log_steps ==0:
                #    print(status)
                data_iter.set_description(status)
                step_n +=1

            ##self.scheduler.step() ## we update every step instead
            if True:
                ## val
                m = self.val(val_dataloader)
                acc = m['acc']
                if best_score < acc:
                    early_n = 0
                    best_score = acc
                    model_path = os.path.join(self.save_dir, 'best.pth')
                    torch.save(self.model.state_dict(), model_path)
                else:
                    early_n += 1
                self.logger.write("steps: {} ,mean ap : {:.4f} , best ap: {:.4f}". \
                                      format(self.global_step, acc, best_score))
                self.logger.write(str(m))
                self.logger.write("=="*50)

                if early_n > self.early_stop_n:
                    print('early stopped!')
                    return best_score
        return  best_score