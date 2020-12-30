#https://github.com/wujiyang/Face_Pytorch/blob/master/margin/CosineMarginProduct.py
#!/usr/bin/env python
# encoding: utf-8
'''
@author: wujiyang
@contact: wujiyang@hust.edu.cn
@file: CosineMarginProduct.py
@time: 2018/12/25 9:13
@desc: additive cosine margin for cosface
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter


class CosineMarginProduct(nn.Module):
    def __init__(self, in_feature=128, out_feature=10575, s=30.0, m=0.35):
        super(CosineMarginProduct, self).__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.s = s
        self.m = m
        self.weight = Parameter(torch.Tensor(out_feature, in_feature))
        nn.init.xavier_uniform_(self.weight)


    def forward(self, input, label):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        if not self.training:
            # For inference
            return self.s * (cosine - self.m) 
        
        one_hot = torch.zeros(cosine.size(), device='cuda' if torch.cuda.is_available() else 'cpu')
        #one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1.0)

        output = self.s * (cosine - one_hot * self.m)
        return output

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

### Model:
model = models.densenet201(pretrained=True)
num_ftrs = model.classifier.in_features
new_classifier = nn.Sequential(
                      nn.Linear(num_ftrs, 512), 
                      nn.ReLU()
                )
model.classifier = new_classifier
model.to(device)
### đầu ra của model là layer 512


### Margin layer , tiếp nối model, đầu vào là 512 phía trước, out_feature là
NUM_CLASSES = 7
margin = CosineMarginProduct(in_feature=512, out_feature=NUM_CLASSES, s=5, m=0.05)
margin.to(device)




### CrossEntropyLoss , optimizer riêng cho model, margin
criterion = torch.nn.CrossEntropyLoss().to(device)
criterion2 = torch.nn.CrossEntropyLoss().to(device)
optimizer_ft = optim.SGD([
  {'params': model.parameters()},
  {'params': margin.parameters(), 'weight_decay': 1e-4}
], lr=0.0001, momentum=0.9, nesterov=False)

MODEL_PATH = "best_model.pth"
MARGIN_PATH = "best_margin.pth"

#### TRAIN FUNCTION FOR LMCL
def train_model(model, criterion, optimizer, scheduler=None, num_epochs=25, MIN_LOSS=999999.0):
    """
    Support function for model training.

    Args:
      model: Model to be trained
      criterion: Optimization criterion (loss)
      optimizer: Optimizer to use for training
      scheduler: Instance of ``torch.optim.lr_scheduler``
      num_epochs: Number of epochs
      device: Device to run the training on. Must be 'cpu' or 'cuda'
    """
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_margin_wts = copy.deepcopy(margin.state_dict())
    best_acc = 0.0
    best_loss = MIN_LOSS
    best_wacc = 0.0
    ### history
    history = dict()
    history["loss"] = []
    history["acc"] = []
    history["val_loss"] = []
    history["val_acc"] = []
    history["wacc"] = []
    history["val_wacc"] = []
    history["precision"] = []
    history["val_precision"] = []
    history["f1"] = []
    history["val_f1"] = []

    margin.train = True

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)
        since_ = time.time()

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
                margin.training = True          
            else:
                model.eval()   # Set model to evaluate mode
                margin.training = False  

            running_loss = 0.0
            running_corrects = 0

            counter = 0
      
            #### Get predictions and true label when training
            y_true = torch.tensor([], dtype=torch.long, device=device)
            all_outputs = torch.tensor([], device=device)

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:        # if use pbar, replace dataloaders[phase] with pbar
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    
                    raw_logits = model(inputs)              ### raw_logits chi moi la 1920->512

                    outputs = margin(raw_logits, labels)    ### 512 -> 7
                    _, preds = torch.max(outputs.data, 1)   ### s*(cos - m)

                    if phase == 'train':
                        loss = criterion(outputs, labels)
                    if phase == 'val':                        ###Test for not use classweight on val
                        loss = criterion2(outputs, labels)

                    y_true = torch.cat((y_true, labels), 0)
                    all_outputs = torch.cat((all_outputs, outputs), 0)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                counter += 1

            if phase == 'train' and scheduler is not None:
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]


            epoch_wacc = 0.0
            #'''
            y_true = y_true.cpu().numpy()  
            _, y_pred = torch.max(all_outputs, 1)
            y_pred = y_pred.cpu().numpy()
            conf = confusion_matrix(y_true, y_pred)
            wacc = conf.diagonal()/conf.sum(axis=1)
            epoch_wacc = np.mean(wacc)   
            #'''
            print('{} Loss: {:.4f} Acc: {:.4f} WAcc: {:.4f}'.format(phase, epoch_loss, epoch_acc, epoch_wacc))
            #print(wacc)
            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_acc = epoch_acc
                best_wacc = epoch_wacc
                best_model_wts = copy.deepcopy(model.state_dict())
                best_margin_wts = copy.deepcopy(margin.state_dict())
                
                ### SAVE BEST STATE:
                torch.save(model.state_dict(), MODEL_PATH)
                torch.save(margin.state_dict(), MARGIN_PATH)
                print("Best state saved with val_loss= {:4f} - val_acc= {:4f} - val_wacc= {:4f}". format(epoch_loss, epoch_acc, epoch_wacc))
                print("Saved at " + MODEL_PATH)
                print("Saved at " + MARGIN_PATH)

            #### UPDATE PLOT VALUE
            if phase == 'train':
                history["loss"].append(epoch_loss)
                history["acc"].append(epoch_acc) 
                history["wacc"].append(epoch_wacc) 
            if phase == 'val':
                history["val_loss"].append(epoch_loss)
                history["val_acc"].append(epoch_acc) 
                history["val_wacc"].append(epoch_wacc) 

        time_elapsed_ = time.time() - since_
        if epoch < 5:
          print('One epoch complete in {:.0f}m {:.0f}s'.format(time_elapsed_ // 60, time_elapsed_ % 60))

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Loss: {:4f} - Best val Acc: {:4f} - Best val WAcc: {:4f}'.format(best_loss, best_acc, best_wacc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    margin.load_state_dict(best_margin_wts)
    return model, history
    
    
##### VAL PREDICT FOR LMCL
#https://discuss.pytorch.org/t/efficient-method-to-gather-all-predictions/8008/5
def pytorch_predict(model, test_loader, device):
    '''
    Make prediction from a pytorch model 
    '''
    # set model to evaluate model
    model.eval()
    #margin.eval()
    margin.training = False
    
    y_true = torch.tensor([], dtype=torch.long, device=device)
    all_outputs = torch.tensor([], device=device)
    
    # deactivate autograd engine and reduce memory usage and speed up computations
    with torch.no_grad():
        for data in test_loader:
            inputs = [i.to(device) for i in data[:-1]]
            labels = data[-1].to(device)
            
            raw_logits = model(*inputs)
            outputs = margin(raw_logits, labels)              #### Day chinh la s*cos - m @@ 

            y_true = torch.cat((y_true, labels), 0)
            all_outputs = torch.cat((all_outputs, outputs), 0)        
    
    y_true = y_true.cpu().numpy()  
    _, y_pred = torch.max(all_outputs, 1)
    y_pred = y_pred.cpu().numpy()
    y_pred_prob = F.softmax(all_outputs, dim=1).cpu().numpy()       #### @@ O day van phai dung SOFTMAX, vi o tren chi moi la s*cos - m
    
    return y_true, y_pred, y_pred_prob

########Example for validation prediction
#Y_TRUE, y_pred, PREDS = pytorch_predict(model, dataloaders["val"], device)




# TEST SINGLE OR MULTI IMAGES WITHOUT LABEL - LMCL version
#https://discuss.pytorch.org/t/how-to-test-image-for-classification-on-my-pretrained-cnn-pth-file/42692/2
test_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

all_outputs = torch.tensor([], device=device)
    
# deactivate autograd engine and reduce memory usage and speed up computations
with torch.no_grad():
    for path in final_test_path:
        img = Image.open(path)  # Load image as PIL.Image
        x = test_transform(img)  # Preprocess image
        x = x.unsqueeze(0).to(device)

        raw_logits = model(x)     #model forward
        outputs = margin(raw_logits, label=None)   #margin forward

        all_outputs = torch.cat((all_outputs, outputs), 0)
        
    
_, y_pred = torch.max(all_outputs, 1)
y_pred = y_pred.cpu().numpy()
PREDS = F.softmax(all_outputs, dim=1).cpu().numpy()
