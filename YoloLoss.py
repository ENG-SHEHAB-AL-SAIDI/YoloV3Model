import torch
import torch.nn as nn
from Utils import intersection_over_union


class YoloLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.entropy = nn.CrossEntropyLoss()
        self.sigmoid = nn.Sigmoid()

        # Constants signifying how much to pay for each respective part of the loss
        self.lambdaClass = 1
        self.lambdaNoObj = 10
        self.lambdaObj = 1
        self.lambdaBox = 10

    def forward(self, predictions, target, anchors):
        # Check where obj and noObj (we ignore if target == -1)
        obj = target[..., 0] == 1  # in paper this is iObj_i
        noObj = target[..., 0] == 0  # in paper this is iNoObj_i

        # ======================= #
        #   FOR NO OBJECT LOSS    #
        # ======================= #

        noObjectLoss = self.bce(
            (predictions[..., 0:1][noObj]), (target[..., 0:1][noObj]),
        )

        # ==================== #
        #   FOR OBJECT LOSS    #
        # ==================== #

        anchors = anchors.reshape(1, 3, 1, 1, 2)
        boxPreds = torch.cat([self.sigmoid(predictions[..., 1:3]), torch.exp(predictions[..., 3:5]) * anchors], dim=-1)
        ious = intersection_over_union(boxPreds[obj], target[..., 1:5][obj]).detach()
        objectLoss = self.mse(self.sigmoid(predictions[..., 0:1][obj]), ious * target[..., 0:1][obj])

        # ======================== #
        #   FOR BOX COORDINATES    #
        # ======================== #

        predictions[..., 1:3] = self.sigmoid(predictions[..., 1:3])  # x,y coordinates
        target[..., 3:5] = torch.log(
            (1e-16 + target[..., 3:5] / anchors)
        )  # width, height coordinates
        boxLoss = self.mse(predictions[..., 1:5][obj], target[..., 1:5][obj])

        # ================== #
        #   FOR CLASS LOSS   #
        # ================== #

        classLoss = self.entropy(
            (predictions[..., 5:][obj]), (target[..., 5][obj].long()),
        )

        #print("__________________________________")
        #print(self.lambdaBox * boxLoss)
        #print(self.lambdaObj * objectLoss)
        #print(self.lambdaNoObj * noObjectLoss)
        #print(self.lambdaClass * classLoss)
        #print("\n")

        return (
                self.lambdaBox * boxLoss
                + self.lambdaObj * objectLoss
                + self.lambdaNoObj * noObjectLoss
                + self.lambdaClass * classLoss
        )
