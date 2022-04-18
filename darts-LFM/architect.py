import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from image_encoder import *


def _concat(xs):
    return torch.cat([x.view(-1) for x in xs])


def calculate_weightage(
    input_train,
    target_train,
    input_valid,
    target_valid,
    model_pretrain,
    img_encoder_v,
    coeff_r,
):
    n = input_train.size(0)

    # visual similarity score
    logits_train = img_encoder_v(input_train)
    logits_valid = img_encoder_v(input_valid)
    visual_scores = torch.matmul(logits_train, logits_valid.transpose(-2, -1))
    visual_scores = F.softmax(visual_scores, dim=-1)

    # label similarity score
    train_label = target_train.unsqueeze(1)
    valid_label = target_valid.unsqueeze(0).repeat(n, 1)
    label_scores = (train_label == valid_label).float()

    # predictive performance score
    with torch.no_grad():
        logits_valid = model_pretrain(input_valid)
    pred_scores = -F.cross_entropy(logits_valid, target_valid, reduction="none")
    pred_scores = pred_scores.unsqueeze(0).repeat(n, 1)

    # calculate the weight of each training example
    scores = visual_scores * label_scores * pred_scores
    a = torch.sigmoid(coeff_r(scores))

    return a


class Architect(object):
    def __init__(self, model, img_encoder_v, coeff_r, optimizer_v, optimizer_r, args):
        self.network_momentum = args.momentum
        self.network_weight_decay = args.weight_decay
        self.args = args
        self.model = model
        self.optimizer = torch.optim.Adam(
            self.model.arch_parameters(),
            lr=args.arch_learning_rate,
            betas=(0.5, 0.999),
            weight_decay=args.arch_weight_decay,
        )
        self.img_encoder_v = img_encoder_v
        self.coeff_r = coeff_r
        self.optimizer_v = optimizer_v
        self.optimizer_r = optimizer_r

    def _compute_unrolled_model(
        self,
        input_train,
        target_train,
        input_valid,
        target_valid,
        model_pretrain,
        eta,
        eta_v,
        eta_r,
        network_optimizer,
    ):
        logits_train = self.model(input_train)
        loss = F.cross_entropy(logits_train, target_train, reduction="none")
        a = calculate_weightage(
            input_train,
            target_train,
            input_valid,
            target_valid,
            model_pretrain,
            self.img_encoder_v,
            self.coeff_r,
        )
        loss = (loss * a).mean()

        theta = _concat(self.model.parameters()).data
        theta_v = _concat(self.img_encoder_v.parameters()).data
        theta_r = _concat(self.coeff_r.parameters()).data
        try:
            moment = _concat(
                network_optimizer.state[v]["momentum_buffer"]
                for v in self.model.parameters()
            ).mul_(self.network_momentum)
            moment_v = _concat(
                self.optimizer_v.state[v]["momentum_buffer"]
                for v in self.img_encoder_v.parameters()
            ).mul_(self.network_momentum)
            moment_r = _concat(
                self.optimizer_r.state[v]["momentum_buffer"]
                for v in self.coeff_r.parameters()
            ).mul_(self.network_momentum)
        except:
            moment = torch.zeros_like(theta)
            moment_v = torch.zeros_like(theta_v)
            moment_r = torch.zeros_like(theta_r)

        loss.backward()

        grad_model = [v.grad.data for v in self.model.parameters()]
        dtheta = _concat(grad_model).data + self.network_weight_decay * theta
        unrolled_model = self._construct_model_from_theta(
            theta.sub(eta, moment + dtheta)
        )
        grad_v = [v.grad.data for v in self.img_encoder_v.parameters()]
        dtheta_v = _concat(grad_v).data + self.args.weight_decay_v * theta_v
        unrolled_model_v = self._construct_model_from_theta_v(
            theta_v.sub(eta_v, moment_v + dtheta_v)
        )
        grad_r = [v.grad.data for v in self.coeff_r.parameters()]
        dtheta_r = _concat(grad_r).data + self.args.weight_decay_r * theta_r
        unrolled_model_r = self._construct_model_from_theta_r(
            theta_r.sub(eta_r, moment_r + dtheta_r)
        )

        return unrolled_model, unrolled_model_v, unrolled_model_r

    def step(
        self,
        input_train,
        target_train,
        input_valid,
        target_valid,
        model_pretrain,
        eta,
        eta_v,
        eta_r,
        network_optimizer,
        unrolled,
    ):
        if unrolled:
            self._backward_step_unrolled(
                input_train,
                target_train,
                input_valid,
                target_valid,
                model_pretrain,
                eta,
                eta_v,
                eta_r,
                network_optimizer,
            )
        else:
            # update the architecture A
            self.optimizer.zero_grad()

            valid_loss = self.model._loss(input_valid, target_valid)
            valid_loss.backward()

            self.optimizer.step()

            # update the image encoder V and coeff r
            self.optimizer_v.zero_grad()
            self.optimizer_r.zero_grad()

            with torch.no_grad():
                logits_train = self.model(input_train)
            train_loss = F.cross_entropy(logits_train, target_train, reduction="none")
            a = calculate_weightage(
                input_train,
                target_train,
                input_valid,
                target_valid,
                model_pretrain,
                self.img_encoder_v,
                self.coeff_r,
            )
            train_loss = (train_loss * a).mean()
            train_loss.backward()

            self.optimizer_v.step()
            self.optimizer_r.step()

    def _backward_step_unrolled(
        self,
        input_train,
        target_train,
        input_valid,
        target_valid,
        model_pretrain,
        eta,
        eta_v,
        eta_r,
        network_optimizer,
    ):
        (
            unrolled_model,
            unrolled_model_v,
            unrolled_model_r,
        ) = self._compute_unrolled_model(
            input_train,
            target_train,
            input_valid,
            target_valid,
            model_pretrain,
            eta,
            eta_v,
            eta_r,
            network_optimizer,
        )

        # update the architecture A
        self.optimizer.zero_grad()

        unrolled_loss = unrolled_model._loss(input_valid, target_valid)
        unrolled_loss.backward()

        dalpha = [v.grad for v in unrolled_model.arch_parameters()]
        vector = [v.grad.data for v in unrolled_model.parameters()]

        implicit_grads = self._hessian_vector_product_theta(
            vector,
            input_train,
            target_train,
            input_valid,
            target_valid,
            model_pretrain,
        )

        for g, ig in zip(dalpha, implicit_grads):
            g.data.sub_(eta, ig.data)

        for v, g in zip(self.model.arch_parameters(), dalpha):
            if v.grad is None:
                v.grad = Variable(g.data)
            else:
                v.grad.data.copy_(g.data)

        self.optimizer.step()

        # update the image encoder V and coeff r based on the updated architecture A
        self.optimizer_v.zero_grad()
        self.optimizer_r.zero_grad()

        logits_train = self.model(input_train)
        train_loss = F.cross_entropy(logits_train, target_train, reduction="none")
        a = calculate_weightage(
            input_train,
            target_train,
            input_valid,
            target_valid,
            model_pretrain,
            unrolled_model_v,
            unrolled_model_r,
        )
        unrolled_train_loss = (train_loss * a).mean()
        unrolled_train_loss.backward()

        d_img_encoder_v = [v.grad.data for v in unrolled_model_v.parameters()]
        d_coeff_r = [v.grad.data for v in unrolled_model_r.parameters()]
        vector = [v.grad.data for v in unrolled_model.parameters()]

        implicit_grads_v = self._hessian_vector_product_theta_v(
            vector,
            input_train,
            target_train,
            input_valid,
            target_valid,
            model_pretrain,
        )
        implicit_grads_r = self._hessian_vector_product_theta_r(
            vector,
            input_train,
            target_train,
            input_valid,
            target_valid,
            model_pretrain,
        )

        # update image encoder V
        for g, ig in zip(d_img_encoder_v, implicit_grads_v):
            g.data.sub_(eta_v, ig.data)

        for v, g in zip(self.img_encoder_v.parameters(), d_img_encoder_v):
            if v.grad is None:
                v.grad = Variable(g.data)
            else:
                v.grad.data.copy_(g.data)

        # update coeff r
        for g, ig in zip(d_coeff_r, implicit_grads_r):
            g.data.sub_(eta_v, ig.data)

        for v, g in zip(self.img_encoder_v.parameters(), d_img_encoder_v):
            if v.grad is None:
                v.grad = Variable(g.data)
            else:
                v.grad.data.copy_(g.data)

        self.optimizer_v.step()
        self.optimizer_r.step()

    def _construct_model_from_theta(self, theta):
        model_new = self.model.new()
        model_dict = self.model.state_dict()

        params, offset = {}, 0
        for k, v in self.model.named_parameters():
            v_length = np.prod(v.size())
            params[k] = theta[offset : offset + v_length].view(v.size())
            offset += v_length

        assert offset == len(theta)
        model_dict.update(params)
        model_new.load_state_dict(model_dict)
        return model_new.cuda()

    def _construct_model_from_theta_v(self, theta):
        if self.args.img_encoder_arch == "18":
            model_new = resnet18().cuda()
        elif self.args.img_encoder_arch == "34":
            model_new = resnet34().cuda()
        elif self.args.img_encoder_arch == "50":
            model_new = resnet50().cuda()
        elif self.args.img_encoder_arch == "101":
            model_new = resnet101().cuda()
        model_dict = self.img_encoder_v.state_dict()

        params, offset = {}, 0
        for k, v in self.img_encoder_v.named_parameters():
            v_length = np.prod(v.size())
            params[k] = theta[offset : offset + v_length].view(v.size())
            offset += v_length

        assert offset == len(theta)
        model_dict.update(params)
        model_new.load_state_dict(model_dict)
        return model_new.cuda()

    def _construct_model_from_theta_r(self, theta):
        model_new = nn.Linear(self.args.batch_size, 1).cuda()
        model_dict = self.coeff_r.state_dict()

        params, offset = {}, 0
        for k, v in self.coeff_r.named_parameters():
            v_length = np.prod(v.size())
            params[k] = theta[offset : offset + v_length].view(v.size())
            offset += v_length

        assert offset == len(theta)
        model_dict.update(params)
        model_new.load_state_dict(model_dict)
        return model_new.cuda()

    def _hessian_vector_product_theta(
        self,
        vector,
        input_train,
        target_train,
        input_valid,
        target_valid,
        model_pretrain,
        r=1e-2,
    ):
        R = r / _concat(vector).norm()

        for p, v in zip(self.model.parameters(), vector):
            p.data.add_(R, v)
        logits_train = self.model(input_train)
        loss = F.cross_entropy(logits_train, target_train, reduction="none")
        a = calculate_weightage(
            input_train,
            target_train,
            input_valid,
            target_valid,
            model_pretrain,
            self.img_encoder_v,
            self.coeff_r,
        )
        loss = (loss * a).mean()
        grads_p = torch.autograd.grad(loss, self.model.arch_parameters())

        for p, v in zip(self.model.parameters(), vector):
            p.data.sub_(2 * R, v)
        logits_train = self.model(input_train)
        loss = F.cross_entropy(logits_train, target_train, reduction="none")
        a = calculate_weightage(
            input_train,
            target_train,
            input_valid,
            target_valid,
            model_pretrain,
            self.img_encoder_v,
            self.coeff_r,
        )
        loss = (loss * a).mean()
        grads_n = torch.autograd.grad(loss, self.model.arch_parameters())

        return [(x - y).div_(2 * R) for x, y in zip(grads_p, grads_n)]

    def _hessian_vector_product_theta_v(
        self,
        vector,
        input_train,
        target_train,
        input_valid,
        target_valid,
        model_pretrain,
        r=1e-2,
    ):
        R = r / _concat(vector).norm()

        for p, v in zip(self.model.parameters(), vector):
            p.data.add_(R, v)
        logits_train = self.model(input_train)
        loss = F.cross_entropy(logits_train, target_train, reduction="none")
        a = calculate_weightage(
            input_train,
            target_train,
            input_valid,
            target_valid,
            model_pretrain,
            self.img_encoder_v,
            self.coeff_r,
        )
        loss = (loss * a).mean()
        grads_p = torch.autograd.grad(loss, self.img_encoder_v.parameters())

        for p, v in zip(self.model.parameters(), vector):
            p.data.sub_(2 * R, v)
        logits_train = self.model(input_train)
        loss = F.cross_entropy(logits_train, target_train, reduction="none")
        a = calculate_weightage(
            input_train,
            target_train,
            input_valid,
            target_valid,
            model_pretrain,
            self.img_encoder_v,
            self.coeff_r,
        )
        loss = (loss * a).mean()
        grads_n = torch.autograd.grad(loss, self.img_encoder_v.parameters())

        return [(x - y).div_(2 * R) for x, y in zip(grads_p, grads_n)]

    def _hessian_vector_product_theta_r(
        self,
        vector,
        input_train,
        target_train,
        input_valid,
        target_valid,
        model_pretrain,
        r=1e-2,
    ):
        R = r / _concat(vector).norm()

        for p, v in zip(self.model.parameters(), vector):
            p.data.add_(R, v)
        logits_train = self.model(input_train)
        loss = F.cross_entropy(logits_train, target_train, reduction="none")
        a = calculate_weightage(
            input_train,
            target_train,
            input_valid,
            target_valid,
            model_pretrain,
            self.img_encoder_v,
            self.coeff_r,
        )
        loss = (loss * a).mean()
        grads_p = torch.autograd.grad(loss, self.coeff_r.parameters())

        for p, v in zip(self.model.parameters(), vector):
            p.data.sub_(2 * R, v)
        logits_train = self.model(input_train)
        loss = F.cross_entropy(logits_train, target_train, reduction="none")
        a = calculate_weightage(
            input_train,
            target_train,
            input_valid,
            target_valid,
            model_pretrain,
            self.img_encoder_v,
            self.coeff_r,
        )
        loss = (loss * a).mean()
        grads_n = torch.autograd.grad(loss, self.coeff_r.parameters())

        return [(x - y).div_(2 * R) for x, y in zip(grads_p, grads_n)]
