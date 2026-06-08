import torch
import torch.nn as nn

class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.block1 = nn.Sequential(nn.Linear(3, 64, bias=False),
                                    nn.BatchNorm1d(64),
                                    nn.Tanh(),
                                    nn.Linear(64, 16, bias=False),
                                    nn.BatchNorm1d(16),
                                    nn.Tanh(),
                                    nn.Linear(16, 64, bias=False),
                                    nn.BatchNorm1d(64),
                                    )

        self.skip1 = nn.Sequential(nn.Linear(3, 64, bias=False), nn.BatchNorm1d(64), )

        self.block2 = nn.Sequential(nn.Linear(64, 128, bias=False),
                                    nn.BatchNorm1d(128),
                                    nn.Tanh(),
                                    nn.Linear(128, 32, bias=False),
                                    nn.BatchNorm1d(32),
                                    nn.Tanh(),
                                    nn.Linear(32, 64, bias=False),
                                    nn.BatchNorm1d(64),
                                    )

        self.block3 = nn.Sequential(nn.Linear(64, 32, bias=False),
                                    nn.BatchNorm1d(32),
                                    nn.Tanh(),
                                    nn.Linear(32, 16, bias=False),
                                    nn.BatchNorm1d(16),
                                    nn.Tanh(),
                                    nn.Linear(16, 32, bias=False),
                                    nn.BatchNorm1d(32),
                                    )

        self.skip2 = nn.Sequential(nn.Linear(64, 32, bias=False), nn.BatchNorm1d(32), )

        self.output = nn.Linear(32, 1)

        self.tanh = nn.Tanh()

        self.log_kappa = nn.Parameter(torch.log(torch.tensor(0.0001)))

    def forward(self, x, y, t):
        inp = torch.cat([x, y, t], dim=1)

        ib1 = self.block1(inp)
        sk1 = self.skip1(inp)
        ibsk1 = self.tanh(ib1 + sk1)

        ib2 = self.block2(ibsk1)
        ibsk2 = self.tanh(ib2 + ibsk1)

        ib3 = self.block3(ibsk2)
        sk2 = self.skip2(ibsk2)
        ibsk3 = self.tanh(ib3 + sk2)

        res = self.output(ibsk3)

        return res

    def get_kappa(self):
        return torch.exp(self.log_kappa)