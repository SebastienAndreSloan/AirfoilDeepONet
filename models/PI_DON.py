from FNN import *

class DeepONet(StructureNN):
    '''Deep operator network.
    Input: [batch size, branch_dim + trunk_dim]
    Output: [batch size, 1]
    '''
    def __init__(self, branch_dim = 3, trunk_dim = 2, branch_depth=3, trunk_depth=3, width=50, k=4, nu=1.5498148291427463e-05, activation='tanh', initializer='Glorot normal', device="cpu"):
        super(DeepONet, self).__init__()
        self.branch_dim = branch_dim
        self.branch_outputd = k * width
        self.trunk_dim = trunk_dim
        self.trunk_outputd = k * width
        self.branch_depth = branch_depth
        self.trunk_depth = trunk_depth
        self.width = width
        self.activation = activation
        self.initializer = initializer
        self.k = k
        self.device = device
        self.loss_fn = torch.nn.MSELoss().to(device)
        self.nu = nu

        self.modus = self.__init_modules()
        self.params = self.__init_params()
        self.__initialize()

    def __init_modules(self):
        modules = nn.ModuleDict()
        # Barnch
        modules['Branch'] = FNN(self.branch_dim, self.branch_outputd, self.branch_depth, self.width,
                                self.activation, self.initializer)

        # Trunk
        modules['TrLinM1'] = nn.Linear(self.trunk_dim, self.width)
        modules['TrActM1'] = self.Act
        for i in range(2, self.trunk_depth):
            modules['TrLinM{}'.format(i)] = nn.Linear(self.width, self.width)
            modules['TrActM{}'.format(i)] = self.Act
        modules['TrLinM{}'.format(i)] = nn.Linear(self.width, self.trunk_outputd)
        modules['TrActM{}'.format(i)] = self.Act
        return modules

    def __init_params(self):
        params = nn.ParameterDict()
        params['bias'] = nn.Parameter(torch.zeros([4]))
        return params

    def __initialize(self):
        for i in range(1, self.trunk_depth):
            self.weight_init_(self.modus['TrLinM{}'.format(i)].weight)
            nn.init.constant_(self.modus['TrLinM{}'.format(i)].bias, 0)

    def forward(self, x):
        x_branch, x_trunk = x[..., :self.branch_dim], x[..., self.branch_dim:]
        x_branch = self.modus['Branch'](x_branch)
        for i in range(1, self.trunk_depth):
            x_trunk = self.modus['TrActM{}'.format(i)](self.modus['TrLinM{}'.format(i)](x_trunk))
        self.out_list = []
        for i in range(self.k):
            output = x_branch[:, i*self.width : (i+1)*self.width] * x_trunk[:, i*self.width : (i+1)*self.width]
            self.out_list.append(torch.sum(output, dim=-1, keepdim=True) + self.params['bias'][i])
        return torch.cat(self.out_list,dim=1)


    def lossData(self, X_train, y_train):
        y_pred = self.forward(X_train)

        loss1 = self.loss_fn(y_pred[:,0], y_train[:,0])
        loss2 = self.loss_fn(y_pred[:,1], y_train[:,1])
        loss3 = self.loss_fn(y_pred[:,2], y_train[:,2])
        loss4 = self.loss_fn(y_pred[:,3], y_train[:,3])

        loss = loss1 + loss2 + loss3 + loss4

        return loss

    def lossPDE(self, X_train):
        g = X_train.clone()
        g.requires_grad=True
        y_pred = self.forward(g)

        u1 = y_pred[:,1]
        u2 = y_pred[:,2]
        nu_t = y_pred[:, 3]

        # derivative of pressure
        p_x = torch.autograd.grad(y_pred[:,0].unsqueeze(dim=-1), g, torch.ones([g.shape[0],1]).to(self.device), retain_graph=True)[0][:,-2:]
        p_x1 = p_x[:,0]
        p_x2 = p_x[:,1]

        # first derivative of velocity (u1)
        u1_x = torch.autograd.grad(y_pred[:,1].unsqueeze(dim=-1), g, torch.ones([g.shape[0],1]).to(self.device), retain_graph=True, create_graph=True)[0][:,-2:]
        u1_x1 = u1_x[:,0]
        u1_x2 = u1_x[:,1]

        # second derivative of velocity (u1)
        u1_x1_x = torch.autograd.grad(u1_x1.unsqueeze(dim=-1), g, torch.ones([g.shape[0],1]).to(self.device), retain_graph=True)[0][:,-2:]
        u1_x1_x1 = u1_x1_x[:,0]
        u1_x2_x = torch.autograd.grad(u1_x2.unsqueeze(dim=-1), g, torch.ones([g.shape[0],1]).to(self.device), retain_graph=True)[0][:,-2:]
        u1_x2_x2 = u1_x2_x[:,1]

        # first derivative of velocity (u2)
        u2_x = torch.autograd.grad(y_pred[:,2].unsqueeze(dim=-1), g, torch.ones([g.shape[0],1]).to(self.device), retain_graph=True, create_graph=True)[0][:,-2:]
        u2_x1 = u2_x[:,0]
        u2_x2 = u2_x[:,1]

        # second derivative of velocity (u2)
        u2_x1_x = torch.autograd.grad(u2_x1.unsqueeze(dim=-1), g, torch.ones([g.shape[0],1]).to(self.device), retain_graph=True)[0][:,-2:]
        u2_x1_x1 = u2_x1_x[:,0]
        u2_x2_x = torch.autograd.grad(u2_x2.unsqueeze(dim=-1), g, torch.ones([g.shape[0],1]).to(self.device), retain_graph=True)[0][:,-2:]
        u2_x2_x2 = u2_x2_x[:,1]

        u_dot_div_u1 = u1 * u1_x1 + u2 * u1_x2
        u_dot_div_u2 = u1 * u2_x1 + u2 * u2_x2

        lp_u1 = u1_x1_x1 + u1_x2_x2
        lp_u2 = u2_x1_x1 + u2_x2_x2

        # Navier-Stokes PDE loss condition
        PDE_loss_1 = u_dot_div_u1 + p_x1 - (self.nu + nu_t) * lp_u1
        PDE_loss_2 = u_dot_div_u2 + p_x2 - (self.nu + nu_t) * lp_u2

        # incompressible condition
        u_div = u1_x1 + u2_x2

        return (u_div, PDE_loss_1, PDE_loss_2)

    def loss(self, X_train, y_train):

        loss_PDE = self.lossPDE(X_train)
        loss_Data = self.lossData(X_train, y_train)

        total_loss = 0.1*loss_Data + self.loss_fn(loss_PDE[0], torch.zeros(loss_PDE[0].shape).to(self.device)) + self.loss_fn(loss_PDE[1], torch.zeros(loss_PDE[1].shape).to(self.device)) + self.loss_fn(loss_PDE[2], torch.zeros(loss_PDE[2].shape).to(self.device))

        return total_loss