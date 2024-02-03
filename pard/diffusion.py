import torch
import torch.nn.functional as F
import math

EPS = 1e-30
LOG_EPS = math.log(EPS)

# ------------------------------------- prob helpers -------------------------------
def sample_uniform_categorical(x_shape, num_classes, device="cuda"):
    return torch.randint(num_classes, size=x_shape, device = device)

def sample_bernoulli(prob, x_shape, device="cuda"):
    assert prob.shape[0] == x_shape[0]
    u = torch.rand(x_shape,device=device)
    b = u.clamp(min=1e-30) < prob
    return b

def sample_categorical(prob):
    """
    Param: prob: (B x N1 x ... x Nk,  x Num_class)
    Return: torch tensor of (B x N1 x ... x Nk)
    """
    shape = prob.shape 
    ind_sample = torch.multinomial(prob.flatten(end_dim=-2), num_samples=1).view(shape[:-1])
    return ind_sample # has the same device as prob 

def get_broadcast_idx(shape):
    return [torch.arange(s).view([1]*i+[-1]+[1]*(len(shape)-1-i)) for i, s in enumerate(shape)]

def index_last_dim(x, idx):
    """
    x     : (B, N1, ..., Nk, C)
    idx   : (B, N1, ..., Nk) with idx.max() < C
    return: (B, N1, ..., Nk)
    """
    assert idx.max() < x.shape[-1]
    broadcast_idx = get_broadcast_idx(idx.shape)
    return x[broadcast_idx + [idx]] 

def set_last_dim(x, idx, value=0, inplace_add=False):
    """
    x     : (B, N1, ..., Nk, C)
    idx   : (B, N1, ..., Nk) with idx.max() < C
    value : (B, N1, ..., Nk) or scalar
    return: (B, N1, ..., Nk, C)
    """
    assert idx.max() < x.shape[-1]
    broadcast_idx = get_broadcast_idx(idx.shape)
    if inplace_add:
        x[broadcast_idx + [idx]] += value
    else:
        x[broadcast_idx + [idx]] = value 
    return x

# --------------------------------------------------------------------------------

def noise_schedule(t_step, s_step=None,
                   schedule_type:str= "cosine",
                   N:int = 1000, # N=0 means continuous 
                   Tmax:float =1,
                   a:float=None, b:float=None, 
                   min_alphabar:float=1e-10, max_beta:float=100, 
                   **kwargs):
    
    assert t_step.max() <= Tmax if N == 0 else t_step.max() <= N
    step_to_time = lambda step: step if N == 0 else step/N * Tmax
    t = step_to_time(t_step)
    s = torch.tensor(0.0) if s_step is None else step_to_time(s_step)

    if schedule_type == "cosine":      
        a = a or 0.008            # set default value
        h = lambda t: torch.cos((t/Tmax + a)/ (1+a) * torch.pi * 0.5)
        h_t, h_s = h(t), h(s) 
        alphabar_t = h_t / h_s
        beta_t = torch.pi * torch.tan((t/Tmax + a) / (1 + a) * torch.pi * 0.5)
        beta_t = beta_t / (2*Tmax*(1+a))
    elif schedule_type == "exponential":
        a, b = a or 0.5, b or 10  # set default value
        b_power = lambda t: torch.exp(t/Tmax * math.log(b))
        b_power_t, b_power_s = b_power(t), b_power(s)
        alphabar_t = torch.exp(a * t * (b_power_s - b_power_t))
        beta_t = a * b_power_t * math.log(b)  
    elif schedule_type == "linear":
        alphabar_t = 1 - t/Tmax
        beta_t = 1/(Tmax - t)
    elif schedule_type == "constant":
        a = a or 0.03
        h = lambda t: torch.exp(-a * t)
        h_t, h_s = h(t), h(s) 
        alphabar_t = h_t / h_s
        beta_t = torch.full_like(t, a) 
    else:
        raise NotImplementedError

    assert alphabar_t.dim() == 1 
    alphabar_t = torch.clip(alphabar_t, min=min_alphabar, max=1-min_alphabar)
    beta_t = torch.clip(beta_t, max=max_beta)  # TODO: revise later

    return alphabar_t, beta_t

def logits_to_prob(logits, dim=-1):
    return F.softmax(logits, dim=dim)

def logits_to_logprob(logits, dim=-1):
    return F.log_softmax(logits, dim=dim)

class DiscreteDiffusion:
    def __init__(self, num_steps, num_classes, noise_schedule_type, noise_schedule_args, ce_only=False):
        self.num_classes = num_classes
        self.num_steps = num_steps # 0 indicates continuous-time diffusion 
        self.noise_schedule_type = noise_schedule_type
        self.noise_schedule_args = noise_schedule_args
        self.ce_only = ce_only

        # TODO: support cache to save computation, e.g., m_dot_xt
        self._cache = {}

    @torch.no_grad()
    def get_m_dot_xt(self, x_t, m=None, recompute=True): ## 4 times call 
        """
        x_t: (B, N1, ..., Nk)
        m  : (B, N1, ..., Nk, C) or None or (C)
        """
        if not recompute and 'm_dot_xt' in self._cache:
            return self._cache['m_dot_xt']
        if m is None:
            m_dot_xt = torch.full_like(x_t, 1/self.num_classes, dtype=torch.float32) 
        elif m.dim() == 1:
            m_dot_xt = m[x_t]
        else:
            m_dot_xt = index_last_dim(m, x_t) 
        assert m_dot_xt.shape == x_t.shape
        self._cache['m_dot_xt'] = m_dot_xt
        return m_dot_xt

    @torch.no_grad()
    def get_alphabar_beta(self, t, s=None): ## 10 times call 
        alphabar_t, beta_t = noise_schedule(t, s, schedule_type=self.noise_schedule_type, N=self.num_steps, **self.noise_schedule_args)
        return alphabar_t, beta_t
    
    @torch.no_grad()
    def get_lambda(self, alphabar_t, alphabar_s, x_t, m=None):  ## 1 times call 
        """
        alphabar_t: (B,1, ..., 1k)
        alphabar_s: (B,1, ..., 1k)
        x_t       : (B, N1, ..., Nk)
        m         : (B, N1, ..., Nk, C) or None or (C)
        return    : (B, N1, ..., Nk)
        """
        assert x_t.dim() == alphabar_t.dim() 
        alpharbar_t_s = alphabar_t / alphabar_s
        m_dot_xt = self.get_m_dot_xt(x_t, m) # (B, N1, ..., Nk)            
        lambda_t_s = (1-alphabar_s)*(1-alpharbar_t_s) * m_dot_xt
        lambda_t_s = lambda_t_s / (alphabar_t + (1-alphabar_t)*m_dot_xt)
        assert (lambda_t_s.shape == x_t.shape)
        return lambda_t_s 
    
    @torch.no_grad()
    def get_mu(self, alphabar_t, alphabar_s): # 3 times call
        """
        alphabar_t: (B,1,...)
        return    : (B,1,...)
        """
        mu_t_s = (1-alphabar_s) / (1-alphabar_t)
        return mu_t_s

    @torch.no_grad()
    def get_mu_times_alphabar(self, alphabar_t, alphabar_s): # 4 times call
        mul = alphabar_s * alphabar_t
        return  (alphabar_t - mul)/(alphabar_s - mul)

    @torch.no_grad()
    def get_gamma_coef(self, alphabar_t, alphabar_s, x_t, m=None): # 2 times call
        # mu - lambda - mu*alphabar
        m_dot_xt = self.get_m_dot_xt(x_t, m) # (B, N1, ..., Nk)  
        coef = self.get_mu_times_alphabar(alphabar_t, alphabar_s) * (alphabar_s - alphabar_t)
        coef = coef / (alphabar_t + (1-alphabar_t)*m_dot_xt) # coef > 0
        assert coef.shape == x_t.shape
        return coef
        
    @torch.no_grad()
    def qt_0_sample(self, x_0, t, m=None, conditional_mask=None):
        """ forward sampling
        x_0: (B, N1, ..., Nk)
        t  : (B,)
        m  : (B, N1, ..., Nk, C), or None, or (C)
        conditional_mask : (B, N1, ..., Nk) or None
        """
        assert x_0.dim() >= 2
        sample_shape = x_0.shape
        alphabar_t, _ = self.get_alphabar_beta(t)  
        alphabar_t = alphabar_t.view([-1]+[1]*(x_0.dim()-1)) #B,N1,....Nk

        #fast sampling from Cat(m)
        if m is None:
            m0 = sample_uniform_categorical(sample_shape, self.num_classes, device=x_0.device) # x_0 shape
        elif m.dim() == 1:
            # sample B x N1 x ... x Nk times with replacement. 
            m0 = torch.multinomial(m, num_samples=sample_shape.numel(), replacement=True).view(sample_shape)
        else:
            assert m.shape[:-1] == sample_shape and m.shape[-1] == self.num_classes
            m0 = sample_categorical(m)
        #sample from the branch indicator function
        bt = sample_bernoulli(alphabar_t, sample_shape, device=x_0.device) # x_0 shape
        #sample of size BxD
        sample = torch.where(bt, x_0, m0)
        if conditional_mask is not None:
            assert conditional_mask.shape == x_0.shape
            sample[conditional_mask] = x_0[conditional_mask]

        return sample
    
    @torch.no_grad() ### ONLY CTMC
    def qt_0_prob(self, x_0, t, m=None, return_beta=False):
        """
        m  : (B, N1, ..., Nk, C) or None or (C)
        """
        shape = [-1]+[1]*(x_0.dim()) #B, 1, ..., 1_k, 1
        alphabar_t, beta_t = self.get_alphabar_beta(t)
        alphabar_t, beta_t = alphabar_t.view(shape), beta_t.view(shape)
        if m is None:
            m = torch.full_like(x_0, 1/self.num_classes, dtype=torch.float32).unsqueeze(-1).repeat_interleave(self.num_classes,-1)
        elif m.dim() == 1:
            m = torch.broadcast_to(m, list(x_0.shape)+[self.num_classes])
        
        prob = (1-alphabar_t) * m
        prob = set_last_dim(prob, x_0, value=alphabar_t.squeeze(-1), inplace_add=True)
        if return_beta:
            return prob, beta_t
        return prob
    
    @torch.no_grad()
    def qs_t0_prob(self, x_t, x_0, t, s, m=None):
        """ forward prob 
        x_0: (B, N1, ..., Nk)
        x_t: (B, N1, ..., Nk)
        s  : (B,) > 0 
        t  : (B,) > s
        m  : (B, N1, ..., Nk, C) or None, or (C)
        """
        shape = [-1]+[1]*(x_0.dim()-1) #B,N1,....Nk
        alphabar_t, _ = self.get_alphabar_beta(t)
        alphabar_s, _ = self.get_alphabar_beta(s)
        alphabar_t, alphabar_s = alphabar_t.view(shape), alphabar_s.view(shape)

        mu_alphabar_t_s = self.get_mu_times_alphabar(alphabar_t, alphabar_s)
        mu_t_s = self.get_mu(alphabar_t, alphabar_s)
        lambda_t_s = self.get_lambda(alphabar_t, alphabar_s, x_t, m) # (B, N1, ..., Nk)

        # compute x_0=x_t prob 
        prob_eq = lambda_t_s[..., None] 
        prob_eq = prob_eq * m if m is not None else (prob_eq/self.num_classes).repeat_interleave(self.num_classes,-1) # (B, N1, ..., Nk, C)
        broadcast_idx = get_broadcast_idx(x_t.shape)
        prob_eq[broadcast_idx+[x_t]] += 1 - lambda_t_s

        # compute x_0!=x_t prob
        prob_neq = (mu_t_s - mu_alphabar_t_s)[...,None]                             # (B, 1, ..., 1k, 1) 
        prob_neq = prob_neq * m if m is not None else prob_neq / self.num_classes   # (B, 1, ..., 1k, C) or (B,N1...Nk,C)
        prob_neq = torch.broadcast_to(prob_neq, list(x_t.shape)+[self.num_classes]).clone() # (B, N1, ..., Nk, C)

        # if m is None:
        #     prob_neq = prob_neq.repeat(1, *list(x_t.shape[1:]), self.num_classes)

        prob_neq[broadcast_idx+[x_t]] += torch.broadcast_to(mu_alphabar_t_s, x_t.shape) # mu'shape is (B,1,...,1k)
        prob_neq[broadcast_idx+[x_0]] += torch.broadcast_to(1-mu_t_s       , x_0.shape)
        
        prob = torch.where((x_t==x_0).unsqueeze(-1), prob_eq, prob_neq)
        return prob 

    def ps_t_logprob(self, flogprob_t, x_t, t, s, m=None):
        """
        m: (B, N1, ..., Nk, C) or None or (C), requires s > 0, t > s. 
        """
        shape = [-1]+[1]*(x_t.dim()-1) #B,N1,....Nk
        alphabar_t, _ = self.get_alphabar_beta(t)
        alphabar_s, _ = self.get_alphabar_beta(s)
        alphabar_t, alphabar_s = alphabar_t.view(shape), alphabar_s.view(shape)

        mu_alphabar_t_s = self.get_mu_times_alphabar(alphabar_t, alphabar_s)
        mu_t_s = self.get_mu(alphabar_t, alphabar_s)

        # term 1 
        logterm1 = flogprob_t + (1-mu_t_s).clip(min=EPS).log()[...,None]
        # compute fprob_dot_xt
        gamma_t_s = self.get_gamma_coef(alphabar_t, alphabar_s, x_t, m) * (index_last_dim(flogprob_t, x_t).exp())
        probterm2 = (mu_t_s - mu_alphabar_t_s - gamma_t_s)[..., None]
        probterm2 = probterm2 * m if m is not None else probterm2/self.num_classes #).repeat_interleave(self.num_classes,-1)
        probterm2 = torch.broadcast_to(probterm2, list(x_t.shape)+[self.num_classes]).clone()
        broadcast_idx = get_broadcast_idx(x_t.shape)
        probterm2[broadcast_idx+[x_t]] = probterm2[broadcast_idx+[x_t]] + gamma_t_s + mu_alphabar_t_s

        logprob = torch.logaddexp(logterm1, probterm2.clip(min=EPS).log())
        return logprob
    
    def ps_t_prob(self, fprob_t, x_t, t, s, m=None):
        """ backward prob
        fprob_t: (B, N1, ..., Nk, C)
        x_t    : (B, N1, ..., Nk)
        s      : (B,)
        t      : (B,)
        m      : (B, N1, ..., Nk, C) or None or (C)
        """
        shape = [-1]+[1]*(x_t.dim()-1) #B,N1,....Nk
        alphabar_t, _ = self.get_alphabar_beta(t)
        alphabar_s, _ = self.get_alphabar_beta(s)
        alphabar_t, alphabar_s = alphabar_t.view(shape), alphabar_s.view(shape)

        # alphabar_t_s = alphabar_t / alphabar_s
        mu_alphabar_t_s = self.get_mu_times_alphabar(alphabar_t, alphabar_s)
        mu_t_s = self.get_mu(alphabar_t, alphabar_s)
        gamma_t_s = self.get_gamma_coef(alphabar_t, alphabar_s, x_t, m) * index_last_dim(fprob_t, x_t)

        prob = (mu_t_s - mu_alphabar_t_s - gamma_t_s)[..., None]
        prob = prob * m if m is not None else (prob/self.num_classes)#.repeat_interleave(self.num_classes,-1)
        prob = torch.broadcast_to(prob, list(x_t.shape)+[self.num_classes])

        prob = prob + (1 - mu_t_s[..., None]) * fprob_t 
        broadcast_idx = get_broadcast_idx(x_t.shape)
        prob[broadcast_idx+[x_t]] = prob[broadcast_idx+[x_t]] + gamma_t_s + mu_alphabar_t_s

        prob = torch.clip(prob, min=0)          # make sure prob >= 0 
        return prob 
    
    def compute_loss(self, logits_t, x_t, x_0, t, m, coeff_ce=1., conditional_mask=None):
        '''
        conditional_mask : (B, N1, ..., Nk) or None

        Compute two Cross-entropy losses
        (1) ce_loss
        (2) vlb_loss
        '''          
        # discrete-time diffusion
        vlb_loss, ce_loss = self._discrete_time_loss(logits_t, x_t, x_0, t, m, conditional_mask)
        coeff_vlb = 1.0
        if self.ce_only:
            coeff_ce = 1.0
            coeff_vlb = 0.0

        loss =  coeff_vlb*vlb_loss + coeff_ce * ce_loss
        
        output_dict = {'loss'    : loss,
                       'vlb_loss': vlb_loss,
                       'ce_loss' : ce_loss,} 
        # clean cache 
        self._cache = {}
        return output_dict
    
    def _prior_at_T(self, x_0, m=None):
        """ Compute KL(q(x_T | x_0) || p(x_T))
        x_0: (B, N1, ..., Nk)
        t  : (B,)
        m  : (B, N1, ..., Nk, C) or None, or (C)
        """
        batch_size = x_0.size(0)
        T = self.num_steps * torch.ones(batch_size, device=x_0.device) # (B,)
        qT_0_prob = self.qt_0_prob(x_0, T, m=m) # (B, N1, ..., Nk, C)
        pT_prob = torch.broadcast_to(m, qT_0_prob.shape) if m is not None else torch.full_like(qT_0_prob, 1/self.num_classes)
        return F.kl_div(pT_prob.clip(min=EPS).log(), qT_0_prob, reduction='none').sum(-1) # (B, N1, ..., Nk)

    def _discrete_time_loss(self, flogits_t, x_t, x_0, t, m=None, conditional_mask=None):
        """
        conditional_mask : (B, N1, ..., Nk) or None

        flogits_t: (B, N1, ..., Nk, C)
        x_0      : (B, N1, ..., Nk)
        x_t      : (B, N1, ..., Nk)
        t        : (B,)
        m        : (B, N1, ..., Nk, C) or None, or (C)
        """
        batch_size = x_0.size(0)
        flogprob_t = logits_to_logprob(flogits_t)
        ce_loss = -index_last_dim(flogprob_t, x_0)

        # ------------- compute vlb loss ----------------
        # if not self.ce_only:
        # get probs 
        assert t.min() >= 1
        ps_t_logprob = self.ps_t_logprob(flogprob_t, x_t, t=t, s=t-1, m=m)
        qs_t0_prob = self.qs_t0_prob(x_t, x_0, t=t, s=t-1, m=m)
        # compute vlb loss and ce loss
        # t >= 2
        # vlb_loss = (-ps_t_logprob * qs_t0_prob).sum(-1)
        vlb_loss = F.kl_div(ps_t_logprob, qs_t0_prob, reduction='none').sum(-1) ### TODO: Double check
        # t = 1
        t0_mask = (t == 1).float().view([-1]+[1]*(x_0.dim()-1)) # min time is 1 
        vlb_loss = t0_mask * ce_loss + (1.0-t0_mask) * vlb_loss # this is not the final vlb, we still need prior loss for discrete case. 
        # prior loss 
        vlb_loss = vlb_loss + (self._prior_at_T(x_0, m=m) / self.num_steps)
        # ---------------------------------------------
        if conditional_mask is not None:
            assert conditional_mask.shape == x_0.shape
            assert (x_t==x_0)[conditional_mask].all()
            vlb_loss = vlb_loss * (~conditional_mask)
            ce_loss = ce_loss * (~conditional_mask)
            assert vlb_loss.shape == x_0.shape

        vlb_loss = vlb_loss.view(batch_size, -1).sum(-1)
        ce_loss = ce_loss.view(batch_size, -1).sum(-1)
        return vlb_loss.sum(), ce_loss.sum() # add all loss togther 
    
    def sample_step(self, fprob_t, x_t, t, s, m=None, conditional_mask=None):
        """ from time step t to time step s  (t>s)
        fprob_t     : (B, N1, ..., Nk, C)
        x_t         : (B, N1, ..., Nk)
        t           : (B,)
        s           : (B,)
        m           : (B, N1, ..., Nk, C) or None or (C)
        mc_step_size: (B,) or None
        conditional_mask : (B, N1, ..., Nk) or None
        """
        # compute P(x_s | x_t)
        prob_s = self.ps_t_prob(fprob_t, x_t, t=t, s=s, m=m)    # (B, N1, ..., Nk, C)
        
        # for s = 0, change prob_s to fprob_t (final step sampling towards x0)
        prob_s[s==0] = fprob_t[s==0]

        # sample x_s 
        x_s = sample_categorical(prob_s)                        # (B, N1, ..., Nk)
        if conditional_mask is not None:
            assert conditional_mask.shape == x_s.shape
            x_s[conditional_mask] = x_t[conditional_mask]
        
        return x_s