import numpy
import scipy
import scipy.stats

N = scipy.stats.norm()

class SpikeSlab:
    def __init__(self, n, p):
        # Gaussian re-parameterization
        self.n = n
        self.mu = numpy.zeros(n)
        self.nu = numpy.zeros(n)

        # Variational parameters Q(.)
        self.alpha = numpy.zeros(p)
        self.beta = numpy.zeros(p)
        self.gamma = numpy.zeros(p)

        # ∇Q
        self.dalpha = numpy.zeros(p)
        self.dbeta = numpy.zeros(p)
        self.dgamma = numpy.zeros(p)

    def sample(self, num_samples=10):
        """Re-parameterize and sample to make expectation differentiable (Kingma,
        Welling arXiv 2014). Return intermediates for gradient calculation.

        """
        sum_F = numpy.zeros(self.n)
        sum_Z1 = numpy.zeros(self.n)
        sum_Z2 = numpy.zeros(self.n)
        # Important intermediates
        # T1 = (Z .* F) / sqrt(nu)
        # T2 = (Z^2 - 1) .* F / nu
        sum_T1 = numpy.zeros(self.n)
        sum_T2 = numpy.zeros(self.n)
        for i in range(num_samples):
            z = N.rvs(size=n)
            eta = self.mu + numpy.sqrt(self.nu) * z
            llik = self.llik(eta)
            sum_F += llik
            sum_Z1 += z
            sum_T1 += z * llik
            w = z * z
            sum_Z2 += w
            sum_T2 += (w - 1) * llik
        # Control variate (Paisley, Jordan, Blei ICML 2012; Ranganath, Gerrish,
        # Blei AIT1TATT1 2014)
        #
        # T1 -= E[T1]
        # T2 -= E[T2]
        T1 = (sum_T1 - sum_Z1 * sum_F) / (numpy.sqrt(self.nu) * nsamples)
        T2 = (sum_T1 - sum_Z2 * sum_F) / self.nu
        return T1, T2

    def gradient(self, num_samples=10):
        """Return ∇q_i

        q ~ N(\mu, \nu)
        \mu = \sum_j X_ij \alpha_j \beta_j
        \nu = \sum_j X_ij^2 (\alpha_j / \gamma_j + \alpha_j (1 - \alpha_j) \beta_j^2)

        """
        T1, T2 = self.sample(num_samples)
        B = self.beta * self.beta
        self.dalpha = T1 * self.beta + .5 * T2 * (1 / self.gamma + B + 2 * self.alpha * B)
        raise NotImplementedError

    def adam(self, step_size=.1e-3, b1=.9, b2=.999, eps=1e-8):
        """Perform stochastic gradient update with adaptive estimation (Adam).

        http://arxiv.org/pdf/1412.6980.pdf

        """
        m = [numpy.zeros(p.shape) for p in (self.alpha, self.beta, self.gamma)]
        v = [numpy.zeros(p.shape) for p in (self.alpha, self.beta, self.gamma)]
        while True:
            t = yield
            nabla = self.gradient()
            m = [(1 - b1) * g + b1 * m for g in nabla]
            v = [(1 - b2) * (numpy.pow(g, 2)) + b2 * v for g in nabla]
            rate = step_size * numpy.sqrt(1 - numpy.pow(b2, t)) / (1 - math.pow(b1, t))
            self.alpha -= rate * m[0] / (numpy.sqrt(v[0]) + eps)
            self.beta -= rate * m[1] / (numpy.sqrt(v[1]) + eps)
            self.gamma -= rate * m[2] / (numpy.sqrt(v[2]) + eps)

class Model:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.params = SpikeSlab(y.shape, x.shape[1])

    def llik(eta):
        """Return a matrix of log-likelihood per sample for (re-parameterized) generative model.

        eta = X theta

        """
        raise NotImplementedError

    def fit(num_iters=1000, step_size=0.001, b1=0.9, b2=0.999, eps=1e-8):
        """Perform stochastic gradient descent with adaptive estimation (Adam)

        http://arxiv.org/pdf/1412.6980.pdf

        """
        m = np.zeros(len(x))
        v = np.zeros(len(x))
        opt = self.params.adam()
        opt.send(None)
        for t in range(num_iters):
            opt.send(t + 1)

class Logit(Model):
    def __init__(self, x, y):
        super().__init__(x, y)

    def llik(eta):
        """Return the model log-likelihood

        logit(y) = eta + e
        eta = X (z .* b)

        """
        return eta * self.y - numpy.log(1 + numpy.exp(eta))
