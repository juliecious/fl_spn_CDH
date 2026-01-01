import numpy as np
from numpy import sqrt
from numpy.linalg import eigh, eigvalsh
from scipy import stats
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import ConstantKernel as C
from sklearn.gaussian_process.kernels import WhiteKernel

from causallearn.utils.KCI.GaussianKernel import GaussianKernel
from causallearn.utils.KCI.Kernel import Kernel
from causallearn.utils.KCI.LinearKernel import LinearKernel
from causallearn.utils.KCI.PolynomialKernel import PolynomialKernel

# from scipy import stats
import random
import math
import time
from numpy.linalg import inv

# import warnings
# import arviz as az
# import matplotlib.pyplot as plt
# import theano.tensor as tt
# warnings.simplefilter(action="ignore", category=FutureWarning)
# MV2_USE_THREAD_WARNING=0
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


class KCI_UInd(object):
    """
    Unconditional Kernel Independence Test (Python Implementation)
    """

    def __init__(
        self,
        kernelX="Gaussian",
        kernelY="Gaussian",
        null_ss=1000,
        approx=True,
        est_width="empirical",
        polyd=2,
        kwidthx=None,
        kwidthy=None,
    ):
        self.kernelX = kernelX
        self.kernelY = kernelY
        self.est_width = est_width
        self.polyd = polyd
        self.kwidthx = kwidthx
        self.kwidthy = kwidthy
        self.nullss = null_ss
        self.thresh = 1e-6
        self.approx = approx

    def compute_pvalue(self, data_x=None, data_y=None):
        Kx, Ky = self.kernel_matrix(data_x, data_y)
        test_stat, Kxc, Kyc = self.HSIC_V_statistic(Kx, Ky)

        if self.approx:
            k_appr, theta_appr = self.get_kappa(Kxc, Kyc)
            pvalue = 1 - stats.gamma.cdf(test_stat, k_appr, 0, theta_appr)
        else:
            null_dstr = self.null_sample_spectral(Kxc, Kyc)
            pvalue = sum(null_dstr.squeeze() > test_stat) / float(self.nullss)
        return pvalue, test_stat

    def compute_pvalue_rf(self, data_x=None, data_y=None):
        raise NotImplementedError(
            "R-based Random Features (RF) disabled to prevent crashes."
        )

    def kernel_matrix(self, data_x, data_y):
        if self.kernelX == "Gaussian":
            if self.est_width == "manual":
                kernelX = GaussianKernel(self.kwidthx)
            else:
                kernelX = GaussianKernel()
                if self.est_width == "median":
                    kernelX.set_width_median(data_x)
                elif self.est_width == "empirical":
                    kernelX.set_width_empirical_hsic(data_x)
        elif self.kernelX == "Polynomial":
            kernelX = PolynomialKernel(self.polyd)
        else:
            kernelX = LinearKernel()

        if self.kernelY == "Gaussian":
            if self.est_width == "manual":
                kernelY = GaussianKernel(self.kwidthy)
            else:
                kernelY = GaussianKernel()
                if self.est_width == "median":
                    kernelY.set_width_median(data_y)
                elif self.est_width == "empirical":
                    kernelY.set_width_empirical_hsic(data_y)
        elif self.kernelY == "Polynomial":
            kernelY = PolynomialKernel(self.polyd)
        else:
            kernelY = LinearKernel()

        data_x = stats.zscore(data_x, ddof=1, axis=0)
        data_x[np.isnan(data_x)] = 0.0
        data_y = stats.zscore(data_y, ddof=1, axis=0)
        data_y[np.isnan(data_y)] = 0.0

        Kx = kernelX.kernel(data_x)
        Ky = kernelY.kernel(data_y)
        return Kx, Ky

    def HSIC_V_statistic(self, Kx, Ky):
        Kxc = Kernel.center_kernel_matrix(Kx)
        Kyc = Kernel.center_kernel_matrix(Ky)
        V_stat = np.sum(Kxc * Kyc)
        return V_stat, Kxc, Kyc

    def null_sample_spectral(self, Kxc, Kyc):
        T = Kxc.shape[0]
        num_eig = np.int(np.floor(T / 2)) if T > 1000 else T
        lambdax = eigvalsh(Kxc)
        lambday = eigvalsh(Kyc)
        lambdax = -np.sort(-lambdax)[:num_eig]
        lambday = -np.sort(-lambday)[:num_eig]

        lambda_prod = np.dot(
            lambdax.reshape(num_eig, 1), lambday.reshape(1, num_eig)
        ).reshape((num_eig**2, 1))
        lambda_prod = lambda_prod[lambda_prod > lambda_prod.max() * self.thresh]
        f_rand = np.random.chisquare(1, (lambda_prod.shape[0], self.nullss))
        return lambda_prod.T.dot(f_rand) / T

    def get_kappa(self, Kx, Ky):
        T = Kx.shape[0]
        mean_appr = np.trace(Kx) * np.trace(Ky) / T
        var_appr = 2 * np.sum(Kx**2) * np.sum(Ky**2) / T / T
        k_appr = mean_appr**2 / var_appr
        theta_appr = var_appr / mean_appr
        return k_appr, theta_appr


class KCI_CInd(object):
    """
    Conditional Kernel Independence Test (Python Implementation)
    """

    def __init__(
        self,
        kernelX="Gaussian",
        kernelY="Gaussian",
        kernelZ="Gaussian",
        nullss=5000,
        est_width="empirical",
        use_gp=False,
        approx=True,
        polyd=2,
        kwidthx=None,
        kwidthy=None,
        kwidthz=None,
    ):
        self.kernelX = kernelX
        self.kernelY = kernelY
        self.kernelZ = kernelZ
        self.est_width = est_width
        self.polyd = polyd
        self.kwidthx = kwidthx
        self.kwidthy = kwidthy
        self.kwidthz = kwidthz
        self.nullss = nullss
        self.epsilon_x = 1e-3
        self.epsilon_y = 1e-3
        self.use_gp = use_gp
        self.thresh = 1e-5
        self.approx = approx

    def compute_pvalue_rf(self, data_x=None, data_y=None, data_z=None):
        raise NotImplementedError(
            "R-based Random Features (RF) disabled to prevent crashes."
        )

    def compute_pvalue(self, data_x=None, data_y=None, data_z=None):
        Kx, Ky, Kzx, Kzy = self.kernel_matrix(data_x, data_y, data_z)
        test_stat, KxR, KyR = self.KCI_V_statistic(Kx, Ky, Kzx, Kzy)
        uu_prod, size_u = self.get_uuprod(KxR, KyR)

        if self.approx:
            k_appr, theta_appr = self.get_kappa(uu_prod)
            pvalue = 1 - stats.gamma.cdf(test_stat, k_appr, 0, theta_appr)
        else:
            null_samples = self.null_sample_spectral(uu_prod, size_u, Kx.shape[0])
            pvalue = sum(null_samples > test_stat) / float(self.nullss)
        return pvalue, test_stat

    def kernel_matrix(self, data_x, data_y, data_z):
        data_x = stats.zscore(data_x, ddof=1, axis=0)
        data_x[np.isnan(data_x)] = 0.0
        data_y = stats.zscore(data_y, ddof=1, axis=0)
        data_y[np.isnan(data_y)] = 0.0
        data_z = stats.zscore(data_z, ddof=1, axis=0)
        data_z[np.isnan(data_z)] = 0.0

        data_x_aug = np.concatenate((data_x, 0.5 * data_z), axis=1)

        # Config Kernel X
        if self.kernelX == "Gaussian":
            if self.est_width == "manual":
                kernelX = GaussianKernel(self.kwidthx)
            else:
                kernelX = GaussianKernel()
                if self.est_width == "median":
                    kernelX.set_width_median(data_x_aug)
                elif self.est_width == "empirical":
                    kernelX.set_width_empirical_kci(data_z)
        else:
            kernelX = (
                PolynomialKernel(self.polyd)
                if self.kernelX == "Polynomial"
                else LinearKernel()
            )

        # Config Kernel Y
        if self.kernelY == "Gaussian":
            if self.est_width == "manual":
                kernelY = GaussianKernel(self.kwidthy)
            else:
                kernelY = GaussianKernel()
                if self.est_width == "median":
                    kernelY.set_width_median(data_y)
                elif self.est_width == "empirical":
                    kernelY.set_width_empirical_kci(data_z)
        else:
            kernelY = (
                PolynomialKernel(self.polyd)
                if self.kernelY == "Polynomial"
                else LinearKernel()
            )

        Kx = kernelX.kernel(data_x_aug)
        Ky = kernelY.kernel(data_y)
        Kx = Kernel.center_kernel_matrix(Kx)
        Ky = Kernel.center_kernel_matrix(Ky)

        # Config Kernel Z
        if self.kernelZ == "Gaussian":
            if not self.use_gp:
                if self.est_width == "manual":
                    kernelZ = GaussianKernel(self.kwidthz)
                else:
                    kernelZ = GaussianKernel()
                    if self.est_width == "median":
                        kernelZ.set_width_median(data_z)
                    elif self.est_width == "empirical":
                        kernelZ.set_width_empirical_kci(data_z)
                Kzx = kernelZ.kernel(data_z)
                Kzx = Kernel.center_kernel_matrix(Kzx)
                Kzy = Kzx
            else:
                # GP Implementation skipped for brevity/safety unless explicitly enabled
                raise NotImplementedError(
                    "GP-based width estimation causing instability, use 'empirical' or 'median'."
                )
        else:
            kernelZ = (
                PolynomialKernel(self.polyd)
                if self.kernelZ == "Polynomial"
                else LinearKernel()
            )
            Kzx = kernelZ.kernel(data_z)
            Kzx = Kernel.center_kernel_matrix(Kzx)
            Kzy = Kzx

        return Kx, Ky, Kzx, Kzy

    def KCI_V_statistic(self, Kx, Ky, Kzx, Kzy):
        KxR, Rzx = Kernel.center_kernel_matrix_regression(Kx, Kzx, self.epsilon_x)
        if self.epsilon_x != self.epsilon_y:
            KyR, _ = Kernel.center_kernel_matrix_regression(Ky, Kzy, self.epsilon_y)
        else:
            KyR = Rzx.dot(Ky.dot(Rzx))
        Vstat = np.sum(KxR * KyR)
        return Vstat, KxR, KyR

    def get_uuprod(self, Kx, Ky):
        wx, vx = eigh(0.5 * (Kx + Kx.T))
        wy, vy = eigh(0.5 * (Ky + Ky.T))

        # Sort desc
        idx = np.argsort(-wx)
        idy = np.argsort(-wy)
        wx = wx[idx]
        vx = vx[:, idx]
        wy = wy[idy]
        vy = vy[:, idy]

        # Filter
        mask_x = wx > np.max(wx) * self.thresh
        mask_y = wy > np.max(wy) * self.thresh
        vx = vx[:, mask_x]
        wx = wx[mask_x]
        vy = vy[:, mask_y]
        wy = wy[mask_y]

        # Scale
        vx = vx.dot(np.diag(np.sqrt(wx)))
        vy = vy.dot(np.diag(np.sqrt(wy)))

        # --- OPTIMIZED VECTORIZED OPERATION ---
        # Previous double loop was:
        # for i in range(num_eigx): for j in range(num_eigy): uu[:, i*...] = vx[:,i] * vy[:,j]
        # This is equivalent to an outer product flattened.
        # vx: [T, Nx], vy: [T, Ny]
        # Result uu: [T, Nx * Ny]

        # 1. Broadcast multiply: [T, Nx, 1] * [T, 1, Ny] = [T, Nx, Ny]
        uu_3d = vx[:, :, None] * vy[:, None, :]

        # 2. Flatten last two dims: [T, Nx * Ny]
        T = Kx.shape[0]
        uu = uu_3d.reshape(T, -1)
        size_u = uu.shape[1]

        if size_u > T:
            uu_prod = uu.dot(uu.T)
        else:
            uu_prod = uu.T.dot(uu)

        return uu_prod, size_u

    def null_sample_spectral(self, uu_prod, size_u, T):
        eig_uu = eigvalsh(uu_prod)
        eig_uu = -np.sort(-eig_uu)
        eig_uu = eig_uu[0 : np.min((T, size_u))]
        eig_uu = eig_uu[eig_uu > np.max(eig_uu) * self.thresh]

        f_rand = np.random.chisquare(1, (eig_uu.shape[0], self.nullss))
        null_dstr = eig_uu.T.dot(f_rand)
        return null_dstr

    def get_kappa(self, uu_prod):
        mean_appr = np.trace(uu_prod)
        var_appr = 2 * np.trace(uu_prod.dot(uu_prod))
        k_appr = mean_appr**2 / var_appr
        theta_appr = var_appr / mean_appr
        return k_appr, theta_appr


# There is no conditional set
class LGM_UInd(object):
    def __init__(self, kernelX="Gaussian", kernelY="Gaussian"):
        self.kernelX = kernelX
        self.kernelY = kernelY

    # data_x is the domain index; data_y is the observed variable.
    def compute_pval(self, data_x=None, data_y=None, K=0):
        # print(f"GMM UI is on!!!!!!! K={K}. x shape: {data_x.shape}. y shape: {data_y.shape}. {type(data_x)}")
        p_value = 1  # independent

        s1, s2 = data_y.shape
        fed_data = data_y.reshape(K, int(s1 / K), s2)

        """
            Using Linear Gaussian Model to get an approximate distribution.
        """
        res_dist = []  # shape: [K clients, 2 parameters]
        for k in range(K):
            mu, sigma = fed_data[k].mean(), fed_data[k].var()
            res_dist.append([mu, sigma])
        res_dist = np.array(res_dist)

        """
            Using Monte Carlo Simulation to recover the samples from Linear Gaussian Model.
        """
        recovered_samples = []  # shape: [K clients, 2 parameters]
        for k in range(K):
            N = 200
            mu, sigma = res_dist[k]
            samples = np.random.normal(mu, sigma, N)
            # random.shuffle(samples)
            recovered_samples.append(samples)
        recovered_samples = np.array(recovered_samples, dtype=np.float64)
        # print("Recovered shape: ", recovered_samples.shape)

        # t_3 = time.time()
        # print(f"Time for getting recovered samples: {t_3-t_2}s.")
        # import matplotlib.pyplot as plt
        # import pandas as pd
        # import seaborn as sns
        # plt.rcParams['figure.figsize'] = [20, 10]
        # ax = plt.figure()
        # for k in range(8):
        #     ax.add_subplot(2,4,k+1)
        #     sns.distplot(fed_data[k,:,:].flatten(), hist=False, kde=True, label="Original")
        #     sns.distplot(recovered_samples[k].flatten(), hist=False, kde=True, label="Recovered")
        # # plt.title('$X_0$')
        # plt.legend(fontsize=12, loc=1)
        # plt.savefig("result/expRes/x1.pdf")

        """
            Do Kolmogorov-Smirnov Test (KS Test) to determine whether two distributions are equal or not, according to their samples.
            Two key parameters: alpha=1e-4, ratio=0.4.
        """
        rand_list = np.array(range(K))
        random.shuffle(rand_list)
        pval_list = []
        pval_rec_list = []
        for k in range(0, K, 2):
            """
            Distribution Matching Test for all pairs.
            """
            i, j = rand_list[k], rand_list[k + 1]
            res1 = stats.ks_2samp(fed_data[i].flatten(), fed_data[j].flatten())
            res2 = stats.ks_2samp(
                recovered_samples[i].flatten(), recovered_samples[j].flatten()
            )
            print(f"GroundTrue: i-{i},j-{j}:", res1)
            print(f"Recovered: i-{i},j-{j}:", res2)
            pval_list.append(res1.pvalue)
            pval_rec_list.append(res2.pvalue)

        count = np.sum([i <= 0.05 for i in pval_list])
        count_rec = np.sum([i <= 1e-4 for i in pval_rec_list])
        print(f"Truth count: ", count)
        print(f"Recover count: ", count_rec)

        p_value = 0 if count / len(pval_list) > 0.3 else 1
        print(f"P-val: {p_value}.")
        p_value_rec = 0 if count_rec / len(pval_rec_list) > 0.4 else 1
        print(f"P-rec-val: {p_value_rec}.")

        # t_4 = time.time()
        # print(f"########### Time for KS test: {t_4-t_3}s.")

        return p_value_rec


# conditional set length =1
class LGM_CInd_1(object):
    def __init__(self, kernelX="Gaussian", kernelY="Gaussian"):
        self.kernelX = kernelX
        self.kernelY = kernelY

    def compute_pval(self, data_x=None, data_y=None, data_z=None, K=0):
        print("Linear Gaussian Model and Conditional Independent Test with |S|=1.")

        s1, s2 = data_y.shape
        fed_data_y = data_y.reshape(K, int(s1 / K), s2)
        s1, s2 = data_z.shape
        fed_data_z = data_z.reshape(K, int(s1 / K), s2)

        """
            Using Linear Gaussian Model to get approximate distributions.
        """
        res_para = []
        for k in range(K):
            dy, dz = fed_data_y[k], fed_data_z[k]
            dy, dz = np.squeeze(dy), np.squeeze(dz)  # [n,1] -> [n]
            cov_mat = np.cov(dy, dz, bias=True)
            inv_cov = inv(cov_mat)
            dy_mu, dz_mu = dy.mean(), dz.mean()

            # according to the formulations.
            coef = inv_cov[0][1] / inv_cov[1][1]
            mu = dy_mu - coef * dz_mu
            sigma = inv_cov[0][0] - inv_cov[0][1] / inv_cov[1][1] * inv_cov[1][0]
            res_para.append([coef, mu, sigma])

        """
            Using Monte Carlo Simulation to recover the samples from Gaussian model.
            We skip the coef term as an approximation.
        """
        recovered_samples = []  # shape: [K clients, N samples]
        for k in range(K):
            N = 200
            coef, mu, sigma = res_para[k]
            samples = np.random.normal(mu, sigma, N)
            recovered_samples.append(samples)
        recovered_samples = np.array(recovered_samples, dtype=np.float64)

        """
            Do Kolmogorov-Smirnov Test (KS Test) to determine whether two distributions are equal or not, according to their samples.
            Two key parameters: alpha=1e-4, ratio=0.4.
        """
        pval_list = []
        rand_list = np.array(range(K))
        random.shuffle(rand_list)
        for k in range(0, K, 2):
            """
            Distribution Matching Test for all pairs.
            """
            i, j = rand_list[k], rand_list[k + 1]
            res = stats.ks_2samp(
                recovered_samples[i].flatten(), recovered_samples[j].flatten()
            )
            pval_list.append(res.pvalue)

        count_rec = np.sum([i <= 1e-4 for i in pval_list])
        print(f"Recovered count: ", count_rec)

        p_value_rec = 0 if count_rec / len(pval_list) > 0.4 else 1
        print(f"P-val: {p_value_rec}.")

        return p_value_rec


# conditional set length >=2
class LGM_CInd_2(object):
    def __init__(self, kernelX="Gaussian", kernelY="Gaussian"):
        self.kernelX = kernelX
        self.kernelY = kernelY

    def compute_pval(self, data_x=None, data_y=None, data_z=None, K=0):
        print("Hi, Linear Gaussian Model. I am in Conditional Independent Test.")

        s1, s2 = data_y.shape
        fed_data_y = data_y.reshape(K, int(s1 / K), s2)
        s1, s2 = data_z.shape
        fed_data_z = data_z.reshape(K, int(s1 / K), s2)

        """
            Seperate y into y1,y2 according to the L2 norm values of z.
            Using Gaussian Mixture Model to get approximate distributions for y1,y2.
            We randomly choose 2 clients and make comparisons.
        """
        res_dist_1 = []
        res_dist_2 = []
        rand_list = np.array(range(K))
        random.shuffle(rand_list)
        for k in range(0, K, 2):
            i, j = rand_list[k], rand_list[k + 1]

            dy1, dy2 = fed_data_y[k], fed_data_y[k + 1]
            dz1, dz2 = fed_data_z[k], fed_data_z[k + 1]
            sep_data_1 = np.linalg.norm(dz1, axis=1, ord=2)
            sep_data_2 = np.linalg.norm(dz2, axis=1, ord=2)

            sep_num = (
                math.floor(
                    (
                        max(sep_data_1)
                        + min(sep_data_1)
                        + max(sep_data_2)
                        + min(sep_data_2)
                    )
                    / 4
                )
                - 0.5
            )  # choose a seperation number
            print(f"Sep_num: {sep_num}", sep_num)

            dy1 = np.squeeze(dy1)  # [n,1] -> [n]
            dy2 = np.squeeze(dy2)  # [n,1] -> [n]

            sep_idx_1_1 = np.where(sep_data_1 > sep_num)
            sep_idx_1_2 = np.where(sep_data_1 <= sep_num)
            dy_1_1 = dy1[sep_idx_1_1].reshape(-1, 1)
            dy_1_2 = dy1[sep_idx_1_2].reshape(-1, 1)
            print(f"dy1 shape: {dy_1_1.shape},{dy_1_2.shape}.")
            print(f"sep idx 1: {sep_idx_1_1}, sep idx 2: {sep_idx_1_2}.")
            print(f"length: {len(sep_idx_1_1)}, {len(sep_idx_1_2)}.")

            sep_idx_2_1 = np.where(sep_data_2 > sep_num)
            sep_idx_2_2 = np.where(sep_data_2 <= sep_num)
            dy_2_1 = dy2[sep_idx_2_1].reshape(-1, 1)
            dy_2_2 = dy2[sep_idx_2_2].reshape(-1, 1)
            print(f"dy2 shape: {dy_2_1.shape},{dy_2_2.shape}.")

            if dy_1_1.shape[0] < 3:
                print("######## Zero happens.")
                dy_1_1 = dy1
            if dy_1_2.shape[0] < 3:
                print("######## Zero happens.")
                dy_1_2 = dy1
            if dy_2_1.shape[0] < 3:
                print("######## Zero happens.")
                dy_2_1 = dy2
            if dy_2_2.shape[0] < 3:
                print("######## Zero happens.")
                dy_2_2 = dy2

            mu, sigma, pi = EM_single(dy_1_1, 2, 1e-8)
            res_dist_1.append([mu, sigma, pi])
            mu, sigma, pi = EM_single(dy_2_1, 2, 1e-8)
            res_dist_1.append([mu, sigma, pi])

            mu, sigma, pi = EM_single(dy_1_2, 2, 1e-8)
            res_dist_2.append([mu, sigma, pi])
            mu, sigma, pi = EM_single(dy_2_2, 2, 1e-8)
            res_dist_2.append([mu, sigma, pi])

        """
            Using Monte Carlo Simulation to recover the samples from GMM.
        """
        recovered_samples_1 = []  # shape: [K clients, 3 parameters, 2 clusters]
        recovered_samples_2 = []
        for k in range(K):
            N = 200
            mus, sigmas, ps = res_dist_1[k]
            samples = np.hstack(
                [
                    np.random.normal(mus[0], sigmas[0], int(ps[0] * N)),
                    np.random.normal(
                        mus[1], sigmas[1], max(int(ps[1] * N), N - int(ps[0] * N))
                    ),
                ]
            )
            random.shuffle(samples)
            recovered_samples_1.append(samples)

            mus, sigmas, ps = res_dist_2[k]
            samples = np.hstack(
                [
                    np.random.normal(mus[0], sigmas[0], int(ps[0] * N)),
                    np.random.normal(
                        mus[1], sigmas[1], max(int(ps[1] * N), N - int(ps[0] * N))
                    ),
                ]
            )
            random.shuffle(samples)
            recovered_samples_2.append(samples)
        recovered_samples_1 = np.array(recovered_samples_1, dtype=np.float64)
        recovered_samples_2 = np.array(recovered_samples_2, dtype=np.float64)

        """
            Do Kolmogorov-Smirnov Test (KS Test) to determine whether two distributions are equal or not, according to their samples.
            Two key parameters: alpha=1e-4, ratio=0.4.
        """
        pval1_list = []
        pval2_list = []
        for k in range(0, K, 2):
            """
            Distribution Matching Test for all pairs.
            """
            # i, j = rand_list[k], rand_list[k+1]
            i, j = k, k + 1
            res1 = stats.ks_2samp(
                recovered_samples_1[i].flatten(), recovered_samples_1[j].flatten()
            )
            res2 = stats.ks_2samp(
                recovered_samples_2[i].flatten(), recovered_samples_2[j].flatten()
            )
            print(
                f"Recovered pvalue1 and pvalue2: i-{i},j-{j}:", res1.pvalue, res2.pvalue
            )
            pval1_list.append(res1.pvalue)
            pval2_list.append(res2.pvalue)

        count_rec_1 = np.sum([i <= 1e-4 for i in pval1_list])
        count_rec_2 = np.sum([i <= 1e-4 for i in pval2_list])
        print(f"Recovered count: ", count_rec_1, count_rec_2)

        p_value_rec_1 = 0 if count_rec_1 / len(pval1_list) > 0.4 else 1
        p_value_rec_2 = 0 if count_rec_2 / len(pval2_list) > 0.4 else 1
        print(f"P-rec-val: {p_value_rec_1, p_value_rec_2}.")

        return p_value_rec_1 * p_value_rec_2


class GMM_UInd(object):
    def __init__(self, kernelX="Gaussian", kernelY="Gaussian"):
        self.kernelX = kernelX
        self.kernelY = kernelY

    def compute_pval(self, data_x=None, data_y=None, K=0):
        # print(f"GMM UI is on!!!!!!! K={K}. x shape: {data_x.shape}. y shape: {data_y.shape}. {type(data_x)}")
        p_value = 1  # independent

        s1, s2 = data_y.shape
        fed_data = data_y.reshape(K, int(s1 / K), s2)

        # data_joint = np.concatenate((data_y, data_y+0.5), axis=1)
        # s1, s2 = data_joint.shape
        # fed_data_joint = data_joint.reshape(K, int(s1/K), s2)

        # t_1 = time.time()

        """
            Using Gaussian Mixture Model to get an approximate distribution.
        """
        res_dist = []  # shape: [K clients, 3 parameters, 2 clusters]
        for k in range(K):
            mu, sigma, pi = EM_single(fed_data[k], 2, 1e-8)
            # mu_j, sigma_j, pi_j = EM_multi(fed_data_joint[k], 3, 1e-8, mode='diag')
            res_dist.append([mu, sigma, pi])
        res_dist = np.array(res_dist)
        # print("Paras shape: ", res_dist.shape)

        # t_2 = time.time()
        # print(f"Time for getting GMM: {t_2-t_1}s.")

        """
            Using Monte Carlo Simulation to recover the samples from GMM.
        """
        recovered_samples = []  # shape: [K clients, 3 parameters, 2 clusters]
        for k in range(K):
            N = 200
            mus, sigmas, ps = res_dist[k]
            samples = np.hstack(
                [
                    np.random.normal(mus[0], sigmas[0], int(ps[0] * N)),
                    np.random.normal(
                        mus[1], sigmas[1], max(int(ps[1] * N), N - int(ps[0] * N))
                    ),
                ]
            )
            random.shuffle(samples)
            recovered_samples.append(samples)
        recovered_samples = np.array(recovered_samples, dtype=np.float64)
        # print("Recovered shape: ", recovered_samples.shape)

        # t_3 = time.time()
        # print(f"Time for getting recovered samples: {t_3-t_2}s.")

        # import matplotlib.pyplot as plt
        # import pandas as pd
        # import seaborn as sns
        # plt.rcParams['figure.figsize'] = [20, 10]
        # ax = plt.figure()
        # for k in range(8):
        #     ax.add_subplot(2,4,k+1)
        #     sns.distplot(fed_data[k,:,:].flatten(), hist=False, kde=True, label="Original")
        #     sns.distplot(recovered_samples[k].flatten(), hist=False, kde=True, label="Recovered")
        # # plt.title('$X_0$')
        # plt.legend(fontsize=12, loc=1)
        # plt.savefig("result/expRes/x1.pdf")

        """
            Do Kolmogorov-Smirnov Test (KS Test) to determine whether two distributions are equal or not, according to their samples.
            Two key parameters: alpha=1e-4, ratio=0.4.
        """
        rand_list = np.array(range(K))
        random.shuffle(rand_list)
        pval_list = []
        pval_rec_list = []
        for k in range(0, K, 2):
            """
            Distribution Matching Test for all pairs.
            """
            i, j = rand_list[k], rand_list[k + 1]
            res1 = stats.ks_2samp(fed_data[i].flatten(), fed_data[j].flatten())
            res2 = stats.ks_2samp(
                recovered_samples[i].flatten(), recovered_samples[j].flatten()
            )
            print(f"GroundTrue: i-{i},j-{j}:", res1)
            print(f"Recovered: i-{i},j-{j}:", res2)
            pval_list.append(res1.pvalue)
            pval_rec_list.append(res2.pvalue)

        count = np.sum([i <= 0.05 for i in pval_list])
        count_rec = np.sum([i <= 1e-4 for i in pval_rec_list])
        print(f"Truth count: ", count)
        print(f"Recover count: ", count_rec)

        p_value = 0 if count / len(pval_list) > 0.3 else 1
        print(f"P-val: {p_value}.")
        p_value_rec = 0 if count_rec / len(pval_rec_list) > 0.4 else 1
        print(f"P-rec-val: {p_value_rec}.")

        # t_4 = time.time()
        # print(f"########### Time for KS test: {t_4-t_3}s.")

        return p_value_rec


def EM_single(data, K, threshold):
    # data shape: [n,1] -> [n]
    data = np.squeeze(data)

    # Initialization
    mu0 = np.ones(K)
    for k in range(K):
        idx = np.int(random.random() * len(data))
        mu0[k] += data[idx]
    sigma0 = np.ones(K) * np.var(data)
    pi0 = np.ones(K) * 1.0 / K
    current_log_likelihood = log_likelihood_single(data, K, mu0, sigma0, pi0)

    # EM method.
    mu, sigma, pi = mu0, sigma0, pi0
    max_iter = 100
    for it in range(max_iter):
        resp = e_step_single(data, K, mu, sigma, pi)
        mu, sigma, pi = m_step_single(data, K, resp)

        new_log_likelihood = log_likelihood_single(data, K, mu, sigma, pi)
        if abs(new_log_likelihood - current_log_likelihood) < threshold:
            # print("iters=%d" % (it))
            break
        current_log_likelihood = new_log_likelihood
        # print(f"Iters:{it}", current_log_likelihood, mu, sigma, pi)
    return mu, sigma, pi


def e_step_single(data, K, mu, sigma, pi):
    idvs = len(data)
    resp = np.zeros((idvs, K))
    for i in range(idvs):
        for k in range(K):
            resp[i][k] = (
                pi[k]
                * gaussian_single(data[i], mu[k], sigma[k])
                / likelihood_single(data[i], K, mu, sigma, pi)
            )
    return resp


def m_step_single(data, K, resp):
    idvs = len(data)
    mu = np.zeros(K)
    sigma = np.zeros(K)
    pi = np.zeros(K)
    marg_resp = np.zeros(K)
    for k in range(K):
        for i in range(idvs):
            marg_resp[k] += resp[i][k]
            mu[k] += (resp[i][k]) * data[i]
        mu[k] /= marg_resp[k]  # update mu.
        for i in range(idvs):
            x_mu = data[i] - mu[k]
            sigma[k] += (resp[i][k] / marg_resp[k]) * x_mu * x_mu  # update sigma.

        pi[k] = marg_resp[k] / idvs  # update pi.
    return mu, sigma, pi


def log_likelihood_single(data, K, mu, sigma, pi):
    log_likelihood = 0.0
    for n in range(len(data)):
        log_likelihood += np.log(
            likelihood_single(data[n], K, mu, sigma, pi)
        )  # log P(X) = log \prod P(X_i) = \sum log P(X_i)
    return log_likelihood


def likelihood_single(x, K, mu, sigma, pi):
    rs = 0.0
    for k in range(K):
        rs += pi[k] * gaussian_single(x, mu[k], sigma[k])
    return rs


def gaussian_single(x, mu, sigma):
    if sigma == 0:
        sigma = 1e-6

    norm_factor = 2 * np.pi * sigma
    norm_factor = 1.0 / np.sqrt(norm_factor)
    rs = norm_factor * np.exp(-0.5 * (x - mu) * (x - mu) / sigma)
    return rs


class GMM_CInd(object):
    def __init__(self, kernelX="Gaussian", kernelY="Gaussian"):
        self.kernelX = kernelX
        self.kernelY = kernelY

    def compute_pval(self, data_x=None, data_y=None, data_z=None, K=0):
        # s1, s2 = data_z.shape
        # dz = 1 if s2==1 else 0

        # data_joint = np.concatenate((data_y, data_z), axis=1)
        # s1, s2 = data_joint.shape
        # fed_data_joint = data_joint.reshape(K, int(s1/K), s2)

        # data_marginal = data_z
        # s1, s2 = data_marginal.shape
        # fed_data_marginal = data_marginal.reshape(K, int(s1/K), s2)
        print("Hi, I am in Conditional Independent Test.")

        s1, s2 = data_y.shape
        fed_data_y = data_y.reshape(K, int(s1 / K), s2)
        s1, s2 = data_z.shape
        fed_data_z = data_z.reshape(K, int(s1 / K), s2)

        """
            Seperate y into y1,y2 according to the L2 norm values of z.
            Using Gaussian Mixture Model to get approximate distributions for y1,y2.
            We randomly choose 2 clients and make comparisons.
        """
        res_dist_1 = []
        res_dist_2 = []
        rand_list = np.array(range(K))
        random.shuffle(rand_list)
        for k in range(0, K, 2):
            i, j = rand_list[k], rand_list[k + 1]

            dy1, dy2 = fed_data_y[k], fed_data_y[k + 1]
            dz1, dz2 = fed_data_z[k], fed_data_z[k + 1]
            sep_data_1 = np.linalg.norm(dz1, axis=1, ord=2)
            sep_data_2 = np.linalg.norm(dz2, axis=1, ord=2)

            sep_num = (
                math.floor(
                    (
                        max(sep_data_1)
                        + min(sep_data_1)
                        + max(sep_data_2)
                        + min(sep_data_2)
                    )
                    / 4
                )
                - 0.5
            )  # choose a seperation number
            print(f"Sep_num: {sep_num}", sep_num)

            dy1 = np.squeeze(dy1)  # [n,1] -> [n]
            dy2 = np.squeeze(dy2)  # [n,1] -> [n]

            sep_idx_1_1 = np.where(sep_data_1 > sep_num)
            sep_idx_1_2 = np.where(sep_data_1 <= sep_num)
            dy_1_1 = dy1[sep_idx_1_1].reshape(-1, 1)
            dy_1_2 = dy1[sep_idx_1_2].reshape(-1, 1)
            print(f"dy1 shape: {dy_1_1.shape},{dy_1_2.shape}.")
            print(f"sep idx 1: {sep_idx_1_1}, sep idx 2: {sep_idx_1_2}.")
            print(f"length: {len(sep_idx_1_1)}, {len(sep_idx_1_2)}.")

            sep_idx_2_1 = np.where(sep_data_2 > sep_num)
            sep_idx_2_2 = np.where(sep_data_2 <= sep_num)
            dy_2_1 = dy2[sep_idx_2_1].reshape(-1, 1)
            dy_2_2 = dy2[sep_idx_2_2].reshape(-1, 1)
            print(f"dy2 shape: {dy_2_1.shape},{dy_2_2.shape}.")

            if dy_1_1.shape[0] < 3:
                print("######## Zero happens.")
                dy_1_1 = dy1
            if dy_1_2.shape[0] < 3:
                print("######## Zero happens.")
                dy_1_2 = dy1
            if dy_2_1.shape[0] < 3:
                print("######## Zero happens.")
                dy_2_1 = dy2
            if dy_2_2.shape[0] < 3:
                print("######## Zero happens.")
                dy_2_2 = dy2

            mu, sigma, pi = EM_single(dy_1_1, 2, 1e-8)
            res_dist_1.append([mu, sigma, pi])
            mu, sigma, pi = EM_single(dy_2_1, 2, 1e-8)
            res_dist_1.append([mu, sigma, pi])

            mu, sigma, pi = EM_single(dy_1_2, 2, 1e-8)
            res_dist_2.append([mu, sigma, pi])
            mu, sigma, pi = EM_single(dy_2_2, 2, 1e-8)
            res_dist_2.append([mu, sigma, pi])

        """
            Using Monte Carlo Simulation to recover the samples from GMM.
        """
        recovered_samples_1 = []  # shape: [K clients, 3 parameters, 2 clusters]
        recovered_samples_2 = []
        for k in range(K):
            N = 200
            mus, sigmas, ps = res_dist_1[k]
            samples = np.hstack(
                [
                    np.random.normal(mus[0], sigmas[0], int(ps[0] * N)),
                    np.random.normal(
                        mus[1], sigmas[1], max(int(ps[1] * N), N - int(ps[0] * N))
                    ),
                ]
            )
            random.shuffle(samples)
            recovered_samples_1.append(samples)

            mus, sigmas, ps = res_dist_2[k]
            samples = np.hstack(
                [
                    np.random.normal(mus[0], sigmas[0], int(ps[0] * N)),
                    np.random.normal(
                        mus[1], sigmas[1], max(int(ps[1] * N), N - int(ps[0] * N))
                    ),
                ]
            )
            random.shuffle(samples)
            recovered_samples_2.append(samples)
        recovered_samples_1 = np.array(recovered_samples_1, dtype=np.float64)
        recovered_samples_2 = np.array(recovered_samples_2, dtype=np.float64)

        """
            Do Kolmogorov-Smirnov Test (KS Test) to determine whether two distributions are equal or not, according to their samples.
            Two key parameters: alpha=1e-4, ratio=0.4.
        """
        pval1_list = []
        pval2_list = []
        for k in range(0, K, 2):
            """
            Distribution Matching Test for all pairs.
            """
            # i, j = rand_list[k], rand_list[k+1]
            i, j = k, k + 1
            res1 = stats.ks_2samp(
                recovered_samples_1[i].flatten(), recovered_samples_1[j].flatten()
            )
            res2 = stats.ks_2samp(
                recovered_samples_2[i].flatten(), recovered_samples_2[j].flatten()
            )
            print(
                f"Recovered pvalue1 and pvalue2: i-{i},j-{j}:", res1.pvalue, res2.pvalue
            )
            pval1_list.append(res1.pvalue)
            pval2_list.append(res2.pvalue)

        count_rec_1 = np.sum([i <= 1e-4 for i in pval1_list])
        count_rec_2 = np.sum([i <= 1e-4 for i in pval2_list])
        print(f"Recovered count: ", count_rec_1, count_rec_2)

        p_value_rec_1 = 0 if count_rec_1 / len(pval1_list) > 0.4 else 1
        p_value_rec_2 = 0 if count_rec_2 / len(pval2_list) > 0.4 else 1
        print(f"P-rec-val: {p_value_rec_1, p_value_rec_2}.")

        # for k in range(K):
        #     """
        #         Using Gaussian Mixture Model to get an approximate distribution, given dataX.
        #     """
        #     mu_j, sigma_j, pi_j = EM_multi(fed_data_joint[k], 3, 1e-8, mode='diag')

        #     if dz==1:
        #         mu_m, sigma_m, pi_m = EM_single(fed_data_marginal[k], 3, 1e-8)
        #     else:
        #         mu_m, sigma_m, pi_m = EM_multi(fed_data_marginal[k], 3, 1e-8, mode='diag')
        return p_value_rec_1 * p_value_rec_2


def random_parameters(data, K, mode=None):
    cols = (data.shape)[1]
    mu = np.zeros((K, cols))
    for k in range(K):
        idx = np.int(random.random() * len(data))
        for col in range(cols):
            mu[k][col] += data[idx, col]

    sigma = np.zeros((K, cols, cols))
    for k in range(K):
        sigma[k] = np.cov(data.T)

    sigma1 = np.zeros((K, cols, cols))
    if mode == "diag":
        for k in range(K):
            sigma1[k] = np.eye(cols) * sigma[k]
    elif mode == "d+s":
        for k in range(K):
            sigma1[k] = np.eye(cols) * np.eys(cols) * sigma[k]
    else:
        sigma1 = sigma
    np.mean(np.array(sigma), axis=0)
    pi = np.ones(K) * 1.0 / K
    return mu, sigma1, pi


def e_step(data, K, mu, sigma, pi):
    idvs = (data.shape)[0]

    resp = np.zeros((idvs, K))

    for i in range(idvs):
        for k in range(K):
            resp[i][k] = (
                pi[k]
                * gaussian(data[i], mu[k], sigma[k])
                / likelihood(data[i], K, mu, sigma, pi)
            )

    return resp


def log_likelihood(data, K, mu, sigma, pi):
    log_likelihood = 0.0
    for n in range(len(data)):
        log_likelihood += np.log(
            likelihood(data[n], K, mu, sigma, pi)
        )  # log P(X) = log \prod P(X_i) = \sum log P(X_i)
    return log_likelihood


def likelihood(x, K, mu, sigma, pi):
    rs = 0.0
    for k in range(K):
        rs += pi[k] * gaussian(x, mu[k], sigma[k])
    return rs


def m_step(data, K, resp, mode=None):
    idvs = (data.shape)[0]
    cols = (data.shape)[1]

    mu = np.zeros((K, cols))
    sigma = np.zeros((K, cols, cols))
    pi = np.zeros(K)

    marg_resp = np.zeros(K)
    for k in range(K):
        for i in range(idvs):
            marg_resp[k] += resp[i][k]
            mu[k] += (resp[i][k]) * data[i]
        mu[k] /= marg_resp[k]

        for i in range(idvs):
            # x_i = (np.zeros((1,cols))+data[k])
            x_mu = np.zeros((1, cols)) + data[i] - mu[k]
            sigma[k] += (resp[i][k] / marg_resp[k]) * x_mu * x_mu.T

        pi[k] = marg_resp[k] / idvs

    sigma1 = np.zeros((K, cols, cols))
    if mode == "diag":
        for k in range(K):
            sigma1[k] = np.eye(cols) * sigma[k]
    elif mode == "share":
        for k in range(K):
            sigma1[k] = np.mean(sigma, axis=0)
    elif mode == "d+s":
        for k in range(K):
            sigma1[k] = np.eye(cols) * np.mean(np.array(sigma), axis=0)
    else:
        sigma1 = sigma

    return mu, sigma1, pi


def gaussian(x, mu, sigma):
    idvs = len(x)
    norm_factor = (2 * np.pi) ** idvs

    norm_factor *= np.linalg.det(sigma)
    norm_factor = 1.0 / np.sqrt(norm_factor)

    x_mu = np.matrix(x - mu)

    rs = norm_factor * np.exp(-0.5 * x_mu * np.linalg.inv(sigma) * x_mu.T)
    return rs


def EM_multi(data, K, threshold, mode):
    mu0, sigma0, pi0 = random_parameters(data, K, mode=mode)
    current_log_likelihood = log_likelihood(data, K, mu0, sigma0, pi0)

    max_iter = 30
    mu, sigma, pi = mu0, sigma0, pi0
    for it in range(max_iter):
        resp = e_step(data, K, mu, sigma, pi)
        mu, sigma, pi = m_step(data, K, resp, mode=mode)

        new_log_likelihood = log_likelihood(data, K, mu, sigma, pi)
        if abs(new_log_likelihood - current_log_likelihood) < threshold:
            print("iters=%d" % (it))
            break
        current_log_likelihood = new_log_likelihood
        print(f"Iters:{it}", current_log_likelihood, pi)

    return mu, sigma, resp
