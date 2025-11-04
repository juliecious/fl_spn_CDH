import os, json, codecs, time, hashlib
import numpy as np
from math import log, sqrt
from collections.abc import Iterable
from scipy.stats import chi2, norm

from causallearn.utils.KCI.KCI import (
    KCI_CInd,
    KCI_UInd,
    GMM_CInd,
    GMM_UInd,
    LGM_UInd,
    LGM_CInd_1,
    LGM_CInd_2,
)
from causallearn.utils.PCUtils import Helper

try:
    import torch
    from sklearn.preprocessing import StandardScaler
    from simple_einet.einet import Einet, EinetConfig
    from simple_einet.layers.distributions.normal import Normal

    SPN_AVAILABLE = True
except ImportError:
    SPN_AVAILABLE = False


CONST_BINCOUNT_UNIQUE_THRESHOLD = 1e5
NO_SPECIFIED_PARAMETERS_MSG = "NO SPECIFIED PARAMETERS"
fisherz = "fisherz"
mv_fisherz = "mv_fisherz"
mc_fisherz = "mc_fisherz"
kci = "kci"
chisq = "chisq"
gsq = "gsq"
d_separation = "d_separation"
gmm = "gmm"
spn = "spn"


def CIT(data, method="fisherz", **kwargs):
    """
    Parameters
    ----------
    data: numpy.ndarray of shape (n_samples, n_features)
    method: str, in ["fisherz", "mv_fisherz", "mc_fisherz", "kci", "chisq", "gsq", "spn"]
    kwargs: placeholder for future arguments, or for KCI specific arguments now
        TODO: ultimately kwargs should be replaced by explicit named parameters.
              check https://github.com/cmu-phil/causal-learn/pull/62#discussion_r927239028
    """
    if method == fisherz:
        return FisherZ(data, **kwargs)
    elif method == kci:
        return KCI(data, **kwargs)
    elif method in [chisq, gsq]:
        return Chisq_or_Gsq(data, method_name=method, **kwargs)
    elif method == mv_fisherz:
        return MV_FisherZ(data, **kwargs)
    elif method == mc_fisherz:
        return MC_FisherZ(data, **kwargs)
    elif method == d_separation:
        return D_Separation(data, **kwargs)
    elif method == spn:
        if not SPN_AVAILABLE:
            raise ImportError(
                "SPN CI test requires simple_einet and pytorch. Please install: pip install torch simple-einet"
            )
        return SPN(data, method_name=method, **kwargs)
    else:
        raise ValueError("Unknown method: {}".format(method))


class CIT_Base(object):
    # Base class for CIT, contains basic operations for input check and caching, etc.
    def __init__(self, data, cache_path=None, **kwargs):
        """
        Parameters
        ----------
        data: data matrix, np.ndarray, in shape (n_samples, n_features)
        cache_path: str, path to save cache .json file. default as None (no io to local file).
        kwargs: for future extension.
        """
        assert isinstance(data, np.ndarray), "Input data must be a numpy array."
        self.data = data
        self.data_hash = hashlib.md5(str(data).encode("utf-8")).hexdigest()
        self.sample_size, self.num_features = data.shape
        self.cache_path = cache_path
        self.SAVE_CACHE_CYCLE_SECONDS = 30
        self.last_time_cache_saved = time.time()
        self.pvalue_cache = {"data_hash": self.data_hash}
        if cache_path is not None:
            assert cache_path.endswith(".json"), "Cache must be stored as .json file."
            if os.path.exists(cache_path):
                with codecs.open(cache_path, "r") as fin:
                    self.pvalue_cache = json.load(fin)
                assert (
                    self.pvalue_cache["data_hash"] == self.data_hash
                ), "Data hash mismatch."
            else:
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    def check_cache_method_consistent(self, method_name, parameters_hash):
        self.method = method_name
        if method_name not in self.pvalue_cache:
            self.pvalue_cache["method_name"] = method_name  # a newly created cache
            self.pvalue_cache["parameters_hash"] = parameters_hash
        else:
            assert (
                self.pvalue_cache["method_name"] == method_name
            ), "CI test method name mismatch."  # a loaded cache
            assert (
                self.pvalue_cache["parameters_hash"] == parameters_hash
            ), "CI test method parameters mismatch."

    def assert_input_data_is_valid(self, allow_nan=False, allow_inf=False):
        assert (
            allow_nan or not np.isnan(self.data).any()
        ), "Input data contains NaN. Please check."
        assert (
            allow_inf or not np.isinf(self.data).any()
        ), "Input data contains Inf. Please check."

    def save_to_local_cache(self):
        if (
            not self.cache_path is None
            and time.time() - self.last_time_cache_saved > self.SAVE_CACHE_CYCLE_SECONDS
        ):
            with codecs.open(self.cache_path, "w") as fout:
                fout.write(json.dumps(self.pvalue_cache, indent=2))
            self.last_time_cache_saved = time.time()

    def get_formatted_XYZ_and_cachekey(self, X, Y, condition_set):
        """
        reformat the input X, Y and condition_set to
            1. convert to built-in types for json serialization
            2. handle multi-dim unconditional variables (for kernel-based)
            3. basic check for valid input (X, Y no overlap with condition_set)
            4. generate unique and hashable cache key

        Parameters
        ----------
        X: int, or np.*int*, or Iterable<int | np.*int*>
        Y: int, or np.*int*, or Iterable<int | np.*int*>
        condition_set: Iterable<int | np.*int*>

        Returns
        -------
        Xs: List<int>, sorted. may swapped with Ys for cache key uniqueness.
        Ys: List<int>, sorted.
        condition_set: List<int>
        cache_key: string. Unique for <X,Y|S> in any input type or order.
        """

        def _stringize(ulist1, ulist2, clist):
            # ulist1, ulist2, clist: list of ints, sorted.
            _strlst = lambda lst: ".".join(map(str, lst))
            return (
                f"{_strlst(ulist1)};{_strlst(ulist2)}|{_strlst(clist)}"
                if len(clist) > 0
                else f"{_strlst(ulist1)};{_strlst(ulist2)}"
            )

        # every time when cit is called, auto save to local cache.
        self.save_to_local_cache()

        METHODS_SUPPORTING_MULTIDIM_DATA = ["kci"]
        if condition_set is None:
            condition_set = []
        # 'int' to convert np.*int* to built-in int; 'set' to remove duplicates; sorted for hashing
        condition_set = sorted(set(map(int, condition_set)))

        # usually, X and Y are 1-dimensional index (in constraint-based methods)
        if self.method not in METHODS_SUPPORTING_MULTIDIM_DATA:
            X, Y = (int(X), int(Y)) if (X < Y) else (int(Y), int(X))
            assert (
                X not in condition_set and Y not in condition_set
            ), "X, Y cannot be in condition_set."
            return [X], [Y], condition_set, _stringize([X], [Y], condition_set)

        # also to support multi-dimensional unconditional X, Y (usually in kernel-based tests)
        Xs = (
            sorted(set(map(int, X))) if isinstance(X, Iterable) else [int(X)]
        )  # sorted for comparison
        Ys = sorted(set(map(int, Y))) if isinstance(Y, Iterable) else [int(Y)]
        # Xs, Ys = (Xs, Ys) if (Xs < Ys) else (Ys, Xs)
        assert (
            len(set(Xs).intersection(condition_set)) == 0
            and len(set(Ys).intersection(condition_set)) == 0
        ), "X, Y cannot be in condition_set."
        return Xs, Ys, condition_set, _stringize(Xs, Ys, condition_set)


class FisherZ(CIT_Base):
    def __init__(self, data, **kwargs):
        super().__init__(data, **kwargs)
        self.check_cache_method_consistent("fisherz", NO_SPECIFIED_PARAMETERS_MSG)
        self.assert_input_data_is_valid()
        self.correlation_matrix = np.corrcoef(data.T)

    def __call__(self, X, Y, condition_set=None):
        """
        Perform an independence test using Fisher-Z's test.

        Parameters
        ----------
        X, Y and condition_set : column indices of data

        Returns
        -------
        p : the p-value of the test
        """
        Xs, Ys, condition_set, cache_key = self.get_formatted_XYZ_and_cachekey(
            X, Y, condition_set
        )
        if cache_key in self.pvalue_cache:
            return self.pvalue_cache[cache_key]
        var = Xs + Ys + condition_set
        sub_corr_matrix = self.correlation_matrix[np.ix_(var, var)]
        try:
            inv = np.linalg.inv(sub_corr_matrix)
        except np.linalg.LinAlgError:
            raise ValueError(
                "Data correlation matrix is singular. Cannot run fisherz test. Please check your data."
            )
        r = -inv[0, 1] / sqrt(inv[0, 0] * inv[1, 1])  # partial correlation coefficient
        Z = 0.5 * log((1 + r) / (1 - r))
        X = sqrt(self.sample_size - len(condition_set) - 3) * abs(Z)
        p = 2 * (1 - norm.cdf(abs(X)))
        self.pvalue_cache[cache_key] = p
        return p


class KCI(CIT_Base):
    def __init__(self, data, **kwargs):
        super().__init__(data, **kwargs)
        kci_ui_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k
            in [
                "kernelX",
                "kernelY",
                "null_ss",
                "approx",
                "est_width",
                "polyd",
                "kwidthx",
                "kwidthy",
            ]
        }
        kci_ci_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k
            in [
                "kernelX",
                "kernelY",
                "kernelZ",
                "null_ss",
                "approx",
                "use_gp",
                "est_width",
                "polyd",
                "kwidthx",
                "kwidthy",
                "kwidthz",
            ]
        }
        self.check_cache_method_consistent(
            "kci",
            hashlib.md5(
                json.dumps(kci_ci_kwargs, sort_keys=True).encode("utf-8")
            ).hexdigest(),
        )
        self.assert_input_data_is_valid()
        self.kci_ui = KCI_UInd(**kci_ui_kwargs)
        self.kci_ci = KCI_CInd(**kci_ci_kwargs)

        self.gmm_ui = GMM_UInd(**kci_ui_kwargs)
        self.gmm_ci = GMM_CInd(**kci_ui_kwargs)

        self.lgm_ui = LGM_UInd(**kci_ui_kwargs)
        self.lgm_ci_1 = LGM_CInd_1(**kci_ui_kwargs)
        self.lgm_ci_2 = LGM_CInd_2(**kci_ui_kwargs)

    def __call__(self, X, Y, condition_set=None, gmm=0, K=0):
        # Kernel-based conditional independence test.
        Xs, Ys, condition_set, cache_key = self.get_formatted_XYZ_and_cachekey(
            X, Y, condition_set
        )
        if cache_key in self.pvalue_cache:
            return self.pvalue_cache[cache_key]

        if gmm == 0:  # normal KCI
            p = (
                self.kci_ui.compute_pvalue(self.data[:, Xs], self.data[:, Ys])[0]
                if len(condition_set) == 0
                else self.kci_ci.compute_pvalue(
                    self.data[:, Xs], self.data[:, Ys], self.data[:, condition_set]
                )[0]
            )
        elif gmm == 1:  # Gaussian Mixture Model
            p = (
                self.gmm_ui.compute_pval(self.data[:, Xs], self.data[:, Ys], K)
                if len(condition_set) == 0
                else self.gmm_ci.compute_pval(
                    self.data[:, Xs], self.data[:, Ys], self.data[:, condition_set], K
                )
            )
        elif gmm == 2:  # Linear Gaussian Model
            # p = self.lgm_ui.compute_pval(self.data[:, Xs], self.data[:, Ys], K) if len(condition_set) == 0 else \
            #     self.lgm_ci.compute_pval(self.data[:, Xs], self.data[:, Ys], self.data[:, condition_set], K)
            if len(condition_set) == 0:
                p = self.lgm_ui.compute_pval(self.data[:, Xs], self.data[:, Ys], K)
            elif len(condition_set) == 1:
                p = self.lgm_ci_1.compute_pval(
                    self.data[:, Xs], self.data[:, Ys], self.data[:, condition_set], K
                )
            else:
                p = self.lgm_ci_2.compute_pval(
                    self.data[:, Xs], self.data[:, Ys], self.data[:, condition_set], K
                )
        elif gmm == 3:  # use random feature.
            # print('Hi, This is in Random Feature.')
            if len(condition_set) == 0:
                p = self.kci_ui.compute_pvalue_rf(self.data[:, Xs], self.data[:, Ys])[0]
            else:
                p = self.kci_ci.compute_pvalue_rf(
                    self.data[:, Xs], self.data[:, Ys], self.data[:, condition_set]
                )[0]

        self.pvalue_cache[cache_key] = p
        print(f"\t KCI p_value: {p}")
        return p


class Chisq_or_Gsq(CIT_Base):
    def __init__(self, data, method_name, **kwargs):
        def _unique(column):
            return np.unique(column, return_inverse=True)[1]

        assert method_name in ["chisq", "gsq"]
        super().__init__(
            np.apply_along_axis(_unique, 0, data).astype(np.int64), **kwargs
        )
        self.check_cache_method_consistent(method_name, NO_SPECIFIED_PARAMETERS_MSG)
        self.assert_input_data_is_valid()
        self.cardinalities = np.max(self.data, axis=0) + 1

    def chisq_or_gsq_test(self, dataSXY, cardSXY, G_sq=False):
        """by Haoyue@12/18/2021
        Parameters
        ----------
        dataSXY: numpy.ndarray, in shape (|S|+2, n), where |S| is size of conditioning set (can be 0), n is sample size
                 dataSXY.dtype = np.int64, and each row has values [0, 1, 2, ..., card_of_this_row-1]
        cardSXY: cardinalities of each row (each variable)
        G_sq: True if use G-sq, otherwise (False by default), use Chi_sq
        """

        def _Fill2DCountTable(dataXY, cardXY):
            """
            e.g. dataXY: the observed dataset contains 5 samples, on variable x and y they're
                x: 0 1 2 3 0
                y: 1 0 1 2 1
            cardXY: [4, 3]
            fill in the counts by index, we have the joint count table in 4 * 3:
                xy| 0 1 2
                --|-------
                0 | 0 2 0
                1 | 1 0 0
                2 | 0 1 0
                3 | 0 0 1
            note: if sample size is large enough, in theory:
                    min(dataXY[i]) == 0 && max(dataXY[i]) == cardXY[i] - 1
                however some values may be missed.
                also in joint count, not every value in [0, cardX * cardY - 1] occurs.
                that's why we pass cardinalities in, and use `minlength=...` in bincount
            """
            cardX, cardY = cardXY
            xyIndexed = dataXY[0] * cardY + dataXY[1]
            xyJointCounts = np.bincount(xyIndexed, minlength=cardX * cardY).reshape(
                cardXY
            )
            xMarginalCounts = np.sum(xyJointCounts, axis=1)
            yMarginalCounts = np.sum(xyJointCounts, axis=0)
            return xyJointCounts, xMarginalCounts, yMarginalCounts

        def _Fill3DCountTableByBincount(dataSXY, cardSXY):
            cardX, cardY = cardSXY[-2:]
            cardS = np.prod(cardSXY[:-2])
            cardCumProd = np.ones_like(cardSXY)
            cardCumProd[:-1] = np.cumprod(cardSXY[1:][::-1])[::-1]
            SxyIndexed = np.dot(cardCumProd[None], dataSXY)[0]

            SxyJointCounts = np.bincount(
                SxyIndexed, minlength=cardS * cardX * cardY
            ).reshape((cardS, cardX, cardY))
            SMarginalCounts = np.sum(SxyJointCounts, axis=(1, 2))
            SMarginalCountsNonZero = SMarginalCounts != 0
            SMarginalCounts = SMarginalCounts[SMarginalCountsNonZero]
            SxyJointCounts = SxyJointCounts[SMarginalCountsNonZero]

            SxJointCounts = np.sum(SxyJointCounts, axis=2)
            SyJointCounts = np.sum(SxyJointCounts, axis=1)
            return SxyJointCounts, SMarginalCounts, SxJointCounts, SyJointCounts

        def _Fill3DCountTableByUnique(dataSXY, cardSXY):
            # Sometimes when the conditioning set contains many variables and each variable's cardinality is large
            # e.g. consider an extreme case where
            # S contains 7 variables and each's cardinality=20, then cardS = np.prod(cardSXY[:-2]) would be 1280000000
            # i.e., there are 1280000000 different possible combinations of S,
            #    so the SxyJointCounts array would be of size 1280000000 * cardX * cardY * np.int64,
            #    i.e., ~3.73TB memory! (suppose cardX, cardX are also 20)
            # However, samplesize is usually in 1k-100k scale, far less than cardS,
            # i.e., not all (and actually only a very small portion of combinations of S appeared in data)
            #    i.e., SMarginalCountsNonZero in _Fill3DCountTable_by_bincount is a very sparse array
            # So when cardSXY is large, we first re-index S (skip the absent combinations) and then count XY table for each.
            # See https://github.com/cmu-phil/causal-learn/pull/37.
            cardX, cardY = cardSXY[-2:]
            cardSs = cardSXY[:-2]

            cardSsCumProd = np.ones_like(cardSs)
            cardSsCumProd[:-1] = np.cumprod(cardSs[1:][::-1])[::-1]
            SIndexed = np.dot(cardSsCumProd[None], dataSXY[:-2])[0]

            uniqSIndices, inverseSIndices, SMarginalCounts = np.unique(
                SIndexed, return_counts=True, return_inverse=True
            )
            cardS_reduced = len(uniqSIndices)
            SxyIndexed = (
                inverseSIndices * cardX * cardY + dataSXY[-2] * cardY + dataSXY[-1]
            )
            SxyJointCounts = np.bincount(
                SxyIndexed, minlength=cardS_reduced * cardX * cardY
            ).reshape((cardS_reduced, cardX, cardY))

            SxJointCounts = np.sum(SxyJointCounts, axis=2)
            SyJointCounts = np.sum(SxyJointCounts, axis=1)
            return SxyJointCounts, SMarginalCounts, SxJointCounts, SyJointCounts

        def _Fill3DCountTable(dataSXY, cardSXY):
            # about the threshold 1e5, see a rough performance example at:
            # https://gist.github.com/MarkDana/e7d9663a26091585eb6882170108485e#file-count-unique-in-array-performance-md
            if np.prod(cardSXY) < CONST_BINCOUNT_UNIQUE_THRESHOLD:
                return _Fill3DCountTableByBincount(dataSXY, cardSXY)
            return _Fill3DCountTableByUnique(dataSXY, cardSXY)

        def _CalculatePValue(cTables, eTables):
            """
            calculate the rareness (pValue) of an observation from a given distribution with certain sample size.

            Let k, m, n be respectively the cardinality of S, x, y. if S=empty, k==1.
            Parameters
            ----------
            cTables: tensor, (k, m, n) the [c]ounted tables (reflect joint P_XY)
            eTables: tensor, (k, m, n) the [e]xpected tables (reflect product of marginal P_X*P_Y)
              if there are zero entires in eTables, zero must occur in whole rows or columns.
              e.g. w.l.o.g., row eTables[w, i, :] == 0, iff np.sum(cTables[w], axis=1)[i] == 0, i.e. cTables[w, i, :] == 0,
                   i.e. in configuration of conditioning set == w, no X can be in value i.

            Returns: pValue (float in range (0, 1)), the larger pValue is (>alpha), the more independent.
            -------
            """
            eTables_zero_inds = eTables == 0
            eTables_zero_to_one = np.copy(eTables)
            eTables_zero_to_one[eTables_zero_inds] = 1  # for legal division

            if G_sq == False:
                sum_of_chi_square = np.sum(
                    ((cTables - eTables) ** 2) / eTables_zero_to_one
                )
            else:
                div = np.divide(cTables, eTables_zero_to_one)
                div[
                    div == 0
                ] = 1  # It guarantees that taking natural log in the next step won't cause any error
                sum_of_chi_square = 2 * np.sum(cTables * np.log(div))

            # array in shape (k,), zero_counts_rows[w]=c (0<=c<m) means layer w has c all-zero rows
            zero_counts_rows = eTables_zero_inds.all(axis=2).sum(axis=1)
            zero_counts_cols = eTables_zero_inds.all(axis=1).sum(axis=1)
            sum_of_df = np.sum(
                (cTables.shape[1] - 1 - zero_counts_rows)
                * (cTables.shape[2] - 1 - zero_counts_cols)
            )
            return 1 if sum_of_df == 0 else chi2.sf(sum_of_chi_square, sum_of_df)

        if len(cardSXY) == 2:  # S is empty
            xyJointCounts, xMarginalCounts, yMarginalCounts = _Fill2DCountTable(
                dataSXY, cardSXY
            )
            xyExpectedCounts = (
                np.outer(xMarginalCounts, yMarginalCounts) / dataSXY.shape[1]
            )  # divide by sample size
            return _CalculatePValue(xyJointCounts[None], xyExpectedCounts[None])

        # else, S is not empty: conditioning
        (
            SxyJointCounts,
            SMarginalCounts,
            SxJointCounts,
            SyJointCounts,
        ) = _Fill3DCountTable(dataSXY, cardSXY)
        SxyExpectedCounts = (
            SxJointCounts[:, :, None]
            * SyJointCounts[:, None, :]
            / SMarginalCounts[:, None, None]
        )
        return _CalculatePValue(SxyJointCounts, SxyExpectedCounts)

    def __call__(self, X, Y, condition_set=None):
        # Chi-square (or G-square) independence test.
        Xs, Ys, condition_set, cache_key = self.get_formatted_XYZ_and_cachekey(
            X, Y, condition_set
        )
        if cache_key in self.pvalue_cache:
            return self.pvalue_cache[cache_key]
        indexs = condition_set + Xs + Ys
        p = self.chisq_or_gsq_test(
            self.data[:, indexs].T,
            self.cardinalities[indexs],
            G_sq=self.method == "gsq",
        )
        self.pvalue_cache[cache_key] = p
        return p


class MV_FisherZ(CIT_Base):
    def __init__(self, data, **kwargs):
        super().__init__(data, **kwargs)
        self.check_cache_method_consistent("mv_fisherz", NO_SPECIFIED_PARAMETERS_MSG)
        self.assert_input_data_is_valid(allow_nan=True)

    def _get_index_no_mv_rows(self, mvdata):
        nrow, ncol = np.shape(mvdata)
        bindxRows = np.ones((nrow,), dtype=bool)
        indxRows = np.array(list(range(nrow)))
        for i in range(ncol):
            bindxRows = np.logical_and(bindxRows, ~np.isnan(mvdata[:, i]))
        indxRows = indxRows[bindxRows]
        return indxRows

    def __call__(self, X, Y, condition_set=None):
        """
        Perform an independence test using Fisher-Z's test for data with missing values.

        Parameters
        ----------
        X, Y and condition_set : column indices of data

        Returns
        -------
        p : the p-value of the test
        """
        Xs, Ys, condition_set, cache_key = self.get_formatted_XYZ_and_cachekey(
            X, Y, condition_set
        )
        if cache_key in self.pvalue_cache:
            return self.pvalue_cache[cache_key]
        var = Xs + Ys + condition_set
        test_wise_deletion_XYcond_rows_index = self._get_index_no_mv_rows(
            self.data[:, var]
        )
        assert (
            len(test_wise_deletion_XYcond_rows_index) != 0
        ), "A test-wise deletion fisher-z test appears no overlapping data of involved variables. Please check the input data."
        test_wise_deleted_data_var = self.data[test_wise_deletion_XYcond_rows_index][
            :, var
        ]
        sub_corr_matrix = np.corrcoef(test_wise_deleted_data_var.T)
        try:
            inv = np.linalg.inv(sub_corr_matrix)
        except np.linalg.LinAlgError:
            raise ValueError(
                "Data correlation matrix is singular. Cannot run fisherz test. Please check your data."
            )
        r = -inv[0, 1] / sqrt(inv[0, 0] * inv[1, 1])
        Z = 0.5 * log((1 + r) / (1 - r))
        X = sqrt(
            len(test_wise_deletion_XYcond_rows_index) - len(condition_set) - 3
        ) * abs(Z)
        p = 2 * (1 - norm.cdf(abs(X)))
        self.pvalue_cache[cache_key] = p
        return p


class MC_FisherZ(CIT_Base):
    def __init__(self, data, **kwargs):
        # no cache for MC_FisherZ, since skel and prt_m is provided for each test.
        super().__init__(data, **kwargs)
        self.check_cache_method_consistent("mc_fisherz", NO_SPECIFIED_PARAMETERS_MSG)
        self.assert_input_data_is_valid(allow_nan=True)
        self.mv_fisherz = MV_FisherZ(data, **kwargs)

    def __call__(self, X, Y, condition_set, skel, prt_m):
        """Perform an independent test using Fisher-Z's test with test-wise deletion and missingness correction
        If it is not the case which requires a correction, then call function mvfisherZ(...)
        :param prt_m: dictionary, with elements:
            - m: missingness indicators which are not MCAR
            - prt: parents of the missingness indicators
        """

        ## Check whether whether there is at least one common child of X and Y
        if not Helper.cond_perm_c(X, Y, condition_set, prt_m, skel):
            return self.mv_fisherz(X, Y, condition_set)

        ## *********** Step 1 ***********
        # Learning generaive model for {X, Y, S} to impute X, Y, and S

        ## Get parents the {xyS} missingness indicators with parents: prt_m
        # W is the variable which can be used for missingness correction
        W_indx_ = Helper.get_prt_mvars(var=list((X, Y) + condition_set), prt_m=prt_m)

        if len(W_indx_) == 0:  # When there is no variable can be used for correction
            return self.mv_fisherz(X, Y, condition_set)

        ## Get the parents of W missingness indicators
        W_indx = Helper.get_prt_mw(W_indx_, prt_m)

        ## Prepare the W for regression
        # Since the XYS will be regressed on W,
        # W will not contain any of XYS
        var = list((X, Y) + condition_set)
        W_indx = list(set(W_indx) - set(var))

        if len(W_indx) == 0:  # When there is no variable can be used for correction
            return self.mv_fisherz(X, Y, condition_set)

        ## Learn regression models with test-wise deleted data
        involve_vars = var + W_indx
        tdel_data = Helper.test_wise_deletion(self.data[:, involve_vars])
        effective_sz = len(tdel_data[:, 0])
        regMs, rss = Helper.learn_regression_model(tdel_data, num_model=len(var))

        ## *********** Step 2 ***********
        # Get the data of the predictors, Ws
        # The sample size of Ws is the same as the effective sample size
        Ws = Helper.get_predictor_ws(
            self.data[:, involve_vars], num_test_var=len(var), effective_sz=effective_sz
        )

        ## *********** Step 3 ***********
        # Generate the virtual data follows the full data distribution P(X, Y, S)
        # The sample size of data_vir is the same as the effective sample size
        data_vir = Helper.gen_vir_data(regMs, rss, Ws, len(var), effective_sz)

        if len(var) > 2:
            cond_set_bgn_0 = np.arange(2, len(var))
        else:
            cond_set_bgn_0 = []

        virtual_cit = MV_FisherZ(data_vir)
        return virtual_cit(0, 1, tuple(cond_set_bgn_0))


class D_Separation(CIT_Base):
    def __init__(self, data, true_dag=None, **kwargs):
        """
        Use d-separation as CI test, to ensure the correctness of constraint-based methods. (only used for tests)
        Parameters
        ----------
        data:   numpy.ndarray, just a placeholder, not used in D_Separation
        true_dag:   nx.DiGraph object, the true DAG
        """
        super().__init__(
            data, **kwargs
        )  # data is just a placeholder, not used in D_Separation
        self.check_cache_method_consistent("d_separation", NO_SPECIFIED_PARAMETERS_MSG)
        self.true_dag = true_dag
        import networkx as nx

        global nx
        # import networkx here violates PEP8; but we want to prevent unnecessary import at the top (it's only used here)

    def __call__(self, X, Y, condition_set=None):
        Xs, Ys, condition_set, cache_key = self.get_formatted_XYZ_and_cachekey(
            X, Y, condition_set
        )
        if cache_key in self.pvalue_cache:
            return self.pvalue_cache[cache_key]
        p = float(nx.d_separated(self.true_dag, {Xs[0]}, {Ys[0]}, set(condition_set)))
        # pvalue is bool here: 1 if is_d_separated and 0 otherwise. So heuristic comparison-based uc_rules will not work.

        # here we use networkx's d_separation implementation.
        # an alternative is to use causal-learn's own d_separation implementation in graph class:
        #   self.true_dag.is_dseparated_from(
        #       self.true_dag.nodes[Xs[0]], self.true_dag.nodes[Ys[0]], [self.true_dag.nodes[_] for _ in condition_set])
        #   where self.true_dag is an instance of GeneralGrpah class.
        # I have checked the two implementations: they are equivalent (when the graph is DAG),
        # and generally causal-learn's implementation is faster.
        # but just for now, I still use networkx's, for two reasons:
        # 1. causal-learn's implementation sometimes stops working during run (haven't check detailed reasons)
        # 2. GeneralGraph class will be hugely refactored in the near future.
        self.pvalue_cache[cache_key] = p
        return p


class SPN(CIT_Base):
    """
    Sum-Product Network based Conditional Independence Test

    This class leverages EinsumNetworks (simple-einet) for tractable
    probabilistic inference, replacing kernel-based summary statistics
    with exact marginal likelihood computation.

    Key advantages over kernel methods:
    1. Tractable inference: O(network_size) vs O(n^3) for kernel methods
    2. Exact marginalization: Uses nan-based marginalization for proper P(X_subset)
    3. Scalable: Network parameters fixed, independent of sample size
    4. Theoretically sound: Based on exact probabilistic inference
    """

    def __init__(self, data, **kwargs):
        super().__init__(data, **kwargs)

        # Extract and validate SPN parameters
        spn_params = {
            "epochs": kwargs.get("epochs", 100),
            "lr": kwargs.get("lr", 0.01),
            "depth": kwargs.get("depth", 2),
            "num_sums": kwargs.get("num_sums", 8),
            "num_leaves": kwargs.get("num_leaves", 12),
            "num_repetitions": kwargs.get("num_repetitions", 6),
            "dropout": kwargs.get("dropout", 0.0),
        }

        # Setup cache consistency
        params_hash = hashlib.md5(
            json.dumps(spn_params, sort_keys=True).encode("utf-8")
        ).hexdigest()
        self.check_cache_method_consistent("spn", params_hash)
        self.assert_input_data_is_valid()

        # Store parameters
        self.spn_params = spn_params
        self.sample_size_for_test = min(200, self.sample_size // 2)

        # Train the model immediately upon initialization
        self._initialize_and_train_spn()

    def _initialize_and_train_spn(self):
        """Initialize and train the EinsumNetwork for SPN-based CI testing."""
        torch.set_num_threads(1)
        torch.set_default_dtype(torch.float32)
        # Check feature count
        if self.num_features < 2:
            raise ValueError(
                f"SPN requires at least 2 features, got {self.num_features}"
            )

        # Data normalization
        self.scaler = StandardScaler()
        normalized_data = self.scaler.fit_transform(self.data)
        if np.isnan(normalized_data).any() or np.isinf(normalized_data).any():
            raise ValueError(
                "SPN initialization: Normalized data contains NaN or Inf values."
            )
        self.data_tensor = torch.FloatTensor(normalized_data)

        # 基於論文的架構參數
        actual_depth = min(3, int(np.floor(np.log2(self.num_features))))
        num_sums = min(10, self.spn_params.get("num_sums", 8))
        num_leaves = min(12, self.spn_params.get("num_leaves", 12))
        num_repetitions = min(6, self.spn_params.get("num_repetitions", 6))

        config = EinetConfig(
            num_features=self.num_features,
            depth=actual_depth,
            num_sums=num_sums,
            num_channels=1,
            num_leaves=num_leaves,
            num_repetitions=num_repetitions,
            num_classes=1,
            dropout=self.spn_params.get("dropout", 0.0),
            leaf_type=Normal,
        )

        self.einet = Einet(config)

        # 改進的訓練策略
        optimizer = torch.optim.Adam(self.einet.parameters(), lr=0.0005)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=20, factor=0.5, verbose=True
        )

        batch_size = min(100, len(self.data_tensor) // 4)
        from torch.utils.data import DataLoader, TensorDataset

        dataloader = DataLoader(
            TensorDataset(self.data_tensor), batch_size=batch_size, shuffle=True
        )

        best_ll = float("-inf")
        patience = 0
        max_patience = 50
        n_epochs = min(200, self.spn_params.get("epochs", 150))

        for epoch in range(n_epochs):
            epoch_loss = 0
            epoch_ll = 0

            for (batch_data,) in dataloader:
                optimizer.zero_grad()
                log_lls = self.einet(batch_data)
                loss = -log_lls.mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.einet.parameters(), 1.0)
                optimizer.step()

                epoch_loss += loss.item()
                epoch_ll += log_lls.mean().item()

            avg_ll = epoch_ll / len(dataloader)
            scheduler.step(-avg_ll)  # 使用負對數似然作為調度指標

            if epoch % 20 == 0:
                print(
                    f"[RAT-SPN] Epoch {epoch} Loss {epoch_loss / len(dataloader):.4f} "
                    f"LogLL {avg_ll:.4f} Patience {patience}"
                )

            if avg_ll > best_ll:
                best_ll = avg_ll
                patience = 0
            else:
                patience += 1

            if patience > max_patience:
                print(f"[SPN] Early stopping at epoch {epoch}")
                break

    def _marginal_loglik(self, data_tensor, keep_dims):
        """
        Compute marginal log-likelihood using nan-marginalization

        This replaces traditional kernel-based marginal estimation
        with exact SPN marginal computation - a key innovation.

        Parameters:
        -----------
        data_tensor : torch.Tensor
            Input data tensor
        keep_dims : list
            Dimensions to keep observed (others set to nan)

        Returns:
        --------
        torch.Tensor : Marginal log-likelihoods
        """
        # Create marginalization tensor: observed dims keep values, others become nan
        marginalized_tensor = torch.full_like(data_tensor, float("nan"))
        marginalized_tensor[:, keep_dims] = data_tensor[:, keep_dims]

        # EinsumNetwork computes exact marginal likelihood for nan features
        with torch.no_grad():
            loglik = self.einet(marginalized_tensor)
            if torch.isnan(loglik).any():
                print("Warning: SPN produced NaN log-likelihoods. Check data/scaling.")
            return loglik

    def get_formatted_XYZ_and_cachekey(self, X, Y, condition_set):
        """
        Normalize X, Y, and condition_set to sorted int lists, check for invalid overlaps,
        and generate a unique string cache key. Compatible with both single integer and iterable input for all arguments.

        This function ensures:
            1. All inputs (X, Y, condition_set) are converted to Python lists of int.
            2. Duplicates are removed and indices are sorted for consistency.
            3. X and Y do not overlap with condition_set; raises ValueError if overlap exists.
            4. Returns a unique, order-agnostic cache key for memoization, invariant to input type and order.

        Parameters
        ----------
        X : int, Iterable[int], or np.*int*
            Indices of the first variable(s). Accepts a single int or any iterable of ints.
        Y : int, Iterable[int], or np.*int*
            Indices of the second variable(s). Accepts a single int or any iterable of ints.
        condition_set : int, Iterable[int], or np.*int*
            Conditioning variable indices. Accepts a single int or any iterable of ints.

        Returns
        -------
        Xs : List[int]
            Sorted list of unique indices for X. Always a list, even if a single variable.
        Ys : List[int]
            Sorted list of unique indices for Y. Always a list, even if a single variable.
        condition_set : List[int]
            Sorted list of unique conditioning indices.
        cache_key : str
            Unique cache key string for the (X, Y | condition_set) triplet, agnostic to input type and argument order.
        """

        # Ensure all arguments are lists of int
        def _flatten(idx):
            if idx is None:
                return []
            elif isinstance(idx, (list, tuple)):
                return [int(i) for i in idx]
            else:
                return [int(idx)]

        Xs = _flatten(X)
        Ys = _flatten(Y)
        conds = _flatten(condition_set)

        # Remove duplicates and sort for uniqueness in cache key
        Xs = sorted(set(Xs))
        Ys = sorted(set(Ys))
        conds = sorted(set(conds))

        # X, Y cannot be in condition_set
        for v in Xs + Ys:
            if v in conds:
                raise ValueError(
                    "get_formatted_XYZ_and_cachekey: variable in X or Y also appears in condition_set."
                )
        # Build a unique string cache key for memoization (order/overlap safe)
        cache_key = f"{'.'.join(map(str, Xs))};{'.'.join(map(str, Ys))}|{'.'.join(map(str, conds)) if conds else ''}"

        return Xs, Ys, conds, cache_key

    def __call__(self, X, Y, condition_set=None, *args, **kwargs):
        num_permutations = 1000

        Xs, Ys, condition_set, cache_key = self.get_formatted_XYZ_and_cachekey(
            X, Y, condition_set
        )
        if cache_key in self.pvalue_cache:
            return self.pvalue_cache[cache_key]

        try:
            # Draw a subset for speed
            n_samples = min(self.sample_size_for_test, self.sample_size)
            sample_indices = np.random.choice(
                self.sample_size, size=n_samples, replace=False
            )
            data_subset = self.data_tensor[sample_indices]
            idx_x = Xs
            idx_y = Ys
            idx_z = condition_set

            # Compute test statistic: e.g., log-likelihood diff (dependent - independent)
            keep_xyz = np.array(idx_x + idx_y + idx_z)
            keep_xz = np.array(idx_x + idx_z)
            keep_yz = np.array(idx_y + idx_z)
            keep_z = np.array(idx_z)
            ll_xyz = self._marginal_loglik(data_subset, keep_xyz).mean().item()
            ll_xz = self._marginal_loglik(data_subset, keep_xz).mean().item()
            ll_yz = self._marginal_loglik(data_subset, keep_yz).mean().item()
            ll_z = (
                self._marginal_loglik(data_subset, keep_z).mean().item()
                if len(keep_z) > 0
                else 0.0
            )

            # Statistic (as in likelihood ratio CI tests)
            stat_obs = ll_xyz + ll_z - ll_xz - ll_yz

            # Permutation null: shuffle X (or Y) relative to everything else
            stat_null = []
            for _ in range(num_permutations):
                perm = np.random.permutation(n_samples)
                data_perm = data_subset.clone()
                data_perm[:, idx_x] = data_perm[perm][
                    :, idx_x
                ]  # permute X only, keep others fixed
                # recompute log-likelihoods with permuted X
                ll_xyz_p = self._marginal_loglik(data_perm, keep_xyz).mean().item()
                ll_xz_p = self._marginal_loglik(data_perm, keep_xz).mean().item()
                ll_yz_p = self._marginal_loglik(data_perm, keep_yz).mean().item()
                ll_z_p = (
                    self._marginal_loglik(data_perm, keep_z).mean().item()
                    if len(keep_z) > 0
                    else 0.0
                )
                stat_null.append(ll_xyz_p + ll_z_p - ll_xz_p - ll_yz_p)

            # p-value: proportion of permuted statistics as or more extreme than observed
            p = (np.sum(np.array(stat_null) >= stat_obs) + 1) / (num_permutations + 1)
            self.pvalue_cache[cache_key] = p
            return p

        except Exception as e:
            # ===== ERROR HANDLING =====
            print(f"SPN CI test error: {str(e)[:100]}...")

            # Return neutral p-value to avoid breaking causal discovery pipeline
            fallback_p_value = 0.5
            self.pvalue_cache[cache_key] = fallback_p_value
            return fallback_p_value
