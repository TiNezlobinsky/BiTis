import warnings
import numpy as np
from scipy import stats
from sklearn.covariance import EmpiricalCovariance, MinCovDet
from .polar_plots import PolarPlots


class DistributionEllipse:
    def __init__(self):
        self.type_name = ''
        self.width = np.nan
        self.height = np.nan
        self.orientation = np.nan

    @property
    def anisotropy(self):
        if self.width is np.nan or self.height is np.nan:
            return np.nan

        return self.width / self.height

    @property
    def full_radius(self):
        r, _ = PolarPlots.rotated_ellipse(0.5 * self.width, 0.5 * self.height, 
                                          self.orientation, n=100)
        return r

    @property
    def full_theta(self):
        _, theta = PolarPlots.rotated_ellipse(0.5 * self.width,
                                              0.5 * self.height,
                                              self.orientation, n=100)
        return theta


class DistributionEllipseBuilder:
    def __init__(self):
        self.dist_ellipse = DistributionEllipse()
        self.cov_estimator = MinCovDet()

    def build(self, r, theta, n_std=2., ellipse_type='error', min_points=5):
        """Build a distribution ellipse from objects properties.

        Parameters
        ----------
        r : np.ndarray
            Radial coordinates.
        theta : np.ndarray
            Angular coordinates.
        n_std : float, optional
            Number of standard deviations.
        ellipse_type : str, optional
            Type of the ellipse ('error' or 'confidence').
        min_points : int, optional
            Minimum number of points to build the ellipse.
        """
        if len(r) < 2 * min_points:
            warnings.warn('Not enough points to build the distribution ellipse.')

            self.dist_ellipse = DistributionEllipse()
            return self.dist_ellipse

        x, y = PolarPlots.polar_to_cartesian(r, theta)
        X = np.array([x, y]).T
        cov = self.cov_estimator.fit(X).covariance_
        eig_vals, eig_vec = self.sorted_eigs(cov)

        if ellipse_type.lower() == 'error':
            width, height = self.error_ellipse(eig_vals, n_std)

        if ellipse_type.lower() == 'confidence':
            width, height = self.confidence_ellipse(eig_vals, n_std)

        theta = np.arctan2(* eig_vec[::-1, 0])
        # theta = np.arctan2(* eig_vec[:, 0])

        self.dist_ellipse = DistributionEllipse()
        self.dist_ellipse.type_name = ellipse_type
        self.dist_ellipse.width = width
        self.dist_ellipse.height = height
        self.dist_ellipse.orientation = theta
        return self.dist_ellipse

    def error_ellipse(self, eig_vals, n_std):
        """Calculate the width and height of an error ellipse.

        Parameters
        ----------
        eig_vals : np.ndarray
            Eigenvalues of the covariance matrix.
        n_std : float
            Number of standard deviations.

        Returns
        -------
        tuple
            Width and height of the error ellipse.
        """
        width, height = 2 * n_std * np.sqrt(eig_vals)
        return width, height

    def confidence_ellipse(self, eig_vals, n_std):
        """Calculate the width and height of a confidence ellipse.

        Parameters
        ----------
        eig_vals : np.ndarray
            Eigenvalues of the covariance matrix.
        n_std : float
            Number of standard deviations.

        Returns
        -------
        tuple
            Width and height of the confidence ellipse.
        """
        # Confidence level
        q = 2 * stats.norm.cdf(n_std) - 1
        r2 = stats.chi2.ppf(q, 2)
        width, height = 2 * np.sqrt(eig_vals * r2)
        return width, height

    def covariance(self, x, y):
        """Calculate the covariance matrix between two variables.

        Parameters
        ----------
        x : np.ndarray
            Input data for x-coordinate.
        y : np.ndarray
            Input data for y-coordinate.

        Returns
        -------
        np.ndarray
            Covariance matrix between x and y.
        """
        return np.cov(np.vstack([x, y]), rowvar=True)

    def sorted_eigs(self, cov):
        """Sort eigenvalues and eigenvectors in descending order.

        Parameters
        ----------
        cov : np.ndarray
            Covariance matrix.

        Returns
        -------
        tuple
            Sorted eigenvalues and eigenvectors.
        """
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:, order]