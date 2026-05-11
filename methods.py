import h3
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.special import sph_harm_y
from sklearn.cluster import DBSCAN
from sklearn.linear_model import LogisticRegression
import pygplates as pygp
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from shapely.geometry import Polygon
import cartopy.crs as crs
from joblib import Parallel, delayed

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

### ----------- data prep ------------------ ###

def make_grid(resolution=3):
    """ set up a global h3 grid at a specified resolution """
    geoJson_hemisphere1 = {'type': 'Polygon', 'coordinates': [[[90,-180],[90,0],[-90,0],[-90,-180]]]}
    geoJson_hemisphere2 = {'type': 'Polygon', 'coordinates': [[[90,0],[90,180],[-90,180],[-90,0]]]}
    hexagons = sorted(list(h3.polyfill(geoJson_hemisphere1, resolution)) + list(h3.polyfill(geoJson_hemisphere2, resolution)))
    centroids = [h3.h3_to_geo(x) for x in hexagons]
    lats = np.array([x[0] for x in centroids])
    lons = np.array([x[1] for x in centroids])
    return hexagons, lats, lons      # return hexagon cell ids + corresponding lats & lons

def add_noise(s, sigma=0.25):
    """ add gaussian noise to observations at specified sigma level """
    np.random.seed(0)
    s_noisy = s + np.random.normal(loc=0.0, scale=sigma, size=s.shape)
    return s_noisy

def make_ternary(s, t1=None, t2=None):
    """ flatten continuous data into ternary set: -1, 0, 1 """
    if not t1: t1 = np.percentile(s, 20)  # use percentile if no value for t1 provided
    if not t2: t2 = np.percentile(s, 80)  # use percentile if no value for t2 provided
    s_ternary = np.where(s < t1, -1, np.where(s > t2, 1, 0))
    return s_ternary

def remove_zeros(s, lats, lons):
    """ remove zero values from dataset """
    mask = (s != 0)
    return s[mask], lats[mask], lons[mask]

def equisphere(n):
    """ generates n equidistributed points on a sphere """
    r = 1.0
    area = 4.0 * np.pi * r**2 / n   # average area per point
    d = np.sqrt(area)               # approximate spacing
    m_lat = round(np.pi / d)        # number of latitude bands
    d_lat = np.pi / m_lat           # latitude step
    d_lon = area / d_lat            # average longitude step
    
    lats, lons = [], []
    for i in range(m_lat):
        v = np.pi * (i + 0.5) / m_lat       
        m_lon = round(2.0 * np.pi * np.sin(v) / d_lon)
        for j in range(m_lon):
            p = (2.0 * np.pi * j) / m_lon 
            lons.append(np.degrees(p))
            lats.append(np.degrees(v) - 90.0)
    return lats, lons

### ----------- analytical functions ------------------ ###

def Y21(lam, lats, lons):
    """ compute expected Y21 values at specified lat/lon positions given a TPW axis location (longitude=lam) """
    lam_rad = np.radians(lam)    # tpw axis longitude
    lats_rad = np.radians(lats)
    lons_rad = np.radians(lons)
    s = np.sin(2*lats_rad) * np.sin(lam_rad - lons_rad)  # expected values
    return s

def Y21_vectorized(lams, lats, lons):
    """ compute expected Y21 values at specified lat/lon positions for an array of TPW axis locations (longitudes=lams) """
    lams_rad = np.radians(lams)[:, None]        # tpw axis locations, shape: (n_lams, 1)
    lats_rad = np.radians(lats)[None, :]        # shape: (1, n_points)
    lons_rad = np.radians(lons)[None, :]        # shape: (1, n_points)
    S = np.sin(2 * lats_rad) * np.sin(lams_rad - lons_rad)  # expected values
    return S                                    # shape (n_lams, n_points)

def fit_logistic_null_fast(y):
    """ quick way to compute null model (just uses the mean value) """
    p = np.clip(np.mean(y), 1e-12, 1-1e-12)
    return np.sum(y * np.log(p) + (1-y) * np.log(1-p))

def fit_logistic(s, y):
    """ fit a logistic regression model (estimate intercept and slope) to binary data y, given predictor values s; return full result object """
    X = sm.add_constant(s)       # add alpha term
    model = sm.Logit(y, X)       # specify model
    result = model.fit(disp=0)   # fit model
    return result                # includes coefficients for alpha and beta and lots of other metrics

def fit_logistic_fast(s, y, return_model=False):
    """ fit a logistic regression model and return either the model or maximized log-likelihood """
    X = np.asarray(s).reshape(-1, 1)  # ensure s is 2D; sklearn expects shape (n_samples, n_features)
    y = np.asarray(y)
    model = LogisticRegression(penalty=None, solver='lbfgs', fit_intercept=True, max_iter=200)  # specify model
    model.fit(X, y)                      # fit model
    if return_model:
        return model
    else:
        p = model.predict_proba(X)[:, 1]     # extract probabilities
        p = np.clip(p, 1e-12, 1 - 1e-12)
        llf = np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)) # log likelihood
        return llf

def invert_lam_logistic(lats, lons, y):
    """ scan TPW longitudes and fit logistic regression model at each orientation to binary data y at locations (lats, lons) """
    results = []
    lambdas = np.linspace(0, 359, 360)        # consider TPW axis at 1-degree intervals
    for lam in lambdas:
        s = Y21(lam, lats, lons)              # compute predictor values 
        result = fit_logistic(s, y)           # fit logistic regression model
        llr = result.llr                      # log likelihood ratio 
        p_llr = result.llr_pvalue             # p-value of log likelihood ratio
        alpha = result.params[0]              # alpha parameter
        beta = result.params[1]               # beta parameter
        conf = result.conf_int()              # confidence bounds
        alpha_conf = conf[0]
        beta_conf = conf[1]
        pr2 = result.prsquared                # pseudo r-squared 
        results.append((lam, llr, p_llr, alpha, beta, alpha_conf, beta_conf, pr2))
    df = pd.DataFrame(results, columns=['lambda', 'llr', 'p_llr', 'alpha', 'beta', 'alpha_conf', 'beta_conf', 'p-rsquared'])
    return df

def eigendecomp(lats, lons):
    """ make design matrix for Y21 harmonic from observation points and retrieve eigenvalues and eigenvectors from it """
    # make design matrix
    rcolats = np.pi/2 - np.radians(lats)
    rlons = np.radians(lons)
    d1 = np.sin(rcolats) * np.cos(rcolats) * np.cos(rlons)
    d2 = np.sin(rcolats) * np.cos(rcolats) * np.sin(rlons)
    D = np.vstack([d1, d2]).T

    # get eigenvalues and eigenvectors
    R = D.T @ D
    evals, evecs = np.linalg.eigh(R)  # evals ascending, evecs as columns
    idx = np.argsort(evals)[::-1]     # reorder to descending
    evals = evals[idx]
    evecs = evecs[:, idx]
    return evals, evecs

### ----------- cluster / permutation functions ------------------ ###

def cluster(lats, lons, eps, min_samples=1):
    """ cluster points (lats/lons) using DBSCAN and a minimum distance specified by eps """
    coords = np.radians(np.column_stack((lats, lons))) 
    db = DBSCAN(eps=np.radians(eps), min_samples=min_samples, metric='haversine').fit(coords)  # eps in degrees
    clusters = db.labels_                                  # array of cluster IDs
    n_clusters = len(np.unique(clusters[clusters != -1]))  # exclude noise
    n_noise = np.sum(clusters == -1)                       # number of noise points
    return clusters, n_clusters, n_noise

def permute(observations, clusters, max_tries=5):
    """ permute observed values according to their occurrence in clusters """
    unique_clusters = np.unique(clusters)                  # get unique cluster labels
    for attempt in range(max_tries):
        randomized_obs = observations.copy()
        for c in unique_clusters:                
            if c == -1: continue                 # skip noise for now
            if np.random.rand() < 0.5:
                randomized_obs[clusters == c] = 1 - randomized_obs[clusters == c]   # flip entire cluster with 50% probability

        noise_mask = (clusters == -1)            # now find noise points
        flips = np.random.rand(noise_mask.sum()) < 0.5    # flip noise points individually with 50% probability
        randomized_obs[noise_mask] = np.where(flips, 1 - randomized_obs[noise_mask], randomized_obs[noise_mask])

        if len(np.unique(randomized_obs)) > 1:  # avoid that result collapses to a single class
            return randomized_obs
    raise RuntimeError('could not generate a valid permutation with 2 classes')
            
def permutation_test(observations, lats, lons, clusters, n=1000):
    """ run n permutations and collect best-fit result from each permutation """
    lambdas = np.linspace(0, 359, 360)          # scan through possible TPW axis locations at 1-degree increments for each permutation
    S = Y21_vectorized(lambdas, lats, lons)     # precomupte S at the given observation locations for each possible value lambda
    best_fits = []
    for i in range(n):                          
        y = (permute(observations, clusters) == 1).astype(int)  # permute
        llnull = fit_null_fast(y)               # fit null
        all_lams = []
        for j, lam in enumerate(lambdas):       # cycle through lambdas and fit logistic each time
            llf = fit_logistic_fast(S[j], y)   
            all_lams.append(2 * (llf - llnull)) # likelihood ratio
        best_fits.append(max(all_lams))         # keep best result (highest likelihood ratio)
    return best_fits

def permutation_test_parallelized(observations, lats, lons, clusters, n=1000, n_jobs=-1):
    """ same as above but set up for parallel execution of permutation runs: run n permutations and collect best-fit result from each permutation """
    lambdas = np.linspace(0, 359, 360)
    S = Y21_vectorized(lambdas, lats, lons)
    
    def one_perm():
        y = (permute(observations, clusters) == 1).astype(int)
        llnull = fit_logistic_null_fast(y)
        all_lams = []
        for j in range(len(lambdas)):
            llf = fit_logistic_fast(S[j], y)
            all_lams.append(2 * (llf - llnull)) # likelihood ratio
        return max(all_lams)

    best_fits = Parallel(n_jobs=n_jobs)(delayed(one_perm)() for _ in range(n))  # allow permutations to be run in parallel
    return best_fits

### -----------spherical harmonics functions ------------------ ###

def sh_design_matrix(lats, lons, lmax):
    """build spherical harmonic design matrix for sparse observations"""    
    # make empty design matrix
    rcolats = np.pi/2 - np.radians(lats)
    rlons = np.radians(lons)
    n_pts  = len(lats)
    n_cols = (lmax + 1) ** 2
    A = np.empty((n_pts, n_cols), dtype=np.float64) # empty design matrix
    l_arr = np.empty(n_cols, dtype=int)             # array of SH degrees

    # populate with basis function values
    col = 0
    for l in range(lmax + 1):
        cache = {}
        for abs_m in range(l + 1):  # compute complex SH for each unique |m| once, reuse for ±m
            cache[abs_m] = sph_harm_y(l, abs_m, rlons, rcolats)
            
        for m in range(-l, l + 1):
            Yc = cache[abs(m)]
            if m == 0:
                A[:, col] = Yc.real
            elif m > 0:
                A[:, col] = Yc.real * np.sqrt(2)
            else:                           # m < 0
                A[:, col] = Yc.imag * np.sqrt(2)
            l_arr[col] = l
            col += 1
    return A, l_arr

def sh_fit(A, data, alpha=0.1, l_array=None, smoothness=0):
    """least-squares SH fit with optional degree-dependent regularization"""
    n = A.shape[1]
    if smoothness == 0 or l_array is None:                                        
        diag = np.full(n, alpha, dtype=float)                                  # proceed with plain Tikhonov (alpha is constant across degrees)
    else:
        diag = alpha * (l_array.astype(float) * (l_array + 1.0)) ** smoothness     # alpha is degree-dependent: penalty ∝ [l(l+1)]^smoothness
        diag[l_array == 0] = alpha                                               # avoid zero-penalization to l=0 term
    c = np.linalg.solve(A.T @ A + np.diag(diag), A.T @ data)                   # solve for coefficients
    return c

def sh_gaussian_taper(l_max, l_cutoff=15):
    """per-degree Gaussian taper weights"""
    ls = np.arange(l_max + 1, dtype=float)
    sigma_sq = -2.0 * np.log(0.5) / (l_cutoff * (l_cutoff + 1))
    return np.exp(-ls * (ls + 1) * sigma_sq / 2.0)

def apply_taper(c, l_array, l_max, l_cutoff=15):
    """apply Gaussian taper to SH coefficients"""
    taper_per_degree = sh_gaussian_taper(l_max, l_cutoff)
    taper_vec = taper_per_degree[l_array]
    return c * taper_vec

def lm_arrays(l_max):
    """return (l_array, m_array) for the column ordering of sh_design_matrix; column ordering: for l in 0..lmax, for m in -l..+l"""
    n_cols = (l_max+1) ** 2
    l_array = np.empty(n_cols, dtype=int)
    m_array = np.empty(n_cols, dtype=int)
    col = 0
    for l in range(l_max + 1):
        for m in range(-l, l + 1):
            l_array[col] = l
            m_array[col] = m
            col += 1
    return l_array, m_array

def sh_evaluate_global(c, l_array, l_max, nlat=181, nlon=361):
    """evaluate SH expansion on a regular 1-degree global grid"""
    rlats = np.linspace(-90, 90, nlat)
    rlons = np.linspace(-180, 180, nlon)
    lon_grid, lat_grid = np.meshgrid(rlons, rlats)
    A_grid, _ = sh_design_matrix(lat_grid.ravel(), lon_grid.ravel(), l_max)
    field = (A_grid @ c).reshape(nlat, nlon)
    return rlats, rlons, field

def sh_data_mask(rlats, rlons, obs_lats, obs_lons, max_dist_deg=8.0):
    """boolean mask of grid cells within max_dist_deg of observation points (True where constrained by data; False elsewhere)"""
    lon_grid, lat_grid = np.meshgrid(rlons, rlats)
    
    def to_xyz(lat, lon):
        latr = np.radians(lat); lonr = np.radians(lon)
        cl = np.cos(latr)
        return np.stack([cl*np.cos(lonr), cl*np.sin(lonr), np.sin(latr)], axis=-1)

    grid_xyz = to_xyz(lat_grid.ravel(), lon_grid.ravel())
    data_xyz = to_xyz(np.asarray(obs_lats), np.asarray(obs_lons))
    chord_thresh_sq = (2.0 * np.sin(np.radians(max_dist_deg) / 2.0)) ** 2
    
    Ng = grid_xyz.shape[0]
    mask_flat = np.zeros(Ng, dtype=bool)
    chunk = 4096
    for i in range(0, Ng, chunk):
        diff = grid_xyz[i:i+chunk, None, :] - data_xyz[None, :, :]
        d2 = np.einsum('ijk,ijk->ij', diff, diff)
        mask_flat[i:i + chunk] = (d2.min(axis=1) <= chord_thresh_sq)
    return mask_flat.reshape(lat_grid.shape)

def sh_power_spectrum(c, l_array, l_max):
    """per-degree power spectrum S(l) = sum_m c_lm^2"""
    degrees = np.arange(l_max+1)
    power = np.bincount(l_array, weights=c**2, minlength=l_max+1).astype(float)
    return degrees, power

def sh_y21_power(c, l_array, m_array):
    """power in Y₂₁ (l=2, m=±1): c[l=2,m=+1]² + c[l=2,m=-1]²"""
    mask = (l_array == 2) & (np.abs(m_array) == 1)
    return float(np.sum(c[mask] ** 2))

def gen_spec(l_max, l_center=30, l_width=15):
    """generate Gaussian-distributed power spectrum"""
    ls = np.arange(l_max+1, dtype=float)
    c_l = np.exp(-0.5 * ((ls - l_center) / l_width) ** 2)
    return c_l

def sample_coeffs(c_l, l_array, m_array, zero_y21=True, rng=None):
    """draw real-basis SH coefficients with variance C_l/(2l+1) per (l,m)"""
    if rng is None:
        rng = np.random.default_rng()
    sigma_per_l = np.sqrt(c_l / (2.0 * np.arange(len(c_l)) + 1.0))
    sigma_per_coeff = sigma_per_l[l_array]
    coeffs = rng.normal(0.0, sigma_per_coeff)
    if zero_y21:
        mask = (l_array == 2) & (np.abs(m_array) == 1)
        coeffs[mask] = 0.0
    return coeffs

def null_y21_sh(A, l_array, m_array, l_max, diff_obs, n_iter=100, null_l_center=30, null_l_width=15, alpha=3.0, smoothness=1, l_cutoff=15, zero_y21=True, seed=0):
    """Monte Carlo null distribution of Y₂₁ power"""
    rng = np.random.default_rng(seed)
    c_l = gen_spec(l_max, null_l_center, null_l_width)

    powers = np.empty(n_iter, dtype=float)
    for i in range(n_iter):
        c_true = sample_coeffs(c_l, l_array, m_array, zero_y21=zero_y21, rng=rng)
        y = A @ c_true
        y = np.where(y >= 0, 1.0, -1.0)
        c_hat = sh_fit(A, y, alpha, l_array, smoothness)
        c_tap = apply_taper(c_hat, l_array, l_max, l_cutoff)
        powers[i] = sh_y21_power(c_tap, l_array, m_array)
    return powers
    
### ----------- reconstruction functions ------------------ ###

def reconstruct_polygons(polygons, rot_model, t, anchor_plate):
    """ reconstruct polygons at specified model time t """
    reconstructed_polygons = []
    pygp.reconstruct(polygons, rot_model, reconstructed_polygons, t, anchor_plate_id = anchor_plate)
    return reconstructed_polygons

### ----------- plotting functions ------------------ ###

def plot_Y21(lam, levels=21):
    reflons = np.linspace(-180, 180, 181)
    reflats = np.linspace(-90, 90, 91)
    rlons, rlats = np.meshgrid(reflons, reflats)
    s = Y21(lam, rlats, rlons)

    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot(1, 1, 1, projection=crs.Robinson())
    ax.set_global()
    ax.gridlines(color='k', linestyle='--')
    cf = ax.contourf(rlons, rlats, s, levels=levels, cmap='RdBu', transform=crs.PlateCarree(), antialiased=True, zorder=0)
    cbar = fig.colorbar(cf, ax=ax, orientation='horizontal', shrink=0.5, pad=0.05, aspect=20)
    cbar.set_label('Y21 response (normalized)')
    ax.scatter(lam, 0, marker='o', s=50, c='orange', edgecolor='k', linewidth=1.5, transform=crs.PlateCarree(), label='TPW axis', zorder=1e6)
    ax.legend(loc='lower right')
    return fig

def plot_flux(lam, lats, lons, s, polygons=None):
    fig = plt.figure(figsize=(12,7))
    ax = fig.add_subplot(1,1,1, projection=crs.Robinson())
    ax.set_global() 
    ax.gridlines()
    
    if polygons == None:
        ax.coastlines()
    else:
        for polygon in polygons:
            geometry = polygon.get_reconstructed_geometry()
            vertices = geometry.to_lat_lon_array()
            ax.plot(vertices[:, 1], vertices[:, 0], transform=crs.Geodetic(),  color='black', linewidth=1, alpha=0.2)
       
    norm = TwoSlopeNorm(vcenter=0, vmin=np.min(s), vmax=np.max(s))
    sc = ax.scatter(lons, lats, c=s, s=2, cmap='RdBu', norm=norm, transform=crs.PlateCarree())
    plt.colorbar(sc, orientation='horizontal', shrink=0.5, pad=0.05, aspect=20, label='predicted normalized sea-level change')
    ax.scatter(lam, 0, marker='o', s=50, c='orange', edgecolor='k', linewidth=1.5, transform=crs.PlateCarree(), label='TPW axis', zorder=1e6)
    ax.legend(loc='lower right')
    return fig

def plot_log_reg(s, y, ax=None):
    new_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 2.5))
        new_fig = True
    else: fig = ax.figure

    x = np.asarray(s).reshape(-1, 1)
    model = fit_logistic_fast(s, y, return_model=True)
    px_vals = np.linspace(x.min(), x.max(), 300).reshape(-1, 1)   # create x vals for smooth curve
    y_prob = model.predict_proba(px_vals)[:, 1]                # probability of class 1
    coeff = model.coef_[0][0]                                  # slope (beta)
    intercept = model.intercept_[0]                            # intercept (alpha)
    x_boundary = -intercept / coeff                               # x at p=0.5 (decision boundary)                         
    
    ax.scatter(x, y, color='k', alpha=0.01, s = 30)
    ax.plot(px_vals, y_prob, color='tab:red', linewidth=1.2, label='Logistic fit')
    ax.axvline(x=x_boundary, color='tab:blue', linestyle='--', linewidth=1, label='Decision boundary (p=0.5)')

    ax.set_xlabel('$Y_2^{1}$ (normalized)', fontsize=10)
    ax.set_ylabel('Probability of transgression', fontsize=10)
    ax.set_title('Best fitting logistic regression')
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc='lower right', bbox_to_anchor=(0.99, 0.05))

    if new_fig:
        fig.subplots_adjust(right=0.7)
        return fig
    else: return ax

def plot_inversion(df, p95=None, ax=None):
    new_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 2))
        new_fig = True
    else: fig = ax.figure
    
    mask = df['beta'] > 0                     # only consider results where the TPW coefficient is positive (negative just give the antipode) 
    best_idx = df.loc[mask, 'llr'].idxmax()   # select the best-fit model
    ax.set_title(f'Best fit TPW axis at {df.loc[best_idx, "lambda"]}°E (anti-clockwise sense); pseudo-R$^2$: {df.loc[best_idx, "p-rsquared"]:.3f}', fontsize=12)
    
    line1, = ax.plot(df['lambda'], df['llr'], linewidth=1.5, color='tab:blue', label=r'll ratio', zorder=10)
    ax.set_xlabel('Longitude of TPW axis (°E)')
    ax.set_ylabel(r'Log likelihood ratio')
    
    ax2 = ax.twinx()
    line2, = ax2.plot(df['lambda'], df['alpha'], linestyle='--', linewidth=0.75, color='purple', label=r'$\alpha$', zorder=8)
    line3, = ax2.plot(df['lambda'], df['beta'], linestyle='--', linewidth=1.5, color='red', label=r'$\beta$', zorder=9)
    
    a_lo = df['alpha_conf'].apply(lambda x: x[0])
    a_hi = df['alpha_conf'].apply(lambda x: x[1])
    ax2.fill_between(df['lambda'], a_lo, a_hi, color='purple', alpha=0.2, zorder=6)
    b_lo = df['beta_conf'].apply(lambda x: x[0])
    b_hi = df['beta_conf'].apply(lambda x: x[1])
    ax2.fill_between(df['lambda'], b_lo, b_hi, color='red', alpha=0.2, zorder=7)
    ax2.set_ylabel(r'$\alpha$, $\beta$ coefficients')
    
    lines = [line1, line2, line3]
    labels = [line.get_label() for line in lines]
    ax.legend(lines, labels, labelspacing=0.2 , loc='upper right', bbox_to_anchor=(1.28, 1), frameon=False)
    
    if p95 is not None:
        if df['llr'].max() > p95:
            ax.axhline(y=p95, color='lightblue', zorder=0)
    p_llr_threshold = df['p_llr'] >= 0.05
    start_idx = None
    for i, val in enumerate(p_llr_threshold):
        if val and start_idx is None:
            start_idx = df['lambda'].iloc[i]
        elif not val and start_idx is not None:
            end_idx = df['lambda'].iloc[i-1]
            ax.axvspan(start_idx, end_idx, color='grey', alpha=0.2, zorder=-1)
            start_idx = None
    if start_idx is not None:
        ax.axvspan(start_idx, df['lambda'].iloc[-1], color='grey', alpha=0.2, zorder=-1)
    
    if new_fig:
        fig.subplots_adjust(right=0.7)
        return fig
    else: return ax

def plot_resolving_pwr(evals, evecs, ref_power, ax=None):
    power = np.sum(evals)
    norm_power = power/ref_power
    
    # re-scale evals to give same ellipsoidal area as unit circle
    axes = np.sqrt(evals)
    scale = np.sqrt(1 / np.prod(axes))
    a, b = axes * scale

    # make circle / ellipse
    pts = np.linspace(0, 2*np.pi, 360)
    circle = np.array([np.sin(pts), -np.cos(pts)])     # unit circle
    ellipse = np.array([a*np.cos(pts), b*np.sin(pts)]) # our ellipse
    ellipse_rot = evecs @ ellipse  # rotate into correct orientation
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(3, 3))
        new_fig = True
    else:
        fig = ax.figure
        new_fig = False

    ax.set_title(f'Relative resolving power: {norm_power:.2f}', pad=20, fontsize=12)
    ax.fill(ellipse_rot[1], -ellipse_rot[0], color='tab:blue', alpha=0.5)
    ax.plot(circle[1], -circle[0], 'k--', linewidth=0.75)
    ax.text(0, 0, f'Observation\nanisotropy', ha='center', va='center', fontsize=10)

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values(): spine.set_visible(False)
    ax.set_aspect('equal')
    r_vert = 1.075
    r_horiz = 1.025
    ax.text(0, -r_vert, '0', ha='center', va='center', fontsize=8)
    ax.text(r_horiz, 0, '90', ha='left', va='center', fontsize=8)
    ax.text(0, r_vert, '180', ha='center', va='center', fontsize=8)
    ax.text(-r_horiz, 0, '270', ha='right', va='center', fontsize=8)

    if new_fig: return fig
    else: return ax

def plot_observations(t, polygons, obs, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(15,10), subplot_kw={'projection': crs.Robinson()})
    ax.set_global()
    ax.gridlines()
    for polygon in polygons:
        geometry = polygon.get_reconstructed_geometry()
        vertices = geometry.to_lat_lon_array()
        ax.plot(vertices[:, 1], vertices[:, 0], transform=crs.Geodetic(),  color='black', linewidth=1, alpha=0.2)

    land = obs[obs['state'] == 1]
    sea = obs[obs['state'] == 0]
    ax.scatter(x=land['lon'], y=land['lat'], color='darkgoldenrod', s=2, transform=crs.PlateCarree(), label='Exposed land')
    ax.scatter(x=sea['lon'], y=sea['lat'], color='dodgerblue', s=2, transform=crs.PlateCarree(), label='Inundated')
    #ax.legend(loc='lower right', bbox_to_anchor=(1.07, 0), markerscale=2)
    return ax

def plot_data_and_model(polygons, lam, rlats, rlons, s, merged, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(15,10), subplot_kw={'projection': crs.Robinson()})
    ax.set_global()
    ax.gridlines(zorder=0)
    for polygon in polygons:
        geometry = polygon.get_reconstructed_geometry()
        vertices = geometry.to_lat_lon_array()
        ax.plot(vertices[:, 1], vertices[:, 0], transform=crs.Geodetic(),  color='black', linewidth=1, alpha=0.2)
        
    for lon in [lam, (lam + 180)]:
        ax.plot([lon]*181, np.linspace(-90, 90, 181), color='k', linewidth=1, linestyle='--', transform=crs.PlateCarree())
    ax.plot(np.linspace(-180, 180, 361), [0]*361, color='k', linewidth=1, linestyle='--', transform=crs.PlateCarree())

    cs = ax.contourf(rlons, rlats, s, levels=21, cmap='RdBu', transform=crs.PlateCarree(), alpha=0.5, antialiased=True, zorder=0)
    cax = inset_axes(ax, width='60%', height='3%', loc='lower center', borderpad=-2)  # negative pad pushes it below
    fig = ax.get_figure()
    fig.colorbar(cs, cax=cax, orientation='horizontal', label='Best-fitting TPW signal (normalized sea-level change)')
    
    drop = merged[merged['diff'] == -1]
    rise = merged[merged['diff'] == 1]
    ax.scatter(x=drop['lon_t0'], y=drop['lat_t0'], color='red', s=5, transform=crs.PlateCarree(), label='Regression')
    ax.scatter(x=rise['lon_t0'], y=rise['lat_t0'], color='blue', s=5, transform=crs.PlateCarree(), label='Transgression')
    ax.scatter(lam, 0, marker='o', s=50, c='orange', edgecolor='k', transform=crs.PlateCarree(), label='TPW axis (acw sense)', zorder=1e6)
    return ax

def plot_data_and_model_wfeatures(continents, margins, basins, lam, rlats, rlons, s, merged, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(15,10), subplot_kw={'projection': crs.Robinson()})
    ax.set_global()
    ax.gridlines(zorder=0)
    for polygon in continents:
        geometry = polygon.get_reconstructed_geometry()
        vertices = geometry.to_lat_lon_array()
        ax.plot(vertices[:, 1], vertices[:, 0], transform=crs.Geodetic(),  color='black', linewidth=1, alpha=0.2)
        
    for lon in [lam, (lam + 180)]:
        ax.plot([lon]*181, np.linspace(-90, 90, 181), color='k', linewidth=1, linestyle='--', transform=crs.PlateCarree())
    ax.plot(np.linspace(-180, 180, 361), [0]*361, color='k', linewidth=1, linestyle='--', transform=crs.PlateCarree())

    #cs = ax.contourf(rlons, rlats, s, levels=21, cmap='RdBu', transform=crs.PlateCarree(), alpha=0.5, antialiased=True, zorder=0)
    #for c in cs.collections: c.set_rasterized(True)
    #cax = inset_axes(ax, width='60%', height='3%', loc='lower center', borderpad=-2)  # negative pad pushes it below
    #fig = ax.get_figure()
    #fig.colorbar(cs, cax=cax, orientation='horizontal', label='Best-fitting TPW signal (normalized sea-level change)')

    wrapped_margins = wrap_features(margins, feature_type='polygons')
    for lons, lats in wrapped_margins:
            polygon = Polygon(zip(lons, lats))
            ax.add_geometries([polygon], crs=crs.PlateCarree(), facecolor='violet', edgecolor='black', linestyle=':', linewidth=1, alpha=0.65)

    wrapped_basins = wrap_features(basins, feature_type='polygons')
    for lons, lats in wrapped_basins:
            polygon = Polygon(zip(lons, lats))
            ax.add_geometries([polygon], crs=crs.PlateCarree(), facecolor='sienna', edgecolor='black', linestyle=':', linewidth=1, alpha=0.65)

    legend_handles = [Patch(facecolor='violet', edgecolor='black', linestyle=':', label='Passive margins', alpha=0.65), 
                      Patch(facecolor='sienna', edgecolor='black', linestyle=':', label='Foreland basins', alpha=0.65)]
    drop = merged[merged['diff'] == -1]
    rise = merged[merged['diff'] == 1]
    ax.scatter(x=drop['lon_t0'], y=drop['lat_t0'], color='red', s=5, transform=crs.PlateCarree())
    ax.scatter(x=rise['lon_t0'], y=rise['lat_t0'], color='blue', s=5, transform=crs.PlateCarree())
    ax.scatter(lam, 0, marker='o', s=50, c='orange', edgecolor='k', transform=crs.PlateCarree(), zorder=1e6)
    return ax, legend_handles

def plot_data_and_sh(polygons, rlats, rlons, field=None, observations=None, mask=None, vmax=None, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(15,10), subplot_kw={'projection': crs.Robinson()})
    ax.set_global()
    ax.gridlines(zorder=0)
    for polygon in polygons:
        geometry = polygon.get_reconstructed_geometry()
        vertices = geometry.to_lat_lon_array()
        ax.plot(vertices[:, 1], vertices[:, 0], transform=crs.Geodetic(),  color='black', linewidth=1, alpha=0.2)

    if field is not None:
        field_to_plot = field.copy()
        if mask is not None:
            field_to_plot = np.ma.array(field_to_plot, mask=~mask)
        lon_grid, lat_grid = np.meshgrid(rlons, rlats)
        if vmax is None:
            vmax = float(np.abs(field_to_plot).max()) or 1.0
        norm = TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=vmax)
        cf = ax.contourf(lon_grid, lat_grid, field_to_plot, levels=21, cmap='RdBu', norm=norm, transform=crs.PlateCarree(), antialiased=True)
        cax = inset_axes(ax, width='60%', height='3%', loc='lower center', borderpad=-2)  # negative pad pushes it below
        fig = ax.get_figure()
        fig.colorbar(cf, cax=cax, orientation='horizontal', label='SH-smoothed sea-level flux')

    if observations is not None:
        drop = observations[observations['diff'] == -1]
        rise = observations[observations['diff'] == 1]
        ax.scatter(x=drop['lon_t0'], y=drop['lat_t0'], color='red', s=2.5, transform=crs.PlateCarree(), label='Regression')
        ax.scatter(x=rise['lon_t0'], y=rise['lat_t0'], color='blue', s=2.5, transform=crs.PlateCarree(), label='Transgression')
    return ax

def plot_permutation_results(t, results, n_clusters=None, noise=None, ax=None):
    arr = np.sort(np.array(results))
    n95 = np.percentile(arr, 95)

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 3))
        new_fig = True
    else:
        fig = ax.figure
        new_fig = False
    
    ax.plot(arr, marker='o', label='best-fits')
    ax.axhline(y=n95, color='lightblue', zorder=0, label='95th percentile')
    ax.set_xlabel('Permutation number')
    ax.set_ylabel(r'Log likelihood ratio')
    ax.legend(loc='lower right')
    ax.set_title(f'Permutation results at {t} Ma', fontsize=12)
    if n_clusters is not None and noise is not None:
        ax.text(0.05, 0.95, f"{n_clusters} clusters; {noise} 'noise' pts.\n95th percentile llr: {n95:.2f}", transform=ax.transAxes, fontsize=10, va='top', ha='left')
    else:
        ax.text(0.05, 0.95, f'95th percentile llr: {n95:.2f}', transform=ax.transAxes, fontsize=10, va='top', ha='left')
    
    if new_fig:
        fig.tight_layout()
        return fig
    else:
        return ax

def wrap_features(features, feature_type):
    """ """
    dateline_wrapper = pygp.DateLineWrapper()
    
    if feature_type == 'polygons':
        polygons_list = []
        for poly in features:   
            geometry = poly.get_reconstructed_geometry()
            if isinstance(geometry, pygp.PolygonOnSphere):
                exterior_ring = list(geometry.get_exterior_ring_points())
                if exterior_ring[0] != exterior_ring[-1]:
                    exterior_ring.append(exterior_ring[0])
                exterior_polygon = pygp.PolygonOnSphere(exterior_ring)
                wrapped_geometry = dateline_wrapper.wrap(exterior_polygon)
                for wgeom in wrapped_geometry:
                    polygon = [point.to_lat_lon() for point in wgeom.get_points()]
                    if len (polygon) < 3: continue
                    if polygon[0] != polygon[-1]:
                        polygon.append(polygon[0])
                    lats, lons = zip(*polygon)
                    polygons_list.append((np.array(lons), np.array(lats)))
        return polygons_list