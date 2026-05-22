# File dedicato alle impostazioni
from pathlib import Path
from dataclasses import dataclass

FEATURES = [
    'original_shape2D_Elongation',
    'original_shape2D_MajorAxisLength',
    'original_shape2D_MinorAxisLength',
    'original_shape2D_Perimeter',
    'original_shape2D_MaximumDiameter',
    'original_shape2D_Sphericity',
    'original_firstorder_Mean',
    'original_firstorder_Median',
    'original_firstorder_Minimum',
    'original_firstorder_Maximum',
    'original_firstorder_Range',
    'original_firstorder_Uniformity',
    'Centroid_X',
    'Centroid_Y',
]

SEED_ALIASES = {
    "nndsvd": "nndsvd",
    "nndsvda": "nndsvd",    # non supportato da nimfa 1.4.0
    "nndsvdar": "nndsvd",   # non supportato da nimfa 1.4.0
    "random": "random",
    "random_c": "random_c",
    "random_vcol": "random_vcol",
    "fixed": "fixed",
}

@dataclass(frozen=True)
class ScalingMethods :      # Permette di implementare altri metodi di scaling in futuro
    MIN_MAX = True
scaling_methods = ScalingMethods

### Flags
RUN_EXPERIMENT = True
PLOT_EXPERIMENT = True
RUN_FINAL = True
RUN_FUZZY = True
PLOT_FUZZY = True


### Settings --- some classes are used as glorified namespaces (intended)
# before choosing k
K_VALUES = list(range(2, 15))   # per 14 feature
MAX_REP = 25                    # WILL fault data experiment (especially cophenetic) if set too low (think less than 5)
# after k is chosen
RANDOM_STATE = 42               # seed deterministco per la NMF finale
CHOSEN_K = 3                    # The k the NMF uses for the interpretation block | chosen (manually) based on results from experiment block
# fuzzy
FUZZY_N_SETS = 5
FUZZY_N_RANDOM_SAMPLES = 10

# Fuzzy Sets
LABEL_MAP = {
    1: "Very Low",
    2: "Low",
    3: "Medium",
    4: "High",
    5: "Very High",
}


@dataclass(frozen=True)
class Nmf :
    INIT : str = 'random_vcol'      # possibili: 'fixed', 'methods', 'nndsvd', 'random', 'random_c', 'random_vcol'
    MAX_ITER : int = 200
    UPDATE : str = 'euclidean'
nmf = Nmf()

class KMeans() :
    INIT : str = 'kmeans++' # possibili: 'kmeans++' o 'random'
    N_INIT : int = 15       # how many times K-Means is run with different initial centroid initializations (before choosing the best one)
    MAX_ITER : int = 200
    TOL: float = 1e-5       # per KMeans va bene 1e-4, però visto che confrontiamo con altri algoritmi...
kmeans = KMeans()

class FCMeans :
    MAX_ITER : int = 200
    M : int = 2.0
    TOL : float = 1e-5
    # consensus mode(s): | implementate: 'hard', 'soft_dot' 'soft_cosine' | attualmente vengono eseguite tutte
    # - "hard": use fcm.predict(X) then connectivity 0/1
    # - "soft_dot": fuzzy co-membership
    # - "soft_cosine": build similarity from membership matrix
    CONSENSUS_MODES = ("hard", "soft_dot", "soft_cosine")
    # sharpening is of course only applied on soft methods
    USE_SHARPENING : bool = True
    SHARPEN_ALPHA : float = 2.0         # simile a M ma applicato a posteriori
fcmeans = FCMeans()


### Paths
PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
RESULTS_DIR = PROJECT_DIR / "results"
NMF_K_SEARCH_DIR = RESULTS_DIR / "nmf_k_search"

def get_nmf_final_root_dir(k: int) -> Path:
    return RESULTS_DIR / f"nmf_final_k_{k}"

def get_csv_dir(run_dir: Path) -> Path:
    return Path(run_dir) / "csv"

def get_numpy_dir(run_dir: Path) -> Path:
    return Path(run_dir) / "numpy"

def get_fuzzy_dir(run_dir: Path) -> Path:
    return Path(run_dir) / "fuzzy"

def get_plots_dir(run_dir: Path) -> Path:
    return Path(run_dir) / "plots"

def relative_to_project(path: Path) -> Path:
    path = Path(path)
    try:
        return path.relative_to(PROJECT_DIR)
    except ValueError:
        return path