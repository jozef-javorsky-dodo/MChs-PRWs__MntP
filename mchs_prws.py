import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple, Literal, TypeVar
import numpy.typing as npt
import uuid

T = TypeVar("T", bound=np.ndarray)
Method = Literal["brute_force", "eigenvalue", "linear_system"]

@dataclass(frozen=True)
class SimulationConfig:
    num_walks: int = 1000
    num_steps: int = 100
    burn_in: int = 1000
    iterations: int = 1_000_000
    methods: Tuple[Method, ...] = ("brute_force", "eigenvalue", "linear_system")
    seed: Optional[int] = None

@dataclass(frozen=True)
class VisualizationConfig:
    figsize: Tuple[int, int] = (10, 6)
    dpi: int = 300
    alpha: float = 0.3
    linewidth: float = 0.8
    grid_alpha: float = 0.3
    hist_alpha: float = 0.4
    hist_bins: int = 30
    x_range: Tuple[float, float] = (-25, 25)
    x_points: int = 100
    subplot_figsize: Tuple[int, int] = (15, 3)

class MarkovChainSimulator:
    def __init__(self, config: Optional[SimulationConfig] = None) -> None:
        self.config = config or SimulationConfig()
        self.rng = np.random.default_rng(self.config.seed)

    def simulate_random_walks(self) -> npt.NDArray[np.float64]:
        steps = self.rng.choice([-1, 1],
                                size=(self.config.num_walks,
                                      self.config.num_steps))
        starts = np.zeros((self.config.num_walks, 1))
        traj = np.hstack((starts, steps))
        return np.cumsum(traj, axis=1)

    def analyze_steady_state(
            self,
            matrix: npt.NDArray[np.float64],
            init_state: npt.NDArray[np.float64]
    ) -> Dict[Method, npt.NDArray[np.float64]]:
        res: Dict[Method, npt.NDArray[np.float64]] = {}
        if "brute_force" in self.config.methods:
            res["brute_force"] = self._compute_brute_force(matrix, init_state)
        if "eigenvalue" in self.config.methods:
            res["eigenvalue"] = self._compute_eigenvalue(matrix)
        if "linear_system" in self.config.methods:
            res["linear_system"] = self._compute_linear_system(matrix)
        return res

    def _compute_brute_force(
            self,
            matrix: npt.NDArray[np.float64],
            init_state: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        state = init_state.copy()
        for _ in range(self.config.iterations):
            state = matrix.T @ state
        return state.flatten()

    def _compute_eigenvalue(
            self, matrix: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        eigvals, eigvecs = np.linalg.eig(matrix.T)
        mask = np.isclose(eigvals, 1.0)
        ss = eigvecs[:, mask].real
        ss /= ss.sum()
        return ss.flatten()

    def _compute_linear_system(
            self, matrix: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        dim = matrix.shape[0]
        sys_mat = np.vstack([matrix.T - np.eye(dim), np.ones(dim)])
        sys_vec = np.zeros(dim + 1)
        sys_vec[-1] = 1.0
        sol, *_ = np.linalg.lstsq(sys_mat, sys_vec, rcond=None)
        return sol

    def monte_carlo_estimate(
            self,
            matrix: npt.NDArray[np.float64],
            init_state: Optional[int] = None
    ) -> npt.NDArray[np.float64]:
        n_states = matrix.shape[0]
        cum_probs = np.cumsum(matrix, axis=1)
        traj = np.zeros(self.config.iterations + 1, dtype=int)
        traj[0] = (
            self.rng.integers(0, n_states)
            if init_state is None else init_state
        )
        for step in range(self.config.iterations):
            r = self.rng.random()
            traj[step + 1] = np.argmax(r < cum_probs[traj[step]])
        eff_traj = traj[self.config.burn_in:]
        return np.array([np.mean(eff_traj == s) for s in range(n_states)])

class MarkovChainVisualizer:
    def __init__(self,
                 config: Optional[VisualizationConfig] = None) -> None:
        self.config = config or VisualizationConfig()

    def plot_walks(self,
                   walks: npt.NDArray[np.float64],
                   show: bool = True,
                   save_path: Optional[str] = None) -> None:
        fig, ax = plt.subplots(figsize=self.config.figsize)
        t = np.arange(walks.shape[1])
        for w in walks:
            ax.plot(t, w, alpha=self.config.alpha,
                    linewidth=self.config.linewidth)
        ax.set_title("Random Walk Trajectories")
        ax.set_xlabel("Step")
        ax.set_ylabel("Displacement")
        ax.grid(True, alpha=self.config.grid_alpha)
        if save_path is not None:
            plt.savefig(save_path, dpi=self.config.dpi,
                        bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close(fig)

    def plot_distribution_snapshots(
            self,
            walks: npt.NDArray[np.float64],
            times: Optional[List[int]] = None,
            show: bool = True) -> None:
        pts = times or list(range(20, 101, 20))
        fig, axes = plt.subplots(1, len(pts),
                                 figsize=self.config.subplot_figsize,
                                 sharex=True)
        xs = np.linspace(self.config.x_range[0],
                         self.config.x_range[1],
                         self.config.x_points)
        for i, t in enumerate(pts):
            axes[i].hist(walks[:, t],
                         bins=self.config.hist_bins,
                         density=True,
                         alpha=self.config.hist_alpha,
                         color="skyblue")
            axes[i].set_title(f"t = {t}")
            p = stats.norm(loc=0, scale=np.sqrt(t)).pdf(xs)
            axes[i].plot(xs, p, "r-",
                         linewidth=self.config.linewidth)
        plt.tight_layout()
        if show:
            plt.show()
        else:
            plt.close(fig)

def run_markov_city_simulation() -> None:
    A = np.array([[0.9, 0.05, 0.05],
                  [0.1, 0.8, 0.1],
                  [0.04, 0.01, 0.95]])
    pop = np.array([300000, 300000, 300000]).reshape(-1, 1)
    cities = ["Raleigh", "Chapel Hill", "Durham"]
    sim = MarkovChainSimulator()
    res = sim.analyze_steady_state(A, pop)
    print("\nCity Migration Simulation Results:")
    print("=" * 40)
    for method, dist in res.items():
        print(f"\n{method.replace('_', ' ').title()} Method:")
        for c, v in zip(cities, dist):
            print(f"  {c}: {v:.0f}")
    mc = sim.monte_carlo_estimate(A)
    print("\nMonte Carlo Estimation:")
    for c, f in zip(cities, mc):
        print(f"  {c}: {f:.5f}")
    tot = pop.sum()
    sc = mc * tot
    print("\nScaled Monte Carlo Estimation (Population):")
    for c, v in zip(cities, sc):
        print(f"  {c}: {v:.0f}")

def main() -> None:
    sim = MarkovChainSimulator()
    vis = MarkovChainVisualizer()
    print("Running Random Walk Simulation...")
    walks = sim.simulate_random_walks()
    out_file = f"markov_walks_{uuid.uuid4().hex}.png"
    vis.plot_walks(walks, save_path=out_file)
    vis.plot_distribution_snapshots(walks)
    print("\nRunning City Migration Markov Chain Simulation...")
    run_markov_city_simulation()

if __name__ == "__main__":
    main()