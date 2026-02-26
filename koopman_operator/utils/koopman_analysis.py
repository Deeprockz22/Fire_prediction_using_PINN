"""
Koopman Spectral Analysis Tools
================================

After training, the Koopman matrix K can be analysed to understand
the dominant dynamical modes of the fire system.

Eigenvalue interpretation:
    |λ| < 1   →  decaying mode   (fire dying down)
    |λ| = 1   →  neutral mode    (steady-state fire)
    |λ| > 1   →  growing mode    (fire growth phase)
    Im(λ) ≠ 0 →  oscillatory     (pulsating fire)

Usage:
    analyser = KoopmanAnalyzer(trained_model)
    analyser.print_spectrum()
    analyser.plot_eigenvalues("koopman_spectrum.png")
"""

import numpy as np
import torch
from typing import Optional


class KoopmanAnalyzer:
    """Spectral analysis of a trained Koopman matrix."""

    def __init__(self, model):
        """
        Args:
            model: A trained KoopmanLSTM whose .koopman_matrix weight
                   is the learned K operator.
        """
        K = model.koopman_matrix.weight.detach().cpu().numpy()
        self.K = K
        self.eigenvalues, self.eigenvectors = np.linalg.eig(K)

        # Sort by magnitude (dominant modes first)
        order = np.argsort(-np.abs(self.eigenvalues))
        self.eigenvalues = self.eigenvalues[order]
        self.eigenvectors = self.eigenvectors[:, order]

    # ------------------------------------------------------------------
    @property
    def magnitudes(self) -> np.ndarray:
        """Eigenvalue magnitudes (growth/decay rates)."""
        return np.abs(self.eigenvalues)

    @property
    def frequencies(self) -> np.ndarray:
        """Angular frequencies from imaginary parts (rad / timestep)."""
        return np.angle(self.eigenvalues)

    @property
    def is_stable(self) -> bool:
        """True if ALL eigenvalue magnitudes ≤ 1 (no unbounded growth)."""
        return bool(np.all(self.magnitudes <= 1.0 + 1e-6))

    # ------------------------------------------------------------------
    def dominant_modes(self, n: int = 5) -> list:
        """
        Return the top-n dominant Koopman modes.

        Each mode is a dict with:
            magnitude   – growth/decay rate
            frequency   – oscillation frequency (rad / timestep)
            eigenvalue  – raw complex eigenvalue
            label       – human-readable interpretation
        """
        modes = []
        for i in range(min(n, len(self.eigenvalues))):
            ev  = self.eigenvalues[i]
            mag = np.abs(ev)
            freq = np.angle(ev)

            if mag > 1.01:
                label = "GROWING"
            elif mag < 0.99:
                label = "DECAYING"
            else:
                label = "STEADY"

            if abs(np.imag(ev)) > 1e-4:
                label += " + OSCILLATORY"

            modes.append({
                "eigenvalue": ev,
                "magnitude":  mag,
                "frequency":  freq,
                "label":      label,
            })
        return modes

    # ------------------------------------------------------------------
    def print_spectrum(self, n: int = 10):
        """Pretty-print the dominant modes."""
        print("\n" + "=" * 65)
        print("  KOOPMAN SPECTRAL ANALYSIS")
        print("=" * 65)
        print(f"  Matrix size:  {self.K.shape[0]}×{self.K.shape[1]}")
        print(f"  Stable:       {self.is_stable}")
        print(f"  Spectral radius: {self.magnitudes[0]:.4f}")
        print("-" * 65)
        print(f"  {'#':>3}  {'|λ|':>8}  {'∠λ (rad)':>10}  {'Re':>10}  {'Im':>10}  Label")
        print("-" * 65)

        for i, m in enumerate(self.dominant_modes(n)):
            ev = m["eigenvalue"]
            print(
                f"  {i+1:>3}  {m['magnitude']:>8.4f}  {m['frequency']:>10.4f}  "
                f"{np.real(ev):>10.4f}  {np.imag(ev):>10.4f}  {m['label']}"
            )
        print("=" * 65 + "\n")

    # ------------------------------------------------------------------
    def plot_eigenvalues(self, save_path: Optional[str] = None):
        """
        Plot eigenvalues on the complex plane with the unit circle.
        Inside the circle = stable, outside = growing.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not installed — skipping plot.")
            return

        fig, ax = plt.subplots(1, 1, figsize=(7, 7))

        # Unit circle
        theta = np.linspace(0, 2 * np.pi, 200)
        ax.plot(np.cos(theta), np.sin(theta), "k--", linewidth=0.8, alpha=0.4,
                label="Unit circle")

        # Eigenvalues
        re = np.real(self.eigenvalues)
        im = np.imag(self.eigenvalues)
        mag = self.magnitudes

        scatter = ax.scatter(re, im, c=mag, cmap="coolwarm", edgecolors="k",
                             s=50, linewidths=0.5, vmin=0.5, vmax=1.5)
        plt.colorbar(scatter, ax=ax, label="|λ|")

        ax.set_xlabel("Re(λ)")
        ax.set_ylabel("Im(λ)")
        ax.set_title("Koopman Eigenvalue Spectrum")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.legend()

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"Saved eigenvalue plot → {save_path}")
        else:
            plt.show()
        plt.close(fig)
