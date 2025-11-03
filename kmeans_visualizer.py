"""
K-Means Clustering Visualization Application
Interactive tool for demonstrating K-means algorithm behavior
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib.patches import Circle
import os


class KMeansVisualizer:
    """Interactive K-means visualization application"""

    def __init__(self):
        """Initialize the application"""
        # Data storage
        self.group1_data = None
        self.group2_data = None
        self.group3_data = None
        self.original_group3_count = 0

        # Original static data (Groups 1 and 2 never change)
        self.original_group1_data = None
        self.original_group2_data = None

        # Centroids
        self.centroid1 = None
        self.centroid2 = None
        self.centroid3 = None

        # Colors for groups
        self.colors = ['#1f77b4', '#2ca02c', '#d62728']  # Blue, Green, Red

        # Dragging state
        self.dragging = False
        self.drag_centroid = None

        # Matplotlib objects
        self.fig = None
        self.ax = None
        self.scatter_plots = []
        self.centroid_plots = []
        self.stats_text = None
        self.button_recalc = None
        self.button_kmeans = None

        # Current cluster assignments (for K-means)
        self.cluster_labels = None

    def load_data(self):
        """Load data from CSV files"""
        # Get the directory where the script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # Load the three groups
        self.group1_data = pd.read_csv(
            os.path.join(script_dir, 'gaussian_group1_data.csv')
        )[['X_Coordinate', 'Y_Coordinate']].values

        self.group2_data = pd.read_csv(
            os.path.join(script_dir, 'gaussian_group2_data.csv')
        )[['X_Coordinate', 'Y_Coordinate']].values

        self.group3_data = pd.read_csv(
            os.path.join(script_dir, 'gaussian_group3_data.csv')
        )[['X_Coordinate', 'Y_Coordinate']].values

        # Store original data for Groups 1 and 2 (these never change)
        self.original_group1_data = self.group1_data.copy()
        self.original_group2_data = self.group2_data.copy()

        # Store original group 3 count for regeneration
        self.original_group3_count = len(self.group3_data)

        # Calculate initial centroids
        self.centroid1 = np.mean(self.group1_data, axis=0)
        self.centroid2 = np.mean(self.group2_data, axis=0)
        self.centroid3 = np.mean(self.group3_data, axis=0)

        print(f"Loaded {len(self.group1_data)} points from Group 1")
        print(f"Loaded {len(self.group2_data)} points from Group 2")
        print(f"Loaded {len(self.group3_data)} points from Group 3")

    def calculate_statistics(self, data):
        """Calculate mean and variance for a dataset"""
        mean_x = np.mean(data[:, 0])
        mean_y = np.mean(data[:, 1])
        var_x = np.var(data[:, 0])
        var_y = np.var(data[:, 1])
        return mean_x, mean_y, var_x, var_y

    def format_statistics_text(self):
        """Format statistics for display"""
        stats1 = self.calculate_statistics(self.group1_data)
        stats2 = self.calculate_statistics(self.group2_data)
        stats3 = self.calculate_statistics(self.group3_data)

        text = "STATISTICS\n" + "="*30 + "\n\n"
        text += f"Group 1 (Blue):\n"
        text += f"  Mean: ({stats1[0]:.2f}, {stats1[1]:.2f})\n"
        text += f"  Var:  ({stats1[2]:.2f}, {stats1[3]:.2f})\n"
        text += f"  Centroid: ({self.centroid1[0]:.2f}, {self.centroid1[1]:.2f})\n\n"

        text += f"Group 2 (Green):\n"
        text += f"  Mean: ({stats2[0]:.2f}, {stats2[1]:.2f})\n"
        text += f"  Var:  ({stats2[2]:.2f}, {stats2[3]:.2f})\n"
        text += f"  Centroid: ({self.centroid2[0]:.2f}, {self.centroid2[1]:.2f})\n\n"

        text += f"Group 3 (Red):\n"
        text += f"  Mean: ({stats3[0]:.2f}, {stats3[1]:.2f})\n"
        text += f"  Var:  ({stats3[2]:.2f}, {stats3[3]:.2f})\n"
        text += f"  Centroid: ({self.centroid3[0]:.2f}, {self.centroid3[1]:.2f})\n"

        return text

    def setup_plot(self):
        """Set up the matplotlib figure and axes"""
        self.fig = plt.figure(figsize=(16, 10))

        # Main plot area (left side)
        self.ax = plt.subplot2grid((10, 10), (0, 0), colspan=7, rowspan=10)

        # Statistics text area (right side)
        self.ax_stats = plt.subplot2grid((10, 10), (0, 7), colspan=3, rowspan=8)
        self.ax_stats.axis('off')

        # Button areas
        self.ax_button_recalc = plt.subplot2grid((10, 10), (8, 7), colspan=3, rowspan=1)
        self.ax_button_kmeans = plt.subplot2grid((10, 10), (9, 7), colspan=3, rowspan=1)

        # Set up main plot
        self.ax.set_xlabel('X Coordinate', fontsize=12)
        self.ax.set_ylabel('Y Coordinate', fontsize=12)
        self.ax.set_title('K-Means Clustering Visualization', fontsize=14, fontweight='bold')
        self.ax.grid(True, alpha=0.3)

        # Create buttons
        self.button_recalc = Button(self.ax_button_recalc, 'Re Calc', color='lightblue', hovercolor='skyblue')
        self.button_recalc.on_clicked(self.on_recalc_clicked)

        self.button_kmeans = Button(self.ax_button_kmeans, 'Implement K-Mean', color='lightgreen', hovercolor='lightcoral')
        self.button_kmeans.on_clicked(self.on_kmeans_clicked)

        # Connect mouse events for dragging
        self.fig.canvas.mpl_connect('button_press_event', self.on_mouse_press)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.fig.canvas.mpl_connect('button_release_event', self.on_mouse_release)

    def plot_data(self):
        """Plot all data points and centroids"""
        self.ax.clear()

        # Plot data points
        self.scatter_plots = []
        self.scatter_plots.append(
            self.ax.scatter(self.group1_data[:, 0], self.group1_data[:, 1],
                          c=self.colors[0], s=30, alpha=0.6, label='Group 1')
        )
        self.scatter_plots.append(
            self.ax.scatter(self.group2_data[:, 0], self.group2_data[:, 1],
                          c=self.colors[1], s=30, alpha=0.6, label='Group 2')
        )
        self.scatter_plots.append(
            self.ax.scatter(self.group3_data[:, 0], self.group3_data[:, 1],
                          c=self.colors[2], s=30, alpha=0.6, label='Group 3')
        )

        # Plot centroids
        self.centroid_plots = []
        self.centroid_plots.append(
            self.ax.scatter(self.centroid1[0], self.centroid1[1],
                          c=self.colors[0], s=500, marker='*',
                          edgecolors='black', linewidths=2, label='Centroid 1', zorder=5)
        )
        self.centroid_plots.append(
            self.ax.scatter(self.centroid2[0], self.centroid2[1],
                          c=self.colors[1], s=500, marker='*',
                          edgecolors='black', linewidths=2, label='Centroid 2', zorder=5)
        )
        self.centroid_plots.append(
            self.ax.scatter(self.centroid3[0], self.centroid3[1],
                          c=self.colors[2], s=500, marker='*',
                          edgecolors='black', linewidths=2, label='Centroid 3 (Draggable)', zorder=5)
        )

        # Add legend and formatting
        self.ax.set_xlabel('X Coordinate', fontsize=12)
        self.ax.set_ylabel('Y Coordinate', fontsize=12)
        self.ax.set_title('K-Means Clustering Visualization', fontsize=14, fontweight='bold')
        self.ax.grid(True, alpha=0.3)
        self.ax.legend(loc='upper right', fontsize=9)

        # Update statistics
        self.update_statistics()

        # Redraw
        self.fig.canvas.draw_idle()

    def update_statistics(self):
        """Update the statistics display"""
        self.ax_stats.clear()
        self.ax_stats.axis('off')

        stats_text = self.format_statistics_text()
        self.ax_stats.text(0.05, 0.95, stats_text,
                          transform=self.ax_stats.transAxes,
                          fontsize=10, verticalalignment='top',
                          fontfamily='monospace',
                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    def on_mouse_press(self, event):
        """Handle mouse press events for centroid dragging"""
        if event.inaxes != self.ax:
            return

        # Check if we clicked near centroid 3 (only this one is draggable)
        if event.xdata is not None and event.ydata is not None:
            dist = np.sqrt((event.xdata - self.centroid3[0])**2 +
                          (event.ydata - self.centroid3[1])**2)

            # If within a reasonable distance, start dragging
            if dist < 1.0:  # Adjust threshold as needed
                self.dragging = True
                self.drag_centroid = 3

    def on_mouse_move(self, event):
        """Handle mouse move events for dragging"""
        if not self.dragging or event.inaxes != self.ax:
            return

        if event.xdata is not None and event.ydata is not None:
            # Update centroid position
            self.centroid3 = np.array([event.xdata, event.ydata])

            # Update the plot
            self.centroid_plots[2].set_offsets([self.centroid3])
            self.update_statistics()
            self.fig.canvas.draw_idle()

    def on_mouse_release(self, event):
        """Handle mouse release events"""
        self.dragging = False
        self.drag_centroid = None

    def on_recalc_clicked(self, event):
        """Handle Re Calc button click - regenerate Group 3 data around current centroid"""
        # Generate new random data for Group 3 centered around the current centroid
        # Using Gaussian distribution with the centroid as mean
        std_dev = 2.0  # Standard deviation for the distribution

        # Generate X and Y coordinates separately
        x_coords = np.random.normal(self.centroid3[0], std_dev, self.original_group3_count)
        y_coords = np.random.normal(self.centroid3[1], std_dev, self.original_group3_count)

        # Combine into data array
        self.group3_data = np.column_stack([x_coords, y_coords])

        # Keep the centroid at its current position (don't recalculate)
        # The centroid stays where the user dragged it
        # Note: The actual mean of generated data will be very close to centroid3 due to Gaussian distribution

        # Reset cluster labels
        self.cluster_labels = None

        # Update plot
        self.plot_data()
        print(f"Generated new Gaussian data for Group 3 ({self.original_group3_count} points) centered at ({self.centroid3[0]:.2f}, {self.centroid3[1]:.2f})")

    def on_kmeans_clicked(self, event):
        """Handle Implement K-Mean button click"""
        # IMPORTANT: Groups 1 and 2 remain static and never change
        # Only Group 3 points get reassigned based on K-means clustering

        # Restore original Groups 1 and 2 data (these never change)
        self.group1_data = self.original_group1_data.copy()
        self.group2_data = self.original_group2_data.copy()

        # Recalculate centroids for Groups 1 and 2 from original data
        self.centroid1 = np.mean(self.group1_data, axis=0)
        self.centroid2 = np.mean(self.group2_data, axis=0)

        # Combine all data
        all_data = np.vstack([self.group1_data, self.group2_data, self.group3_data])

        # Initial centroids (use current centroid positions)
        centroids = np.array([self.centroid1, self.centroid2, self.centroid3])

        # Run K-means
        print("Running K-means algorithm...")
        final_centroids, labels = self.kmeans(all_data, centroids, max_iterations=100)

        # Store results
        self.cluster_labels = labels

        # Find indices for each original group
        n1 = len(self.group1_data)
        n2 = len(self.group2_data)
        n3 = len(self.group3_data)

        # Get cluster assignments for Group 3 points only
        group3_start_idx = n1 + n2
        group3_end_idx = n1 + n2 + n3
        group3_labels = labels[group3_start_idx:group3_end_idx]

        # Group 3 data remains the same physical points, just need to track which cluster they belong to
        # For visualization purposes, we'll keep the data in group3_data
        # The statistics will still be based on the original groups

        # Update centroids based on K-means result
        self.centroid3 = final_centroids[2]

        # Update plot
        self.plot_data()

        # Count how many Group 3 points were assigned to each cluster
        group3_to_cluster0 = np.sum(group3_labels == 0)
        group3_to_cluster1 = np.sum(group3_labels == 1)
        group3_to_cluster2 = np.sum(group3_labels == 2)

        print(f"K-means completed.")
        print(f"  Group 1 (Blue): {len(self.group1_data)} points (unchanged)")
        print(f"  Group 2 (Green): {len(self.group2_data)} points (unchanged)")
        print(f"  Group 3 (Red): {len(self.group3_data)} points")
        print(f"    - {group3_to_cluster0} assigned to Cluster 0")
        print(f"    - {group3_to_cluster1} assigned to Cluster 1")
        print(f"    - {group3_to_cluster2} assigned to Cluster 2")

    def kmeans(self, data, initial_centroids, max_iterations=100, tolerance=0.001):
        """
        K-means clustering algorithm

        Args:
            data: numpy array of shape (n_samples, n_features)
            initial_centroids: numpy array of shape (k, n_features)
            max_iterations: maximum number of iterations
            tolerance: convergence tolerance

        Returns:
            final_centroids: numpy array of shape (k, n_features)
            labels: numpy array of shape (n_samples,)
        """
        centroids = initial_centroids.copy()
        k = len(centroids)

        for iteration in range(max_iterations):
            # Assign each point to nearest centroid
            distances = np.zeros((len(data), k))
            for i in range(k):
                distances[:, i] = np.linalg.norm(data - centroids[i], axis=1)

            labels = np.argmin(distances, axis=1)

            # Calculate new centroids
            new_centroids = np.zeros_like(centroids)
            for i in range(k):
                cluster_points = data[labels == i]
                if len(cluster_points) > 0:
                    new_centroids[i] = np.mean(cluster_points, axis=0)
                else:
                    # If cluster is empty, keep the old centroid
                    new_centroids[i] = centroids[i]

            # Check for convergence
            centroid_shift = np.linalg.norm(new_centroids - centroids)
            centroids = new_centroids

            if centroid_shift < tolerance:
                print(f"Converged after {iteration + 1} iterations")
                break

        return centroids, labels

    def run(self):
        """Run the application"""
        print("Loading data...")
        self.load_data()

        print("Setting up visualization...")
        self.setup_plot()
        self.plot_data()

        print("Application ready!")
        print("\nInstructions:")
        print("- Drag the red centroid (Centroid 3) to move it to a new location")
        print("- Click 'Re Calc' to generate new Gaussian data for Group 3 centered around the current centroid position")
        print("- Click 'Implement K-Mean' to run clustering algorithm on all data points")

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # Create and run the application
    app = KMeansVisualizer()
    app.run()
