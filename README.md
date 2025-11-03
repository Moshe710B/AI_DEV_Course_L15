# K-Means Clustering Visualization

An interactive Python application for visualizing and understanding the K-means clustering algorithm through hands-on experimentation.

## Overview

This educational tool demonstrates how K-means clustering works by allowing you to:
- Visualize three groups of data points in 2D space
- Manually reposition cluster centroids
- Generate new data distributions
- Run the K-means algorithm and observe the results

## Features

- **Interactive Visualization**: Real-time scatter plot showing all data points and centroids
- **Draggable Centroids**: Click and drag the red centroid (Group 3) to any position
- **Dynamic Data Generation**: Generate new Gaussian-distributed data centered around the current centroid position
- **K-Means Algorithm**: Execute clustering algorithm with customizable initial positions
- **Statistics Display**: View mean, variance, and centroid positions for each group
- **Static Reference Groups**: Groups 1 and 2 remain unchanged, providing consistent reference points

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone or download this repository to your local machine

2. Navigate to the project directory:
```bash
cd "L15 - K Mean"
```

3. Install required dependencies:
```bash
pip install -r requirements.txt
```

The application requires:
- `matplotlib` - For interactive visualization
- `numpy` - For numerical computations and K-means algorithm
- `pandas` - For CSV data loading

## Usage

### Running the Application

Execute the main Python script:

```bash
python kmeans_visualizer.py
```

### How to Use

The application window will display:
- **Left Panel**: Interactive scatter plot with data points and centroids
- **Right Panel**: Statistics showing mean, variance, and centroid positions
- **Buttons**: "Re Calc" and "Implement K-Mean"

### Workflow

1. **Initial View (First Stage)**
   - The application loads three groups of Gaussian-distributed data from CSV files
   - Each group is displayed in a different color (Blue, Green, Red)
   - Centroids are shown as large star markers

![First Stage](First%20Stage.jpg)
*First Stage: Initial data with three distinct groups*

2. **Drag Centroid 3**
   - Click and drag the red centroid (Centroid 3) to any position on the graph
   - Statistics update in real-time as you drag

3. **Generate New Data (Second Stage)**
   - Click the **"Re Calc"** button to generate new random data for Group 3
   - New data points are generated as a Gaussian distribution centered around the current Group 3 centroid position
   - Groups 1 and 2 remain unchanged

![Second Stage](Second%20Stage%20-%20Group%203%20is%20distinct.jpg)
*Second Stage: Group 3 regenerated in a distinct location (upper right)*

4. **Run K-Means Clustering (Third Stage)**
   - Click the **"Implement K-Mean"** button to execute the clustering algorithm
   - The algorithm reassigns data points to the nearest centroid
   - Centroids are recalculated based on the new cluster assignments
   - **Important**: Groups 1 and 2 always maintain their original data and statistics

![Third Stage](Third%20Stage%20-%20Group%203%20moved%20between%201%20and%202%20and%20is%20not%20distinct.jpg)
*Third Stage: Group 3 positioned between Groups 1 and 2, showing cluster overlap*

5. **Experiment**
   - Drag the centroid to different positions
   - Click "Re Calc" to generate new data around that position
   - Click "Implement K-Mean" to see how the algorithm clusters the data
   - Observe how centroid initialization affects the final clustering

## Understanding the Results

### Statistics Panel

For each group, the statistics panel displays:
- **Mean**: The average (μx, μy) position of all points in the group
- **Variance**: The spread (σ²x, σ²y) of the data in X and Y directions
- **Centroid**: The center point used for clustering

### Key Observations

- **Groups 1 and 2** (Blue and Green):
  - Always maintain their original data from the CSV files
  - Mean and variance remain constant across all operations
  - Serve as static reference points

- **Group 3** (Red):
  - Can be regenerated with the "Re Calc" button
  - New data is generated as a Gaussian distribution around the current centroid
  - Data and statistics change based on generation and clustering

### K-Means Behavior

The visualization demonstrates:
- How initial centroid position affects clustering results
- The iterative convergence of the algorithm
- How overlapping distributions are separated
- The impact of cluster initialization on final assignments

## File Structure

```
L15 - K Mean/
│
├── kmeans_visualizer.py                    # Main application
├── requirements.txt                        # Python dependencies
├── README.md                               # This file
├── PRD_K-Means_Visualization.md           # Product Requirements Document
│
├── gaussian_group1_data.csv               # Group 1 data (2000 points)
├── gaussian_group2_data.csv               # Group 2 data (2000 points)
├── gaussian_group3_data.csv               # Group 3 data (2000 points)
│
├── First Stage.jpg                        # Screenshot: Initial view
├── Second Stage - Group 3 is distinct.jpg # Screenshot: After regenerating Group 3
└── Third Stage - Group 3 moved between... # Screenshot: After K-means clustering
```

## Algorithm Details

### K-Means Implementation

The application uses the standard K-means (Lloyd's algorithm) with:
- **K = 3** clusters
- **Distance Metric**: Euclidean distance
- **Max Iterations**: 100
- **Convergence Criteria**: Centroids move less than 0.001 units

### Data Generation

When clicking "Re Calc":
- Generates 2000 new points for Group 3
- Uses Gaussian (normal) distribution
- Mean: Current centroid position (μx, μy)
- Standard deviation: 2.0 for both X and Y coordinates

## Educational Use Cases

This tool is ideal for:
- Learning how K-means clustering works
- Understanding the importance of centroid initialization
- Visualizing the impact of data distribution on clustering
- Experimenting with different cluster configurations
- Teaching machine learning concepts in classrooms

## Technical Notes

- The application uses Matplotlib's interactive backend for real-time updates
- Mouse events are captured for centroid dragging functionality
- The visualization automatically rescales to fit all data points
- Console output provides detailed information about clustering results

## Troubleshooting

**Issue**: Application window doesn't appear
- **Solution**: Ensure you have a GUI backend installed for Matplotlib. On some systems, you may need to install `python-tk`

**Issue**: "ModuleNotFoundError" when running
- **Solution**: Install all dependencies using `pip install -r requirements.txt`

**Issue**: Dragging centroid is not smooth
- **Solution**: This may be due to system performance. The application is optimized for datasets up to 10,000 points.

**Issue**: Console shows many warnings
- **Solution**: These are typically matplotlib deprecation warnings and can be safely ignored.

## Contributing

This is an educational project. Feel free to:
- Modify the code for your learning purposes
- Experiment with different parameters
- Extend functionality (e.g., add more clusters, different algorithms)

## License

This project is provided for educational purposes. Feel free to use and modify as needed for learning and teaching.

## Acknowledgments

Built using:
- Python 3.13
- Matplotlib for visualization
- NumPy for numerical computations
- Pandas for data handling

## Version History

- **v1.0** (2025-11-03): Initial release with basic K-means visualization
- **v1.1** (2025-11-03): Updated to keep Groups 1 and 2 static during clustering
- **v1.2** (2025-11-03): Modified "Re Calc" to generate data around current centroid position

## Contact

For questions, issues, or suggestions about this educational tool, please refer to the PRD document for detailed specifications and requirements.

---

**Happy Clustering!** 🎯
