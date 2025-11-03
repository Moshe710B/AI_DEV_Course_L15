# Product Requirements Document (PRD)
## K-Means Clustering Visualization Application

---

### 1. Overview

**Product Name:** K-Means Interactive Visualization Tool

**Version:** 1.0

**Date:** November 3, 2025

**Purpose:** An interactive educational application that demonstrates K-means clustering algorithm behavior by allowing users to manipulate data points and observe real-time clustering results.

---

### 2. Objectives

- Provide an intuitive visual demonstration of K-means clustering algorithm
- Enable interactive manipulation of cluster centroids
- Allow dynamic data generation and re-clustering
- Display statistical information (mean, variance, centroids) for each cluster
- Serve as an educational tool for understanding clustering behavior

---

### 3. Data Specifications

**Input Data:**
- Three CSV files containing Gaussian-distributed data groups:
  - `gaussian_group1_data.csv` (Static - Read Only)
  - `gaussian_group2_data.csv` (Static - Read Only)
  - `gaussian_group3_data.csv` (Dynamic - Can be regenerated)

**Data Structure:**
- Each CSV contains columns: `Group`, `Sample_Index`, `X_Coordinate`, `Y_Coordinate`
- Data points are 2D coordinates (X, Y) suitable for scatter plot visualization

---

### 4. Functional Requirements

#### 4.1 Initial Visualization (MVP Feature)

**FR-001: Display All Data Points**
- **Description:** Display all points from the three groups on a 2D scatter plot
- **Acceptance Criteria:**
  - Each group is displayed with a distinct color
  - Group 1: Color #1 (e.g., Blue)
  - Group 2: Color #2 (e.g., Green)
  - Group 3: Color #3 (e.g., Red)
  - All points are clearly visible and distinguishable
  - Axes are properly labeled (X and Y)

**FR-002: Display Centroids**
- **Description:** Calculate and display the centroid for each group
- **Acceptance Criteria:**
  - Centroids are displayed as distinct markers (e.g., large stars or crosses)
  - Centroid markers use the same color scheme as their respective groups
  - Centroids are clearly distinguishable from regular data points

**FR-003: Display Statistics Panel**
- **Description:** Show statistical information for each group
- **Acceptance Criteria:**
  - Display mean (μ) for X and Y coordinates per group
  - Display variance (σ²) for X and Y coordinates per group
  - Statistics are updated in real-time when data changes
  - Format:
    ```
    Group 1:
      Mean: (μx, μy)
      Variance: (σ²x, σ²y)
    Group 2:
      Mean: (μx, μy)
      Variance: (σ²x, σ²y)
    Group 3:
      Mean: (μx, μy)
      Variance: (σ²x, σ²y)
    ```

---

#### 4.2 Interactive Centroid Manipulation

**FR-004: Draggable Group 3 Centroid**
- **Description:** Enable users to click and drag the centroid of Group 3 to any position on the graph
- **Acceptance Criteria:**
  - Only Group 3's centroid is draggable (Groups 1 and 2 remain static)
  - Cursor changes to indicate draggable element on hover
  - Centroid position updates smoothly during drag operation
  - New centroid position is immediately reflected in the statistics panel
  - Centroid can be placed anywhere within the visible graph area

**Technical Notes:**
- This feature simulates manual initialization of a cluster center
- The dragged position will serve as the initial centroid for Group 3 when K-means is executed

---

#### 4.3 Data Regeneration

**FR-005: "Re Calc" Button**
- **Description:** Generate new random data points for Group 3 centered around the current centroid position
- **Acceptance Criteria:**
  - Button labeled "Re Calc" is prominently displayed on the interface
  - On click, generates new random data points for Group 3 only
  - New data points are generated as a Gaussian distribution centered around the current Group 3 centroid position
  - If the centroid has been dragged to a new location, data is generated around that new position
  - Number of generated points should match the original Group 3 data count
  - Groups 1 and 2 remain unchanged
  - Graph automatically updates to show new Group 3 points
  - Statistics panel updates to reflect new Group 3 statistics
  - Group 3 centroid position remains at the dragged location (or recalculated mean if not dragged)

**Data Generation Specifications:**
- Distribution: Gaussian (normal) distribution centered on current Group 3 centroid
- Standard deviation: 2.0 for both X and Y coordinates (configurable)
- This creates a cluster of points around the centroid position
- Default point count: Same as original `gaussian_group3_data.csv` (or configurable)
- The centroid acts as the mean of the distribution (μx, μy)

---

#### 4.4 K-Means Implementation

**FR-006: "Implement K-Mean" Button**
- **Description:** Execute K-means clustering algorithm on all data points using current centroids as initialization
- **Acceptance Criteria:**
  - Button labeled "Implement K-Mean" is prominently displayed
  - Uses 3 clusters (K=3)
  - Initial centroids are:
    - Current centroid of Group 1 (always calculated from original Group 1 data)
    - Current centroid of Group 2 (always calculated from original Group 2 data)
    - Current centroid of Group 3 (may be manually positioned or from regenerated data)
  - **IMPORTANT: Groups 1 and 2 remain static (read-only)**
    - The original data points in Groups 1 and 2 NEVER change
    - Their mean and variance remain constant across all operations
    - Only Group 3 data can be modified or reassigned
  - Algorithm runs K-means on all data points to find optimal cluster assignments
  - Only Group 3 points may be reassigned to different clusters
  - Centroids are recalculated based on final cluster assignments
  - Process iterates until convergence or max iterations reached
  - Statistics panel always shows statistics for the original Groups 1 and 2
  - Statistics for Group 3 update based on its current data

**FR-007: Visual Feedback During K-Means**
- **Description:** Provide visual indication of clustering process
- **Acceptance Criteria:**
  - Option 1: Animate the clustering process showing each iteration
  - Option 2: Show loading indicator during calculation
  - Final result clearly shows new cluster assignments with updated colors
  - User can distinguish which points have changed clusters

**K-Means Algorithm Specifications:**
- Algorithm: Standard K-means (Lloyd's algorithm)
- K = 3 (three clusters)
- Distance metric: Euclidean distance
- Max iterations: 100 (or until convergence)
- Convergence criteria: Centroids move less than 0.001 units

---

### 5. User Interface Requirements

#### 5.1 Layout

**UI-001: Main Layout Structure**
- **Left/Center Panel:** Interactive scatter plot (70-80% of width)
- **Right Panel:** Statistics display (20-30% of width)
- **Bottom/Top Bar:** Control buttons

**UI-002: Graph Requirements**
- Grid lines for easier coordinate reading
- Axis labels and title
- Legend showing group/cluster colors
- Zoom and pan capabilities (optional but recommended)
- Responsive to window resizing

**UI-003: Control Buttons**
- "Re Calc" button: Primary action style
- "Implement K-Mean" button: Primary action style
- Buttons should be clearly labeled and easily accessible
- Tooltips on hover explaining each button's function

---

### 6. Non-Functional Requirements

**NFR-001: Performance**
- Graph should render within 1 second for datasets up to 10,000 points
- K-means calculation should complete within 2 seconds
- UI interactions (dragging centroid) should feel smooth (60 fps)

**NFR-002: Usability**
- Interface should be intuitive without requiring extensive documentation
- Error messages should be clear and actionable
- Controls should follow standard UI conventions

**NFR-003: Compatibility**
- Application should run on Windows, macOS, and Linux
- Support modern web browsers (if web-based) or desktop environments

**NFR-004: Maintainability**
- Code should be well-documented
- Modular architecture allowing easy feature additions
- Clear separation between data processing and visualization logic

---

### 7. Technical Stack (Selected Solution)

**Python Desktop Application with Matplotlib**

This is the simplest and most effective solution for this PC, requiring minimal setup and leveraging Python's excellent scientific computing ecosystem.

**Core Components:**
- **Language:** Python 3.8+
- **Visualization:** Matplotlib with interactive backend (matplotlib.pyplot)
- **GUI Framework:** Matplotlib's built-in event handling (no additional GUI framework needed)
- **Data Processing:**
  - NumPy (for numerical operations and K-means algorithm)
  - Pandas (for CSV file reading and data manipulation)

**Required Libraries:**
```bash
pip install matplotlib numpy pandas
```

**Why This Solution:**
1. **Simplicity:** All libraries are standard in data science/ML workflows
2. **Minimal Setup:** No complex framework installation required
3. **Interactive Features:** Matplotlib supports mouse events for dragging centroids
4. **Fast Development:** Matplotlib can handle buttons, updates, and real-time visualization
5. **Self-Contained:** Single Python script can run the entire application
6. **Performance:** Efficient for datasets with thousands of points
7. **Windows Compatible:** Works seamlessly on Windows with standard Python installation

**Architecture:**
- Single Python file (`kmeans_visualizer.py`) containing:
  - Data loading functions (Pandas)
  - K-means algorithm implementation (NumPy)
  - Visualization and interaction logic (Matplotlib)
  - Event handlers for buttons and dragging

**Matplotlib Interactive Features Used:**
- `plt.scatter()` - Display data points
- `plt.plot()` - Display centroids
- `plt.text()` - Display statistics
- `Button` widget - For "Re Calc" and "Implement K-Mean" buttons
- `pick_event` - For detecting centroid selection
- `motion_notify_event` - For dragging centroids
- `button_release_event` - For completing drag operation
- `plt.draw()` - For real-time updates

---

### 8. User Stories

**US-001:** As a student, I want to see how K-means clustering works visually, so that I can better understand the algorithm.

**US-002:** As an educator, I want to manipulate the initial centroid position, so that I can demonstrate how initialization affects clustering results.

**US-003:** As a user, I want to generate new random data, so that I can experiment with different data distributions.

**US-004:** As a user, I want to see statistics for each cluster, so that I can understand the mathematical properties of the groups.

**US-005:** As a user, I want to run K-means on mixed data, so that I can see how the algorithm separates overlapping groups.

---

### 9. Success Metrics

- Application successfully loads and displays all three data groups
- Centroid dragging works smoothly with <100ms latency
- Data regeneration completes in <1 second
- K-means algorithm successfully converges and displays results
- Users can complete a full workflow (view → drag → regenerate → cluster) without errors

---

### 10. Future Enhancements (Out of Scope for v1.0)

- **FE-001:** Support for variable K (number of clusters)
- **FE-002:** Multiple initialization methods (K-means++, random)
- **FE-003:** Export clustered data to CSV
- **FE-004:** Load custom datasets
- **FE-005:** 3D visualization support
- **FE-006:** Comparison mode showing before/after clustering
- **FE-007:** Animation showing iteration-by-iteration clustering
- **FE-008:** Elbow method visualization for optimal K selection
- **FE-009:** Support for other clustering algorithms (DBSCAN, Hierarchical)

---

### 11. Assumptions and Constraints

**Assumptions:**
- Users have basic understanding of clustering concepts
- Data files are properly formatted CSV files
- System has sufficient memory to load all data points

**Constraints:**
- Group 1 and Group 2 data remain static (read-only)
- Only Group 3 can be regenerated
- Only Group 3's centroid can be manually repositioned
- K is fixed at 3 clusters
- 2D visualization only (X, Y coordinates)

---

### 12. Risks and Mitigations

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Large datasets cause performance issues | High | Medium | Implement data sampling or pagination |
| K-means doesn't converge | Medium | Low | Set max iterations limit (100) |
| UI becomes unresponsive during calculation | Medium | Medium | Implement async processing with progress indicator |
| Browser compatibility issues (if web-based) | Low | Low | Test on major browsers, use standard libraries |

---

### 13. Acceptance Criteria Summary

The application will be considered complete when:
1. All three data groups are displayed with distinct colors
2. Centroids and statistics are correctly calculated and displayed
3. Group 3 centroid can be dragged to any position on the graph
4. "Re Calc" button generates new random data for Group 3 (range: -10 to +10)
5. "Implement K-Mean" button successfully re-clusters all data points
6. Visual feedback clearly shows cluster assignments
7. Statistics update correctly after all operations
8. Application runs without crashes or errors during normal use

---

### 14. Glossary

- **K-means:** A clustering algorithm that partitions data into K clusters
- **Centroid:** The center point of a cluster (mean of all points in the cluster)
- **Cluster:** A group of data points that are similar to each other
- **Convergence:** When the algorithm reaches a stable state (centroids stop moving)
- **Variance:** A measure of how spread out the data points are

---

**Document Status:** Draft v1.0
**Author:** Product Team
**Reviewed By:** [Pending]
**Approved By:** [Pending]
