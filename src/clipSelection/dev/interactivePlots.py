import numpy as np
import umap.umap_ as umap
import plotly.graph_objs as go

def create_interactive_umap_plot(valid_image_paths, embeddings, dimension='2d',
                                 cluster_labels=None,
                                 highlight_indices=None,
                                 output_file='umap_plot.html'):
    """
    Create an interactive UMAP visualization with cluster-based coloring.
    Points corresponding to highlight_indices will have a black outline.

    Parameters
    ----------
    valid_image_paths : list of pathlib.Path
        Paths to the images corresponding to each embedding.
    embeddings : np.ndarray
        Array of embeddings with shape (N, D).
    dimension : str
        '2d' or '3d'. Determines whether to do a 2D or 3D UMAP projection.
    cluster_labels : array-like or None
        Cluster labels for each embedding. Should be numeric if you want a continuous colorscale.
        If None, all points will be the same color.
    highlight_indices : set or list
        A set of indices that should be highlighted with a border.
    output_file : str
        HTML file to write the interactive plot to.
    """
    if highlight_indices is None:
        highlight_indices = set()

    # Run UMAP
    n_components = 2 if dimension == '2d' else 3
    reducer = umap.UMAP(n_components=n_components, random_state=42)
    embedding_2d_or_3d = reducer.fit_transform(embeddings)

    # Extract coordinates
    x = embedding_2d_or_3d[:, 0]
    y = embedding_2d_or_3d[:, 1]
    z = embedding_2d_or_3d[:, 2] if dimension == '3d' else None

    # Determine hover text
    if cluster_labels is not None:
        hover_text = [f"Cluster: {cl}" for cl in cluster_labels]
    else:
        hover_text = [f"Point {i}" for i in range(len(valid_image_paths))]

    # Marker outlines for highlighted points
    marker_line_color = ['black' if i in highlight_indices else 'rgba(0,0,0,0)'
                         for i in range(len(valid_image_paths))]
    marker_line_width = [2 if i in highlight_indices else 0
                         for i in range(len(valid_image_paths))]

    # Set color arguments
    if cluster_labels is not None:
        # Ensure cluster_labels is numeric for a continuous colorscale
        cluster_labels = np.array(cluster_labels, dtype=float)
        color_kwargs = dict(
            mode='markers',
            hoverinfo='text',
            text=hover_text,
            marker=dict(
                size=6,
                line=dict(color=marker_line_color, width=marker_line_width),
                color=cluster_labels,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title='Cluster')
            )
        )
    else:
        # Single color for all points
        color_kwargs = dict(
            mode='markers',
            hoverinfo='text',
            text=hover_text,
            marker=dict(
                size=6,
                line=dict(color=marker_line_color, width=marker_line_width),
                color='rgba(100,100,100,0.7)'
            )
        )

    # Create the figure
    if dimension == '2d':
        trace = go.Scattergl(x=x, y=y, **color_kwargs)
        fig = go.Figure(data=[trace])
        fig.update_layout(
            title='2D UMAP Embeddings',
            xaxis_title='UMAP 1',
            yaxis_title='UMAP 2',
            hovermode='closest'
        )
    else:
        trace = go.Scatter3d(x=x, y=y, z=z, **color_kwargs)
        fig = go.Figure(data=[trace])
        fig.update_layout(
            title='3D UMAP Embeddings',
            scene=dict(
                xaxis_title='UMAP 1',
                yaxis_title='UMAP 2',
                zaxis_title='UMAP 3'
            ),
            hovermode='closest'
        )

    # Save the figure to an HTML file
    fig.write_html(output_file)
    print(f"Interactive UMAP plot saved to {output_file}. Open this file in a web browser to view.")

