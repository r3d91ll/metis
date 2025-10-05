#!/usr/bin/env python3
"""Create visualizations for case study analysis."""

import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from umap import UMAP

from utils import load_config, setup_logging


def load_embeddings_from_db(db, collection_name: str, logger) -> tuple[list, list, list]:
    """Load embeddings from ArangoDB.

    Args:
        db: Database client
        collection_name: Collection name
        logger: Logger instance

    Returns:
        Tuple of (embeddings, labels, metadata)
    """
    collection = db.collection(collection_name)
    documents = collection.all()

    embeddings = []
    labels = []
    metadata = []

    for doc in documents:
        # Check if this is a paper document
        if "sections" in doc:
            # Add each section as a separate point
            for section_name, section_data in doc["sections"].items():
                if "embedding" in section_data and section_data["embedding"]:
                    embeddings.append(section_data["embedding"])
                    labels.append(f"{doc['paper_name']}: {section_name}")
                    metadata.append(
                        {
                            "type": "paper_section",
                            "paper": doc["paper_name"],
                            "section": section_name,
                            "title": doc["title"],
                        }
                    )
        elif "embedding" in doc:
            # Boundary object
            embeddings.append(doc["embedding"])
            labels.append(f"{doc['paper_name']}: {doc['type']}")
            metadata.append(
                {
                    "type": "boundary_object",
                    "paper": doc["paper_name"],
                    "object_type": doc["type"],
                    "source": doc["source"],
                }
            )

    logger.info(f"Loaded {len(embeddings)} embeddings from database")
    return embeddings, labels, metadata


def reduce_dimensions(embeddings: list, logger, n_components: int = 2, random_state: int = 42) -> np.ndarray:
    """Reduce embedding dimensions using UMAP.

    Args:
        embeddings: List of embedding vectors
        logger: Logger instance
        n_components: Number of dimensions to reduce to
        random_state: Random seed

    Returns:
        Reduced embeddings
    """
    logger.info(f"Reducing {len(embeddings)} embeddings to {n_components}D using UMAP")

    embeddings_array = np.array(embeddings)

    reducer = UMAP(n_components=n_components, random_state=random_state, n_neighbors=15, min_dist=0.1)
    reduced = reducer.fit_transform(embeddings_array)

    logger.info(f"Reduction complete: {reduced.shape}")
    return reduced


def create_paper_sections_map(
    reduced_embeddings: np.ndarray, labels: list, metadata: list, output_path: Path, logger
):
    """Create semantic map of paper sections only.

    Args:
        reduced_embeddings: 2D reduced embeddings
        labels: Point labels
        metadata: Point metadata
        output_path: Output file path
        logger: Logger instance
    """
    logger.info("Creating paper sections semantic map")

    # Filter to paper sections only
    indices = [i for i, meta in enumerate(metadata) if meta["type"] == "paper_section"]

    if not indices:
        logger.warning("No paper sections found")
        return

    coords = reduced_embeddings[indices]
    section_metadata = [metadata[i] for i in indices]

    # Create color mapping by paper
    papers = list(set(meta["paper"] for meta in section_metadata))
    color_map = {paper: i for i, paper in enumerate(papers)}

    # Create hover text
    hover_text = [
        f"<b>{meta['title']}</b><br>"
        f"Section: {meta['section']}<br>"
        f"Paper: {meta['paper']}"
        for meta in section_metadata
    ]

    # Create plot
    fig = go.Figure()

    for paper in papers:
        paper_indices = [i for i, meta in enumerate(section_metadata) if meta["paper"] == paper]
        paper_coords = coords[paper_indices]
        paper_hover = [hover_text[i] for i in paper_indices]

        fig.add_trace(
            go.Scatter(
                x=paper_coords[:, 0],
                y=paper_coords[:, 1],
                mode="markers+text",
                name=paper.capitalize(),
                text=[section_metadata[i]["section"][:10] for i in paper_indices],
                textposition="top center",
                hovertext=paper_hover,
                hoverinfo="text",
                marker=dict(size=12, line=dict(width=2, color="white")),
            )
        )

    fig.update_layout(
        title="Semantic Map: Paper Sections",
        xaxis_title="UMAP Dimension 1",
        yaxis_title="UMAP Dimension 2",
        hovermode="closest",
        width=1200,
        height=800,
        template="plotly_white",
    )

    # Save
    fig.write_html(str(output_path))
    logger.info(f"Saved paper sections map to {output_path}")

    # Also save as PNG
    png_path = output_path.with_suffix(".png")
    fig.write_image(str(png_path))
    logger.info(f"Saved PNG to {png_path}")


def create_full_ecosystem_map(
    reduced_embeddings: np.ndarray, metadata: list, output_path: Path, logger
):
    """Create full ecosystem map with papers and boundary objects.

    Args:
        reduced_embeddings: 2D reduced embeddings
        labels: Point labels
        metadata: Point metadata
        output_path: Output file path
        logger: Logger instance
    """
    logger.info("Creating full ecosystem semantic map")

    # Create hover text
    hover_text = []
    for meta in metadata:
        if meta["type"] == "paper_section":
            text = (
                f"<b>{meta['title']}</b><br>"
                f"Section: {meta['section']}<br>"
                f"Paper: {meta['paper']}"
            )
        else:
            text = (
                f"<b>Boundary Object</b><br>"
                f"Type: {meta['object_type']}<br>"
                f"Source: {meta['source']}<br>"
                f"Paper: {meta['paper']}"
            )
        hover_text.append(text)

    # Create color mapping
    type_map = {}
    color_idx = 0
    for meta in metadata:
        if meta["type"] == "paper_section":
            key = f"paper_{meta['paper']}"
        else:
            key = f"bo_{meta['paper']}"

        if key not in type_map:
            type_map[key] = color_idx
            color_idx += 1

    colors = []
    for meta in metadata:
        if meta["type"] == "paper_section":
            key = f"paper_{meta['paper']}"
        else:
            key = f"bo_{meta['paper']}"
        colors.append(type_map[key])

    # Create plot
    fig = go.Figure()

    # Group by type and paper
    for paper in set(meta["paper"] for meta in metadata):
        # Paper sections
        paper_indices = [
            i
            for i, meta in enumerate(metadata)
            if meta["paper"] == paper and meta["type"] == "paper_section"
        ]
        if paper_indices:
            coords = reduced_embeddings[paper_indices]
            hover = [hover_text[i] for i in paper_indices]

            fig.add_trace(
                go.Scatter(
                    x=coords[:, 0],
                    y=coords[:, 1],
                    mode="markers",
                    name=f"{paper.capitalize()} (Paper)",
                    hovertext=hover,
                    hoverinfo="text",
                    marker=dict(size=15, symbol="circle", line=dict(width=2, color="white")),
                )
            )

        # Boundary objects
        bo_indices = [
            i
            for i, meta in enumerate(metadata)
            if meta["paper"] == paper and meta["type"] == "boundary_object"
        ]
        if bo_indices:
            coords = reduced_embeddings[bo_indices]
            hover = [hover_text[i] for i in bo_indices]

            fig.add_trace(
                go.Scatter(
                    x=coords[:, 0],
                    y=coords[:, 1],
                    mode="markers",
                    name=f"{paper.capitalize()} (Docs)",
                    hovertext=hover,
                    hoverinfo="text",
                    marker=dict(size=10, symbol="diamond", line=dict(width=1, color="white")),
                )
            )

    fig.update_layout(
        title="Semantic Map: Full Ecosystem (Papers + Boundary Objects)",
        xaxis_title="UMAP Dimension 1",
        yaxis_title="UMAP Dimension 2",
        hovermode="closest",
        width=1400,
        height=900,
        template="plotly_white",
    )

    # Save
    fig.write_html(str(output_path))
    logger.info(f"Saved full ecosystem map to {output_path}")

    # Also save as PNG
    png_path = output_path.with_suffix(".png")
    fig.write_image(str(png_path))
    logger.info(f"Saved PNG to {png_path}")


def create_citation_timeline_plot(config: dict, output_path: Path, logger):
    """Create citation timeline comparison plot.

    Args:
        config: Configuration
        output_path: Output file path
        logger: Logger instance
    """
    logger.info("Creating citation timeline plot")

    citations_dir = Path("data/case_study/citations")

    # Load citation data
    citation_data = {}
    for paper_name in config["papers"].keys():
        citation_file = citations_dir / f"{paper_name}_citations.json"
        if citation_file.exists():
            with open(citation_file) as f:
                citation_data[paper_name] = json.load(f)

    if not citation_data:
        logger.warning("No citation data found")
        return

    # Create plot
    fig = go.Figure()

    for paper_name, data in citation_data.items():
        monthly_data = data["monthly_citations"]

        # Convert to dates and cumulative counts
        dates = [f"{m['year']}-{m['month']:02d}-01" for m in monthly_data]
        counts = [m["count"] for m in monthly_data]
        cumulative = np.cumsum(counts)

        fig.add_trace(
            go.Scatter(
                x=dates,
                y=cumulative,
                mode="lines+markers",
                name=paper_name.capitalize(),
                line=dict(width=3),
                marker=dict(size=6),
            )
        )

    fig.update_layout(
        title="Citation Timeline Comparison",
        xaxis_title="Date",
        yaxis_title="Cumulative Citations",
        hovermode="x unified",
        width=1200,
        height=600,
        template="plotly_white",
        showlegend=True,
    )

    # Save
    fig.write_html(str(output_path))
    logger.info(f"Saved citation timeline to {output_path}")

    png_path = output_path.with_suffix(".png")
    fig.write_image(str(png_path))
    logger.info(f"Saved PNG to {png_path}")


def create_repository_comparison_plot(config: dict, output_path: Path, logger):
    """Create repository statistics comparison plot.

    Args:
        config: Configuration
        output_path: Output file path
        logger: Logger instance
    """
    logger.info("Creating repository comparison plot")

    repos_dir = Path("data/case_study/implementations")

    # Load repository data
    repo_data = {}
    for paper_name in config["papers"].keys():
        repo_file = repos_dir / f"{paper_name}_repos.json"
        if repo_file.exists():
            with open(repo_file) as f:
                repo_data[paper_name] = json.load(f)

    if not repo_data:
        logger.warning("No repository data found")
        return

    # Create subplots
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Repository Count by Type", "Total Stars Comparison"),
    )

    # Type distribution
    for paper_name, data in repo_data.items():
        types = list(data["type_counts"].keys())
        counts = list(data["type_counts"].values())

        fig.add_trace(
            go.Bar(name=paper_name.capitalize(), x=types, y=counts), row=1, col=1
        )

    # Total stars
    papers = []
    total_stars = []
    for paper_name, data in repo_data.items():
        papers.append(paper_name.capitalize())
        stars = sum(repo["stars"] for repo in data["repositories"])
        total_stars.append(stars)

    fig.add_trace(go.Bar(x=papers, y=total_stars, showlegend=False), row=1, col=2)

    fig.update_layout(
        height=500,
        width=1200,
        title_text="GitHub Repository Comparison",
        template="plotly_white",
    )

    # Save
    fig.write_html(str(output_path))
    logger.info(f"Saved repository comparison to {output_path}")

    png_path = output_path.with_suffix(".png")
    fig.write_image(str(png_path))
    logger.info(f"Saved PNG to {png_path}")


def main():
    """Main entry point."""
    # Setup
    logger = setup_logging()
    config = load_config()

    # Output directory
    output_dir = Path("data/case_study/visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize database
    try:
        from metis.database import ArangoDBClient

        logger.info("Connecting to ArangoDB")
        import os
        socket_path = os.getenv("ARANGODB_SOCKET", "/run/hades/readwrite/arangod.sock")
        db = ArangoDBClient(socket_path=socket_path)

    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        raise

    # Load embeddings
    collection_name = config["database"]["collection_name"]
    embeddings, labels, metadata = load_embeddings_from_db(db, collection_name, logger)

    if embeddings:
        # Reduce dimensions
        reduced = reduce_dimensions(embeddings, logger)

        # Save coordinates
        coords_file = output_dir / "embedding_coordinates.json"
        coords_data = {
            "coordinates": reduced.tolist(),
            "labels": labels,
            "metadata": metadata,
        }
        with open(coords_file, "w") as f:
            json.dump(coords_data, f, indent=2)
        logger.info(f"Saved coordinates to {coords_file}")

        # Create visualizations
        logger.info("\n" + "=" * 60)
        logger.info("Creating semantic maps")
        logger.info("=" * 60)

        create_paper_sections_map(
            reduced, labels, metadata, output_dir / "semantic_map_papers.html", logger
        )

        create_full_ecosystem_map(
            reduced, metadata, output_dir / "full_ecosystem.html", logger
        )

    # Create timeline plots
    logger.info("\n" + "=" * 60)
    logger.info("Creating timeline plots")
    logger.info("=" * 60)

    create_citation_timeline_plot(config, output_dir / "citation_timeline.html", logger)

    create_repository_comparison_plot(config, output_dir / "repository_comparison.html", logger)

    logger.info("\n" + "=" * 60)
    logger.info("Visualization generation complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
