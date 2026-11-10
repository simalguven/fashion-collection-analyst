"""Read-only collection queries and their LangChain tool wrappers."""

import json

import numpy as np
import pandas as pd
from langchain_core.tools import tool

from .colours import REFERENCE_COLOURS
from .config import ARTIFACT_DIR
from .embeddings import ClipTextEncoder

MAX_RESULTS = 20
COLOUR_ALIASES = {"navy": "Navy Blue", "gray": "Grey", "burgundy": "Maroon", "violet": "Purple"}
ITEM_FIELDS = ["articleType", "gender", "baseColour", "season", "usage", "cluster", "colour_name"]


class CollectionError(ValueError):
    """A request the collection cannot answer; the message is shown to the agent."""


class CollectionStore:
    """In-memory view of the pipeline artifacts."""

    def __init__(self, artifact_dir=ARTIFACT_DIR) -> None:
        self.items = pd.read_parquet(artifact_dir / "items.parquet")
        self.row_by_id = {int(i): row for row, i in enumerate(self.items["id"])}
        self.clusters = {
            c["cluster_id"]: c for c in json.loads((artifact_dir / "clusters.json").read_text())
        }
        self.visual = np.load(artifact_dir / "embeddings_dinov2.npy")
        self.clip = np.load(artifact_dir / "embeddings_clip.npy")
        self._text_encoder = None

    def _cluster(self, cluster_id: int) -> dict:
        if cluster_id not in self.clusters:
            raise CollectionError(
                f"Cluster {cluster_id} does not exist. Valid clusters are 0 to {len(self.clusters) - 1}."
            )
        return self.clusters[cluster_id]

    def _row(self, item_id: int) -> int:
        if item_id not in self.row_by_id:
            raise CollectionError(f"Item {item_id} is not in this collection sample.")
        return self.row_by_id[item_id]

    def _item_record(self, row: int, score: float | None = None) -> dict:
        item = self.items.iloc[row]
        record = {"item_id": int(item["id"]), "name": item["productDisplayName"]}
        record.update({field: item[field] for field in ITEM_FIELDS})
        record["cluster"] = int(record["cluster"])
        if score is not None:
            record["similarity"] = round(float(score), 3)
        return record

    @staticmethod
    def _check_limit(limit: int) -> None:
        if not 1 <= limit <= MAX_RESULTS:
            raise CollectionError(f"limit must be between 1 and {MAX_RESULTS}, got {limit}.")

    def list_clusters(self) -> dict:
        return {
            "total_items": int(len(self.items)),
            "clusters": [
                {"cluster_id": c["cluster_id"], "name": c["name"], "size": c["size"]}
                for c in sorted(self.clusters.values(), key=lambda c: -c["size"])
            ],
        }

    def describe_cluster(self, cluster_id: int) -> dict:
        cluster = self._cluster(cluster_id)
        fields = ("cluster_id", "name", "size", "top_article_types", "gender", "usage", "season")
        return {field: cluster[field] for field in fields}

    def cluster_colours(self, cluster_id: int) -> dict:
        cluster = self._cluster(cluster_id)
        return {"cluster_id": cluster_id, "name": cluster["name"], "palette": cluster["palette"]}

    def resolve_colour(self, colour: str) -> str:
        cleaned = colour.strip().lower()
        canonical = {name.lower(): name for name in REFERENCE_COLOURS}
        if cleaned in canonical:
            return canonical[cleaned]
        if cleaned in COLOUR_ALIASES:
            return COLOUR_ALIASES[cleaned]
        raise CollectionError(
            f"Colour '{colour}' is not tracked. Tracked colours: {', '.join(REFERENCE_COLOURS)}."
        )

    def clusters_by_colour(self, colour: str) -> dict:
        name = self.resolve_colour(colour)
        shares = self.items["colour_name"].eq(name).groupby(self.items["cluster"]).mean()
        ranked = shares.sort_values(ascending=False, kind="stable").head(5)
        return {
            "colour": name,
            "clusters_ranked_by_item_share": [
                {"cluster_id": int(c), "name": self.clusters[int(c)]["name"], "item_share": round(float(s), 3)}
                for c, s in ranked.items()
            ],
        }

    def get_item(self, item_id: int) -> dict:
        return self._item_record(self._row(item_id))

    def similar_items(self, item_id: int, limit: int = 5) -> dict:
        self._check_limit(limit)
        row = self._row(item_id)
        scores = self.visual @ self.visual[row]
        scores[row] = -np.inf
        top = np.argsort(-scores)[:limit]
        return {"query_item": item_id, "results": [self._item_record(i, scores[i]) for i in top]}

    def search_by_text(self, query: str, limit: int = 5) -> dict:
        self._check_limit(limit)
        if not query.strip():
            raise CollectionError("The search query is empty.")
        if self._text_encoder is None:
            self._text_encoder = ClipTextEncoder()
        scores = self.clip @ self._text_encoder.encode(query)
        top = np.argsort(-scores)[:limit]
        return {"query": query, "results": [self._item_record(i, scores[i]) for i in top]}


def _respond(function, *args) -> str:
    """Return tool results as JSON, turning known failures into readable errors."""
    try:
        return json.dumps(function(*args), default=str)
    except CollectionError as error:
        return json.dumps({"error": str(error)})


def build_tools(store: CollectionStore) -> list:
    @tool
    def list_clusters() -> str:
        """List every style cluster with its id, name and item count, largest first."""
        return _respond(store.list_clusters)

    @tool
    def describe_cluster(cluster_id: int) -> str:
        """Describe the product types, gender, usage and seasons found in one cluster."""
        return _respond(store.describe_cluster, cluster_id)

    @tool
    def cluster_colours(cluster_id: int) -> str:
        """Return one cluster's main colours, ordered by the share of its items in each colour."""
        return _respond(store.cluster_colours, cluster_id)

    @tool
    def find_clusters_by_colour(colour: str) -> str:
        """Rank clusters by the share of their items whose main colour is `colour`."""
        return _respond(store.clusters_by_colour, colour)

    @tool
    def get_item(item_id: int) -> str:
        """Look up one item: name, catalogue labels, cluster and detected colour (colour_name)."""
        return _respond(store.get_item, item_id)

    @tool
    def find_similar_items(item_id: int, limit: int = 5) -> str:
        """Find items that look most like item `item_id`. `limit` must be 1 to 20."""
        return _respond(store.similar_items, item_id, limit)

    @tool
    def search_items_by_text(query: str, limit: int = 5) -> str:
        """Find items matching a visual description in plain English. `limit` must be 1 to 20."""
        return _respond(store.search_by_text, query, limit)

    return [
        list_clusters,
        describe_cluster,
        cluster_colours,
        find_clusters_by_colour,
        get_item,
        find_similar_items,
        search_items_by_text,
    ]
