"""
In-memory vector store with cosine and Euclidean distance search.

Stores (vector, document) pairs and retrieves the k nearest neighbours
for a given query vector or query string (when an embedding function
is supplied).

No external dependencies beyond the Python standard library.
"""

import math
from typing import Optional, Any, List, Dict, Tuple


class VectorIndex:

    def __init__(self, distance_metric: str = "cosine", embedding_fn=None):
        """
        Create an empty vector index.

        Args:
            distance_metric: Similarity measure to use — 'cosine' or 'euclidean'.
            embedding_fn:    Optional callable that maps a string to a list of
                             floats. Required when passing string queries to
                             search() or when using add_document().
        """
        if distance_metric not in ["cosine", "euclidean"]:
            raise ValueError("distance_metric must be 'cosine' or 'euclidean'")

        self.vectors: List[List[float]] = []
        self.documents: List[Dict[str, Any]] = []
        self._vector_dim: Optional[int] = None
        self._distance_metric = distance_metric
        self._embedding_fn = embedding_fn

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_document(self, document: Dict[str, Any]):
        """
        Embed the document's 'content' field and store it in the index.

        Requires an embedding function to have been provided at initialisation.

        Args:
            document: Dict that must contain a 'content' key with a string value.
                      Any extra keys are stored alongside the vector and returned
                      in search results.
        """
        if not self._embedding_fn:
            raise ValueError("Embedding function not provided during initialization.")
        if not isinstance(document, dict):
            raise TypeError("Document must be a dictionary.")
        if "content" not in document:
            raise ValueError("Document dictionary must contain a 'content' key.")

        content = document["content"]
        if not isinstance(content, str):
            raise TypeError("Document 'content' must be a string.")

        # embed the raw text, then delegate to add_vector for storage
        vector = self._embedding_fn(content)
        self.add_vector(vector=vector, document=document)

    def add_vector(self, vector: List[float], document: Dict[str, Any]):
        """
        Store a pre-computed vector together with its source document.

        The first vector added fixes the expected dimension for all
        subsequent additions.

        Args:
            vector:   List of floats representing the document embedding.
            document: Dict that must contain a 'content' key.
        """
        if not isinstance(vector, list) or not all(
            isinstance(x, (int, float)) for x in vector
        ):
            raise TypeError("Vector must be a list of numbers.")
        if not isinstance(document, dict):
            raise TypeError("Document must be a dictionary.")
        if "content" not in document:
            raise ValueError("Document dictionary must contain a 'content' key.")

        # lock in the vector dimension on the first insertion
        if not self.vectors:
            self._vector_dim = len(vector)
        elif len(vector) != self._vector_dim:
            raise ValueError(
                f"Inconsistent vector dimension. Expected {self._vector_dim}, got {len(vector)}"
            )

        self.vectors.append(list(vector))
        self.documents.append(document)

    def search(self, query: Any, k: int = 1) -> List[Tuple[Dict[str, Any], float]]:
        """
        Return the k documents whose vectors are closest to the query.

        Args:
            query: Either a natural-language string (requires embedding_fn) or
                   a pre-computed list of floats matching the stored dimension.
            k:     Number of results to return (default 1).

        Returns:
            List of (document_dict, distance) tuples sorted by ascending
            distance — the most similar document comes first.
        """
        if not self.vectors:
            return []

        # accept either a raw string or a pre-computed vector
        if isinstance(query, str):
            if not self._embedding_fn:
                raise ValueError("Embedding function not provided for string query.")
            query_vector = self._embedding_fn(query)
        elif isinstance(query, list) and all(isinstance(x, (int, float)) for x in query):
            query_vector = query
        else:
            raise TypeError("Query must be either a string or a list of numbers.")

        if self._vector_dim is None:
            return []

        if len(query_vector) != self._vector_dim:
            raise ValueError(
                f"Query vector dimension mismatch. Expected {self._vector_dim}, got {len(query_vector)}"
            )

        if k <= 0:
            raise ValueError("k must be a positive integer.")

        # pick the right distance function based on the chosen metric
        dist_func = self._cosine_distance if self._distance_metric == "cosine" else self._euclidean_distance

        # compute distance to every stored vector, then sort ascending
        distances = [
            (dist_func(query_vector, stored_vector), self.documents[i])
            for i, stored_vector in enumerate(self.vectors)
        ]
        distances.sort(key=lambda item: item[0])

        return [(doc, dist) for dist, doc in distances[:k]]

    # ------------------------------------------------------------------
    # Distance functions
    # ------------------------------------------------------------------

    def _euclidean_distance(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Straight-line distance between two vectors.

        Args:
            vec1: First vector.
            vec2: Second vector (must have the same length as vec1).

        Returns:
            Non-negative float — 0.0 means identical vectors.
        """
        if len(vec1) != len(vec2):
            raise ValueError("Vectors must have the same dimension")
        return math.sqrt(sum((p - q) ** 2 for p, q in zip(vec1, vec2)))

    def _cosine_distance(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Angular distance derived from cosine similarity (1 − similarity).

        Returns 0.0 for identical vectors and 1.0 for orthogonal ones.
        Two zero vectors are treated as identical (distance = 0.0).

        Args:
            vec1: First vector.
            vec2: Second vector (must have the same length as vec1).

        Returns:
            Float in [0.0, 2.0] — lower means more similar.
        """
        if len(vec1) != len(vec2):
            raise ValueError("Vectors must have the same dimension")

        mag1 = self._magnitude(vec1)
        mag2 = self._magnitude(vec2)

        # guard against zero-magnitude vectors
        if mag1 == 0 and mag2 == 0:
            return 0.0
        elif mag1 == 0 or mag2 == 0:
            return 1.0

        cosine_similarity = self._dot_product(vec1, vec2) / (mag1 * mag2)
        # clamp to [-1, 1] to guard against floating-point drift
        cosine_similarity = max(-1.0, min(1.0, cosine_similarity))

        return 1.0 - cosine_similarity

    # ------------------------------------------------------------------
    # Vector math helpers
    # ------------------------------------------------------------------

    def _dot_product(self, vec1: List[float], vec2: List[float]) -> float:
        if len(vec1) != len(vec2):
            raise ValueError("Vectors must have the same dimension")
        return sum(p * q for p, q in zip(vec1, vec2))

    def _magnitude(self, vec: List[float]) -> float:
        return math.sqrt(sum(x * x for x in vec))

    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.vectors)

    def __repr__(self) -> str:
        has_embed_fn = "Yes" if self._embedding_fn else "No"
        return (
            f"VectorIndex(count={len(self)}, dim={self._vector_dim}, "
            f"metric='{self._distance_metric}', has_embedding_fn='{has_embed_fn}')"
        )


###############
# Example usage
###############

#index = VectorIndex(distance_metric="cosine")
#
#index.add_vector([1.0, 0.0, 0.0], {"content": "red"})
#index.add_vector([0.0, 1.0, 0.0], {"content": "green"})
#index.add_vector([0.0, 0.0, 1.0], {"content": "blue"})
#
#results = index.search([1.0, 0.1, 0.0], k=2)
#for doc, distance in results:
#    print(f"Distance: {distance:.4f}  |  Content: {doc['content']}")
