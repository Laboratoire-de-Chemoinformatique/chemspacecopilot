"""Numerical and representation regressions for chemical similarity tools."""

import numpy as np
import pytest
from rdkit import DataStructs

from cs_copilot.tools.chemistry.base_chemistry import calc_morgan_fp
from cs_copilot.tools.chemistry.similarity_toolkit import ChemicalSimilarityToolkit, SimilarityError


@pytest.fixture
def toolkit():
    return ChemicalSimilarityToolkit()


def test_euclidean_distance_and_nearest_neighbor_order(toolkit):
    fp1 = toolkit.generate_fingerprint("CCO", "morgan")
    fp2 = toolkit.generate_fingerprint("CCN", "morgan")
    expected = (fp1 ^ fp2).GetNumOnBits() ** 0.5
    assert toolkit.calculate_euclidean_distance("CCO", "CCN", "morgan", False) == pytest.approx(
        expected
    )
    assert toolkit.calculate_euclidean_distance("CCO", "CCN", "morgan") == pytest.approx(
        expected / fp1.GetNumBits() ** 0.5
    )
    matches = toolkit.find_most_similar("CCO", ["CCN", "CCO"], metric="euclidean")
    assert len(matches) == 2
    assert matches[0] == ("CCO", 0.0, 1)
    assert toolkit.calculate_all_similarities("CCO", "CCO")["euclidean_distance"] == 0


@pytest.mark.parametrize("dtype", [np.bool_, np.uint8, np.uint32, np.int64, np.float32])
def test_binary_matrix_matches_rdkit(toolkit, dtype):
    a = np.array([[1, 0, 1], [0, 0, 0]], dtype=dtype)
    b = np.array([[1, 1, 0], [0, 0, 0], [1, 0, 1]], dtype=dtype)
    result = toolkit.calculate_similarity_matrix(a, b, "binary")
    expected = [
        [
            DataStructs.TanimotoSimilarity(
                DataStructs.CreateFromBitString("".join(str(int(v)) for v in x)),
                DataStructs.CreateFromBitString("".join(str(int(v)) for v in y)),
            )
            for y in b
        ]
        for x in a
    ]
    np.testing.assert_allclose(result["matrix"], expected)
    assert result["metric"] == "tanimoto"
    assert result["is_distance"] is False


def test_count_matrix_preserves_multiplicity(toolkit):
    result = toolkit.calculate_similarity_matrix([[2, 1], [0, 0]], [[1, 1], [0, 0]], "count")
    np.testing.assert_allclose(result["matrix"], [[0.75, 0], [0, 1]])
    # Cast before multiplication: uint8 squaring would overflow.
    result = toolkit.calculate_similarity_matrix(
        np.array([[255, 0]], dtype=np.uint8), [[255, 0], [0, 255]], "count"
    )
    np.testing.assert_allclose(result["matrix"], [[1, 0]])


def test_morgan_count_smiles_uses_same_representation_as_encoder(toolkit):
    a, b = (calc_morgan_fp(s, 2048).astype(float) for s in ("CCCO", "CCO"))
    expected = float(a @ b / (a @ a + b @ b - a @ b))
    assert toolkit.calculate_tanimoto_similarity("CCCO", "CCO", "morgan_count") == pytest.approx(
        expected
    )


def test_embedding_metrics_and_direction(toolkit):
    result = toolkit.calculate_similarity_matrix([[0, 0], [3, 4]], [[0, 0], [-3, -4]], "embedding")
    np.testing.assert_allclose(result["matrix"], [[0, 5], [5, 10]])
    assert result["metric"] == "euclidean"
    assert result["is_distance"] is True
    result = toolkit.calculate_similarity_matrix(
        [[1, 0]], [[1, 0], [0, 1], [-1, 0]], "embedding", "cosine"
    )
    np.testing.assert_allclose(result["matrix"], [[1, 0, -1]])
    assert result["is_distance"] is False


@pytest.mark.parametrize(
    "a,b,kind,metric",
    [
        ([[1, 0]], [[1]], "binary", "auto"),
        ([], [[1]], "count", "auto"),
        ([[np.nan]], [[1]], "embedding", "auto"),
        ([[np.inf]], [[1]], "embedding", "auto"),
        ([[-1]], [[1]], "count", "auto"),
        ([[2]], [[1]], "binary", "auto"),
        ([[1]], [[1]], "unknown", "auto"),
        ([[1]], [[1]], "embedding", "tanimoto"),
        ([[1]], [[1]], "count", "euclidean"),
        ([[0, 0]], [[1, 0]], "embedding", "cosine"),
    ],
)
def test_invalid_descriptors_or_metric_fail_clearly(toolkit, a, b, kind, metric):
    with pytest.raises(SimilarityError):
        toolkit.calculate_similarity_matrix(a, b, kind, metric)
