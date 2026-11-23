from dexposure_agent.data_health import compute_data_health


def test_healthy_graph(sample_graph, sample_config):
    result = compute_data_health(sample_graph, sample_config)
    assert result.score > sample_config.tau_data
    assert result.safe_mode is False


def test_empty_graph_triggers_safe_mode(sample_config):
    from dexposure_agent.types import GraphSnapshot
    empty = GraphSnapshot(date="2025-01-01", nodes={}, edges=[])
    result = compute_data_health(empty, sample_config)
    assert result.score < sample_config.tau_data
    assert result.safe_mode is True


def test_missing_features_lowers_score(sample_graph, sample_config):
    sample_graph.nodes["aave-v3"].missing_features = ["log_size", "num_tokens"]
    result = compute_data_health(sample_graph, sample_config)
    assert result.score < 1.0
    assert "missingness" in result.checks


def test_prev_tvl_change_is_part_of_missingness_contract(sample_graph, sample_config):
    sample_graph.nodes["aave-v3"].missing_features = ["prev_tvl_change"]
    result = compute_data_health(sample_graph, sample_config)
    assert result.checks["missingness"] < 1.0


def test_legitimate_zero_features_are_not_missing(sample_graph, sample_config):
    sample_graph.nodes["aave-v3"].log_size = 0.0
    sample_graph.nodes["aave-v3"].num_tokens = 0
    result = compute_data_health(sample_graph, sample_config)
    assert result.checks["missingness"] == 1.0


def test_single_node_graph(sample_config):
    from dexposure_agent.types import GraphSnapshot, NodeFeatures
    single = GraphSnapshot(
        date="2025-01-01",
        nodes={"aave-v3": NodeFeatures(log_size=8.5, num_tokens=3, max_share=0.6, entropy=1.2, category="lending")},
        edges=[],
    )
    result = compute_data_health(single, sample_config)
    assert 0.0 <= result.score <= 1.0
    assert "topology" in result.checks


def test_discontinuity_is_omitted_without_previous_graph(sample_graph, sample_config):
    result = compute_data_health(sample_graph, sample_config)
    assert "discontinuity" not in result.checks


def test_discontinuity_uses_previous_graph(sample_graph, sample_config):
    previous = sample_graph.model_copy(deep=True)
    current = sample_graph.model_copy(deep=True)
    current.edges = []
    result = compute_data_health(current, sample_config, previous_graph=previous)
    assert "discontinuity" in result.checks
    assert result.checks["discontinuity"] < 1.0
    assert "timestamp_validity" in result.checks


def test_invalid_source_edge_records_lower_data_health(sample_graph, sample_config):
    graph = sample_graph.model_copy(
        update={
            "source_edge_records_total": 10,
            "source_edge_records_invalid": 4,
        }
    )
    result = compute_data_health(graph, sample_config)
    assert result.checks["edge_record_validity"] == 0.6
    assert result.score < 1.0


def test_invalid_source_node_records_lower_data_health(sample_graph, sample_config):
    graph = sample_graph.model_copy(
        update={
            "source_node_records_total": 5,
            "source_node_records_invalid": 2,
        }
    )
    result = compute_data_health(graph, sample_config)
    assert result.checks["node_record_validity"] == 0.6
    assert result.score < 1.0
