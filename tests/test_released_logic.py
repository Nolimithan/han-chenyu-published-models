import numpy as np

from dsm2025_dual_market.dual_market import simulate_dual_market
from dsm2025_dual_market.labeling import binary_action_labels
from dsm2025_dual_market.single_market import simulate_single_market
from eswa2025_robo_advisor.labeling import ternary_labels
from eswa2025_robo_advisor.metrics import risk_metrics
from eswa2025_robo_advisor.reconstruct_2d import to_image_tensor
from eswa2025_robo_advisor.trading import simulate_trading


def test_binary_cost_screen():
    labels = binary_action_labels([100, 100, 100], [101, 100.4, 99])
    assert list(labels["action"]) == [1, 0, 0]
    assert list(labels["inaction"]) == [0, 1, 1]


def test_ternary_window_matches_notebook_index():
    labels = ternary_labels([1, 3, 2, 0, 4])
    assert list(map(tuple, labels.to_numpy())) == [(1, 0, 0), (0, 1, 0), (0, 0, 1)]


def test_image_drops_first_mode():
    row = np.concatenate([
        np.arange(11),
        np.arange(100, 111),
    ])
    image = to_image_tensor(row.reshape(1, -1), n_indicators=2)
    assert image.shape == (1, 10, 2)
    assert np.allclose(image[0, :, 0], np.arange(1, 11))
    assert np.allclose(image[0, :, 1], np.arange(101, 111))


def test_image_ignores_saved_index_column():
    indexed = np.concatenate([[999], np.arange(22)]).reshape(1, -1)
    image = to_image_tensor(indexed, n_indicators=2)
    assert image.shape == (1, 10, 2)
    assert image[0, 0, 0] == 1


def test_single_market_inaction_holds_cash():
    result = simulate_single_market([10, 10], [11, 11], [1, 1], 210000)
    assert result["trades"] == 0
    assert result["final_capital"] == 210000


def test_robo_advisor_holds_when_not_buy():
    result = simulate_trading([10, 11, 12], [1, 1, 0], 10000)
    assert result["trades"] == 0
    assert result["capital"][0] == 10000


def test_dual_market_runs_four_label_states():
    frame = {
        "open_1": [10, 10, 10, 10],
        "close_1": [11, 11, 11, 11],
        "open_2": [20, 20, 20, 20],
        "close_2": [21, 21, 21, 21],
        "CNYCNH_open": [1, 1, 1, 1],
        "CNYCNH_close": [1, 1, 1, 1],
        "USDCNH_open": [7, 7, 7, 7],
        "USDCNH_close": [7, 7, 7, 7],
        "prediction_1": [0, 0, 1, 1],
        "prediction_2": [0, 1, 0, 1],
    }
    result = simulate_dual_market(frame, "prediction_1", "prediction_2", 210000, variant="prediction")
    assert result["final_capital"] > 0
    assert len(result["capital"]) == 4


def test_risk_metrics_are_finite():
    stats = risk_metrics([10000, 10100, 10050, 10300])
    assert np.isfinite(stats["Sharpe Ratio"])
    assert stats["MDD(%)"] >= 0
