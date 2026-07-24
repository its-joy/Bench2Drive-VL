from fault_injector.lane_position_metrics import compute_lane_position_metrics


def _record(timestamp, distance, preferred, marking="Broken", junction=False):
    return {
        "timestamp_s": timestamp,
        "driving_state": {
            "ego": {
                "lane_id": -2 if preferred else -1,
                "is_junction": junction,
                "lateral_offset_from_lane_center_m": 0.3,
            },
            "route": {
                "distance_to_junction_m": distance,
                "preferred_approach_lane_id": -2,
                "current_lane_is_preferred": preferred,
            },
            "lane_topology": {
                "right_lane_marking_type": marking,
                "current_lane_marking_right_type": marking,
            },
        },
    }


def test_identical_lane_entries_are_not_late():
    clean = [_record(0.0, 30.0, True)]
    faulted = [_record(0.0, 30.0, True)]
    metrics = compute_lane_position_metrics(clean, faulted)
    assert metrics["late_lane_positioning_detected"] is False
    assert metrics["lane_entry_delay_m"] == 0.0


def test_delayed_lane_entry_is_detected():
    clean = [_record(1.0, 31.0, True)]
    faulted = [_record(1.0, 25.0, False), _record(3.0, 13.0, True)]
    metrics = compute_lane_position_metrics(clean, faulted)
    assert metrics["late_lane_positioning_detected"] is True
    assert metrics["lane_entry_delay_m"] == 18.0
    assert metrics["entered_correct_lane_before_junction"] is True


def test_wrong_lane_junction_entry_is_rejected():
    clean = [_record(1.0, 31.0, True)]
    faulted = [_record(1.0, 20.0, False), _record(2.0, 0.0, False, junction=True)]
    metrics = compute_lane_position_metrics(clean, faulted)
    assert metrics["late_lane_positioning_detected"] is False
    assert metrics["wrong_lane_junction_entry"] is True


def test_solid_marking_crossing_is_rejected():
    clean = [_record(1.0, 31.0, True)]
    faulted = [_record(1.0, 25.0, False, marking="Solid"), _record(3.0, 13.0, True, marking="Solid")]
    metrics = compute_lane_position_metrics(clean, faulted)
    assert metrics["late_lane_positioning_detected"] is False
    assert metrics["crossed_solid_marking"] is True
