"""
Reasoning Context Template System
==================================
Templates keyed on (cdp_type, compliance_outcome_or_inferred_cause).
All placeholders map directly to event log fields — no inference in templates.
"""

# ────────────────────────────────────────────────────────────────────
# TRAFFIC LIGHT – compliance_outcome only, no inferred_cause
# ────────────────────────────────────────────────────────────────────

TRAFFIC_LIGHT_TEMPLATES = {

    'ran_red_light':
        "ego proceeded through junction at {speed_start_kmh}km/h "
        "with red traffic light at {traffic_light_distance_m}m; "
        "cross-traffic present: {cross_traffic_summary}",

    'complied_with_red':
        "ego stopped at red traffic light {traffic_light_distance_m}m ahead; "
        "speed reduced to {speed_end_kmh}km/h",

    'proceeded_at_yellow':
        "ego proceeded through junction at {speed_start_kmh}km/h "
        "with yellow traffic light at {traffic_light_distance_m}m",

    'stopped_at_yellow':
        "ego stopped at yellow traffic light {traffic_light_distance_m}m ahead; "
        "speed reduced to {speed_end_kmh}km/h",

    'complied_with_green':
        "ego proceeded through junction at {speed_start_kmh}km/h "
        "with green traffic light at {traffic_light_distance_m}m",

    'speeding_through_green':
        "ego proceeded through junction at {speed_start_kmh}km/h "
        "exceeding speed limit of {speed_limit_kmh}km/h; "
        "green traffic light at {traffic_light_distance_m}m",
}

# ────────────────────────────────────────────────────────────────────
# LANE POSITION – compliance_outcome, nested in junction_entry
# ────────────────────────────────────────────────────────────────────

LANE_POSITION_TEMPLATES = {

    'correct_lane_right_turn':
        "ego in correct rightmost lane (lane_id={lane_id}) "
        "for intended right turn",

    'correct_lane_left_turn':
        "ego in correct leftmost lane (lane_id={lane_id}) "
        "for intended left turn",

    'wrong_lane_right_turn':
        "ego in lane_id={lane_id} attempting right turn; "
        "rightmost available lane is {correct_lane_id}; "
        "positional error compounds junction risk",

    'wrong_lane_left_turn':
        "ego in lane_id={lane_id} attempting left turn; "
        "leftmost available lane is {correct_lane_id}; "
        "positional error compounds junction risk",

    'correct_lane_straight':
        "ego in lane_id={lane_id} proceeding straight through junction",
}

# ────────────────────────────────────────────────────────────────────
# JUNCTION ENTRY – inferred_cause (tagged inference)
# ────────────────────────────────────────────────────────────────────

JUNCTION_ENTRY_TEMPLATES = {

    'executing_target_maneuver':
        "ego entered junction at {speed_start_kmh}km/h "
        "to execute target maneuver: {maneuver}",

    'forced_entry':
        "ego entered junction at {speed_start_kmh}km/h "
        "with red traffic light at {traffic_light_distance_m}m; "
        "proceeded without stopping",
}

# ────────────────────────────────────────────────────────────────────
# LANE CHANGE – inferred_cause (tagged inference)
# ────────────────────────────────────────────────────────────────────

LANE_CHANGE_TEMPLATES = {

    'positioning_for_maneuver':
        "ego changed {direction} at {speed_start_kmh}km/h; "
        "no blocking agent in current lane; "
        "closest target lane agent: {target_agent_summary}",

    'obstacle_avoidance':
        "ego changed {direction} at {speed_start_kmh}km/h; "
        "agent {blocking_agent_id} at {blocking_agent_gap}m ahead",

    'unsafe_gap':
        "ego changed {direction} at {speed_start_kmh}km/h; "
        "agent at {target_agent_gap}m in target lane "
        "closing at {closing_speed}km/h",

    'unknown':
        "ego changed {direction} at {speed_start_kmh}km/h; "
        "cause unverifiable from available data",
}

# ────────────────────────────────────────────────────────────────────
# MASTER LOOKUP
# ────────────────────────────────────────────────────────────────────

ALL_TEMPLATES = {
    'traffic_light':        TRAFFIC_LIGHT_TEMPLATES,
    'lane_position':        LANE_POSITION_TEMPLATES,
    'junction_entry':       JUNCTION_ENTRY_TEMPLATES,
    'lane_change':          LANE_CHANGE_TEMPLATES,
}


def get_reasoning_context(cdp_type, outcome_or_cause, field_values):
    """
    cdp_type:          one of the keys in ALL_TEMPLATES
    outcome_or_cause:  compliance_outcome or inferred_cause string
    field_values:      dict of placeholder values from event log fields

    Returns filled reasoning_context string.
    Returns None if template not found.
    """
    if cdp_type not in ALL_TEMPLATES:
        return None

    templates = ALL_TEMPLATES[cdp_type]

    if outcome_or_cause not in templates:
        return None

    template = templates[outcome_or_cause]

    try:
        return template.format(**field_values)
    except KeyError:
        return None


def build_cross_traffic_summary(agents_involved):
    """Build compact cross-traffic string from agents_involved list."""
    if not agents_involved:
        return "none detected"

    cross = [
        a for a in agents_involved
        if a.get('relative_position') and
           a['relative_position'][0] in ('left', 'right') and
           (a.get('agent_speed_kmh') or 0) > 5.0
    ]
    if not cross:
        return "none detected"

    return ', '.join(
        f"id={a['agent_id']} {a['relative_position'][0]}/{a['relative_position'][1]} "
        f"{a['agent_speed_kmh']}km/h at {a['distance_m']}m"
        for a in cross[:3]
    )
