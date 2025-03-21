import polars as pl
from awpy import Demo


class DemoParser:
    """
    Parse a CS2 demo file and enable interogation of demo events.
    """

    def __init__(self, path: str, verbose=False):
        """
        Initializes the DemoParser from a demo file.

        Args:
        path: Path to the demo file.
        """
        self.verbose = verbose
        self.parser = Demo(path=path, tickrate=64, verbose=self.verbose)
        self.parser.parse(
            player_props=[
                "X",
                "Y",
                "Z",
                "health",
                "score",
                "mvps",
                "is_alive",
                "balance",
                "inventory",
                "inventory_as_ids",
                "life_state",
                "pitch",
                "yaw",
                "is_auto_muted",
                "crosshair_code",
                "pending_team_num",
                "player_color",
                "ever_played_on_team",
                "is_coach_team",
                "rank",
                "rank_if_win",
                "rank_if_loss",
                "rank_if_tie",
                "comp_wins",
                "comp_rank_type",
                "is_controlling_bot",
                "has_controlled_bot_this_round",
                "can_control_bot",
                "has_defuser",
                "has_helmet",
                "spawn_time",
                "death_time",
                "game_time",
                "is_connected",
                "player_name",
                "player_steamid",
                "fov",
                "start_balance",
                "total_cash_spent",
                "cash_spent_this_round",
                "music_kit_id",
                "leader_honors",
                "teacher_honors",
                "friendly_honors",
                "ping",
                "move_collide",
                "move_type",
                "team_num",
                "active_weapon",
                "looking_at_weapon",
                "holding_look_at_weapon",
                "next_attack_time",
                "duck_time_ms",
                "max_speed",
                "max_fall_velo",
                "duck_amount",
                "duck_speed",
                "duck_overrdie",
                "old_jump_pressed",
                "jump_until",
                "jump_velo",
                "fall_velo",
                "in_crouch",
                "crouch_state",
                "ducked",
                "ducking",
                "in_duck_jump",
                "allow_auto_movement",
                "jump_time_ms",
                "last_duck_time",
                "is_rescuing",
                "weapon_purchases_this_match",
                "weapon_purchases_this_round",
                "spotted",
                "approximate_spotted_by",
                "time_last_injury",
                "direction_last_injury",
                "player_state",
                "passive_items",
                "is_scoped",
                "is_walking",
                "resume_zoom",
                "is_defusing",
                "is_grabbing_hostage",
                "blocking_use_in_progess",
                "molotov_damage_time",
                "moved_since_spawn",
                "in_bomb_zone",
                "in_buy_zone",
                "in_no_defuse_area",
                "killed_by_taser",
                "move_state",
                "which_bomb_zone",
                "in_hostage_rescue_zone",
                "stamina",
                "direction",
                "shots_fired",
                "armor_value",
                "velo_modifier",
                "ground_accel_linear_frac_last_time",
                "flash_duration",
                "flash_max_alpha",
                "wait_for_no_attack",
                "last_place_name",
                "is_strafing",
                "round_start_equip_value",
                "current_equip_value",
                "velocity",
                "velocity_X",
                "velocity_Y",
                "velocity_Z",
                "agent_skin",
                "user_id",
                "entity_id",
                "is_airborne",
                "aim_punch_angle",
                "aim_punch_angle_vel",
            ]
        )

    def prefilter_relevant_ticks(
        self,
        max_lookback: int = 64,
    ):
        """
        Prefilter tick events to only include ticks that are within max_lookback ticks
        before any damage event, and only for players involved in damage events.

        Parameters:
        - damage_events_df: Polars DataFrame with damage events
        - tick_events_df: Polars DataFrame with tick events
        - max_lookback: Maximum number of ticks to look back from each damage event

        Returns:
        - Filtered tick events DataFrame
        """

        damage_df = self.parser.damages
        ticks_df = self.parser.ticks

        # Get unique player steamids involved in damage events
        attacker_ids = damage_df["attacker_steamid"].unique()
        victim_ids = damage_df["victim_steamid"].unique()

        # Combine into a single list of all player IDs
        all_player_ids = list(set(attacker_ids.to_list() + victim_ids.to_list()))

        # Calculate tick boundaries
        damage_ticks = damage_df["tick"]
        min_tick = max(0, damage_ticks.min() - max_lookback)
        max_tick = damage_ticks.max()

        # Filter tick events
        filtered_ticks = ticks_df.filter(
            (pl.col("steamid").is_in(all_player_ids))
            & (pl.col("tick") >= min_tick)
            & (pl.col("tick") <= max_tick)
        )

        return filtered_ticks
