import sys

import polars as pl
from awpy import Demo
from loguru import logger


class DemoParser:
    """
    Parse a CS2 demo file and enable interogation of demo events.
    """

    def __init__(self, path: str, weapons_path: str, verbose=False):
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
        if verbose:
            logger.remove()
            logger.add(sys.stdout, colorize=True, level="DEBUG")
        else:
            logger.remove()
        self.weapons = pl.scan_csv(weapons_path)

    def prefilter_damage_events(
        self,
        burst_exclusion_ticks: int = 7,  # ~0.1s at 64 tick rate for 600 RPM
    ):
        """
        Prefilter damage events to:
        1. Exclude those that happen in rapid succession (burst fire) for the same attacker-victim pair
        2. Exclude damage from utility or the bomb (non-weapon damage)

        Parameters:
        - burst_exclusion_ticks: Number of ticks to exclude after a damage event

        Returns:
        - Filtered damage events DataFrame
        """

        damage_df = self.parser.damages

        # Sort damage events by tick for chronological processing
        damage_df = damage_df.sort("tick")

        # Track which events to keep
        keep_events = []

        # Track last damage tick for each attacker-victim pair
        last_damage_tick = {}  # (attacker_steamid, victim_steamid) -> tick
        burst_exclusion_count = 0
        utility_exclusion_count = 0

        # Process damage events chronologically
        for damage_event in damage_df.iter_rows(named=True):
            attacker_steamid = damage_event["attacker_steamid"]
            victim_steamid = damage_event["victim_steamid"]
            damage_tick = damage_event["tick"]
            weapon_name = damage_event["weapon"]

            # Check if this is a legitimate weapon (not utility or bomb)
            weapon_class = self.get_weapon_class_by_weapon_name(weapon_name)
            if weapon_class is None:
                utility_exclusion_count += 1
                continue

            # Check if this is a burst fire event (rapid succession)
            attacker_victim_pair = (attacker_steamid, victim_steamid)

            if attacker_victim_pair in last_damage_tick:
                last_tick = last_damage_tick[attacker_victim_pair]
                if damage_tick - last_tick <= burst_exclusion_ticks:
                    # This is a burst fire event, exclude it
                    burst_exclusion_count += 1
                    # Update the last damage tick
                    last_damage_tick[attacker_victim_pair] = damage_tick
                    continue

            # This is not a burst fire event, keep it
            keep_events.append(damage_event)

            # Update the last damage tick for this attacker-victim pair
            last_damage_tick[attacker_victim_pair] = damage_tick

        logger.debug(f"Original damage events: {len(damage_df)}")
        logger.debug(f"Excluded utility/bomb damage events: {utility_exclusion_count}")
        logger.debug(f"Excluded burst fire events: {burst_exclusion_count}")
        logger.debug(f"Filtered damage events: {len(keep_events)}")

        # Convert results to a DataFrame
        if not keep_events:
            logger.warning("No damage events left after filtering")
            return pl.DataFrame(schema=damage_df.schema)

        return pl.DataFrame(keep_events)

    def get_weapon_class_by_weapon_name(self, weapon_name: str) -> str:
        """
        Get weapon class from the weapons table based on the provided weapon name.

        Parameters:
            weapon_name (str): Name of the weapon to filter on.

        Returns:
            str: A string containing the weapon class, or None if weapon not found.
        """
        weapon_details = self.weapons.filter(pl.col("Name") == weapon_name).collect()
        if len(weapon_details) == 0:
            return None

        weapon_class = weapon_details["Class"]
        return weapon_class[0]

    def prefilter_relevant_ticks(
        self,
        max_lookback: int = 64,
        burst_exclusion_ticks: int = 7,
    ):
        """
        Prefilter tick events to only include ticks that are within max_lookback ticks
        before any damage event, and only for players involved in damage events.

        Also excludes damage events that:
        1. Happen in rapid succession (burst fire)
        2. Are caused by utility or the bomb

        Parameters:
        - max_lookback: Maximum number of ticks to look back from each damage event
        - burst_exclusion_ticks: Number of ticks to exclude after a damage event

        Returns:
        - Filtered tick events DataFrame, Filtered damage events DataFrame
        """

        # First, filter out utility damage and burst fire damage events
        filtered_damage_df = self.prefilter_damage_events(
            burst_exclusion_ticks=burst_exclusion_ticks
        )

        ticks_df = self.parser.ticks

        # Get unique player steamids involved in damage events
        attacker_ids = filtered_damage_df["attacker_steamid"].unique()
        victim_ids = filtered_damage_df["victim_steamid"].unique()

        # Combine into a single list of all player IDs
        all_player_ids = list(set(attacker_ids.to_list() + victim_ids.to_list()))

        # Calculate tick boundaries
        damage_ticks = filtered_damage_df["tick"]

        # Handle the case where no damage events are left after filtering
        if len(damage_ticks) == 0:
            logger.warning(
                "No damage events left after filtering, returning empty DataFrame"
            )
            return pl.DataFrame(schema=ticks_df.schema), filtered_damage_df

        min_tick = max(0, damage_ticks.min() - max_lookback)
        max_tick = damage_ticks.max()

        # Filter tick events
        filtered_ticks = ticks_df.filter(
            (pl.col("steamid").is_in(all_player_ids))
            & (pl.col("tick") >= min_tick)
            & (pl.col("tick") <= max_tick)
        )

        logger.debug(f"Original tick events: {len(ticks_df)}")
        logger.debug(f"Filtered tick events: {len(filtered_ticks)}")

        return filtered_ticks, filtered_damage_df
