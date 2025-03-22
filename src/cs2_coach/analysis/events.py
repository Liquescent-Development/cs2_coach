import sys

import polars as pl
from loguru import logger

from cs2_coach.analysis.visibility import VisibilityCalculator
from cs2_coach.demo.parser import DemoParser


class EventMetrics:
    """
    Calculate event metrics for players in a Counter-Strike 2 demo.
    """

    def __init__(
        self,
        visibility_calculator: VisibilityCalculator,
        demo_parser: DemoParser,
        verbose: bool = False,
    ):
        """
        Initializes EventMetrics.

        """
        self.visibility_calculator = visibility_calculator
        self.demo_parser = demo_parser

        if verbose:
            logger.remove()
            logger.add(sys.stdout, colorize=True, level="DEBUG")
        else:
            logger.remove()

    def _calculate_player_metrics(self, enriched_damage_events: pl.DataFrame):
        """
        Calculate the median time to damage and crosshair placement per player
        """
        return (
            enriched_damage_events.lazy()
            .group_by("attacker_steamid", "attacker_name")
            .agg(
                [
                    pl.col("time_to_damage").median().alias("median_time_to_damage"),
                    pl.col("crosshair_placement")
                    .median()
                    .alias("median_crosshair_placement"),
                    pl.count("time_to_damage").alias("damage_events_count"),
                ]
            )
            .collect()
        )

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
        import logging

        logger = logging.getLogger(__name__)

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
        import logging

        logger = logging.getLogger(__name__)

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

    def analyze_player_performance(
        self,
        max_lookback: int = 64,
        burst_exclusion_ticks: int = 7,
    ):
        """
        Main function to calculate player performance metrics with prefiltering
        """
        import logging

        logger = logging.getLogger(__name__)

        # Prefilter tick events and damage events
        logger.debug("Prefiltering tick events and damage events")
        filtered_ticks, filtered_damages = self.demo_parser.prefilter_relevant_ticks(
            max_lookback=max_lookback, burst_exclusion_ticks=burst_exclusion_ticks
        )

        # Check if we have any damage events left after filtering
        if len(filtered_damages) == 0:
            logger.warning(
                "No damage events left after filtering, skipping performance analysis"
            )
            return None, pl.DataFrame()

        # Find all first sightings and calculate metrics using filtered ticks
        logger.debug("Finding first sightings and calculating metrics")
        enriched_damage_events = (
            self.visibility_calculator.find_first_sightings_and_metrics(
                filtered_damages, filtered_ticks, max_lookback=max_lookback
            )
        )

        # Check if we have any enriched damage events
        if len(enriched_damage_events) == 0:
            logger.warning(
                "No enriched damage events found, skipping player metrics calculation"
            )
            return None, enriched_damage_events

        # Calculate player metrics from the enriched damage events
        logger.debug("Calculating player metrics")
        player_metrics = self._calculate_player_metrics(enriched_damage_events)

        return player_metrics, enriched_damage_events
