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

    def analyze_player_performance(
        self,
    ):
        """
        Main function to calculate player performance metrics with prefiltering
        """
        # Prefilter tick events to only include relevant ones
        logger.debug("Prefiltering tick events to only include relevant ones")
        filtered_ticks = self.demo_parser.prefilter_relevant_ticks()

        # Find all first sightings and calculate metrics using filtered ticks
        logger.debug("Finding first sightings and calculating metrics")
        enriched_damage_events = (
            self.visibility_calculator.find_first_sightings_and_metrics(
                self.demo_parser.parser.damages, filtered_ticks
            )
        )

        # Calculate player metrics from the enriched damage events
        logger.debug("Calculating player metrics")
        player_metrics = self._calculate_player_metrics(enriched_damage_events)

        return player_metrics, enriched_damage_events
