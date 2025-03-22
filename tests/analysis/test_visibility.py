import sys
import unittest

import polars as pl
from awpy.data import TRIS_DIR
from awpy.vector import Vector3
from loguru import logger

from cs2_coach.analysis.events import EventMetrics
from cs2_coach.analysis.visibility import VisibilityCalculator
from cs2_coach.demo.parser import DemoParser


class TestEventMetrics(unittest.TestCase):
    """
    Test suite for the EventMetrics class.
    """

    def setup_class(cls):
        """
        Set up the test environment.
        """

        logger.remove()
        logger.add(sys.stdout, colorize=True, level="DEBUG")

        map_tri = TRIS_DIR / "de_mirage.tri"
        vc = VisibilityCalculator(path=map_tri, verbose=True)
        tris = VisibilityCalculator.read_tri_file(map_tri)
        vc = VisibilityCalculator(triangles=tris, verbose=True)
        demo_parser = DemoParser(
            path="/home/kiener/testdemo.dem",
            weapons_path="/home/kiener/cs2weapons.csv",
            verbose=True,
        )
        event_metrics = EventMetrics(vc, demo_parser, verbose=True)

        cls.visibility_calculator = vc
        cls.demo_parser = demo_parser
        cls.event_metrics = event_metrics

        cls.leetify_stats_df = pl.scan_csv(
            "/home/kiener/demo_test_leetify.csv"
        ).collect()

    def test_is_target_visible_mirage_tick_153202(self):
        """
        Test visibility from Mirage Stairs to Main -- Tick 153202 -- Bullet Damage Event -- Head Barely Visible *****"
        """

        shooter_pos = Vector3(
            -768.6340942382812, -1738.987548828125, -179.46734619140625
        )
        shooter_yaw = 7.985687255859375
        shooter_pitch = 2.8049468994140625
        shooter_is_crouched = False
        victim_pos = Vector3(743.3181152343785, -1526.9077148375, -263.9549560546875)
        victim_yaw = -172.77511596679688
        victim_pitch = -3.396148681640625
        victim_is_crouched = False

        is_visible = self.visibility_calculator.is_target_visible(
            shooter_pos,
            shooter_yaw,
            shooter_pitch,
            shooter_is_crouched,
            victim_pos,
            victim_yaw,
            victim_pitch,
            victim_is_crouched,
        )
        self.assertEqual(is_visible, True)

    # TODO: This belongs in the test_events.py file not here
    def test_analyze_player_performance_simple(self):
        """
        Test basic visibility between two players with no obstacles.
        """

        player_metrics, enriched_events = (
            self.event_metrics.analyze_player_performance()
        )

        self.assertGreater(len(player_metrics), 0)

        # Check that we have enriched events
        self.assertGreater(len(enriched_events), 0)

        # Basic structure checks
        self.assertIn("median_time_to_damage", player_metrics.columns)
        self.assertIn("median_crosshair_placement", player_metrics.columns)
        self.assertIn("damage_events_count", player_metrics.columns)

        # PlayerA should have at least one event
        player_shmeeny = player_metrics.filter(pl.col("attacker_name") == "shmeeny")
        if len(player_shmeeny) > 0:
            self.assertGreaterEqual(player_shmeeny[0, "damage_events_count"], 1)

        print(player_metrics)

    def test_compare_with_leetify(self):
        """
        Test that our calculated metrics are within +/-15% of Leetify's values.
        """

        # Run our player performance analysis
        player_metrics, enriched_events = (
            self.event_metrics.analyze_player_performance()
        )

        # Ensure we have results
        self.assertGreater(len(player_metrics), 0)
        self.assertGreater(len(enriched_events), 0)

        # Leetify data
        leetify_data = self.leetify_stats_df

        # Prepare our data for comparison
        # Focus only on columns that exist in both dataframes
        common_columns = [
            "attacker_steamid",
            "attacker_name",
            "median_time_to_damage",
            "median_crosshair_placement",
        ]

        # Ensure all common columns exist in both dataframes
        for col in common_columns:
            self.assertIn(col, player_metrics.columns)
            self.assertIn(col, leetify_data.columns)

        # Select only common columns for comparison
        our_data = player_metrics.select(common_columns)
        leetify_data = leetify_data.select(common_columns)

        # Sort both dataframes by attacker_steamid for consistent comparison
        our_data = our_data.sort("attacker_steamid")
        leetify_data = leetify_data.sort("attacker_steamid")

        # Check that we have the same players
        our_steamids = set(our_data["attacker_steamid"].to_list())
        leetify_steamids = set(leetify_data["attacker_steamid"].to_list())

        # Log any differences in players
        if our_steamids != leetify_steamids:
            missing_in_ours = leetify_steamids - our_steamids
            missing_in_leetify = our_steamids - leetify_steamids

            if missing_in_ours:
                logger.warning(
                    f"Players in Leetify but not in our results: {missing_in_ours}"
                )

            if missing_in_leetify:
                logger.warning(
                    f"Players in our results but not in Leetify: {missing_in_leetify}"
                )

        # Only compare players that exist in both datasets
        common_steamids = our_steamids.intersection(leetify_steamids)
        self.assertGreater(
            len(common_steamids), 0, "No common players found to compare"
        )

        # Filter both datasets to only include common players
        our_data = our_data.filter(pl.col("attacker_steamid").is_in(common_steamids))
        leetify_data = leetify_data.filter(
            pl.col("attacker_steamid").is_in(common_steamids)
        )

        # Create a comparison DataFrame
        comparison_data = []

        # Track overall pass/fail counts
        total_comparisons = 0
        passed_comparisons = 0

        # Compare each player's metrics
        for our_row in our_data.iter_rows(named=True):
            steamid = our_row["attacker_steamid"]
            name = our_row["attacker_name"]

            # Find matching Leetify row
            leetify_row = leetify_data.filter(pl.col("attacker_steamid") == steamid)

            if len(leetify_row) == 0:
                continue

            leetify_row = leetify_row.row(0, named=True)

            # Compare metrics columns
            for metric in ["median_time_to_damage", "median_crosshair_placement"]:
                our_value = our_row[metric]
                leetify_value = leetify_row[metric]

                # Skip if either value is None
                if our_value is None or leetify_value is None:
                    logger.warning(
                        f"Skipping comparison for {name} - {metric}: our={our_value}, leetify={leetify_value}"
                    )
                    continue

                # Calculate percentage difference
                if leetify_value == 0:
                    # Handle division by zero
                    percent_diff = float("inf") if our_value != 0 else 0
                else:
                    percent_diff = abs(our_value - leetify_value) / leetify_value * 100

                # Check if within 15% tolerance
                within_tolerance = percent_diff <= 15

                # Track pass/fail
                total_comparisons += 1
                if within_tolerance:
                    passed_comparisons += 1

                # Add to comparison data for logging
                comparison_data.append(
                    {
                        "attacker_name": name,
                        "metric": metric,
                        "our_value": our_value,
                        "leetify_value": leetify_value,
                        "percent_diff": percent_diff,
                        "within_tolerance": within_tolerance,
                    }
                )

        # Create comparison DataFrame for logging
        comparison_df = pl.DataFrame(comparison_data)

        # Log the comparison results
        logger.info("\nComparison Results:")
        logger.info(f"Total comparisons: {total_comparisons}")
        logger.info(f"Passed comparisons: {passed_comparisons}")
        logger.info(f"Pass rate: {passed_comparisons / total_comparisons * 100:.2f}%")

        # Log detailed comparison for each metric
        for metric in ["median_time_to_damage", "median_crosshair_placement"]:
            metric_df = comparison_df.filter(pl.col("metric") == metric)
            pass_rate = (
                metric_df.filter(pl.col("within_tolerance")).height
                / max(1, metric_df.height)
            ) * 100

            logger.info(f"\n{metric} comparison:")
            logger.info(f"Pass rate: {pass_rate:.2f}%")

            # Sort by percent difference to see biggest discrepancies first
            sorted_df = metric_df.sort("percent_diff", descending=True)
            for row in sorted_df.iter_rows(named=True):
                status = "✓" if row["within_tolerance"] else "✗"
                logger.info(
                    f"{status} {row['attacker_name']}: Ours={row['our_value']:.4f}, "
                    f"Leetify={row['leetify_value']:.4f}, "
                    f"Diff={row['percent_diff']:.2f}%"
                )

        # Assert that at least 80% of comparisons pass
        pass_rate = passed_comparisons / total_comparisons * 100
        self.assertGreaterEqual(
            pass_rate,
            80,
            f"Only {pass_rate:.2f}% of metrics are within 15% of Leetify values (expected ≥80%)",
        )

        # Print the full comparison DataFrame
        print("\nDetailed Comparison:")
        print(comparison_df)
