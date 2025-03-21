import unittest

import polars as pl
from awpy.data import TRIS_DIR
from awpy.vector import Vector3

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
        map_tri = TRIS_DIR / "de_mirage.tri"
        vc = VisibilityCalculator(path=map_tri, verbose=True)
        tris = VisibilityCalculator.read_tri_file(map_tri)
        vc = VisibilityCalculator(triangles=tris, verbose=True)
        demo_parser = DemoParser("/home/kiener/testdemo.dem", verbose=True)
        event_metrics = EventMetrics(vc, demo_parser, verbose=True)

        cls.visibility_calculator = vc
        cls.demo_parser = demo_parser
        cls.event_metrics = event_metrics

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
