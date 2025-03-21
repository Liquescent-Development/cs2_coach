import math
import sys
from pathlib import Path
from typing import List

import polars as pl
from awpy.vector import Vector3
from awpy.visibility import Triangle, VisibilityChecker
from loguru import logger


def monkey_patch_vector3():
    """
    Monkey patches the Vector3 class to overload the * operator for scalar multiplication.
    """

    def __mul__(self, scalar):
        if isinstance(scalar, (int, float)):
            return Vector3(self.x * scalar, self.y * scalar, self.z * scalar)
        else:
            raise TypeError("Can only multiply Vector3 by a scalar (int or float).")

    Vector3.__mul__ = __mul__


monkey_patch_vector3()


class VisibilityCalculator(VisibilityChecker):
    """
    Calculates visibility between players in a Counter-Strike 2 demo.
    """

    def __init__(
        self,
        path: Path | None = None,
        triangles: list[Triangle] | None = None,
        verbose: bool = False,
    ) -> None:
        super().__init__(
            path=path, triangles=triangles
        )  # Pass parameters to the superclass constructor
        """Initialize the visibility calculator with a list of triangles.

        Args:
            path (pathlib.Path | None, optional): Path to a .tri file to read
                triangles from.
            triangles (list[Triangle] | None, optional): List of triangles to
                build the BVH from.
            verbose (bool, optional): If True, print debug information.
        """
        if verbose:
            logger.remove()
            logger.add(sys.stdout, colorize=True, level="DEBUG")
        else:
            logger.remove()

    def _calculate_eye_height(self, player_vec: Vector3, is_crouched: bool) -> Vector3:
        """Calculates the eye height vector based on the player's position and crouching state."""
        result = Vector3(player_vec.x, player_vec.y, player_vec.z)
        if is_crouched:
            result.z += 64.093811
        else:
            result.z += 46.076218
        return result

    def _calculate_direction_vector(self, yaw: float, pitch: float) -> Vector3:
        """Convert yaw and pitch to a direction vector."""
        yaw_rad = math.radians(yaw)
        pitch_rad = math.radians(pitch)
        x = math.cos(yaw_rad) * math.cos(pitch_rad)
        y = math.sin(yaw_rad) * math.cos(pitch_rad)
        z = math.sin(pitch_rad)
        return Vector3(x, y, z)

    def _vector_norm(self, vec: Vector3) -> float:
        """Calculates the norm (magnitude) of a Vector3.

        Args:
          vec: The Vector3 to calculate the norm of.

        Returns:
          The norm (magnitude) of the vector as a float.
        """
        return math.sqrt(vec.x**2 + vec.y**2 + vec.z**2)

    def _generate_target_sample_points(
        self, target_pos: Vector3, forward: Vector3
    ) -> List[Vector3]:
        """Generates sample points around a target position.
        Args:
            target_pos: The target position (Vector3).
            forward: The forward vector (Vector3).
        Returns:
            A list of Vector3 representing the sample points.
        """
        # Standard player dimensions in model space
        player_height = 72.0
        player_width = 32.0
        arm_length = 26.0
        # Calculate up and right vectors based on forward vector
        # Corrected for Python coordinate system
        up = Vector3(0, 0, 1)  # Z is up in Python
        right = forward.cross(up).normalize()
        if self._vector_norm(right) < 0.01:
            # If forward is parallel to up, use a different axis
            right = Vector3(1, 0, 0)  # X is right in Python
        up = right.cross(forward).normalize()
        points = [
            # Center position (default)
            target_pos,
            # Head points - more detailed coverage
            target_pos + up * player_height,  # Top of head
            target_pos + up * (player_height * 0.95),  # Upper head
            target_pos + up * (player_height * 0.90),  # Eye level
            target_pos + up * (player_height * 0.90) + right * 5,  # Right side of head
            target_pos + up * (player_height * 0.90) - right * 5,  # Left side of head
            target_pos + up * (player_height * 0.90) + forward * 5,  # Front of head
            target_pos + up * (player_height * 0.90) - forward * 5,  # Back of head
            # Neck area
            target_pos + up * (player_height * 0.70),
            # Shoulder points
            target_pos
            + up * (player_height * 0.65)
            + right * (player_width - 0.4),  # Right shoulder
            target_pos
            + up * (player_height * 0.65)
            - right * (player_width - 0.4),  # Left shoulder
            # Upper chest
            target_pos + up * (player_height * 0.6) + right * (player_width - 0.25),
            target_pos + up * (player_height * 0.6) - right * (player_width - 0.25),
            # Mid chest level (upper body)
            target_pos + up * (player_height * 0.55) + right * (player_width - 0.35),
            target_pos + up * (player_height * 0.55) - right * (player_width - 0.35),
            # Lower chest
            target_pos + up * (player_height * 0.5),
            # Waist level (mid body)
            target_pos + up * (player_height * 0.45) + right * (player_width - 0.35),
            target_pos + up * (player_height * 0.45) - right * (player_width - 0.35),
            # Hips
            target_pos + up * (player_height * 0.4),
            # Upper legs (lower body)
            target_pos + up * (player_height * 0.35) + right * (player_width - 0.2),
            target_pos + up * (player_height * 0.35) - right * (player_width - 0.2),
            # Mid legs
            target_pos + up * (player_height * 0.25),
            # Lower legs
            target_pos + up * (player_height * 0.15),
            # Feet
            target_pos + up * (player_height * 0.05),
            # Arms - Extended right arm
            target_pos
            + up * (player_height * 0.65)
            + right * (player_width + arm_length * 0.25),  # Right upper arm
            target_pos
            + up * (player_height * 0.65)
            + right * (player_width + arm_length * 0.5),  # Right mid arm
            target_pos
            + up * (player_height * 0.65)
            + right * (player_width + arm_length * 0.75),  # Right forearm
            target_pos
            + up * (player_height * 0.65)
            + right * (player_width + arm_length),  # Right hand
            # Arms - Extended left arm
            target_pos
            + up * (player_height * 0.65)
            - right * (player_width + arm_length * 0.25),  # Left upper arm
            target_pos
            + up * (player_height * 0.65)
            - right * (player_width + arm_length * 0.5),  # Left mid arm
            target_pos
            + up * (player_height * 0.65)
            - right * (player_width + arm_length * 0.75),  # Left forearm
            target_pos
            + up * (player_height * 0.65)
            - right * (player_width + arm_length),  # Left hand
            # Gun position (typically held in front)
            target_pos + up * (player_height * 0.55) + forward * (player_width - 0.6),
            # Front, back, and sides at key heights
            target_pos
            + up * (player_height * 0.6)
            + forward * (player_width - 0.35),  # Front at chest height
            target_pos
            + up * (player_height * 0.6)
            - forward * (player_width - 0.35),  # Back at chest height
            target_pos
            + up * (player_height * 0.45)
            + forward * (player_width - 0.35),  # Front at waist
            target_pos
            + up * (player_height * 0.45)
            - forward * (player_width - 0.35),  # Back at waist
            # Diagonal points at key heights
            target_pos
            + up * (player_height * 0.6)
            + right * (player_width - 0.3)
            + forward * (player_width - 0.3),  # Front-right
            target_pos
            + up * (player_height * 0.6)
            + right * (player_width - 0.3)
            - forward * (player_width - 0.3),  # Back-right
            target_pos
            + up * (player_height * 0.6)
            - right * (player_width - 0.3)
            + forward * (player_width - 0.3),  # Front-left
            target_pos
            + up * (player_height * 0.6)
            - right * (player_width - 0.3)
            - forward * (player_width - 0.3),  # Back-left
            # Head level diagonals
            target_pos
            + up * (player_height * 0.9)
            + right * 4
            + forward * 4,  # Front-right of head
            target_pos
            + up * (player_height * 0.9)
            + right * 4
            - forward * 4,  # Back-right of head
            target_pos
            + up * (player_height * 0.9)
            - right * 4
            + forward * 4,  # Front-left of head
            target_pos
            + up * (player_height * 0.9)
            - right * 4
            - forward * 4,  # Back-left of head
        ]
        return points

    def is_target_visible(
        self,
        player_vec: Vector3,
        player_yaw: float,
        player_pitch: float,
        player_is_crouched: bool,
        target_vec: Vector3,
        target_yaw: float,
        target_pitch: float,
        target_is_crouched: bool,
        fov: float = 180.00,
    ) -> bool:
        """Check if the player at the start vector can see the player at the end vector."""

        player_eye_vec = self._calculate_eye_height(player_vec, player_is_crouched)
        target_eye_vec = self._calculate_eye_height(target_vec, target_is_crouched)
        target_forward = self._calculate_direction_vector(
            target_yaw, target_pitch
        ).normalize()
        sample_points = self._generate_target_sample_points(
            target_eye_vec, target_forward
        )

        direction_vec = self._calculate_direction_vector(
            player_yaw, player_pitch
        ).normalize()

        for sample_point in sample_points:
            if not self._is_within_fov(direction_vec, sample_point, fov):
                continue

            direction = (sample_point - player_eye_vec).normalize()
            distance = (sample_point - player_eye_vec).length()

            if distance < 1e-6:
                return True

            if not self._traverse_bvh(self.root, player_eye_vec, direction, distance):
                return True

        return False

    def _is_within_fov(
        self, direction_vec: Vector3, target_vec: Vector3, fov: float
    ) -> bool:
        """Check if the target vector is within the field of view."""
        rel_dir = (
            target_vec - Vector3(0, 0, 0)
        ).normalize()  # Assuming origin for relative direction
        dot_product = direction_vec.dot(rel_dir)
        dot_product = max(-1.0, min(1.0, dot_product))
        angle_between = math.degrees(math.acos(dot_product))
        return angle_between <= fov / 2

    def _calculate_angle_difference(
        self,
        attacker_yaw: float,
        attacker_pitch: float,
        initial_yaw: float,
        initial_pitch: float,
    ) -> float:
        """Calculate the angular difference between two aim positions in degrees."""
        # Calculate yaw difference (handle 360 degree wrapping)
        yaw_diff = min(
            abs(attacker_yaw - initial_yaw), 360 - abs(attacker_yaw - initial_yaw)
        )

        # Calculate pitch difference
        pitch_diff = abs(attacker_pitch - initial_pitch)

        # Calculate 3D angular difference (Pythagorean theorem in spherical coordinates)
        return math.sqrt(yaw_diff**2 + pitch_diff**2)

    def find_first_sightings_and_metrics(
        self,
        damage_events_df: pl.DataFrame,
        tick_events_df: pl.DataFrame,
        max_lookback: int = 64,
    ):
        """
        For each damage event:
        1. Find the first tick when the attacker saw the victim
        2. Calculate crosshair placement (angle difference)
        3. Calculate time to damage

        Returns enriched damage events with these metrics
        """
        # Ensure we're using lazy API
        damage_lazy = damage_events_df.lazy()
        tick_lazy = tick_events_df.lazy()

        # Process each damage event (this has to be done row by row)
        results = []

        # Collect the damage events to iterate through them
        damage_events = damage_lazy.collect()
        tick_events = tick_lazy.collect()

        for damage_event in damage_events.iter_rows(named=True):
            attacker_steamid = damage_event["attacker_steamid"]
            victim_steamid = damage_event["victim_steamid"]
            damage_tick = damage_event["tick"]

            # Get relevant tick range (from damage_tick - max_lookback to damage_tick)
            start_tick = max(0, damage_tick - max_lookback)

            # Filter tick events for the relevant tick range, attacker and victim
            attacker_ticks = tick_events.filter(
                (pl.col("tick").is_between(start_tick, damage_tick))
                & (pl.col("steamid") == attacker_steamid)
            ).sort("tick")

            victim_ticks = tick_events.filter(
                (pl.col("tick").is_between(start_tick, damage_tick))
                & (pl.col("steamid") == victim_steamid)
            ).sort("tick")

            first_sight_tick = None
            first_sight_yaw = None
            first_sight_pitch = None

            # Find the first tick where attacker saw victim
            for a_tick in attacker_ticks.iter_rows(named=True):
                # Find the closest victim tick
                v_tick = victim_ticks.filter(pl.col("tick") == a_tick["tick"])
                if len(v_tick) == 0:
                    continue

                v_tick = v_tick.row(0, named=True)

                # Check visibility
                is_visible = self.is_target_visible(
                    player_vec=Vector3(a_tick["X"], a_tick["Y"], a_tick["Z"]),
                    player_yaw=a_tick["yaw"],
                    player_pitch=a_tick["pitch"],
                    player_is_crouched=a_tick["in_crouch"],
                    target_vec=Vector3(v_tick["X"], v_tick["Y"], v_tick["Z"]),
                    target_yaw=v_tick["yaw"],
                    target_pitch=v_tick["pitch"],
                    target_is_crouched=v_tick["in_crouch"],
                )

                if is_visible:
                    first_sight_tick = a_tick["tick"]
                    first_sight_yaw = a_tick["yaw"]
                    first_sight_pitch = a_tick["pitch"]
                    break

            # Skip if no visibility found
            if first_sight_tick is None:
                continue

            # Calculate time to damage (in seconds, assuming 64 tick rate)
            tick_difference = damage_tick - first_sight_tick
            time_to_damage = tick_difference / 64.0  # seconds

            # If TTD > 1s, exclude as per requirements
            if time_to_damage > 1.0:
                continue

            # Calculate crosshair placement (angle difference)
            angle_diff = self._calculate_angle_difference(
                damage_event["attacker_yaw"],
                damage_event["attacker_pitch"],
                first_sight_yaw,
                first_sight_pitch,
            )

            # Add results to our list
            result_row = {
                **damage_event,
                "first_sight_tick": first_sight_tick,
                "time_to_damage": time_to_damage,
                "crosshair_placement": angle_diff,
            }
            results.append(result_row)

        # Convert results to a DataFrame
        if not results:
            return pl.DataFrame(
                schema={
                    **damage_events.schema,
                    "first_sight_tick": pl.Int64,
                    "time_to_damage": pl.Float64,
                    "crosshair_placement": pl.Float64,
                }
            )

        return pl.DataFrame(results)
