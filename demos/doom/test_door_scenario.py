from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import play
import vizdoom as vzd


@unittest.skipUnless(
    os.environ.get("JET_DOOM_ENGINE_SMOKE") == "1",
    "set JET_DOOM_ENGINE_SMOKE=1 to run the real Doom door regression",
)
class DoorScenarioTests(unittest.TestCase):
    def test_approach_then_use_opens_a_door_that_distant_use_cannot(self):
        # These fixture coordinates describe the first door in the Freedoom2 MAP01
        # bundled with ViZDoom 1.3.1. Geometry is only an assertion oracle: no model
        # receives positions or sectors, and no model process is started here.
        door_sector_id = 27
        initial_x = 576.0
        use_only = tuple(float(name == "USE") for name in play.BUTTON_NAMES)
        forward_only = tuple(
            float(name == "MOVE_FORWARD") for name in play.BUTTON_NAMES
        )

        with tempfile.TemporaryDirectory(prefix="jet-doom-door-") as directory:
            game = play.configure_game(
                SimpleNamespace(
                    output_dir=Path(directory),
                    iwad=None,
                    map="map01",
                    skill=1,
                    seed=7,
                )
            )
            # Synchronous ticks make phase timing reproducible without waiting for
            # wall time; production still uses ASYNC_PLAYER and actual deadlines.
            game.set_mode(vzd.Mode.PLAYER)
            game.set_sectors_info_enabled(True)
            game.add_game_args("-nomonsters")

            def door():
                state = game.get_state()
                self.assertIsNotNone(state)
                return next(s for s in state.sectors if s.id == door_sector_id)

            def position_x():
                return game.get_game_variable(vzd.GameVariable.POSITION_X)

            try:
                game.init()
                self.assertEqual(game.get_game_variable(vzd.GameVariable.ANGLE), 0)
                game.send_game_command("warp 576 -160")
                game.make_action(list(play.RELEASE), 1)
                self.assertAlmostEqual(position_x(), initial_x)
                self.assertEqual(door().floor_height, -64)
                self.assertEqual(door().ceiling_height, -64)
                closed_image = game.get_state().screen_buffer.copy()

                # The door line is x=704: standing 128 units away cannot use it.
                # Release between presses so all five attempts are distinct USE
                # edges, rather than one continuously held button.
                for _ in range(5):
                    game.make_action(list(use_only), 4)
                    game.make_action(list(play.RELEASE), 14)
                    self.assertAlmostEqual(position_x(), initial_x)
                    self.assertEqual(door().ceiling_height, -64)

                controls = play.Controls()
                transitions = []
                use_positions = []
                for attempt in range(2):
                    now = game.get_episode_time() / 35.0
                    frame = play.PendingFrame(
                        f"door-attempt-{attempt + 1}",
                        controls.episode,
                        now,
                        game.get_episode_time(),
                    )
                    buttons = controls.apply(frame, "use", now, 3.0)
                    self.assertEqual(buttons, forward_only)
                    game.set_action(list(buttons))
                    transitions.append(buttons)
                    released = False
                    for _ in range(70):
                        game.advance_action(1)
                        now = game.get_episode_time() / 35.0
                        buttons = controls.release_due(now)
                        if buttons is not None:
                            transitions.append(buttons)
                            game.set_action(list(buttons))
                            if buttons == use_only:
                                use_positions.append(position_x())
                            if buttons == play.RELEASE:
                                released = True
                                self.assertFalse(controls.ready_for_observation(now))
                        if controls.ready_for_observation(now):
                            self.assertTrue(released)
                            break
                    else:
                        self.fail("door macro failed to release and settle within 2 s")

                    if door().ceiling_height > door().floor_height:
                        break

                self.assertIn(use_only, transitions)
                self.assertEqual(transitions[-1], play.RELEASE)
                self.assertTrue(all(x > initial_x + 64 for x in use_positions))
                for buttons in transitions:
                    self.assertFalse(
                        buttons[play.BUTTON_NAMES.index("MOVE_FORWARD")]
                        and buttons[play.BUTTON_NAMES.index("USE")],
                        "approach and USE must be separate phases",
                    )
                self.assertGreater(door().ceiling_height, door().floor_height + 56)
                self.assertFalse(
                    (game.get_state().screen_buffer == closed_image).all(),
                    "the actual screen should reflect the opened door",
                )
            finally:
                game.close()


if __name__ == "__main__":
    unittest.main()
