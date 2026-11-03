# 2D Space Shooter MVP — Implementation & Test Plan

## 1. Project Overview
This document serves as the specification and implementation guide for an AI coding agent or developer to build a **Minimum Viable Prototype (MVP) 2D Space Shooter** on **Ubuntu Linux** using **Python Arcade**. 

The core objective is to deliver a simple, playable, single-file prototype (`main.py`) with zero external asset downloads required, while enforcing architectural patterns that allow seamless future expansion (e.g., menus, new enemy types, boss fights, multi-file refactoring).

---

## 2. Environment & Dependencies (Ubuntu Linux)
The coding agent must configure an isolated Python virtual environment and install the required dependencies:

```bash
# 1. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install Python Arcade and Pytest for unit testing
pip install arcade pytest
```

* **Assets Strategy:** Use Python Arcade's built-in `:resources:` handle for all initial sprites and audio so the game runs out-of-the-box without external files.
  * Player Sprite: `:resources:images/space_shooter/playerShip1_orange.png`
  * Enemy Sprite: `:resources:images/space_shooter/playerShip1_green.png`
  * Laser Sprite: `:resources:images/space_shooter/laserBlue01.png`
  * Laser Audio: `:resources:sounds/hurt5.wav`

---

## 3. Core Architectural Requirements
To ensure the MVP is easy to extend later without rewriting core logic, the coding agent **must** adhere to these five design rules:

1. **Use `arcade.View` instead of subclassing `arcade.Window`:** Encapsulate gameplay inside `GameView(arcade.View)`. This ensures future `MenuView`, `PauseView`, or `GameOverView` screens can be swapped in via `window.show_view()`.
2. **Subclass `arcade.Sprite` for Entities:** Create distinct `Player` and `Enemy` classes that inherit from `arcade.Sprite` and manage their own state (such as cooldown timers, bounds clamping, and hit points) inside their `on_update(self, delta_time)` methods.
3. **Set-Based Input Tracking:** Track active keys using `self.keys_pressed = set()`. Calculate player velocity inside `GameView.on_update()` based on the set state to avoid input stutter when switching directions.
4. **Hardware-Accelerated Text & Batching:** Always store sprites in `arcade.SpriteList()` for GPU batch rendering, and use `arcade.Text()` initialized inside `setup()` rather than calling `arcade.draw_text()` every frame.
5. **Frame-Rate Independent Cooldowns:** Use `delta_time` (seconds elapsed since the last frame) for weapon fire cooldowns rather than counting frames.

---

## 4. Target Directory Structure
Keep the initial MVP flat and self-contained:

```text
space_shooter_mvp/
├── main.py                  # Core game implementation (Entities + GameView + Entrypoint)
├── test_game.py             # Automated unit test suite (pytest)
└── implementation_plan.md   # This specification document
```

---

## 5. Reference Implementation (`main.py`)
The coding agent should generate `main.py` following this structure:

```python
import arcade

# --- Constants ---
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
SCREEN_TITLE = "Extensible 2D Shooter MVP"
PLAYER_SPEED = 5
BULLET_SPEED = 10
FIRE_COOLDOWN = 0.25  # Seconds between shots


class Player(arcade.Sprite):
    """Custom Player class keeps player state and logic self-contained."""
    def __init__(self):
        super().__init__(":resources:images/space_shooter/playerShip1_orange.png", scale=0.5)
        self.center_x = SCREEN_WIDTH // 2
        self.center_y = 50
        self.shoot_timer = 0.0

    def on_update(self, delta_time: float = 1 / 60):
        # Move and clamp to screen bounds
        self.center_x += self.change_x
        self.center_x = max(30, min(SCREEN_WIDTH - 30, self.center_x))
        if self.shoot_timer > 0:
            self.shoot_timer = max(0.0, self.shoot_timer - delta_time)


class Enemy(arcade.Sprite):
    """Custom Enemy class makes it easy to add HP or AI movement later."""
    def __init__(self, x: float, y: float):
        super().__init__(":resources:images/space_shooter/playerShip1_green.png", scale=0.5)
        self.center_x = x
        self.center_y = y
        self.angle = 180
        self.hp = 1


class GameView(arcade.View):
    """Using a View makes adding Start/Game Over screens effortless later."""
    def __init__(self):
        super().__init__()
        arcade.set_background_color(arcade.color.DARK_MIDNIGHT_BLUE)
        self.laser_sound = arcade.load_sound(":resources:sounds/hurt5.wav")

        self.player_list = None
        self.bullet_list = None
        self.enemy_list = None
        self.player = None
        self.keys_pressed = set()
        self.score = 0
        self.score_text = None

    def setup(self):
        """Initialize or reset the game state."""
        self.player_list = arcade.SpriteList()
        self.bullet_list = arcade.SpriteList()
        self.enemy_list = arcade.SpriteList()
        self.keys_pressed.clear()

        self.player = Player()
        self.player_list.append(self.player)

        for x in range(100, 750, 120):
            self.enemy_list.append(Enemy(x, SCREEN_HEIGHT - 80))

        self.score = 0
        if self.window:
            self.score_text = arcade.Text(
                f"Score: {self.score}", 10, 10, arcade.color.WHITE, 16
            )

    def on_draw(self):
        self.clear()
        self.player_list.draw()
        self.bullet_list.draw()
        self.enemy_list.draw()
        if self.score_text:
            self.score_text.draw()

    def on_update(self, delta_time: float):
        # Smooth input handling from key set
        left = arcade.key.LEFT in self.keys_pressed or arcade.key.A in self.keys_pressed
        right = arcade.key.RIGHT in self.keys_pressed or arcade.key.D in self.keys_pressed
        self.player.change_x = (right - left) * PLAYER_SPEED

        if arcade.key.SPACE in self.keys_pressed and self.player.shoot_timer <= 0:
            self.shoot()

        # Update sprites
        self.player_list.on_update(delta_time)
        self.bullet_list.update()
        self.enemy_list.on_update(delta_time)

        # Collisions and cleanup
        for bullet in list(self.bullet_list):
            hit_list = arcade.check_for_collision_with_list(bullet, self.enemy_list)
            if hit_list:
                bullet.remove_from_sprite_lists()
                for enemy in hit_list:
                    enemy.hp -= 1
                    if enemy.hp <= 0:
                        enemy.remove_from_sprite_lists()
                        self.score += 1
                        if self.score_text:
                            self.score_text.text = f"Score: {self.score}"
            elif bullet.bottom > SCREEN_HEIGHT:
                bullet.remove_from_sprite_lists()

    def shoot(self, play_audio: bool = True):
        self.player.shoot_timer = FIRE_COOLDOWN
        bullet = arcade.Sprite(":resources:images/space_shooter/laserBlue01.png", scale=0.8)
        bullet.angle = 90
        bullet.center_x = self.player.center_x
        bullet.bottom = self.player.top
        bullet.change_y = BULLET_SPEED
        self.bullet_list.append(bullet)
        if play_audio and self.laser_sound:
            arcade.play_sound(self.laser_sound)

    def on_key_press(self, key, modifiers):
        self.keys_pressed.add(key)
        if key == arcade.key.R:
            self.setup()

    def on_key_release(self, key, modifiers):
        self.keys_pressed.discard(key)


def main():
    window = arcade.Window(SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_TITLE)
    game_view = GameView()
    game_view.setup()
    window.show_view(game_view)
    arcade.run()


if __name__ == "__main__":
    main()
```

---

## 6. Automated Unit Test Suite (`test_game.py`)
The coding agent must generate `test_game.py` and run `pytest` to verify entity boundaries, cooldown timers, projectile spawning, and collision/score logic without requiring manual human interaction.

```python
import arcade
import pytest
from main import (
    SCREEN_WIDTH,
    SCREEN_HEIGHT,
    PLAYER_SPEED,
    FIRE_COOLDOWN,
    Player,
    Enemy,
    GameView,
)


@pytest.fixture
def game_view():
    """Fixture that initializes GameView in headless/offscreen state for unit testing."""
    view = GameView()
    view.setup()
    return view


def test_initial_setup(game_view):
    """Verify initial game state after setup()."""
    assert len(game_view.player_list) == 1
    assert len(game_view.enemy_list) == 6
    assert len(game_view.bullet_list) == 0
    assert game_view.score == 0
    assert game_view.player.center_x == SCREEN_WIDTH // 2


def test_player_movement_and_bounds():
    """Verify player moves properly and clamps within screen boundaries."""
    player = Player()

    # Test moving left past the boundary
    player.center_x = 35
    player.change_x = -PLAYER_SPEED * 5
    player.on_update(1 / 60)
    assert player.center_x == 30  # Clamped at left minimum

    # Test moving right past the boundary
    player.center_x = SCREEN_WIDTH - 35
    player.change_x = PLAYER_SPEED * 5
    player.on_update(1 / 60)
    assert player.center_x == SCREEN_WIDTH - 30  # Clamped at right maximum


def test_key_set_simultaneous_input(game_view):
    """Verify holding both LEFT and RIGHT cancels movement, and releasing one resumes."""
    game_view.on_key_press(arcade.key.LEFT, 0)
    game_view.on_key_press(arcade.key.RIGHT, 0)
    game_view.on_update(1 / 60)
    assert game_view.player.change_x == 0

    # Release LEFT while still holding RIGHT
    game_view.on_key_release(arcade.key.LEFT, 0)
    game_view.on_update(1 / 60)
    assert game_view.player.change_x == PLAYER_SPEED


def test_shooting_and_cooldown(game_view):
    """Verify shooting spawns a bullet, sets cooldown, and respects delta_time."""
    game_view.shoot(play_audio=False)
    assert len(game_view.bullet_list) == 1
    assert game_view.player.shoot_timer == pytest.approx(FIRE_COOLDOWN)

    # Advance half the cooldown time; player should not fire again yet
    game_view.on_key_press(arcade.key.SPACE, 0)
    game_view.on_update(FIRE_COOLDOWN / 2)
    assert len(game_view.bullet_list) == 1

    # Advance past the remaining cooldown time; a second bullet should spawn
    game_view.on_update(FIRE_COOLDOWN)
    assert len(game_view.bullet_list) == 2


def test_bullet_enemy_collision_and_scoring(game_view):
    """Verify bullet hitting an enemy removes both and increments score."""
    target_enemy = game_view.enemy_list[0]
    initial_enemy_count = len(game_view.enemy_list)

    # Spawn a bullet and position it directly on top of the first enemy
    game_view.shoot(play_audio=False)
    bullet = game_view.bullet_list[0]
    bullet.center_x = target_enemy.center_x
    bullet.center_y = target_enemy.center_y

    # Run update to trigger collision check
    game_view.on_update(1 / 60)

    assert len(game_view.bullet_list) == 0
    assert len(game_view.enemy_list) == initial_enemy_count - 1
    assert game_view.score == 1


def test_offscreen_bullet_cleanup(game_view):
    """Verify bullets that travel above SCREEN_HEIGHT are garbage-collected."""
    game_view.shoot(play_audio=False)
    bullet = game_view.bullet_list[0]
    bullet.bottom = SCREEN_HEIGHT + 10

    game_view.on_update(1 / 60)
    assert len(game_view.bullet_list) == 0
```

---

## 7. End-to-End (E2E) Verification & Test Plan
After implementing `main.py` and `test_game.py`, execute the following three-stage verification plan to ensure the end-to-end result works properly on Ubuntu Linux.

### Stage 1: Automated Unit Tests
Run `pytest` from the terminal:
```bash
pytest -v
```
* **Pass Criteria:** All 6 unit tests (`test_initial_setup`, `test_player_movement_and_bounds`, `test_key_set_simultaneous_input`, `test_shooting_and_cooldown`, `test_bullet_enemy_collision_and_scoring`, `test_offscreen_bullet_cleanup`) pass with `0` errors.

### Stage 2: Smoke & Window Launch Test
Run the game on the Ubuntu desktop session:
```bash
python3 main.py
```
* **Pass Criteria:**
  1. An $800 \times 600$ window titled `"Extensible 2D Shooter MVP"` opens smoothly with a dark midnight blue background.
  2. The orange player ship appears centered at the bottom ($x = 400, y = 50$).
  3. Six green enemy ships appear in a top row facing downward.
  4. `"Score: 0"` is rendered cleanly in the bottom-left corner.

### Stage 3: Interactive Gameplay Verification Checklist
Perform the following manual checks during the E2E run:

| Test Case | User Action | Expected Visual & Audio Result |
| :--- | :--- | :--- |
| **TC-01: Smooth Direction Swap** | Hold `A` (move left), then hold `D` while still holding `A`, then release `D`. | Ship moves left, stops while both are held, and immediately resumes moving left when `D` is released (no stuttering). |
| **TC-02: Screen Clamping** | Hold `A` for 5 seconds, then hold `D` for 5 seconds. | Player ship stops cleanly at the left and right edges without disappearing off-screen. |
| **TC-03: Continuous Firing** | Hold down `SPACE`. | Blue laser bolts fire upward at a steady interval ($4\text{ shots/sec}$) accompanied by the laser sound effect. |
| **TC-04: Enemy Destruction & Score** | Align ship under each green enemy and fire. | Upon impact, both the laser and the enemy disappear, and the bottom-left score counter increments from `0` up to `6`. |
| **TC-05: Instant Reset** | Press `R` after destroying several enemies. | All 6 enemies respawn in their starting positions, active bullets clear, player recenters, and `Score` resets to `0`. |

---

## 8. Future Extension Roadmap (Post-MVP)
Once Stage 1–3 pass, the coding agent can extend the game incrementally in future iterations without modifying the core architecture:
1. **Enemy Movement & Wave Spawning:** Override `Enemy.on_update(self, delta_time)` to move enemies downward and use `arcade.schedule()` in `GameView.setup()` to spawn new waves periodically.
2. **Start & Game Over Views:** Add `MenuView(arcade.View)` and `GameOverView(arcade.View)` classes and transition via `self.window.show_view()`.
3. **Multi-File Split:** Once `main.py` exceeds 250–300 lines, move `Player` and `Enemy` into `sprites.py` and constants into `constants.py`.