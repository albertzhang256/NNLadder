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

